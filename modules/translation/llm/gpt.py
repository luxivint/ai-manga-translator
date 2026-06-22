from typing import Any
import os
import numpy as np
import requests
import json
import uuid

from .base import BaseLLMTranslation
from ...utils.translator_utils import MODEL_MAP


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _price_for_model(model: str) -> tuple[float, float, float]:
    model_key = (model or "").lower()
    if os.environ.get("OPENAI_INPUT_PRICE_PER_1M") or os.environ.get("OPENAI_OUTPUT_PRICE_PER_1M"):
        return (
            _float_env("OPENAI_INPUT_PRICE_PER_1M", 0.75),
            _float_env("OPENAI_OUTPUT_PRICE_PER_1M", 4.50),
            _float_env("OPENAI_CACHED_INPUT_PRICE_PER_1M", 0.075),
        )
    if "5.4-mini" in model_key:
        return 0.75, 4.50, 0.075
    if "5.1-mini" in model_key or "4.1-mini" in model_key:
        return 0.40, 1.60, 0.10
    return 0.75, 4.50, 0.075


def _usage_int(usage: dict, key: str) -> int:
    try:
        return int(usage.get(key) or 0)
    except (TypeError, ValueError):
        return 0


def _estimate_cost_usd(model: str, usage: dict) -> tuple[int, int, int, int, float]:
    prompt_tokens = _usage_int(usage, "prompt_tokens")
    completion_tokens = _usage_int(usage, "completion_tokens")
    total_tokens = _usage_int(usage, "total_tokens") or prompt_tokens + completion_tokens
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    cached_tokens = _usage_int(details, "cached_tokens")
    input_price, output_price, cached_price = _price_for_model(model)
    billed_input_tokens = max(0, prompt_tokens - cached_tokens)
    cost = (
        billed_input_tokens * input_price
        + cached_tokens * cached_price
        + completion_tokens * output_price
    ) / 1_000_000
    return prompt_tokens, completion_tokens, cached_tokens, total_tokens, cost


def _ensure_usage_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS openai_usage_events (
              id text PRIMARY KEY,
              provider text NOT NULL DEFAULT 'openai',
              model text NOT NULL,
              operation text NOT NULL DEFAULT 'translation',
              manga_id text,
              chapter_id text,
              pipeline_job_id text,
              source_file text,
              prompt_tokens integer NOT NULL DEFAULT 0,
              completion_tokens integer NOT NULL DEFAULT 0,
              cached_prompt_tokens integer NOT NULL DEFAULT 0,
              total_tokens integer NOT NULL DEFAULT 0,
              estimated_cost_usd numeric(12, 6) NOT NULL DEFAULT 0,
              metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
              created_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )


def _log_openai_usage(model: str, usage: dict) -> None:
    if not usage or os.environ.get("LOG_OPENAI_USAGE", "true").lower() != "true":
        return
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        return
    try:
        import psycopg2
        import psycopg2.extras

        prompt, completion, cached, total, cost = _estimate_cost_usd(model, usage)
        metadata = {
            "source": os.environ.get("OPENAI_USAGE_SOURCE", "manga-translation-worker"),
            "raw_prefix": os.environ.get("TRANSLATION_RAW_PREFIX", ""),
            "output_prefix": os.environ.get("TRANSLATION_OUTPUT_PREFIX", ""),
            "chapter_number": os.environ.get("TRANSLATION_CHAPTER_NUMBER", ""),
            "page_index": os.environ.get("TRANSLATION_PAGE_INDEX", ""),
            "raw_usage": usage,
        }
        with psycopg2.connect(database_url) as conn:
            _ensure_usage_table(conn)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO openai_usage_events (
                      id, provider, model, operation, manga_id, chapter_id, pipeline_job_id,
                      source_file, prompt_tokens, completion_tokens, cached_prompt_tokens,
                      total_tokens, estimated_cost_usd, metadata, created_at
                    )
                    VALUES (%s, 'openai', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, now())
                    """,
                    [
                        f"ou_{uuid.uuid4().hex}",
                        model,
                        os.environ.get("OPENAI_USAGE_OPERATION", "translation"),
                        os.environ.get("TRANSLATION_MANGA_ID") or None,
                        os.environ.get("TRANSLATION_CHAPTER_ID") or None,
                        os.environ.get("PIPELINE_JOB_ID") or None,
                        os.environ.get("TRANSLATION_SOURCE_FILE") or None,
                        prompt,
                        completion,
                        cached,
                        total,
                        round(cost, 6),
                        json.dumps(metadata, ensure_ascii=False),
                    ],
                )
    except Exception as exc:
        if os.environ.get("OPENAI_USAGE_DEBUG", "false").lower() == "true":
            print(f"openai usage log skipped: {exc}", flush=True)


class GPTTranslation(BaseLLMTranslation):
    """Translation engine using OpenAI GPT models through direct REST API calls."""
    
    def __init__(self):
        super().__init__()
        self.model_name = None
        self.api_key = None
        self.api_base_url = "https://api.openai.com/v1"
        self.supports_images = True
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, model_name: str, **kwargs) -> None:
        """
        Initialize GPT translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            model_name: GPT model name
        """
        super().initialize(settings, source_lang, target_lang, **kwargs)
        
        self.model_name = model_name
        credentials = settings.get_credentials(settings.ui.tr('Open AI GPT'))
        self.api_key = credentials.get('api_key', '') or os.environ.get("OPENAI_API_KEY", "")
        self.model = MODEL_MAP.get(self.model_name)

        if not self.api_key:
            raise RuntimeError(
                "OpenAI API key is missing. Add it in Settings > Advanced > "
                "Open AI GPT, or set the OPENAI_API_KEY environment variable."
            )
    
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using direct REST API calls to OpenAI.
        
        Args:
            user_prompt: Text prompt from user
            system_prompt: System instructions
            image: Image as numpy array
            
        Returns:
            Translated text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.supports_images and self.img_as_llm_input:
            # Use the base class method to encode the image
            encoded_image, mime_type = self.encode_image(image)
            
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "system", 
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": user_prompt}]
                }
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "top_p": self.top_p,
        }

        return self._make_api_request(payload, headers)
    
    def _make_api_request(self, payload, headers):
        """
        Make API request and process response
        """
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            response.raise_for_status()
            response_data = response.json()
            _log_openai_usage(str(payload.get("model") or self.model or self.model_name or ""), response_data.get("usage") or {})
            
            return response_data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f" - {json.dumps(error_details)}"
                except:
                    error_msg += f" - Status code: {e.response.status_code}"
            raise RuntimeError(error_msg)
