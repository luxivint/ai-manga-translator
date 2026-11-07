from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from modules.utils.textblock import TextBlock, adjust_text_line_coordinates
from modules.utils.openai_endpoint import get_openai_chat_completions_url, get_openai_model

from modules.utils import gateway
from modules.utils.gateway import GatewayError, OCRFailedError

from .base import OCREngine


class QwenGridOCR(OCREngine):
    """Qwen visual OCR via one ID-labelled contact-sheet request."""

    _usage_events: list[dict[str, Any]] = []

    def __init__(self):
        self.api_key = ""
        self.model = "qwen3-vl-flash"
        self.api_base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.max_tokens = 2500
        self.max_tokens_parameter = "max_tokens"
        self.input_price_per_1m = 0.10
        self.output_price_per_1m = 0.40

    def initialize(self, api_key: str = "", model: str = "qwen3-vl-flash", **kwargs) -> None:
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.model = model
        self.api_base_url = os.environ.get("DASHSCOPE_BASE_URL", self.api_base_url).rstrip("/") + "/chat/completions"
        if self.api_base_url.endswith("/v1/chat/completions/chat/completions"):
            self.api_base_url = self.api_base_url.replace("/chat/completions/chat/completions", "/chat/completions")
        self.max_tokens = int(kwargs.get("max_tokens", os.environ.get("QWEN_GRID_OCR_MAX_TOKENS", "2500")))
        self.input_price_per_1m = float(os.environ.get("QWEN_INPUT_PRICE_PER_1M", "0.10"))
        self.output_price_per_1m = float(os.environ.get("QWEN_OUTPUT_PRICE_PER_1M", "0.40"))

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        if not blk_list:
            return blk_list
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY is missing for QwenGridOCR.")

        sheet, id_to_idx = self._make_sheet(img, blk_list)
        if sheet is None:
            return blk_list

        prompt = (
            "You are an OCR engine for comic/webtoon text crops. "
            "The image is a contact sheet. Each crop has a yellow ID label like ID 1, ID 2, etc. "
            "Read ONLY the comic text inside each crop, not the yellow ID label. Do NOT translate. "
            "Only return English/Latin alphabet dialogue, narration, and signs. "
            "If the crop is Korean, Japanese, Chinese, raw SFX, watermark text, or has no readable English text, return an empty string. "
            "Preserve the original wording as much as possible. Use spaces instead of line breaks. "
            "If a crop has no readable text, return an empty string for it. "
            "For EVERY ID also transcribe original_text VERBATIM in the original writing system, "
            "and classify script (latin, hangul, han, kana, mixed, none) and kind "
            "(dialogue, narration, sign, sfx, watermark, other). "
            "Do not romanize or guess English words from Korean shapes. Large decorative effects are sfx, "
            "even if a shape resembles 'OF' or 'IN'. Ignore neighbouring captions crossing the crop edge. "
            'Return strict JSON only: [{"id":1,"original_text":"...","script":"latin",'
            '"kind":"dialogue","text":"..."}]. Non-Latin and sfx crops must have text="".'
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{self.encode_image(np.asarray(sheet), 'jpg')}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            self.max_tokens_parameter: self.max_tokens,
            "temperature": 0,
        }

        max_attempts = max(1, int(os.environ.get("QWEN_GRID_OCR_RETRIES", "2")))
        parsed: list[dict[str, Any]] = []
        data: dict[str, Any] = {}
        seconds = 0.0
        content = ""
        for attempt in range(1, max_attempts + 1):
            started = time.time()
            # Transport resilience (retry/backoff, circuit breaker, OCR_CONCURRENCY slot) lives in gateway.
            # A failed call raises OCRFailedError; it is never turned into "no text on this page".
            try:
                response = gateway.resilient_post(
                    self.api_base_url,
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                    data=json.dumps(payload),
                    timeout=int(os.environ.get("QWEN_GRID_OCR_TIMEOUT", "90")),
                    kind="ocr",
                    limiter=gateway.ocr_limiter(),
                    cache="manual",  # identical crop sheet + prompt -> stored answer, no new request
                )
            except GatewayError as exc:
                print(f"Qwen grid OCR failed: {exc}", flush=True)
                raise OCRFailedError(str(exc)) from exc
            seconds = time.time() - started
            if response.status_code != 200:
                print(f"Qwen grid OCR API error: {response.status_code} {response.text[:300]}", flush=True)
                raise OCRFailedError(f"OCR API error {response.status_code}")

            data = response.json()
            content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
            parsed = self._parse_json(content)
            if not self._looks_like_failed_batch(parsed, len(id_to_idx), set(id_to_idx)):
                gateway.cache_commit()
                break
            gateway.cache_discard()  # a suspicious answer must never be replayed from the cache
            if attempt < max_attempts:
                print(
                    f"Qwen grid OCR retry {attempt}/{max_attempts}: suspicious empty result "
                    f"for {len(id_to_idx)} crops",
                    flush=True,
                )

        failed = self._looks_like_failed_batch(parsed, len(id_to_idx), set(id_to_idx))
        if failed and not self._looks_like_failed_batch(
            parsed, len(id_to_idx), set(id_to_idx), check_empty_ratio=False
        ):
            # A structurally COMPLETE sheet can legitimately consist of SFX,
            # Korean text or decoration. Verify its empty crops independently;
            # do not drop missing IDs or trust a suspicious all-empty sheet.
            print(f"Qwen grid OCR: verifying blank crops individually ({len(id_to_idx)} total)", flush=True)
            for item in parsed:
                if not str(item['text'] or '').strip() and not self._preserve_source(item):
                    block = blk_list[id_to_idx[int(item['id'])]].deep_copy()
                    self.process_image(img, [block])
                    item['text'] = block.text
                    if getattr(block, 'ocr_source_verified', False):
                        item.update(script=block.source_script, kind=block.source_kind,
                                    original_text=block.original_text)
            failed = False

        if failed:
            # A missing crop ID is not an empty string: it is lost dialogue.
            # Fail the page (and retain its raw source) instead of silently
            # publishing a partly untranslated or misaligned double balloon.
            gateway.cache_discard()
            seen = {int(item["id"]) for item in parsed if isinstance(item, dict)
                    and str(item.get("id", "")).isdigit()}
            missing = sorted(set(id_to_idx) - seen)
            raise OCRFailedError(
                f"OCR incomplete after {max_attempts} attempt(s): "
                f"missing crop IDs {missing}, expected {sorted(id_to_idx)}"
            )

        # Old cache entries without provenance must not bypass source-script
        # validation on free lettering. Normal legacy dialogue stays compatible.
        missing = [item for item in parsed if self._needs_source_verification(
            item, blk_list[id_to_idx[int(item['id'])]])]
        if missing:
            self._verify_source_metadata(img, blk_list, id_to_idx, missing)

        for item in parsed:
            try:
                crop_id = int(item.get("id"))
            except Exception:
                continue
            idx = id_to_idx.get(crop_id)
            if idx is not None and 0 <= idx < len(blk_list):
                block = blk_list[idx]
                metadata = self._source_metadata(item)
                block.ocr_source_verified = metadata is not None
                if metadata:
                    block.source_script, block.source_kind, block.original_text = metadata
                block.preserve_original = self._preserve_source(item)
                block.text = '' if block.preserve_original else self._normalize_text(str(item.get("text") or ""))
                if block.preserve_original:
                    block.translation = ''
                    print(f'OCR source kept: id={crop_id} script={block.source_script} kind={block.source_kind}', flush=True)

        self._log_usage(data.get("usage") or {}, seconds, len(blk_list))
        return blk_list

    @staticmethod
    def _source_metadata(item):
        script, kind, original = item.get('script'), item.get('kind'), item.get('original_text')
        if script not in {'latin', 'hangul', 'han', 'kana', 'mixed', 'none'} or \
                kind not in {'dialogue', 'narration', 'sign', 'sfx', 'watermark', 'other'} or not isinstance(original, str):
            return None
        return script, kind, original

    @classmethod
    def _preserve_source(cls, item):
        metadata = cls._source_metadata(item)
        if metadata is None:
            return False
        non_latin = bool(re.search(r'[\u1100-\u11ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', metadata[2]))
        return non_latin or metadata[0] != 'latin' or metadata[1] in {'sfx', 'watermark', 'other'}

    @classmethod
    def _needs_source_verification(cls, item, block):
        if cls._source_metadata(item) is None:
            return True
        if cls._preserve_source(item) or getattr(block, 'text_class', '') != 'text_free':
            return False
        text = str(item.get('text') or '').strip()
        box = block.xyxy
        # A huge isolated 'OF'/'IN' or a couple of words over art is a common
        # transliteration hallucination. Verify the original independently;
        # genuine Latin titles/signs remain eligible after that check.
        return bool(text and len(text.split()) <= 4 and float(box[3])-float(box[1]) >= 96)

    def _verify_source_metadata(self, image, blocks, id_to_idx, items):
        selected = [blocks[id_to_idx[int(item['id'])]] for item in items]
        sheet, _ = self._make_sheet(image, selected)
        prompt = ('Identify the ORIGINAL writing system and role in EACH ID-labelled crop. '
                  'Transcribe original_text VERBATIM, including Hangul; NEVER translate or romanize. '
                  'Korean effects can resemble OF or IN: do not interpret those glyphs as English. '
                  'Report script: latin/hangul/han/kana/mixed/none and '
                  'kind: dialogue/narration/sign/sfx/watermark/other. '
                  'Return one JSON object per ID: [{"id":1,"script":"hangul","kind":"sfx","original_text":"..."}].')
        payload = {'model': self.model, 'temperature': 0,
            self.max_tokens_parameter: self.max_tokens,
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + self.encode_image(np.asarray(sheet), 'jpg')}},
                {'type': 'text', 'text': prompt}]}]}
        for attempt in range(2):
            try:
                response = gateway.resilient_post(self.api_base_url,
                    headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'},
                    data=json.dumps(payload), timeout=90, kind='ocr-source', limiter=gateway.ocr_limiter(), cache='manual')
                response.raise_for_status()
                data = response.json()
                verified = self._parse_json(data['choices'][0]['message']['content'])
                ids = [int(item['id']) for item in verified]
                if len(ids) != len(items) or set(ids) != set(range(1, len(items)+1)) or \
                        any(self._source_metadata(item) is None for item in verified):
                    raise ValueError('incomplete source metadata')
                by_id = {int(item['id']): item for item in verified}
                for index, item in enumerate(items, 1):
                    item.update({key: by_id[index][key] for key in ('script', 'kind', 'original_text')})
                    if not self._preserve_source(item):
                        if not item['original_text'].strip():
                            raise ValueError('Latin text has no original transcription')
                        item['text'] = item['original_text']
                gateway.cache_commit()
                return
            except (GatewayError, requests.RequestException, ValueError, KeyError, TypeError, IndexError):
                gateway.cache_discard()
        raise OCRFailedError('Original script/role could not be verified; source graphics kept')

    @staticmethod
    def _is_valid_empty_answer(content: str) -> bool:
        """True when the model really answered with an empty list ([] / {"items": []})."""
        return bool(re.search(r"\[\s*\]", content or ""))

    @staticmethod
    def _looks_like_failed_batch(
        parsed: list[dict[str, Any]],
        crop_count: int,
        expected_ids: set[int] | None = None,
        *, check_empty_ratio: bool = True,
    ) -> bool:
        """Reject missing, repeated or malformed crop IDs, even for two balloons.

        Explicit {"id": N, "text": ""} is accepted for an unreadable crop,
        while a crop omitted from the JSON is never interpreted as blank.
        """
        if not parsed:
            return crop_count > 0
        expected = expected_ids if expected_ids is not None else set(range(1, crop_count + 1))
        found = set()
        empty = 0
        for item in parsed:
            if not isinstance(item, dict) or "id" not in item or "text" not in item:
                return True
            try:
                crop_id = int(item["id"])
            except (ValueError, TypeError):
                return True
            if crop_id not in expected or crop_id in found:
                return True
            found.add(crop_id)
            if not str(item["text"] or "").strip() and not QwenGridOCR._preserve_source(item):
                empty += 1
        if found != expected:
            return True
        # Large sheets with almost exclusively empty reads may be gateway
        # failures; small sheets can legitimately contain an empty bubble.
        return check_empty_ratio and crop_count > 2 and empty / max(1, crop_count) >= float(
            os.environ.get("QWEN_GRID_OCR_RETRY_EMPTY_RATIO", "0.85")
        )

    def _make_sheet(self, img: np.ndarray, blk_list: list[TextBlock]) -> tuple[Image.Image | None, dict[int, int]]:
        source = Image.fromarray(img).convert("RGB")
        max_cell_w = int(os.environ.get("QWEN_GRID_OCR_CELL_WIDTH", "360"))
        pad = int(os.environ.get("QWEN_GRID_OCR_CROP_PAD", "12"))
        cols = int(os.environ.get("QWEN_GRID_OCR_COLS", "2"))
        gap = 16
        header_h = 34
        cells: list[tuple[int, int, Image.Image]] = []
        id_to_idx: dict[int, int] = {}
        width, height = source.size

        for idx, blk in enumerate(blk_list):
            xyxy = adjust_text_line_coordinates(blk.xyxy, 12, 20, img)
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = source.crop((x1, y1, x2, y2))
            if crop.width > max_cell_w:
                ratio = max_cell_w / crop.width
                crop = crop.resize((max_cell_w, max(1, int(crop.height * ratio))), Image.LANCZOS)
            crop_id = len(cells) + 1
            id_to_idx[crop_id] = idx
            cell = Image.new("RGB", (max(crop.width, 130), crop.height + header_h), "white")
            cell.paste(crop, (0, header_h))
            draw = ImageDraw.Draw(cell)
            draw.rectangle((0, 0, cell.width, header_h), fill=(255, 218, 0))
            draw.text((8, 7), f"ID {crop_id}", fill=(0, 0, 0), font=self._font(22))
            cells.append((cell.width, cell.height, cell))

        if not cells:
            return None, {}

        rows = (len(cells) + cols - 1) // cols
        col_w = [0] * cols
        row_h = [0] * rows
        for i, (w, h, _cell) in enumerate(cells):
            c = i % cols
            r = i // cols
            col_w[c] = max(col_w[c], w)
            row_h[r] = max(row_h[r], h)
        sheet = Image.new("RGB", (sum(col_w) + gap * (cols + 1), sum(row_h) + gap * (rows + 1)), (32, 32, 32))
        y = gap
        for r in range(rows):
            x = gap
            for c in range(cols):
                i = r * cols + c
                if i >= len(cells):
                    break
                _w, _h, cell = cells[i]
                sheet.paste(cell, (x, y))
                x += col_w[c] + gap
            y += row_h[r] + gap
        return sheet, id_to_idx

    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        for path in [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf"]:
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    def _log_usage(self, usage: dict[str, Any], seconds: float, block_count: int) -> None:
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or prompt_tokens + completion_tokens)
        cost = (
            prompt_tokens * self.input_price_per_1m / 1_000_000
            + completion_tokens * self.output_price_per_1m / 1_000_000
        )
        event = {
            "model": self.model,
            "blocks": block_count,
            "seconds": round(seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 8),
        }
        self._usage_events.append(event)
        path = os.environ.get("QWEN_GRID_OCR_USAGE_PATH")
        if path:
            aggregate = {
                "events": self._usage_events,
                "totals": {
                    "calls": len(self._usage_events),
                    "blocks": sum(e["blocks"] for e in self._usage_events),
                    "seconds": round(sum(e["seconds"] for e in self._usage_events), 3),
                    "prompt_tokens": sum(e["prompt_tokens"] for e in self._usage_events),
                    "completion_tokens": sum(e["completion_tokens"] for e in self._usage_events),
                    "total_tokens": sum(e["total_tokens"] for e in self._usage_events),
                    "estimated_cost_usd": round(sum(e["estimated_cost_usd"] for e in self._usage_events), 8),
                },
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(aggregate, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _parse_json(text: str) -> list[dict[str, Any]]:
        text = text.strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
        if fence:
            text = fence.group(1).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end == -1 or end <= start:
                return []
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return []
        if isinstance(parsed, dict):
            parsed = parsed.get("items") or parsed.get("blocks") or []
        return parsed if isinstance(parsed, list) else []

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()


class OpenAICompatibleGridOCR(QwenGridOCR):
    """Grid OCR routed through the configured OpenAI-compatible endpoint."""

    def initialize(self, api_key: str = "", model: str = "", **kwargs) -> None:
        self.max_tokens_parameter = "max_completion_tokens"
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = get_openai_model(model)
        self.api_base_url = get_openai_chat_completions_url()
        self.max_tokens = int(kwargs.get("max_tokens", os.environ.get("QWEN_GRID_OCR_MAX_TOKENS", "2500")))
        # Do not report an invented per-token cost for an arbitrary gateway model.
        self.input_price_per_1m = float(os.environ.get("OPENAI_INPUT_PRICE_PER_1M", "0"))
        self.output_price_per_1m = float(os.environ.get("OPENAI_OUTPUT_PRICE_PER_1M", "0"))
