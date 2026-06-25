import json
import hashlib

from .base import OCREngine
from .qwen_grid_ocr import QwenGridOCR


class OCRFactory:
    _engines = {}

    @classmethod
    def create_engine(cls, settings, source_lang_english: str, ocr_model: str, backend=None) -> OCREngine:
        cache_key = cls._create_cache_key(ocr_model, source_lang_english, settings)
        if cache_key in cls._engines:
            return cls._engines[cache_key]
        engine = cls._create_new_engine(settings, ocr_model)
        cls._engines[cache_key] = engine
        return engine

    @classmethod
    def _create_cache_key(cls, ocr_key: str, source_lang: str, settings) -> str:
        base = f"{ocr_key}_{source_lang}"
        creds = settings.get_credentials(ocr_key)
        if not creds:
            return base
        extras_json = json.dumps(creds, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(extras_json.encode()).hexdigest()
        return f"{base}_{digest}"

    @classmethod
    def _create_new_engine(cls, settings, ocr_model: str) -> OCREngine:
        qwen_models = {
            'Qwen3-VL-Flash Grid OCR': 'qwen3-vl-flash',
            'Qwen3-VL-Plus Grid OCR': 'qwen3-vl-plus',
            'Qwen3.6-Flash Grid OCR': 'qwen3.6-flash',
            'Qwen3.6-Plus Grid OCR': 'qwen3.6-plus',
        }
        if ocr_model in qwen_models:
            credentials = settings.get_credentials("Qwen")
            engine = QwenGridOCR()
            engine.initialize(api_key=credentials.get("api_key", ""), model=qwen_models[ocr_model])
            return engine
        raise ValueError(f"Unsupported OCR model: {ocr_model!r}. Supported: {list(qwen_models)}")
