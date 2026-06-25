import json
import hashlib

from .base import TranslationEngine
from .llm.gpt import GPTTranslation


class TranslationFactory:
    _engines = {}

    @classmethod
    def create_engine(cls, settings, source_lang: str, target_lang: str, translator_key: str) -> TranslationEngine:
        cache_key = cls._create_cache_key(translator_key, source_lang, target_lang, settings)
        if cache_key in cls._engines:
            return cls._engines[cache_key]
        engine = GPTTranslation()
        engine.initialize(settings, source_lang, target_lang, translator_key)
        cls._engines[cache_key] = engine
        return engine

    @classmethod
    def _create_cache_key(cls, translator_key: str, source_lang: str, target_lang: str, settings) -> str:
        base = f"{translator_key}_{source_lang}_{target_lang}"
        extras = {}
        creds = settings.get_credentials(translator_key)
        if creds:
            extras["credentials"] = creds
        llm_settings = settings.get_llm_settings()
        if llm_settings:
            extras["llm"] = llm_settings
        if not extras:
            return base
        extras_json = json.dumps(extras, sort_keys=True, separators=(",", ":"), default=str)
        digest = hashlib.sha256(extras_json.encode()).hexdigest()
        return f"{base}_{digest}"
