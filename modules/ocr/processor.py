import numpy as np
import os
import re
from typing import Any

from ..utils.textblock import TextBlock
from ..utils.language_utils import language_codes
from .factory import OCRFactory


class OCRProcessor:
    """
    Processor for OCR operations using various engines.
    
    Uses a factory pattern to create and utilize the appropriate OCR engine
    based on settings and language.
    """
    
    def __init__(self):
        self.main_page = None
        self.settings = None
        self.source_lang = None
        self.source_lang_english = None
        
    def initialize(self, main_page: Any, source_lang: str) -> None:
        """
        Initialize the OCR processor with settings and language.
        
        Args:
            main_page: The main application page with settings
            source_lang: The source language for OCR
        """
        self.main_page = main_page
        self.settings = main_page.settings_page
        self.source_lang = source_lang
        self.source_lang_english = self._get_english_lang(source_lang)
        self.ocr_key = self._get_ocr_key(self.settings.get_tool_selection('ocr'))
        
    def _get_english_lang(self, translated_lang: str) -> str:
        return self.main_page.lang_mapping.get(translated_lang, translated_lang)

    def process(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Process image with appropriate OCR engine.
        
        Args:
            img: Input image as numpy array
            blk_list: List of TextBlock objects to update with OCR text
            
        Returns:
            Updated list of TextBlock objects with recognized text
        """

        self._set_source_language(blk_list)
        engine = OCRFactory.create_engine(self.settings, self.source_lang_english, self.ocr_key)
        blk_list = engine.process_image(img, blk_list)
        self._recover_empty_bubbles_with_local_recognition(img, blk_list, engine)
        return blk_list

    def _recover_empty_bubbles_with_local_recognition(self, img: np.ndarray, blk_list: list[TextBlock], engine: Any) -> None:
        if os.environ.get("LOCAL_OCR_EMPTY_BUBBLE_FALLBACK", "true").lower() != "true":
            return
        if not hasattr(engine, "_rec_infer"):
            return

        candidates = []
        crops = []
        pad = int(os.environ.get("LOCAL_OCR_EMPTY_PAD", "8"))
        height, width = img.shape[:2]
        for blk in blk_list:
            if getattr(blk, "text_class", None) != "text_bubble":
                continue
            if (getattr(blk, "text", "") or "").strip():
                continue
            x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(width, x2 + pad)
            y2 = min(height, y2 + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            candidates.append(blk)
            crops.append(crop)

        if not crops:
            return

        try:
            texts, confidences = engine._rec_infer(crops)
        except Exception as exc:
            print(f"Local OCR empty-bubble fallback skipped: {exc}", flush=True)
            return

        min_confidence = float(os.environ.get("LOCAL_OCR_EMPTY_MIN_CONF", "0.35"))
        for blk, text, confidence in zip(candidates, texts, confidences):
            cleaned = self._normalize_local_ocr_text(text)
            if cleaned and confidence >= min_confidence:
                blk.text = cleaned

    @staticmethod
    def _normalize_local_ocr_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\.{4,}", "...", text)
        text = re.sub(r"\s+([.,!?…])", r"\1", text)
        return text
            
    def _set_source_language(self, blk_list: list[TextBlock]) -> None:
        source_lang_code = language_codes.get(self.source_lang_english, 'en')
        for blk in blk_list:
            blk.source_lang = source_lang_code

    def _get_ocr_key(self, localized_ocr: str) -> str:
        translator_map = {
            self.settings.ui.tr('GPT-4.1-mini'): 'GPT-4.1-mini',
            self.settings.ui.tr('GPT-5.4-mini'): 'GPT-5.4-mini',
            self.settings.ui.tr('Microsoft OCR'): 'Microsoft OCR',
            self.settings.ui.tr('Google Cloud Vision'): 'Google Cloud Vision',
            self.settings.ui.tr('Gemini-2.0-Flash'): 'Gemini-2.0-Flash',
            self.settings.ui.tr('Default'): 'Default',
        }
        return translator_map.get(localized_ocr, localized_ocr)
