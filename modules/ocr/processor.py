import numpy as np
import os
import re
from typing import Any

from ..utils.textblock import TextBlock
from ..utils.language_utils import language_codes
from .factory import OCRFactory


class OCRProcessor:
    def __init__(self):
        self.main_page = None
        self.settings = None
        self.source_lang = None
        self.source_lang_english = None

    def initialize(self, main_page: Any, source_lang: str) -> None:
        self.main_page = main_page
        self.settings = main_page.settings_page
        self.source_lang = source_lang
        self.source_lang_english = source_lang
        self.ocr_key = self.settings.get_tool_selection('ocr')

    def process(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        self._set_source_language(blk_list)
        engine = OCRFactory.create_engine(self.settings, self.source_lang_english, self.ocr_key)
        blk_list = engine.process_image(img, blk_list)
        return blk_list

    def _set_source_language(self, blk_list: list[TextBlock]) -> None:
        source_lang_code = language_codes.get(self.source_lang_english, 'en')
        for blk in blk_list:
            blk.source_lang = source_lang_code
