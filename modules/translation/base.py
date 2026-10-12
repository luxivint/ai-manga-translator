import os
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

from ..utils.textblock import TextBlock
from ..utils.language_utils import is_no_space_lang


class TranslationEngine(ABC):
    """
    Abstract base class for all translation engines.
    Defines common interface and utility methods.
    """
    
    @abstractmethod
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the translation engine with necessary parameters.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        pass
    
    def get_language_code(self, language: str) -> str:
        """
        Get standardized language code from language name.
        
        Args:
            language: Language name
            
        Returns:
            Standardized language code
        """
        from ..utils.language_utils import get_language_code
        return get_language_code(language)
    
    def preprocess_text(self, blk_text: str, source_lang_code: str) -> str:
        """
        PreProcess text based on language:
        - Remove spaces for Chinese and Japanese languages
        - Remove all newline/carriage-return characters
        - Keep original text for other languages (aside from the newline removal)
        
        Args:
            blk_text (str): The input text to process
            source_lang_code (str): Language code of the source text
        
        Returns:
            str: Processed text
        """
        # Remove newline and carriage‐return characters
        text = blk_text.replace('\r', '').replace('\n', '')

        # 2) If No-Space Language, also remove all spaces
        if is_no_space_lang(source_lang_code):
            return text.replace(' ', '')
        # 3) Otherwise, return the text (with newlines already removed)
        else:
            return text


class TraditionalTranslation(TranslationEngine):
    """Base class for traditional translation engines (non-LLM)."""
    
    @abstractmethod
    def translate(self, blk_list: list[TextBlock]) -> list[TextBlock]:
        """
        Translate text blocks using non-LLM translators.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            
        Returns:
            List of updated TextBlock objects with translations
        """
        pass

    def preprocess_language_code(self, lang_code: str) -> str:
        """
        Preprocess language codes to match the specific translation API requirements.
        By default, returns the original language code.
        
        Args:
            lang_code: The language code to preprocess
            
        Returns:
            Preprocessed language code supported by the translation API
        """
        return lang_code  # Default implementation just returns the original code


class LLMTranslation(TranslationEngine):
    """Base class for LLM-based translation engines."""
    
    @abstractmethod
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects containing text to translate
            image: Image as numpy array (for context)
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        pass
    
    def get_system_prompt(self, source_lang: str, target_lang: str, ad_detection: bool = False, own_brand: str = "") -> str:
        """
        Get system prompt for LLM translation.
        
        Args:
            source_lang: Source language
            target_lang: Target language
            
        Returns:
            Formatted system prompt. ad_detection=True (batch path) appends the scanlation-ad rule.
        """
        turkish_rule = ""
        if target_lang and "turk" in target_lang.lower():
            turkish_rule = (
                "\n        IMPORTANT (Turkish output): Use ONLY the modern Turkish alphabet "
                "(a b c ç d e f g ğ h ı i j k l m n o ö p r s ş t u ü v y z). "
                "NEVER use circumflex/accented letters such as â, î, û (and Â, Î, Û). "
                "These are NOT in the render font and come out as blank/empty boxes. "
                "Always write the plain letter instead: â->a, î->i, û->u "
                "(e.g. 'kâğıt'->'kağıt', 'âlem'->'alem', 'hâlâ'->'hala', 'rüzgâr'->'rüzgar')."
            )

        style_rule = ""
        if os.environ.get("RENDER_STYLE_TAGS", "true").lower() == "true":
            style_rule = (
                "\n        STYLE TAGS: each value of the returned json must be an object "
                "{\"text\": \"<translation>\", \"style\": \"<tag>\"} instead of a plain string. "
                "<tag> is exactly one of: normal (ordinary speech), shout (yelling/emphasis), whisper, "
                "thought (inner monologue), narration (caption box), sfx (sound effect), sign (shop/board/"
                "banner/poster/graffiti text in the scene), phone (phone/computer/game UI, status window, "
                "chat message, system text), handwriting (notes/letters), title (chapter/series/section title)."
            )

        block_rules = (
            "\n        BLOCK RULES (very important):"
            "\n        - Translate EVERY block separately. NEVER merge two blocks into one, never move text from one "
            "block into another, never drop a block. Return exactly the same keys you were given."
            "\n        - Translate everything that is readable text: UI/status windows, board and sign titles, "
            "questions, chat lines, captions. Only these stay exactly as written: personal names, usernames, "
            "nicknames and handles (e.g. 'WILD GRYLLS'), URLs, codes and pure numbers/symbols. UI field labels such as "
            "'[NAME]' or 'LEVEL' ARE translated ('[NAME] - ADRIAN WHITE WOLF' -> '[İSİM] - ADRIAN WHITE WOLF')."
            "\n        - Proper nouns keep their capital letter (e.g. Shin, Ricky, Adrian White Wolf). Do not lowercase names."
            "\n        - The OCR often drops a lone 'I' (e.g. 'IF HAD TO SUM UP' or 'WHERE THE HELL AM , DAMMIT'). "
            "Restore the missing first-person subject so the Turkish keeps the right person."
            "\n        - The source is usually ALL CAPS because of comic lettering. Do NOT copy that: write natural "
            "Turkish with normal capitalisation (sentence case, capital after . ! ?). Only use ALL CAPS for "
            "shouts, sound effects, signs and titles. Use proper Turkish casing (İ/ı) everywhere."
            "\n        - If a block's text contains several lines/entries (status window, list, menu, stats), "
            "translate it entry by entry and put each entry on its own line separated by \\n, in the original order."
            "\n        - The OCR sometimes merges several unrelated pieces of text into one block (for example a menu "
            "bar, a question and a username). Look at the page image: translate EVERY piece, never omit any part "
            "(even a short opening phrase), and put each visually separate piece of text on its own line separated "
            "by \\n in reading order (top to bottom). Words that sit on the same row of the image (menu items) "
            "stay together on one line."
            "\n        - Keep the speaker's person and tone consistent between neighbouring blocks."
        )
        style_rule = block_rules + style_rule
        if ad_detection:
            style_rule += (
                "\n        AD DETECTION: Mark scanlation/fansub promo blocks as ads - do NOT translate them; "
                "output exactly the string \"[[AD]]\" as that block's text. A block is an AD if it contains ANY of: "
                "a website/domain name (e.g. something.com / .net / .org / .gg), a URL or invite/link code "
                "(discord.gg/xxx, /AbCdEf123), a QR/scan instruction, or it pitches reading chapters elsewhere "
                "('read/continue at...', 'premium/advance chapters', 'more at our site', 'join/support us'). "
                "THIS APPLIES EVEN IF the text is phrased like in-story dialogue - real comic dialogue NEVER "
                f"contains real-world website domains or invite links. '{own_brand or 'TrendManga'}' is our OWN brand; "
                "never mark our own brand as an ad."
            )

        return f"""You are an expert translator who translates {source_lang} to {target_lang}. You pay attention to style, formality, idioms, slang etc and try to convey it in the way a {target_lang} speaker would understand.
        BE MORE NATURAL. NEVER USE 당신, 그녀, 그 or its Japanese equivalents.
        Specifically, you will be translating text OCR'd from a comic. The OCR is not perfect and as such you may receive text with typos or other mistakes.
        To aid you and provide context, You may be given the image of the page and/or extra context about the comic. You will be given a json string of the detected text blocks and the text to translate. Return the json string with the texts translated. DO NOT translate the keys of the json. For each block:
        - If it's already in {target_lang} or looks like gibberish, OUTPUT IT AS IT IS instead
        - DO NOT give explanations{turkish_rule}{style_rule}
        Do Your Best! I'm really counting on you."""
    