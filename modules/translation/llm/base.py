import os
import re
from typing import Any
import numpy as np
from abc import abstractmethod
import base64
import imkit as imk

from ..base import LLMTranslation
from ...utils.textblock import TextBlock
import json
from modules.utils import gateway
from ...utils.translator_utils import (
    apply_translation_json, get_raw_text, redistribute_lines, set_texts_from_json, translation_temperature,
)


class BaseLLMTranslation(LLMTranslation):
    """Base class for LLM-based translation engines with shared functionality."""
    
    def __init__(self):
        self.source_lang = None
        self.target_lang = None
        self.api_key = None
        self.api_url = None
        self.model = None
        self.img_as_llm_input = False
        self.temperature = None
        self.top_p = None
        self.max_tokens = None
        self.timeout = 30  
    
    def initialize(self, settings: Any, source_lang: str, target_lang: str, **kwargs) -> None:
        """
        Initialize the LLM translation engine.
        
        Args:
            settings: Settings object with credentials
            source_lang: Source language name
            target_lang: Target language name
            **kwargs: Engine-specific initialization parameters
        """
        llm_settings = settings.get_llm_settings()
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.img_as_llm_input = llm_settings.get('image_input_enabled', True)
        self.temperature = translation_temperature()
        self.top_p = 0.95
        self.max_tokens = 5000
        
    def translate(self, blk_list: list[TextBlock], image: np.ndarray, extra_context: str) -> list[TextBlock]:
        """
        Translate text blocks using LLM.
        
        Args:
            blk_list: List of TextBlock objects to translate
            image: Image as numpy array
            extra_context: Additional context information for translation
            
        Returns:
            List of updated TextBlock objects with translations
        """
        crops: list = []
        extra_context = f"{extra_context}{self.row_hint_context({f'block_{i}': b for i, b in enumerate(blk_list)}, image, crops)}"
        entire_raw_text = get_raw_text(blk_list)
        system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
        user_prompt = f"{extra_context}\nMake the translation sound as natural as possible.\nTranslate this:\n{entire_raw_text}"
        
        send = [image] + crops if (crops and self.supports_multi_image and image is not None) else image
        entire_translated_text = self._perform_translation(user_prompt, system_prompt, send)
        set_texts_from_json(blk_list, entire_translated_text)
        self.fix_line_counts(blk_list, image, extra_context)
            
        return blk_list

    # ------------------------------------------------------------------ shared by the batch path
    supports_multi_image = False

    def translate_keyed(self, raw: dict[str, str], images=None, extra_context: str = "",
                        ad_detection: bool = False, own_brand: str = "") -> dict:
        """Translate {key: text} with the SAME system prompt / user prompt / temperature as translate().

        Used by the batch pipeline (keys like 'p3_b7'). Returns the parsed JSON dict
        {key: str | {"text", "style"}}. `images`: page image(s) shown to the model (only sent when the
        engine supports it)."""
        system_prompt = self.get_system_prompt(self.source_lang, self.target_lang,
                                               ad_detection=ad_detection, own_brand=own_brand)
        user_prompt = (f"{extra_context}\nMake the translation sound as natural as possible.\n"
                       f"Translate this:\n{json.dumps(raw, ensure_ascii=False, indent=4)}")
        if images is None:
            image = None
        elif self.supports_multi_image:
            image = list(images)
        else:
            image = images[0] if images else None
        saved = self.max_tokens
        self.max_tokens = max(self.max_tokens or 0, 16000 if len(raw) > 30 else 8000, len(raw) * 300)
        try:
            content = self._perform_translation(user_prompt, system_prompt, image)
        finally:
            self.max_tokens = saved
        m = re.search(r"\{[\s\S]*\}", content or "")
        try:
            if not m:
                raise RuntimeError(f"no JSON in the translation response (len={len(content or '')})")
            parsed = json.loads(m.group(0))
        except Exception:
            gateway.cache_discard()  # a bad answer (also a cached one) is asked again next time
            raise
        gateway.cache_commit()
        return parsed

    # ---------------------------------------------------------------- line-count enforcement
    @staticmethod
    def row_hint_context(keyed_blocks: dict, image, crops: list | None = None) -> str:
        """Tell the model how many text rows a free-standing multi-row block has in the image (the OCR text
        has no line breaks), so lists / status windows come back entry by entry."""
        if image is None:
            return ""
        from modules.rendering.text_style import source_line_count
        notes: list[str] = []
        zoom_keys: list[str] = []
        from modules.rendering.text_style import looks_merged, text_groups
        for key, blk in keyed_blocks.items():
            if not (blk.text or "").strip():
                continue
            try:
                groups = text_groups(image, blk.xyxy)
                if looks_merged(groups):  # OCR merged several separate texts (menu bar, titles, question, username)
                    if crops is not None:
                        x1, y1, x2, y2 = [int(v) for v in blk.xyxy]
                        crop = image[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
                        if crop.size and crop.shape[1] > 900:
                            import cv2
                            crop = cv2.resize(crop, (900, max(1, int(crop.shape[0] * 900 / crop.shape[1]))), interpolation=cv2.INTER_AREA)
                        crops.append(crop)
                        zoom_keys.append(key)
                    where = ", ".join(f"{i + 1}: y {g['box'][1]}-{g['box'][3]}" for i, g in enumerate(groups))
                    notes.append(f"{key} merges {len(groups)} visually separate text groups, top to bottom ({where}); "
                                 f"translate each group and return exactly {len(groups)} lines separated by \\n, one per "
                                 f"group (a group wrapped over several rows is ONE line), never dropping or merging groups")
                    continue
                if getattr(blk, "text_class", "") == "text_bubble":
                    continue
                n = source_line_count(image, blk.xyxy)
            except Exception:
                continue
            if n >= 3:
                from modules.rendering.text_style import source_rows
                rows = source_rows(image, blk.xyxy, blk.text)
                split = f" (approximate row split: {' | '.join(rows)})" if len(rows) == n else ""
                notes.append(f"{key} shows {n} text rows in the image{split}")
        if not notes:
            return ""
        if zoom_keys:
            notes.append("zoomed crops of " + ", ".join(zoom_keys) + " are attached after the page image(s), in that order")
        return ("\nROW HINT: " + "; ".join(notes) + ". If the rows are separate entries (list, status window, menu) "
                "return exactly that many lines separated by \\n, one per row, in order, never moving words between "
                "rows; if they are one wrapped sentence, return it as a single line.")

    LINE_STYLES = {"phone", "sign", "title"}

    @staticmethod
    def expected_lines(blk, image) -> int:
        """Visual row count of a free-standing UI/sign text block (status window, list, menu); 0 = not a list."""
        if image is None or (getattr(blk, "style", "") or "") not in BaseLLMTranslation.LINE_STYLES:
            return 0
        from modules.rendering.text_style import looks_merged, source_line_count, text_groups
        try:
            have = len([ln for ln in (blk.translation or "").split("\n") if ln.strip()])
            if have >= 2:  # a merged board / menu: one line per visually separate group
                groups = text_groups(image, blk.xyxy)
                if looks_merged(groups, 3):
                    return len(groups)
            if getattr(blk, "text_class", "") == "text_bubble":
                return 0
            return source_line_count(image, blk.xyxy)
        except Exception:
            return 0

    def _translate_rows(self, blk, n: int, image) -> list[str]:
        """Last resort for a list block: translate every approximate source row on its own (n rows or []).
        The rows come from splitting the OCR text by the ink width of each image row."""
        from modules.rendering.text_style import source_rows
        try:
            rows = source_rows(image, blk.xyxy, blk.text)
            if len(rows) != n:
                return []
            system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
            raw = {f"block_{i}": r for i, r in enumerate(rows)}
            user_prompt = ("These are separate rows of one status window / list. Translate each row on its own, "
                           "one translated row per key.\nTranslate this:\n"
                           f"{json.dumps(raw, ensure_ascii=False, indent=4)}")
            content = self._perform_translation(user_prompt, system_prompt, None)
            tmp = {k: type("R", (), {"translation": "", "style": ""})() for k in raw}
            found = apply_translation_json(tmp, content)
            out = [tmp[k].translation.replace("\n", " ").strip() for k in raw if k in found]
            return out if len(out) == n and all(out) else []
        except Exception as exc:
            print(f"per-row translation failed: {exc}", flush=True)
            return []

    def fix_line_counts(self, blk_list, image, extra_context: str = "") -> int:
        """A list-like block with N visual rows must come back with N lines: one retry with an explicit line
        count, then a deterministic redistribution. Returns the number of blocks that needed fixing."""
        todo: dict[str, tuple] = {}
        for idx, blk in enumerate(blk_list):
            n = self.expected_lines(blk, image)
            have = len([ln for ln in (blk.translation or "").split("\n") if ln.strip()])
            if n >= 2 and have and have != n and (have >= 2 or n >= 4):
                todo[f"block_{idx}"] = (blk, n)
        if not todo:
            return 0
        raw = {k: blk.text for k, (blk, _) in todo.items()}
        rules = "; ".join(f"{k} shows exactly {n} separate text rows in the image (top to bottom)"
                          for k, (_, n) in todo.items())
        retry_context = (f"{extra_context}\nLINE COUNT: {rules}. Return that block as ONE string containing exactly "
                         "that many lines separated by \\n, one entry per row of the image, same order. "
                         "Do not merge rows and do not split a row.")
        try:
            system_prompt = self.get_system_prompt(self.source_lang, self.target_lang)
            user_prompt = (f"{retry_context}\nMake the translation sound as natural as possible.\n"
                           f"Translate this:\n{json.dumps(raw, ensure_ascii=False, indent=4)}")
            content = self._perform_translation(user_prompt, system_prompt, image)
            apply_translation_json({k: blk for k, (blk, _) in todo.items()}, content)
        except Exception as exc:  # network/JSON problems: the redistribution below still guarantees N lines
            print(f"line-count retry failed: {exc}", flush=True)
        for blk, n in todo.values():
            lines = [ln for ln in (blk.translation or "").split("\n") if ln.strip()]
            if len(lines) == n:
                continue
            per_row = self._translate_rows(blk, n, image)
            blk.translation = "\n".join(per_row) if per_row else "\n".join(redistribute_lines(lines, n))
        return len(todo)

    @abstractmethod
    def _perform_translation(self, user_prompt: str, system_prompt: str, image: np.ndarray) -> str:
        """
        Perform translation using specific LLM.
        
        Args:
            user_prompt: User prompt for LLM
            system_prompt: System prompt for LLM
            image: Image as numpy array
            
        Returns:
            Translated JSON text
        """
        pass

    def encode_image(self, image: np.ndarray, ext=".jpg"):
        """
        Encode CV2/numpy image directly to base64 string using cv2.imencode.
        
        Args:
            image: Numpy array representing the image
            ext: Extension/format to encode the image as (".png" by default for higher quality)
                
        Returns:
            Tuple of (Base64 encoded string, mime_type)
        """
        # Direct encoding from numpy/cv2 format to bytes
        buffer = imk.encode_image(image, ext.lstrip('.'))
        
        # Convert to base64
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Map extension to mime type
        mime_types = {
            ".jpg": "image/jpeg", 
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(ext.lower(), f"image/{ext[1:].lower()}")
        
        return img_str, mime_type