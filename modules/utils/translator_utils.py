import base64
import json
import os
import re
import jieba
import janome.tokenizer
import numpy as np
from pythainlp.tokenize import word_tokenize
from .textblock import TextBlock
import imkit as imk


MODEL_MAP = {
    "Custom": "",  
    "Deepseek-v3": "deepseek-chat", 
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1-mini": "gpt-4.1-mini",
    "GPT-5.4-mini": "gpt-5.4-mini",
    "GPT-5.4-nano": "gpt-5.4-nano",
    "Claude-4.6-Sonnet": "claude-sonnet-4-6",
    "Claude-4.5-Haiku": "claude-haiku-4-5-20251001",
    "Gemini-2.0-Flash": "gemini-2.0-flash",
    "Gemini-3.0-Flash": "gemini-3-flash-preview",
    "Gemini-2.5-Pro": "gemini-2.5-pro"
}

TR_UPPER_MAP = str.maketrans({
    "i": "İ",
    "ı": "I",
})


def locale_upper(text: str, lang_code: str) -> str:
    if (lang_code or "").lower().startswith("tr"):
        return text.translate(TR_UPPER_MAP).upper()
    return text.upper()


def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode('utf-8')

def get_raw_text(blk_list: list[TextBlock]):
    rw_txts_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_txts_dict[block_key] = blk.text
    
    raw_texts_json = json.dumps(rw_txts_dict, ensure_ascii=False, indent=4)
    
    return raw_texts_json

def get_raw_translation(blk_list: list[TextBlock]):
    rw_translations_dict = {}
    for idx, blk in enumerate(blk_list):
        block_key = f"block_{idx}"
        rw_translations_dict[block_key] = blk.translation
    
    raw_translations_json = json.dumps(rw_translations_dict, ensure_ascii=False, indent=4)
    
    return raw_translations_json

def translation_temperature(default: float = 0.3) -> float:
    """Sampling temperature shared by the single-page and the batch translation path."""
    try:
        return float(os.environ.get("TRANSLATION_TEMPERATURE", default))
    except (TypeError, ValueError):
        return default


def parse_translation_value(value) -> tuple[str, str]:
    """(text, style) from one JSON value: plain string or the style-tag object {"text", "style"}."""
    style = ""
    if isinstance(value, dict):
        style = str(value.get("style") or "").strip().lower()
        value = value.get("text", "")
    if value is None:
        value = ""
    return str(value), style


def apply_translation_json(keyed_blocks: dict, json_string: str) -> set[str]:
    """Write the translations in `json_string` into {key: TextBlock}; returns the keys that were found.

    Sets blk.translation (explicit \n line breaks preserved) and blk.style (style-tag schema)."""
    match = re.search(r"\{[\s\S]*\}", json_string or "")
    if not match:
        print("No JSON found in the input string.")
        return set()
    translation_dict = json.loads(match.group(0))
    found = set()
    for key, blk in keyed_blocks.items():
        if key in translation_dict:
            blk.translation, style = parse_translation_value(translation_dict[key])
            if style or isinstance(translation_dict[key], dict):
                blk.style = style
            found.add(key)
        else:
            print(f"Warning: {key} not found in JSON string.")
    return found


def set_texts_from_json(blk_list: list[TextBlock], json_string: str):
    apply_translation_json({f"block_{idx}": blk for idx, blk in enumerate(blk_list)}, json_string)


def redistribute_lines(lines: list[str], n: int) -> list[str]:
    """Force `lines` into exactly n lines: merge the shortest adjacent pair while there are too many,
    split the longest multi-word line (balanced) while there are too few."""
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    if n <= 0 or not lines:
        return lines
    while len(lines) > n:
        i = min(range(len(lines) - 1), key=lambda k: len(lines[k]) + len(lines[k + 1]))
        lines[i:i + 2] = [lines[i] + " " + lines[i + 1]]
    while len(lines) < n:
        cand = [k for k, ln in enumerate(lines) if len(ln.split()) > 1]
        if not cand:
            break
        i = max(cand, key=lambda k: len(lines[k]))
        words = lines[i].split()
        best = min(range(1, len(words)), key=lambda c: abs(len(" ".join(words[:c])) - len(" ".join(words[c:]))))
        lines[i:i + 1] = [" ".join(words[:best]), " ".join(words[best:])]
    return lines


def set_upper_case(blk_list: list[TextBlock], upper_case: bool, lang_code: str = ""):
    for blk in blk_list:
        translation = blk.translation
        if translation is None:
            continue
        if upper_case and not translation.isupper():
            blk.translation = locale_upper(translation, lang_code)
        elif not upper_case and translation.isupper():
            blk.translation = translation.lower().capitalize()
        else:
            blk.translation = translation

def get_chinese_tokens(text):
    return list(jieba.cut(text, cut_all=False))

def get_japanese_tokens(text):
    tokenizer = janome.tokenizer.Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text)]

def format_translations(blk_list: list[TextBlock], trg_lng_cd: str, upper_case: bool = True):
    for blk in blk_list:
        translation = blk.translation
        trg_lng_code_lower = trg_lng_cd.lower()
        seg_result = []

        if 'zh' in trg_lng_code_lower:
            seg_result = get_chinese_tokens(translation)

        elif 'ja' in trg_lng_code_lower:
            seg_result = get_japanese_tokens(translation)

        elif 'th' in trg_lng_code_lower:
            seg_result = word_tokenize(translation)

        if seg_result:
            blk.translation = ''.join(word if word in ['.', ','] else f' {word}' for word in seg_result).lstrip()
        else:
            # apply casing/formatting for this single block when no segmentation is done
            if translation is None:
                continue
            if upper_case and not translation.isupper():
                blk.translation = locale_upper(translation, trg_lng_cd)
            elif not upper_case and translation.isupper():
                blk.translation = translation.lower().capitalize()
            else:
                blk.translation = translation

def is_there_text(blk_list: list[TextBlock]) -> bool:
    return any(blk.text for blk in blk_list)
