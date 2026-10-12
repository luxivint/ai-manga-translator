"""Font library for Turkish typesetting: glyph coverage scan and style -> font selection."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .typeset import TURKISH_LETTERS

FONT_DIRS = ("FONT KLASÖR", "assets")
FONT_EXTS = {".ttf", ".otf"}
FALLBACK_FONT = "assets/Comic Geek.ttf"

# style tag -> ordered list of category folders (or exact relative font files)
DEFAULT_STYLE_MAP: dict[str, list[str]] = {
    "normal": ["FONT KLASÖR/Kare/CC_VictorySpeechLower.otf", "konuşma"],
    "whisper": ["FONT KLASÖR/Kare/CC_VictorySpeechLower.otf", "konuşma"],
    "thought": ["FONT KLASÖR/Kare/Chinacat.ttf", "konuşma"],
    "shout": ["bağırma"],
    "sfx": ["sfx"],
    "narration": ["FONT KLASÖR/Kare/CCTALLTALES-REGULAR.OTF", "Kare"],
    "sign": ["elektronik alet"],
    "phone": ["elektronik alet"],
    "handwriting": ["el yazısı tarihi"],
}


def covers_turkish(cmap: Iterable[int]) -> bool:
    have = set(cmap)
    return all(ord(c) in have for c in TURKISH_LETTERS)


def _font_cmap(path: Path) -> set[int]:
    from fontTools.ttLib import TTFont

    font = TTFont(str(path), lazy=True, fontNumber=0)
    try:
        return set(font.getBestCmap() or {})
    finally:
        font.close()


def scan_fonts(root: Path) -> dict[str, bool]:
    """{relative posix path: has all Turkish letters} for every font under FONT_DIRS."""
    result: dict[str, bool] = {}
    for d in FONT_DIRS:
        base = Path(root) / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() not in FONT_EXTS:
                continue
            rel = p.relative_to(root).as_posix()
            try:
                result[rel] = covers_turkish(_font_cmap(p))
            except Exception:  # unreadable/corrupt font: treat as unusable
                result[rel] = False
    return result


def load_style_map(path: Optional[Path]) -> dict[str, list[str]]:
    mapping = {k: list(v) for k, v in DEFAULT_STYLE_MAP.items()}
    if path and Path(path).exists():
        try:
            user = json.loads(Path(path).read_text(encoding="utf-8"))
            for k, v in user.items():
                mapping[k] = [v] if isinstance(v, str) else list(v)
        except (OSError, ValueError):
            pass
    return mapping


def pick_font(style: str, coverage: dict[str, bool], mapping: dict[str, list[str]], root: Path) -> str:
    """Deterministic: first covering font of the first matching entry, else fallback."""
    style = (style or "normal").lower()
    for entry in mapping.get(style) or mapping.get("normal", []):
        if entry in coverage:
            if coverage[entry]:
                return entry
            continue
        folder = entry.strip("/")
        for rel in sorted(coverage):
            parts = rel.split("/")
            if len(parts) >= 3 and parts[1] == folder and coverage[rel]:
                return rel
    return FALLBACK_FONT
