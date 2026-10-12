"""Inventory + specimen sheets for every font under FONT KLASOR/ and assets/.

python scripts/font_analysis.py   -> render_test/render_fix/font_inventory.json + specimens/*.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.rendering import font_metrics as fm  # noqa: E402
from modules.rendering.font_library import FONT_DIRS, FONT_EXTS  # noqa: E402

OUT = ROOT / "render_test" / "render_fix"
LINES = [
    "Şu güzel çocuk İstanbul'da ığdır ışığı yedi. Ağaç, öğün, ürün?!",
    "ŞU GÜZEL ÇOCUK İSTANBUL'DA IĞDIR IŞIĞI YEDİ. AĞAÇ, ÖĞÜN, ÜRÜN?!",
    "iİıI  … ? ! “ ” ’ – —  0123  Shin: hiçbir şey olmadı...",
]


def collect():
    fonts = []
    for d in FONT_DIRS:
        base = ROOT / d
        for p in sorted(base.rglob("*")):
            if p.suffix.lower() in FONT_EXTS:
                fonts.append(p)
    return fonts


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "specimens").mkdir(exist_ok=True)
    inv = {}
    by_cat: dict[str, list[Path]] = {}
    for p in collect():
        rel = p.relative_to(ROOT).as_posix()
        cat = p.parent.name
        rec = {"category": cat}
        try:
            rec.update(fm.font_names(p))
            rec.update(fm.glyph_coverage(p))
            rec.update(fm.font_traits(p))
        except Exception as exc:
            rec["error"] = str(exc)[:100]
        inv[rel] = rec
        by_cat.setdefault(cat, []).append(p)
    (OUT / "font_inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1), encoding="utf-8")
    keep = ("capem", "stem", "xh", "wid", "cw", "slant", "caps", "serif", "contrast", "turkish_ok", "punct_ok",
            "missing_turkish", "missing_punct", "family", "weight_class")
    compact = {k: {f: v[f] for f in keep if f in v} for k, v in inv.items()}
    (ROOT / "modules" / "rendering" / "font_traits.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=1), encoding="utf-8")
    label = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 15)
    for cat, paths in by_cat.items():
        rows = []
        for p in paths:
            rel = p.relative_to(ROOT).as_posix()
            rec = inv[rel]
            f = ImageFont.truetype(str(p), 34)
            h = 24 + 3 * 48
            im = Image.new("RGB", (1400, h), "white")
            d = ImageDraw.Draw(im)
            tag = (f"{p.name} | {rec.get('family')} {rec.get('style')} | w{rec.get('weight_class')} "
                   f"stem={rec.get('stem')} xh={rec.get('xh')} cw={rec.get('cw')} slant={rec.get('slant')} "
                   f"caps={rec.get('caps')} serif={rec.get('serif')} TR={rec.get('turkish_ok')} P={rec.get('punct_ok')}"
                   f" miss={''.join(rec.get('missing_turkish', []) + rec.get('missing_punct', []))}")
            d.text((4, 2), tag, font=label, fill=(200, 0, 0))
            for i, line in enumerate(LINES):
                d.text((8, 26 + i * 48), line, font=f, fill="black")
            d.line([(0, h - 1), (1400, h - 1)], fill=(160, 160, 160))
            rows.append(im)
        sheet = Image.new("RGB", (1400, sum(r.height for r in rows)), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        name = "".join(c if c.isalnum() else "_" for c in cat)
        sheet.save(OUT / "specimens" / f"{name}.png")
    print("fonts", len(inv))


if __name__ == "__main__":
    main()
