"""Measurable typographic traits for fonts and for source text pixels.

Used by the font analysis script and by original-font matching. Pure
PIL/numpy/fontTools (no Qt), so everything is unit-testable.

Trait vector (all dimensionless):
  stem   stem width / cap height          (weight; ~0.10 light, 0.18 regular, 0.3 heavy)
  xh     x-height / cap height            (small-x-height fonts look tiny in lowercase)
  wid    average advance of 'HOEn' / cap height (condensed < 0.62 < wide)
  slant  tan(angle) of the vertical stems (italic > 0.12)
  caps   True if lowercase letters are drawn as capitals (caps-only / small-caps)
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PUNCT = "…?!“”’‘–—.,:;-'\"()"
TURKISH = "ğĞşŞıİçÇöÖüÜ"
_SZ = 160


def _glyph(font: ImageFont.FreeTypeFont, ch: str, size: int = _SZ) -> np.ndarray | None:
    """Binary ink mask of one glyph, tightly cropped (None if empty/notdef-ish)."""
    img = Image.new("L", (size * 3, size * 3), 0)
    ImageDraw.Draw(img).text((size // 2, size // 2), ch, font=font, fill=255)
    a = np.asarray(img) > 127
    ys, xs = np.nonzero(a)
    if len(xs) == 0:
        return None
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _iou(a: np.ndarray, b: np.ndarray, n: int = 40) -> float:
    import cv2

    ra = cv2.resize(a.astype(np.uint8) * 255, (n, n), interpolation=cv2.INTER_AREA) > 127
    rb = cv2.resize(b.astype(np.uint8) * 255, (n, n), interpolation=cv2.INTER_AREA) > 127
    return float((ra & rb).sum() / max(1, (ra | rb).sum()))


def _first_run(row: np.ndarray) -> int:
    xs = np.nonzero(row)[0]
    if len(xs) == 0:
        return 0
    n = 1
    for a, b in zip(xs, xs[1:]):
        if b != a + 1:
            break
        n += 1
    return n


def slant_of_mask(mask: np.ndarray) -> float:
    """tan(slant) from second-order moments of a (stem-dominated) glyph mask; + = leaning right."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 20:
        return 0.0
    yc, xc = ys - ys.mean(), xs - xs.mean()
    var_y = float((yc ** 2).mean())
    if var_y < 1e-6:
        return 0.0
    return float(-(xc * yc).mean() / var_y)


def font_traits(path: str | Path) -> dict:
    font = ImageFont.truetype(str(path), _SZ)
    H, x, I, l = (_glyph(font, c) for c in ("H", "x", "I", "l"))
    a_low, a_up = _glyph(font, "a"), _glyph(font, "A")
    o = _glyph(font, "o")
    out: dict = {}
    if H is None:
        return {"stem": 0.18, "xh": 0.7, "wid": 0.7, "cw": 0.8, "slant": 0.0, "caps": False, "serif": False, "contrast": 1.0}
    cap = H.shape[0]
    stem = _first_run(H[int(cap * 0.25)]) / cap
    if l is not None and l.shape[0] > 0:
        # 'l' is a plain stem in most fonts; take the thinner estimate
        stem = min(stem, np.median(l.sum(axis=1)) / cap) if stem > 0 else np.median(l.sum(axis=1)) / cap
    out["stem"] = round(float(stem), 3)
    out["capem"] = round(cap / _SZ, 4)  # drawn capital height as a fraction of the em
    out["xh"] = round((x.shape[0] / cap) if x is not None else 0.7, 3)
    wids = [font.getlength(c) for c in "HOEn"]
    out["wid"] = round(float(np.mean(wids)) / cap, 3)
    boxes = [_glyph(font, c) for c in "HEOANDRSTMBKCUPG"]
    out["cw"] = round(float(np.median([g.shape[1] / g.shape[0] for g in boxes if g is not None] or [0.8])), 3)
    sl = [slant_of_mask(g) for g in (_glyph(font, c) for c in "HIlEFLNDBPRK") if g is not None]
    out["slant"] = round(float(np.median(sl)), 3) if sl else 0.0
    caps = False
    ious = []
    for ch in "abdefghnqrtyk":
        lo, up = _glyph(font, ch), _glyph(font, ch.upper())
        if lo is None or up is None:
            continue
        ious.append(_iou(lo, up))
    if ious:
        # a caps-only font draws 'a' as 'A', 'b' as 'B'...: shapes match after normalising size
        caps = bool(np.median(ious) > 0.72) or (out["xh"] > 0.95)
    out["caps"] = caps
    i_run = I.shape[1] if I is not None else 0
    out["serif"] = bool(I is not None and stem > 0 and i_run > 1.7 * stem * cap)
    if o is not None and o.shape[0] > 8:
        mid_r, mid_c = o.shape[0] // 2, o.shape[1] // 2
        side = _first_run(o[mid_r])
        top = _first_run(o[:, mid_c])
        out["contrast"] = round(float(side) / max(1, top), 2)
    else:
        out["contrast"] = 1.0
    return out


def glyph_coverage(path: str | Path) -> dict:
    from fontTools.ttLib import TTFont

    f = TTFont(str(path), lazy=True, fontNumber=0)
    try:
        cmap = f.getBestCmap() or {}
    finally:
        f.close()
    missing_tr = [c for c in TURKISH if ord(c) not in cmap]
    missing_p = [c for c in PUNCT if ord(c) not in cmap]
    return {"missing_turkish": missing_tr, "missing_punct": missing_p,
            "turkish_ok": not missing_tr, "punct_ok": not missing_p}


def font_names(path: str | Path) -> dict:
    from fontTools.ttLib import TTFont

    f = TTFont(str(path), lazy=True, fontNumber=0)
    try:
        name = f["name"]
        fam = name.getDebugName(1) or ""
        sub = name.getDebugName(2) or ""
        full = name.getDebugName(4) or ""
        os2 = f["OS/2"] if "OS/2" in f else None
        return {"family": fam, "style": sub, "full": full,
                "weight_class": getattr(os2, "usWeightClass", None),
                "width_class": getattr(os2, "usWidthClass", None),
                "italic_angle": float(f["post"].italicAngle) if "post" in f else 0.0}
    finally:
        f.close()


# ---------------------------------------------------------------- source text
def source_traits(gray_text_mask: np.ndarray, line_height: float | None = None) -> dict:
    """Traits of source lettering from a binary glyph mask (True = ink).

    stem   median stroke width (2 * distance-transform mode) / glyph height
    slant  tan of lean from component moments
    caps   True when almost all component heights are within 15 percent of the max
    wid    mean component width / height (cheap proxy of condensed vs wide)
    """
    import cv2

    m = (gray_text_mask > 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return {"stem": 0.18, "slant": 0.0, "caps": False, "wid": 0.7, "height": 0.0}
    comps = [(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], i) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 12]
    if not comps:
        return {"stem": 0.18, "slant": 0.0, "caps": False, "wid": 0.7, "height": 0.0}
    hs = np.array([c[1] for c in comps], dtype=float)
    ws = np.array([c[0] for c in comps], dtype=float)
    cap_h = float(np.percentile(hs, 90))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    skel_vals = dist[(dist > 0) & (dist >= cv2.dilate(dist, np.ones((3, 3), np.uint8)) - 1e-3)]
    stem = float(np.median(skel_vals)) * 2 / max(cap_h, 1.0) if len(skel_vals) else 0.18
    slants = []
    for _, h, i in comps:
        if h < 0.6 * cap_h:
            continue
        slants.append(slant_of_mask(lab[st[i, 1]:st[i, 1] + h, st[i, 0]:st[i, 0] + st[i, 2]] == i))
    caps = bool(len(hs) >= 4 and (hs > 0.85 * cap_h).mean() > 0.85)
    return {"stem": round(stem, 3), "slant": round(float(np.median(slants)) if slants else 0.0, 3),
            "caps": caps, "wid": round(float(np.median(ws[hs > 0.6 * cap_h] / hs[hs > 0.6 * cap_h])) if (hs > 0.6 * cap_h).any() else 0.7, 3),
            "height": cap_h}


def trait_distance(src: dict, font: dict) -> float:
    """Smaller = closer look. Weights chosen so weight and slant dominate."""
    d = 0.0
    d += 4.0 * abs(src["stem"] - font["stem"])
    d += 3.0 * abs(abs(src["slant"]) - abs(font["slant"])) if abs(src["slant"]) > 0.1 or abs(font["slant"]) > 0.1 else 0.0
    d += 1.2 * abs(src["wid"] - font.get("cw", font["wid"])) / max(font.get("cw", 0.8), 0.3)
    return d
