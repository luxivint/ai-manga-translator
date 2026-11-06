"""Deterministic white speech-balloon cleanup, independent of glyph segmentation.

Only repaint a proven, enclosed, uniformly bright balloon interior. Keep
untranslated balloons, bubble outlines, panel edges and textured art intact.
"""
from __future__ import annotations

import cv2
import numpy as np


def _rect(box, width, height):
    x1, y1, x2, y2 = (int(v) for v in box)
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)


def _interior(original, bubble_box, text_boxes):
    """Return (page-box, safely inset interior mask, background RGB) or None."""
    height, width = original.shape[:2]
    x1, y1, x2, y2 = _rect(bubble_box, width, height)
    if x2 - x1 < 65 or y2 - y1 < 65:
        return None
    crop = original[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
    # Only plain paper-white interiors; no tinted/illustrated balloons.
    bright = ((gray >= 215) & (chroma <= 38)).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
    contenders = []
    for k in range(1, n):
        area = int(stats[k, cv2.CC_STAT_AREA])
        if area < 0.16 * cw * ch or area > 0.96 * cw * ch:
            continue
        # A component reaching the crop edge might be the surrounding PAGE,
        # especially near a dashed/open balloon outline.
        piece = labels == k
        border_hits = int(piece[0].sum() + piece[-1].sum()
                          + piece[:, 0].sum() + piece[:, -1].sum())
        if border_hits > max(5, int(0.022 * 2 * (cw + ch))):
            continue
        count, coverage = 0, 0.0
        for box in text_boxes:
            tx1, ty1, tx2, ty2 = _rect(box, width, height)
            lx1, ly1 = max(0, tx1 - x1), max(0, ty1 - y1)
            lx2, ly2 = min(cw, tx2 - x1), min(ch, ty2 - y1)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            # Glyphs are dark, but most space around them is paper-white.
            whiteness = float(piece[ly1:ly2, lx1:lx2].mean())
            if whiteness >= 0.28:
                count += 1
                coverage += whiteness
        if count == len(text_boxes):
            contenders.append((coverage, area, k))
    if not contenders:
        return None
    _, _, selected = max(contenders)
    chosen = (labels == selected).astype(np.uint8)
    # Fill external contour to recover ALL dark glyph holes, even those
    # completely omitted by the YOLO text segmentation mask.
    contours, _ = cv2.findContours(chosen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inside = np.zeros((ch, cw), np.uint8)
    cv2.drawContours(inside, contours, -1, 255, cv2.FILLED)
    inset = max(4, min(9, int(0.012 * min(ch, cw))))
    inside = cv2.erode(inside, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)))
    if (inside > 0).sum() < 0.14 * cw * ch:
        return None
    sample = crop[chosen > 0].astype(np.float32)
    if sample.size == 0 or float(np.max(np.std(sample, axis=0))) > 14.0:
        return None  # Texture/gradient: keep the existing mask path.
    fill = np.median(sample, axis=0).astype(np.uint8)
    return (x1, y1, x2, y2), inside > 0, fill


def clean_white_balloons(original, cleaned, blocks, details=None):
    """Fill verified white interiors; preserve outlines and untranslated text.

    Optional details dict records full-area successes separately from weaker
    text-box fallbacks, allowing a safe YOLO bypass only for proven interiors.
    """
    result = cleaned.copy()
    groups = {}
    for block in blocks or []:
        bubble = getattr(block, "bubble_xyxy", None)
        if bubble is None or getattr(block, "text_class", None) != "text_bubble":
            continue
        key = tuple(int(v) for v in bubble)
        groups.setdefault(key, []).append(block)
    repainted = 0
    for bubble, members in groups.items():
        # One detector box may enclose two speakers. Never erase an untranslated
        # source block in that shared box.
        if any(len((getattr(m, "translation", "") or "").strip()) <= 1 for m in members):
            continue
        safe = _interior(original, bubble, [m.xyxy for m in members])
        if safe is None:
            # An open outline or white page surrounding the balloon makes a
            # full interior ambiguous. Clean just the confirmed text-box
            # interiors instead of silently doing nothing.
            result, count = _fallback_white_text_regions(original, result, bubble, members)
            repainted += count
            continue
        (x1, y1, x2, y2), mask, fill = safe
        result[y1:y2, x1:x2][mask] = fill
        if details is not None:
            details.setdefault("full_balloons", set()).add(bubble)
        repainted += 1
    return result, repainted


def _outline_guard(original, bubble_box):
    """Protect long connected outline/panel strokes, never normal small letters."""
    height, width = original.shape[:2]
    x1, y1, x2, y2 = _rect(bubble_box, width, height)
    protect = np.zeros((height, width), dtype=bool)
    if x2 <= x1 or y2 <= y1:
        return protect
    crop = original[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
    dark = ((gray <= 150) & (chroma < 80)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    outline = np.zeros((h, w), np.uint8)
    for k in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in stats[k])
        if area < 55 or (cw < 0.24 * w and ch < 0.24 * h):
            continue
        extent = area / max(1, cw * ch)
        if extent < 0.48 or cw > 0.65 * w or ch > 0.65 * h:
            outline[labels == k] = 255
    if outline.any():
        outline = cv2.dilate(outline, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    protect[y1:y2, x1:x2] = outline > 0
    return protect


def _fallback_white_text_regions(original, cleaned, bubble_box, blocks):
    """Repaint OCR text footprints on white paper when full segmentation fails.

    A joined balloon may have a shared detector box whose outer edge is open,
    cropped, or surrounded by another white page region. Rather than abandon
    cleaning, use high-confidence local white support around each translated
    OCR box, protecting all long black outline strokes and adjacent artwork.
    """
    height, width = original.shape[:2]
    bx1, by1, bx2, by2 = _rect(bubble_box, width, height)
    if bx2 <= bx1 or by2 <= by1:
        return cleaned, 0
    protect = _outline_guard(original, bubble_box)
    if int(protect[by1:by2, bx1:bx2].sum()) < max(100, 0.0015 * (bx2-bx1) * (by2-by1)):
        return cleaned, 0  # no surrounding outline: refuse to erase an open page

    result = cleaned.copy()
    cleaned_boxes = 0
    for block in blocks:
        tx1, ty1, tx2, ty2 = _rect(block.xyxy, width, height)
        bw, bh = tx2 - tx1, ty2 - ty1
        if bw < 14 or bh < 10 or bw * bh > 0.65 * (bx2-bx1) * (by2-by1):
            continue
        pad_x = max(6, min(18, round(bw * 0.07)))
        pad_y = max(6, min(16, round(bh * 0.12)))
        x1, y1 = max(bx1, tx1-pad_x), max(by1, ty1-pad_y)
        x2, y2 = min(bx2, tx2+pad_x), min(by2, ty2+pad_y)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = original[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
        white = (gray >= 222) & (chroma <= 38)
        if float(white.mean()) < 0.42:
            continue
        # Don't repaint coloured illustrations that happen to lie in an OCR box.
        if float(((chroma > 65) & (gray < 225)).mean()) > 0.025:
            continue
        sample = crop[white].astype(np.float32)
        if not sample.size or float(np.std(sample, axis=0).max()) > 15:
            continue
        fill = np.median(sample, axis=0).astype(np.uint8)

        # Unlike a rectangular fill, local white support avoids painting a
        # square over the grey comic panel beneath a balloon's tail. Dense
        # lettering still passes even when individual pixels are near black.
        white_support = cv2.blur(white.astype(np.float32), (29, 29))
        paint = (white_support >= 0.36) & ~protect[y1:y2, x1:x2]
        # Separate untranslated lettering in the same bubble remains intact.
        for other in blocks:
            if other is block or (getattr(other, "translation", "") or "").strip():
                continue
            ox1, oy1, ox2, oy2 = _rect(other.xyxy, width, height)
            paint[max(0, oy1-y1):min(y2-y1, oy2-y1),
                  max(0, ox1-x1):min(x2-x1, ox2-x1)] = False
        if not paint.any():
            continue
        result[y1:y2, x1:x2][paint] = fill
        cleaned_boxes += 1
    return result, cleaned_boxes
