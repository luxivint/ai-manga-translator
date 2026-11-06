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


def clean_white_balloons(original, cleaned, blocks):
    """Fill verified white interiors; preserve outlines and untranslated text."""
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
            continue
        (x1, y1, x2, y2), mask, fill = safe
        result[y1:y2, x1:x2][mask] = fill
        repainted += 1
    return result, repainted
