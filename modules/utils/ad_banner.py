"""Cheap, OCR-free detector for a competitor promo banner strip at the top/bottom edge of a page.

Scanlation groups end a chapter with a dark banner ("READ ONLY AT THUNDERSCANS ...") that, once a long page is
sliced, starts in the middle of the last story page. GPT-Vision gives only a rough percentage (it said 46 % for a
54 % banner) and OCR returns nothing for watermark text, so the strip survived. This detector measures the real
edge instead:

  * a band of dark rows (>= 15 % near-black pixels per row) glued to the page edge,
  * a hard, full-width light -> dark step at its inner border,
  * strongly coloured violet/purple pixels inside it (the banner artwork), unless `strict=False`.

Only meant for the first/last few pages of a chapter. A plain dark story panel has no light->dark step with violet
artwork, so it is left alone. Returns the number of pixel rows to cut from (top, bottom)."""
from __future__ import annotations

import os

import cv2
import numpy as np


def enabled() -> bool:
    return os.environ.get("AD_BANNER_DETECT", "true").strip().lower() not in ("false", "0", "no", "off")


def _bottom_band(img: np.ndarray, strict: bool) -> int:
    h, w = img.shape[:2]
    if h < 200 or w < 100:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    dark_thr = 40 if strict else 25
    row_dark = (gray <= dark_thr).mean(axis=1) >= 0.15
    rowmed = np.median(gray, axis=1).astype(np.float32)
    top = h
    gap = 0
    i = h - 1
    while i >= 0:
        if row_dark[i]:
            top, gap = i, 0
        else:
            gap += 1
            if gap > 10:
                break
        i -= 1
    band_h = h - top
    min_h = max(48, int(h * 0.03)) if strict else 8
    max_h = int(h * (0.65 if strict else 0.08))
    if band_h < min_h or band_h > max_h or top < 14:
        return 0
    above = float(rowmed[max(0, top - 10):top - 2].mean())
    below = float(rowmed[top + 2:min(h, top + 10)].mean())
    if above < 140 or above - below < 100:
        return 0  # no hard light->dark step: ordinary dark panel
    if strict:
        hsv = cv2.cvtColor(img[top:], cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        purple = ((hue >= 115) & (hue <= 160) & (sat > 90) & (val > 100)).mean()
        if purple < 0.015:
            return 0
    return min(h, band_h + 2)


def detect_edge_banner(image: np.ndarray, strict: bool = True) -> tuple[int, int]:
    """(rows to cut from the top, rows to cut from the bottom) for an RGB uint8 page."""
    if not enabled() or image is None or image.ndim != 3 or image.shape[2] != 3:
        return 0, 0
    bottom = _bottom_band(image, strict)
    top = _bottom_band(np.ascontiguousarray(image[::-1]), strict)
    return top, bottom
