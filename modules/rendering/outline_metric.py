"""Offline QA metric for text fit: how much of the rendered ink of a balloon lies on / next to the balloon outline.

Pure numpy/cv2, no Qt. ink = pixels the renderer changed (final vs wiped page) inside the balloon box; outline = dark
pixels of the wiped page inside the same box (dashed outlines included); a band of `band_px` around the outline is the
'too close' zone. Only light-filled balloons are measured (dark narration boxes have no dark outline to compare with).
"""
from __future__ import annotations

import cv2
import numpy as np


def balloon_outline_overlap(cleaned: np.ndarray, final: np.ndarray, bubble_xyxy, band_px: int = 4,
                            dark_luma: float = 140.0, ink_delta: float = 60.0, skip_rects=()):
    """(ink_pixels, ink_pixels_in_outline_band) for one balloon box, or None (no box / dark fill / no ink)."""
    H, W = cleaned.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bubble_xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    c = cleaned[y1:y2, x1:x2].astype(np.int16)
    f = final[y1:y2, x1:x2].astype(np.int16)
    luma = c @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    fill = float(np.median(luma))
    if fill < 170.0:
        return None
    dark_luma = min(dark_luma, fill - 90.0)
    ink = np.abs(f - c).max(axis=2) > ink_delta
    for rx1, ry1, rx2, ry2 in skip_rects:  # lettering that is deliberately drawn over the outline (sound effects, free text)
        ink[max(0, int(ry1) - y1):max(0, int(ry2) - y1), max(0, int(rx1) - x1):max(0, int(rx2) - x1)] = False
    n_ink = int(ink.sum())
    if n_ink < 30:
        return None
    outline = (luma < dark_luma) & ~ink
    k = np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8)
    band = cv2.dilate(outline.astype(np.uint8), k) > 0
    return n_ink, int((ink & band).sum())
