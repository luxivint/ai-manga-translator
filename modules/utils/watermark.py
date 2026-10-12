"""Competitor scanlation watermark / banner handling (WATERMARK_POLICY=clean).

`match_watermark` decides WHAT is a watermark (never the site's own brand), `region_mask` decides WHERE to erase:
the whole banner, not just the glyphs, so LaMa repaints a clean area instead of smudging half a logo."""
from __future__ import annotations

import os
import re

import cv2
import numpy as np

WATERMARK_PATTERNS = (
    r"vortex\s*scans",
    r"[a-z0-9-]+\.(?:com|net|org|io|me|gg|xyz|co|to)\b",
    r"read\s+(?:this\s+series\s+)?first\s+at",
    r"discord\.gg|t\.me/|www\.|https?://",
    r"\b(?:patreon|ko-?fi)\b",
)


def own_brand_keys() -> list[str]:
    keys = {os.environ.get("SEO_SITE_NAME", "TrendManga"), os.environ.get("SEO_IMAGE_BRAND", ""), "TrendManga"}
    return [re.sub(r"\s+", "", k).lower() for k in keys if k.strip()]


def match_watermark(text: str) -> str | None:
    """Name of the matching watermark rule, or None. The site's own brand is never a watermark."""
    txt = (text or "").strip().lower()
    if not txt:
        return None
    squashed = re.sub(r"\s+", "", txt)
    if any(k and k in squashed for k in own_brand_keys()):
        return None
    extra = os.environ.get("WATERMARK_PATTERNS", "").strip()
    for pat in list(WATERMARK_PATTERNS) + ([extra] if extra else []):
        try:
            if re.search(pat, txt, re.IGNORECASE):
                return f"pattern:{pat[:28]}"
        except re.error:
            continue
    return None


def _plate_hull(image: np.ndarray, box: tuple, pad: int, tol: float) -> np.ndarray | None:
    """Convex hull (full-page uint8 mask) of the uniformly coloured plate/tag the text sits on, with an attached
    round logo when it shares the colour. None when there is no bounded plate (text straight on artwork or on a
    huge flat area): the flood fill then reaches the search window border."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = box
    th = max(1, y2 - y1)
    rx1, ry1 = max(0, x1 - 3 * th), max(0, y1 - 2 * th)
    rx2, ry2 = min(W, x2 + 3 * th), min(H, y2 + 2 * th)
    roi = image[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).astype(np.float32)
    bx1, by1, bx2, by2 = x1 - rx1, y1 - ry1, x2 - rx1, y2 - ry1
    pix = lab[by1:by2, bx1:bx2].reshape(-1, 3)
    if len(pix) == 0:
        return None
    keys, counts = np.unique((pix // 6).astype(np.int32), axis=0, return_counts=True)
    plate = keys[counts.argmax()] * 6 + 3
    near = (np.linalg.norm(lab - plate.astype(np.float32), axis=2) < tol).astype(np.uint8)
    near = cv2.morphologyEx(near, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _n, labels = cv2.connectedComponents(near)
    ring = np.zeros_like(near)
    ring[max(0, by1 - pad):by2 + pad, max(0, bx1 - pad):bx2 + pad] = 1
    ring[by1:by2, bx1:bx2] = 0
    ids = np.unique(labels[(ring > 0) & (near > 0)])
    ids = ids[ids > 0]
    if len(ids) == 0:
        return None
    comp = np.isin(labels, ids)
    h, w = comp.shape
    # leaked to the window edge (not merely the page edge) -> not a bounded plate
    if ((rx1 > 0 and comp[:, 0].any()) or (rx2 < W and comp[:, -1].any())
            or (ry1 > 0 and comp[0, :].any()) or (ry2 < H and comp[-1, :].any())):
        return None
    ys, xs = np.nonzero(comp)
    if len(xs) < 0.25 * (x2 - x1) * th:
        return None
    hull = cv2.convexHull(np.stack([xs, ys], axis=1).astype(np.int32))
    filled = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(filled, hull, 255)
    if (filled > 0).sum() > 12 * (x2 - x1) * th:  # absurdly large for a tag
        return None
    out = np.zeros((H, W), np.uint8)
    out[ry1:ry2, rx1:rx2] = filled
    return out


def region_mask(image: np.ndarray, box, pad: int | None = None, tol: float | None = None) -> np.ndarray:
    """Full-page uint8 mask (255 = erase) covering the whole watermark: the plate/tag hull when the text sits on
    one, always united with the padded text box."""
    H, W = image.shape[:2]
    pad = int(os.environ.get("WATERMARK_PAD", "12")) if pad is None else pad
    tol = float(os.environ.get("WATERMARK_PLATE_TOL", "14")) if tol is None else tol
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    mask = np.zeros((H, W), np.uint8)
    if x2 <= x1 or y2 <= y1:
        return mask
    hull = _plate_hull(image, (x1, y1, x2, y2), pad, tol)
    if hull is not None:
        mask = cv2.dilate(hull, np.ones((5, 5), np.uint8))  # cover the anti-aliased rim
        # a plate is only a little taller than its text: never let a huge flat area (black page top) be erased
        band = max(pad, int(0.6 * (y2 - y1)))
        mask[:max(0, y1 - band)] = 0
        mask[min(H, y2 + band):] = 0
    mask[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)] = 255
    return mask


def flat_fill(original: np.ndarray, cleaned: np.ndarray, wm_mask: np.ndarray, max_dev: float = 24.0, min_frac: float = 0.7) -> np.ndarray:
    """Where the erased area is surrounded by an (almost) flat colour (e.g. a black banner page) paint it with that
    colour instead of trusting the inpainter, which invents textured grey on flat backgrounds."""
    if wm_mask is None or not wm_mask.any():
        return cleaned
    out = cleaned.copy()
    n, labels = cv2.connectedComponents((wm_mask > 0).astype(np.uint8))
    k = np.ones((31, 31), np.uint8)
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        ring = (cv2.dilate(comp, k) > 0) & (comp == 0) & (wm_mask == 0)
        pix = original[ring]
        if len(pix) < 50:
            continue
        med = np.median(pix, axis=0)
        near = (np.abs(pix.astype(np.float32) - med).max(axis=1) <= max_dev).mean()
        if near >= min_frac:  # most of the surround is one flat colour (stray glow/outline pixels tolerated)
            out[comp > 0] = med.astype(out.dtype)
    return out
