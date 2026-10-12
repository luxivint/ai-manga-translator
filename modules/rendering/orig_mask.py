"""Original-page text mask registry.

The cleaning stage builds a pixel mask of the lettering on the ORIGINAL page (segmenter + tight glyph mask).
The renderer needs exactly that mask to measure the source lettering (cap height, colours, gradient, outline,
line groups) without being fooled by whatever the inpainter painted into the cleaned page.

`remember(image, mask)` is called by the cleaning stage, `lookup(image)` by the renderer (keyed by the page
array identity; the entry disappears with the array). No dependency on the cleaning code.
"""
from __future__ import annotations

import os
import threading
import weakref

import cv2
import numpy as np

_LOCK = threading.Lock()
_STORE: dict[int, np.ndarray] = {}  # id(page image) -> mask; entry dropped when the page array is freed


def remember(image: np.ndarray, mask: np.ndarray | None) -> None:
    if mask is None or image is None or mask.shape[:2] != image.shape[:2]:
        return
    key = id(image)
    with _LOCK:
        fresh = key not in _STORE
        _STORE[key] = mask
    if fresh:
        weakref.finalize(image, lambda k=key: _STORE.pop(k, None))


def lookup(image: np.ndarray) -> np.ndarray | None:
    with _LOCK:
        return _STORE.get(id(image))


def tight_page_mask(mask: np.ndarray) -> np.ndarray:
    """Undo the cleaner's final dilation (MASK_DILATE, default 9) so the mask hugs the glyphs again."""
    k = int(os.environ.get("MASK_DILATE", "9"))
    if k <= 1:
        return mask > 0
    return cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


def glyph_mask(image: np.ndarray, xyxy, page_mask: np.ndarray | None = None):
    """(bool mask of the box, (x1, y1)) from the original-page text mask, or None when none is registered."""
    pm = page_mask if page_mask is not None else lookup(image)
    if pm is None:
        return None
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), bool), (x1, y1)
    pad = 6  # erosion needs context at the box edge
    cx1, cy1, cx2, cy2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    tight = tight_page_mask(pm[cy1:cy2, cx1:cx2])
    return tight[y1 - cy1:y2 - cy1, x1 - cx1:x2 - cx1].copy(), (x1, y1)
