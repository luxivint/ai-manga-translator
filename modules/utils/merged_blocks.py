"""Split OCR blocks that merged several visually separate texts (menu bar, titles, question, username).

The groups come from the ORIGINAL page (local ink + rule lines, see rendering.text_style.text_groups); every
group is re-read by the normal OCR engine as its own block, so translation, cleaning and rendering see one block
per text group instead of one long run-on string.
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

MIN_GROUPS = 4


def split_merged_ocr_blocks(blocks: list, image: np.ndarray, ocr, skip: Callable | None = None, pad: int = 4) -> list:
    """Returns a new block list; merged blocks are replaced by one OCR'd block per text group."""
    from modules.rendering.text_style import looks_merged, text_groups

    H, W = image.shape[:2]
    out: list = []
    for blk in blocks:
        text = (getattr(blk, "text", "") or "").strip()
        if not text or (skip is not None and skip(blk)):
            out.append(blk)
            continue
        try:
            groups = text_groups(image, blk.xyxy)
        except Exception:
            groups = []
        if not looks_merged(groups, MIN_GROUPS):
            out.append(blk)
            continue
        subs = []
        for g in groups:
            x1, y1, x2, y2 = g["box"]
            sub = copy.copy(blk)
            sub.xyxy = np.array([max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)])
            sub.bubble_xyxy = None
            sub.text_class = "text_free"
            sub.text, sub.texts, sub.translation, sub.lines = "", [], "", None
            subs.append(sub)
        try:
            ocr.process(image, subs)
        except Exception as exc:
            print(f"  [merged-split] OCR of groups failed ({exc}); block kept whole", flush=True)
            out.append(blk)
            continue
        for s in subs:  # UI divider bars ("| REAL-TIME BOARD") are not text and trip the noisy-OCR filter
            s.text = (s.text or "").strip().strip("|").strip()
        read = [s for s in subs if s.text]
        keys = ["".join(ch for ch in s.text.lower() if ch.isalnum()) for s in read]
        if any(a and b and (a in b or b in a) for i, a in enumerate(keys) for b in keys[i + 1:]):
            print(f"  [merged-split] {list(map(int, blk.xyxy))}: duplicate group texts (OCR echo), block kept whole", flush=True)
            out.append(blk)
            continue
        if len(read) < 2:
            out.append(blk)
            continue
        print(f"  [merged-split] {list(map(int, blk.xyxy))}: {len(read)} groups: " + " | ".join(s.text[:24] for s in read), flush=True)
        out.extend(read)
    return out
