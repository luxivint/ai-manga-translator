"""Erase white caption glyphs only inside a proven uniform black surround."""
import numpy as np


def clean_flat_dark_captions(original, cleaned, blocks):
    result = cleaned
    height, width = original.shape[:2]
    for block in blocks:
        if (getattr(block, 'preserve_original', False) or
                not (getattr(block, 'translation', '') or '').strip() or
                getattr(block, 'bubble_xyxy', None) is not None or
                getattr(block, 'style', '') != 'narration'):
            continue
        x1, y1, x2, y2 = map(int, block.xyxy)
        pad = 6
        x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(width, x2+pad), min(height, y2+pad)
        if x1 < pad or y1 < pad or x2+pad > width or y2+pad > height or min(x2-x1, y2-y1) < 20:
            continue
        patch = original[y1-pad:y2+pad, x1-pad:x2+pad, :3]
        ring = np.ones(patch.shape[:2], dtype=bool); ring[pad:-pad, pad:-pad] = False
        surround = patch[ring].astype(np.int16)
        fill = np.median(surround, axis=0)
        # Reject gradients, artwork and frames. The original pixels, not the
        # inpainted result, must prove the complete surrounding ring is flat.
        if fill.max() > 25 or np.mean(np.max(np.abs(surround-fill), axis=1) <= 8) < .995:
            continue
        inside = original[y1:y2, x1:x2, :3].astype(np.int16)
        dark = np.max(np.abs(inside-fill), axis=2) <= 12
        white = inside.min(axis=2) >= 190
        if dark.mean() < .65 or white.mean() < .015 or np.mean(dark | white) < .94:
            continue
        if result is cleaned:
            result = cleaned.copy()
        result[y1:y2, x1:x2] = fill.astype(np.uint8)
    return result
