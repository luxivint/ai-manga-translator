"""Assign detected text to its speech balloon before OCR.

A detector may return one text rectangle covering two touching balloons.
Split at the inter-balloon gap so OCR and translation receive separate blocks.
"""
from __future__ import annotations

import numpy as np


def _area(box) -> float:
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def _intersection(a, b) -> float:
    return max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )


def _centre(box) -> tuple[float, float]:
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def _best_owner(box, bubbles):
    """Choose the best-overlapping balloon, not the first one detected."""
    matches = []
    for index, bubble in enumerate(bubbles):
        shared = _intersection(box, bubble) / max(1.0, _area(box))
        cx, cy = _centre(box)
        centre_inside = bubble[0] <= cx <= bubble[2] and bubble[1] <= cy <= bubble[3]
        if shared >= 0.55 or (shared >= 0.2 and centre_inside):
            matches.append((shared, -_area(bubble), index))
    return max(matches)[2] if matches else None


def _gap_cut(image: np.ndarray, box, axis: int, expected: float) -> int:
    """Prefer a low-ink row/column near the two balloon centres' midpoint."""
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = image[y1:y2, x1:x2]
    lo, hi = (y1, y2) if axis == 1 else (x1, x2)
    span = hi - lo
    center = max(lo, min(hi, int(round(expected))))
    radius = max(8, int(span * 0.16))
    left = max(lo + int(span * 0.28), center - radius)
    right = min(hi - int(span * 0.28), center + radius)
    if crop.size == 0 or right <= left:
        return center
    ink = crop[..., :3].mean(axis=2) < 165
    profile = ink.mean(axis=1 if axis == 1 else 0)
    rolling = np.convolve(profile, np.ones(5) / 5.0, mode="same")
    choices = np.arange(left, right + 1, dtype=int)
    scores = rolling[choices - lo] + 0.12 * np.abs(choices - center) / max(1, radius)
    return int(choices[np.argmin(scores)])


def assign_balloon_regions(image: np.ndarray, text_boxes, bubble_boxes):
    """Return (text_box, balloon_box | None, split_from_double_balloon).

    Split only when TWO DISTINCT balloon boxes each cover a large part of one
    text box. Normal isolated balloons and duplicate detections stay unchanged.
    """
    bubbles = [np.asarray(b, dtype=np.int32) for b in bubble_boxes]
    results = []
    for original in text_boxes:
        box = np.asarray(original, dtype=np.int32)
        area = _area(box)
        overlaps = [
            (i, _intersection(box, bubble) / max(1.0, area))
            for i, bubble in enumerate(bubbles)
            if _intersection(box, bubble) >= 0.24 * area
        ]
        overlaps.sort(key=lambda item: item[1], reverse=True)
        if len(overlaps) == 2 and overlaps[0][1] < 0.90:
            ia, ib = overlaps[0][0], overlaps[1][0]
            a, b = bubbles[ia], bubbles[ib]
            shared = _intersection(a, b)
            iou = shared / max(1.0, _area(a) + _area(b) - shared)
            ac, bc = _centre(a), _centre(b)
            axis = 1 if abs(ac[1] - bc[1]) >= abs(ac[0] - bc[0]) else 0
            separation = abs(ac[axis] - bc[axis])
            extent = min(float(a[axis + 2] - a[axis]), float(b[axis + 2] - b[axis]))
            if iou < 0.35 and separation >= 0.40 * extent:
                cut = _gap_cut(image, box, axis, (ac[axis] + bc[axis]) / 2.0)
                start, end = (int(box[1]), int(box[3])) if axis == 1 else (int(box[0]), int(box[2]))
                if min(cut - start, end - cut) >= max(25, int(0.20 * (end - start))):
                    first, second = box.copy(), box.copy()
                    first[axis + 2] = cut
                    second[axis] = cut
                    order = sorted((ia, ib), key=lambda i: _centre(bubbles[i])[axis])
                    parts = []
                    for part, index in zip((first, second), order):
                        owner = bubbles[index]
                        part[0] = max(part[0], owner[0])
                        part[1] = max(part[1], owner[1])
                        part[2] = min(part[2], owner[2])
                        part[3] = min(part[3], owner[3])
                        parts.append((part, owner, True))
                    if all(_area(part) >= 0.12 * area and part[2] - part[0] >= 24
                           and part[3] - part[1] >= 24 for part, _, _ in parts):
                        results.extend(parts)
                        continue
        owner = _best_owner(box, bubbles)
        if owner is not None:
            from modules.rendering.linked_lobes import split_source_paragraphs
            parts = split_source_paragraphs(image, box, bubbles[owner])
            if parts is not None:
                results.extend((part, bubbles[owner], True) for part in parts)
                continue
        results.append((box, bubbles[owner] if owner is not None else None, False))
    return results
