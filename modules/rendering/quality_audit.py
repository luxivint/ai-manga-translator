"""Read-only quality diagnostics for the manga translation pipeline.

The audit never edits the rendered image and never rejects a chapter on a
heuristic alone. Operators can inspect JSON reports before deciding to rerun a
page. This is deliberately conservative: artwork and punctuation can resemble
residual glyphs.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def _crop_rect(box, height, width):
    try:
        x1, y1, x2, y2 = (int(v) for v in box)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def audit_cleaning(original: np.ndarray, cleaned: np.ndarray, blocks: list) -> dict:
    """Potential residuals inside bright original OCR boxes before new text."""
    if original.shape != cleaned.shape:
        raise ValueError("original and cleaned images must match")
    height, width = original.shape[:2]
    warnings = []
    measured = 0
    translated = 0
    for index, block in enumerate(blocks):
        source = (getattr(block, "text", "") or "").strip()
        target = (getattr(block, "translation", "") or "").strip()
        if not target or len(target) <= 1:
            if len(source) >= 3 and not getattr(block, "is_ad", False):
                warnings.append({"kind": "missing_translation", "block": index})
            continue
        translated += 1
        if getattr(block, "text_class", "") != "text_bubble":
            continue
        rect = _crop_rect(getattr(block, "xyxy", None), height, width)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        if x2-x1 < 10 or y2-y1 < 10:
            continue
        before = original[y1:y2, x1:x2]
        after = cleaned[y1:y2, x1:x2]
        gray_before = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        white_ratio = float((gray_before > 230).mean())
        # Colourful/illustrated balloons need shape-aware review instead of a
        # misleading black-ink statistic.
        if white_ratio < 0.60:
            continue
        dark_before = gray_before < 145
        total = int(dark_before.sum())
        if total < 12:
            continue
        measured += 1
        same_dark = dark_before & (gray_after < 155)
        remaining = float(same_dark.sum()) / total
        if remaining > 0.15:
            warnings.append({"kind": "possible_residual_ink", "block": index,
                             "fraction": round(remaining, 3)})
    return {"translated_blocks": translated,
            "measured_white_text_blocks": measured,
            "warnings": warnings}


def audit_layout(items: list, image_shape: tuple) -> list[dict]:
    """Flag only clear unrotated canvas overflows; soft warning, not a failure."""
    height, width = image_shape[:2]
    warnings = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if not (item.get("text") or "").strip():
            warnings.append({"kind": "empty_render_item", "item": index})
            continue
        try:
            if abs(float(item.get("rotation") or 0)) > 0.01:
                continue
            x, y = item.get("position", (0, 0))
            x, y = float(x), float(y)
            w, h = float(item.get("width") or 0), float(item.get("height") or 0)
            if w > 0 and h > 0 and (x < -2 or y < -2 or x + w > width + 2 or y + h > height + 2):
                warnings.append({"kind": "outside_page", "item": index,
                                 "box": [round(x), round(y), round(w), round(h)]})
        except (TypeError, ValueError):
            continue
    return warnings


def write_report(path, original, cleaned, blocks, items) -> dict:
    report = audit_cleaning(original, cleaned, blocks)
    report["render_items"] = len(items)
    report["warnings"].extend(audit_layout(items, original.shape))
    if report["translated_blocks"] and not items:
        report["warnings"].append({"kind": "no_render_items"})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return report
