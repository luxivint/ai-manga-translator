"""Residual pass: catch source text the first cleaning mask missed (small / rotated / low-contrast / UI text).

After LaMa the page is segmented again. Text-like components that sit close to the already cleaned area (same panel),
that are not inside a block we deliberately left untouched (no translation / kept watermark) and not on a bubble
outline, are added to a second mask (glyph-hugging via local_batch._tight_glyph_mask) and inpainted with the same
fill logic. Rotated lines are measured with cv2.minAreaRect, not axis-aligned boxes.

Knobs (env, defaults ON): CLEAN_RESIDUAL_PASS (true/false), CLEAN_RESIDUAL_PASSES (max 2), CLEAN_RESIDUAL_NEAR (px
from the first mask, default 150), CLEAN_RESIDUAL_CONF (segmenter confidence, 0.1), CLEAN_RESIDUAL_MAX_FRAC (thickest
allowed component as a share of the page width, 0.25), CLEAN_RESIDUAL_MIN_CONTRAST (Lab distance, 28), CLEAN_RESIDUAL_GROW (extra mask dilation px, 5).
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def enabled() -> bool:
    return os.environ.get("CLEAN_RESIDUAL_PASS", "true").strip().lower() not in ("false", "0", "no", "off")


def _ell(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def untouched_boxes(blocks, h: int, w: int, pad: int = 6) -> np.ndarray:
    """Blocks cleaning deliberately skips (no translation, kept watermark): never add residual text inside them."""
    out = np.zeros((h, w), np.uint8)
    for blk in blocks or []:
        tr = (getattr(blk, "translation", "") or "").strip()
        if len(tr) > 1 and not getattr(blk, "is_ad", False):
            continue
        x1, y1, x2, y2 = [int(v) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
        out[max(0, y1 - pad):min(h, y2 + pad), max(0, x1 - pad):min(w, x2 + pad)] = 255
    return out


def select_components(seg: np.ndarray, cleaned: np.ndarray, first_mask: np.ndarray, exclude: np.ndarray,
                      protected: np.ndarray | None = None, debug: list | None = None) -> tuple[np.ndarray, int]:
    """Accepted text-like coarse components (uint8 mask 0/255) and how many there are. Pure function (testable with a
    fake segmenter mask)."""
    h, w = seg.shape[:2]
    near_px = int(_f("CLEAN_RESIDUAL_NEAR", 150))
    max_frac = _f("CLEAN_RESIDUAL_MAX_FRAC", 0.25)
    min_con = _f("CLEAN_RESIDUAL_MIN_CONTRAST", 28.0)
    near = cv2.dilate((first_mask > 0).astype(np.uint8), _ell(2 * near_px + 1)) > 0
    lab = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB).astype(np.float32)
    grouped = cv2.dilate((seg > 0).astype(np.uint8), _ell(9))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    keep = np.zeros((h, w), np.uint8)
    count = 0
    why = (lambda *a: debug.append(a)) if debug is not None else (lambda *a: None)
    for i in range(1, n):
        x, y, cw, ch, _area = [int(v) for v in stats[i]]
        comp = labels[y:y + ch, x:x + cw] == i
        s_roi = (seg[y:y + ch, x:x + cw] > 0) & comp
        if s_roi.sum() < 30:
            why((x, y, cw, ch), 'tiny')
            continue
        if not near[y:y + ch, x:x + cw][comp].any():
            why((x, y, cw, ch), 'far')
            continue
        if exclude[y:y + ch, x:x + cw][s_roi].mean() > 0.3:
            why((x, y, cw, ch), 'excluded')
            continue
        if protected is not None and protected[y:y + ch, x:x + cw][s_roi].mean() > 0.3:
            why((x, y, cw, ch), 'protected')
            continue
        pts = np.column_stack(np.nonzero(s_roi))[:, ::-1].astype(np.float32)
        (_, _), (rw, rh), _ang = cv2.minAreaRect(pts)  # rotation-safe size
        long_side, short_side = max(rw, rh), min(rw, rh)
        if short_side < 5 or short_side > max_frac * min(h, w):  # lines may span the page; blobs may not
            why((x, y, cw, ch), 'size', long_side, short_side)
            continue
        pad = 14
        y0, y1, x0, x1 = max(0, y - pad), min(h, y + ch + pad), max(0, x - pad), min(w, x + cw + pad)
        ring = (cv2.dilate(comp.astype(np.uint8), _ell(2 * pad + 1)) > 0)
        full = np.zeros((y1 - y0, x1 - x0), bool)
        full[y - y0:y - y0 + ch, x - x0:x - x0 + cw] = comp
        ring = (cv2.dilate(full.astype(np.uint8), _ell(2 * pad + 1)) > 0) & ~full
        if ring.sum() < 40:
            why((x, y, cw, ch), 'ring')
            continue
        roi = lab[y0:y1, x0:x1]
        bg = np.median(roi[ring], axis=0)
        diff = np.linalg.norm(roi - bg, axis=2)
        sfull = np.zeros_like(full)
        sfull[y - y0:y - y0 + ch, x - x0:x - x0 + cw] = s_roi
        glyph = sfull & (diff > min_con)
        frac = glyph.sum() / max(1, sfull.sum())
        if glyph.sum() < 25 or frac < 0.08 or frac > 0.75:
            why((x, y, cw, ch), 'ink', int(glyph.sum()), round(float(frac), 2))
            continue  # no real ink / a solid blob (art)
        keep[y:y + ch, x:x + cw][comp] = 255
        count += 1
    return keep, count


def fringe_components(cleaned: np.ndarray, done: np.ndarray, exclude: np.ndarray, protected: np.ndarray | None = None,
                      debug: list | None = None) -> tuple[np.ndarray, int]:
    """Pixel-based leftovers the segmenter is blind to: cut-off glyph pieces / dots hugging the cleaned area (short
    letters at the edge of a line, a trailing '.', an 'at'). A component of ink (clearly different from the local
    flat background) that touches the neighbourhood of `done` is accepted when it is small (letter sized), sits on a
    flat background (its ring is one colour, so no art / hair / line work) and is not an outline or an untouched block.
    Pure function (testable on synthetic pages). Knobs: CLEAN_FRINGE_PASS, CLEAN_FRINGE_REACH (px, 20),
    CLEAN_FRINGE_MAX (longest side px, 160), CLEAN_FRINGE_CONTRAST (Lab distance, 40)."""
    h, w = cleaned.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    why = (lambda *a: debug.append(a)) if debug is not None else (lambda *a: None)
    reach = int(_f("CLEAN_FRINGE_REACH", 20))
    max_side = int(_f("CLEAN_FRINGE_MAX", 160))
    con = _f("CLEAN_FRINGE_CONTRAST", 40.0)
    if not (done > 0).any():
        return keep, 0
    near = cv2.dilate((done > 0).astype(np.uint8), _ell(2 * reach + 1)) > 0
    ys, xs = np.nonzero(near)
    y0, y1, x0, x1 = max(0, ys.min() - 20), min(h, ys.max() + 21), max(0, xs.min() - 20), min(w, xs.max() + 21)
    sub = np.ascontiguousarray(cleaned[y0:y1, x0:x1])
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)
    bg = cv2.medianBlur(lab, 41).astype(np.float32)
    ink = (np.linalg.norm(lab.astype(np.float32) - bg, axis=2) > con).astype(np.uint8)
    ink &= (near[y0:y1, x0:x1]).astype(np.uint8)
    ink[(done[y0:y1, x0:x1] > 0)] = 0
    if protected is not None:  # bubble outline never belongs to a leftover glyph (and must not glue pieces together)
        ink[protected[y0:y1, x0:x1] > 0] = 0
    grouped = cv2.dilate(ink, _ell(5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    labf = lab.astype(np.float32)
    count = 0
    for i in range(1, n):
        x, y, cw, ch, _a = [int(v) for v in stats[i]]
        comp = labels[y:y + ch, x:x + cw] == i
        core = comp & (ink[y:y + ch, x:x + cw] > 0)
        if core.sum() < 20:
            why((x + x0, y + y0, cw, ch), 'tiny')
            continue
        if max(cw, ch) > max_side or min(cw, ch) > 80:
            why((x + x0, y + y0, cw, ch), 'size')
            continue
        gx, gy = x + x0, y + y0
        if exclude[gy:gy + ch, gx:gx + cw][core].mean() > 0.3:
            why((gx, gy, cw, ch), 'excluded')
            continue
        if protected is not None and protected[gy:gy + ch, gx:gx + cw][core].mean() > 0.3:
            why((gx, gy, cw, ch), 'protected')
            continue
        pad = 12
        by0, by1, bx0, bx1 = max(0, y - pad), min(y1 - y0, y + ch + pad), max(0, x - pad), min(x1 - x0, x + cw + pad)
        full = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
        full[y - by0:y - by0 + ch, x - bx0:x - bx0 + cw] = comp
        ring = (cv2.dilate(full, _ell(2 * pad + 1)) > 0) & (full == 0) & (ink[by0:by1, bx0:bx1] == 0)  # other ink is judged on its own
        rl = labf[by0:by1, bx0:bx1][ring]
        if len(rl) < 40:
            why((gx, gy, cw, ch), 'ring')
            continue
        flat = (np.linalg.norm(rl - np.median(rl, axis=0), axis=1) <= 14.0).mean()
        if flat < 0.6:
            why((gx, gy, cw, ch), 'busy', round(float(flat), 2))
            continue
        keep[gy:gy + ch, gx:gx + cw][core] = 255
        count += 1
    return keep, count


def residual_pass(lb, image: np.ndarray, cleaned: np.ndarray, first_mask: np.ndarray, blocks, config) -> tuple[np.ndarray, dict]:
    """Run up to CLEAN_RESIDUAL_PASSES residual rounds. Returns (cleaned, report{added, passes, px, mask}); report['mask']
    is the union of all added masks (uint8) so the fill QA can treat them like the first mask."""
    rep = dict(added=0, passes=0, px=0, mask=None)
    if not enabled() or first_mask is None or not (first_mask > 0).any():
        return cleaned, rep
    try:
        h, w = cleaned.shape[:2]
        conf = _f("CLEAN_RESIDUAL_CONF", 0.1)
        exclude = untouched_boxes(blocks, h, w)
        protected = lb.bubble_outline_protection_mask(image, blocks)
        if protected is not None:
            protected = cv2.dilate((protected > 0).astype(np.uint8), _ell(5))
        done = (first_mask > 0).astype(np.uint8) * 255
        cur = cleaned
        for _ in range(int(max(1, min(2, _f("CLEAN_RESIDUAL_PASSES", 2))))):
            seg, fine = lb._tiled_segment(lb._get_segmenter(), cur, conf)
            keep, n = select_components(seg, cur, done, exclude, protected)
            if n == 0:
                break
            tb = cv2.dilate(keep, _ell(29))
            add = lb._tight_glyph_mask(cur, cv2.bitwise_and(seg, keep), int(os.environ.get("MASK_CLOSE", "7")),
                                       int(os.environ.get("MASK_DILATE", "9")), cv2.bitwise_and(fine, keep), tb)
            grow = int(_f("CLEAN_RESIDUAL_GROW", 5))
            if grow > 0:  # leftovers are usually partly cut glyphs with dark anti-aliased halos
                add = cv2.dilate(add, _ell(2 * grow + 1))
            add[exclude > 0] = 0
            if protected is not None:
                add[protected > 0] = 0
            if not add.any():
                break
            filled = lb.block_inpaint(cur, add, config)
            cur = filled
            rep["mask"] = add if rep["mask"] is None else cv2.bitwise_or(rep["mask"], add)
            done = cv2.bitwise_or(done, add)
            rep["added"] += n
            rep["px"] += int((add > 0).sum())
            rep["passes"] += 1
        if os.environ.get("CLEAN_FRINGE_PASS", "true").strip().lower() not in ("false", "0", "no", "off"):
            keep, n = fringe_components(cur, done, exclude, protected)
            if n:
                add = cv2.dilate(keep, _ell(2 * int(_f("CLEAN_RESIDUAL_GROW", 5)) + 1))
                add[exclude > 0] = 0
                if protected is not None:
                    add[protected > 0] = 0
                if add.any():
                    cur = lb.block_inpaint(cur, add, config)
                    rep["mask"] = add if rep["mask"] is None else cv2.bitwise_or(rep["mask"], add)
                    rep["added"] += n
                    rep["fringe"] = n
                    rep["px"] += int((add > 0).sum())
        return cur, rep
    except Exception as exc:  # best effort
        rep["error"] = str(exc)
        return cleaned, rep
