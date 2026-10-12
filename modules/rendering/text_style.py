"""Measure the source lettering of a text block from its glyph pixels.

Everything works on the difference between the original page and the cleaned page
(`glyph_mask`), so no OCR/detection data is needed. Pure numpy/cv2.

  glyph_mask      pixels the cleaner removed (text + outline + glow)
  cap_height      robust letter height in px (90th percentile of component heights)
  text_angle      rotation of the text line (deg, + = clockwise) from minAreaRect
  analyze_colors  fill colour / vertical-or-horizontal gradient / outline / glow colour
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def glyph_mask(orig: np.ndarray, cleaned: np.ndarray, xyxy, thr: int = 120) -> tuple[np.ndarray, tuple[int, int]]:
    """(bool mask of the box, (x1, y1) offset)."""
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), bool), (x1, y1)
    delta = orig[y1:y2, x1:x2].astype(np.int16) - cleaned[y1:y2, x1:x2].astype(np.int16)
    # the cleaner sometimes re-tints the whole panel: remove the box-wide median shift so only
    # pixels that changed MORE than their surroundings (the glyphs) count
    shift = np.median(delta.reshape(-1, 3), axis=0)
    d = np.abs(delta - shift).sum(axis=2)
    # keep the strongest ink only: a soft glow/halo is weaker than the glyphs themselves
    thr = max(thr, int(0.5 * np.percentile(d, 97)))
    m = d > thr
    if int(m.sum()) < 40:  # faint lettering (light grey on white...): retry with a lower threshold
        m = d > 70
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = _drop_fat_blobs(m).astype(bool)
    return m, (x1, y1)


def ink_mask_original(orig: np.ndarray, xyxy) -> tuple[np.ndarray, tuple[int, int]]:
    """Ink (text) pixels of a box measured on the ORIGINAL page only (no cleaned image needed).

    Background = median Lab colour of the box rim; ink = pixels clearly different from it (light-on-dark and
    dark-on-light both work). Returns (bool mask, (x1, y1))."""
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((1, 1), bool), (x1, y1)
    lab = cv2.cvtColor(np.ascontiguousarray(orig[y1:y2, x1:x2]), cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    b = max(2, int(0.06 * min(h, w)))
    rim = np.ones((h, w), bool)
    rim[b:h - b, b:w - b] = False
    bg = np.median(lab[rim], axis=0)
    d = np.linalg.norm(lab - bg, axis=2)
    t, _ = cv2.threshold(np.clip(d, 0, 255).astype(np.uint8).reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return d > max(28.0, min(float(t), 70.0)), (x1, y1)


def row_runs(mask: np.ndarray, min_height: int = 4) -> list[tuple[int, int]]:
    """Text rows of a mask as (y_start, y_end) runs of its row projection (mask coordinates)."""
    if mask.size == 0 or not mask.any():
        return []
    prof = mask.sum(axis=1).astype(float)
    on = prof > max(2.0, 0.03 * prof.max())
    on = cv2.morphologyEx(on.astype(np.uint8)[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))[:, 0].astype(bool)
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= min_height]


def split_words_by_weights(text: str, weights: list[float]) -> list[str]:
    """Split text into len(weights) consecutive word groups whose character counts follow the weights."""
    import re as _re
    words: list[str] = []
    for tok in text.split():  # numbers / dashes belong to the label before them: never start a row with one
        if words and (_re.fullmatch(r"[-–:/%.,\d]+", tok) or words[-1] in {"-", "–"}):
            words[-1] += " " + tok
        else:
            words.append(tok)
    n = len(weights)
    if n <= 1 or len(words) < n:
        return [text.strip()] if n <= 1 else []
    total_w, total_c = float(sum(weights)) or 1.0, float(sum(len(w) + 1 for w in words))
    out, i, cum_w, cum_c = [], 0, 0.0, 0.0
    for g, w in enumerate(weights):
        cum_w += w
        target = total_c * cum_w / total_w
        j = i
        take = []
        while j < len(words) - (n - g - 1):
            c2 = cum_c + len(words[j]) + 1
            if take and abs(c2 - target) > abs(cum_c - target):
                break
            take.append(words[j])
            cum_c = c2
            j += 1
        if g == n - 1:
            take += words[j:]
        out.append(" ".join(take))
        i = j
    return out


def source_rows(orig: np.ndarray, xyxy, text: str) -> list[str]:
    """Approximate source text per visual row: the OCR text split by the ink width of each row (empty if unmeasurable)."""
    mask, _ = ink_mask_original(orig, xyxy)
    runs = row_runs(mask)
    if len(runs) < 2:
        return []
    widths = []
    for a, b in runs:
        xs = np.nonzero(mask[a:b].any(axis=0))[0]
        widths.append(float(xs.max() - xs.min() + 1) if len(xs) else 1.0)
    return split_words_by_weights(text, widths)


def source_line_count(orig: np.ndarray, xyxy) -> int:
    """Number of visual text rows of the original lettering inside `xyxy` (0 when unmeasurable)."""
    mask, _ = ink_mask_original(orig, xyxy)
    return len(row_runs(mask))


def _drop_fat_blobs(m: np.ndarray, frac: float = 0.17) -> np.ndarray:
    """Remove areas too thick to be lettering (the cleaner re-painted background around the glyphs).

    Strokes are thin: any region that can hold a disk of radius frac * min(box h, w) is background.
    """
    h, w = m.shape
    r = max(6, int(frac * min(h, w)))
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    fat = (dist > r).astype(np.uint8)
    if not fat.any():
        return m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 3, 2 * r + 3))
    big = cv2.dilate(fat, k)
    return (m & (1 - big)).astype(np.uint8)


def components(mask: np.ndarray, min_area: int = 12):
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    return lab, st, comps


def cap_height(mask: np.ndarray) -> float:
    """Letter height in px: 90th percentile of component heights (punctuation and dots are shorter)."""
    _, st, comps = components(mask)
    if not comps:
        return 0.0
    hs = np.array([st[i, cv2.CC_STAT_HEIGHT] for i in comps], dtype=float)
    if len(hs) >= 4:
        hs = hs[hs >= 0.4 * hs.max()]  # drop punctuation/dots before taking the percentile
    return float(np.percentile(hs, 75 if len(hs) >= 6 else 90))


def line_pitch(mask: np.ndarray) -> tuple[float, int]:
    """(pitch between text lines px, number of lines) from the row projection of the mask."""
    prof = mask.sum(axis=1).astype(float)
    if prof.max() <= 0:
        return 0.0, 1
    on = prof > 0.12 * prof.max()
    on = cv2.morphologyEx(on.astype(np.uint8)[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8))[:, 0].astype(bool)
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= 4]
    if len(runs) < 2:
        return 0.0, max(1, len(runs))
    centers = np.array([(s + e) / 2 for s, e in runs])
    return float(np.median(np.diff(centers))), len(runs)


def text_angle(mask: np.ndarray, min_aspect: float = 1.8, max_abs: float = 38.0) -> tuple[float, tuple[float, float, float, float] | None]:
    """(angle deg clockwise, (cx, cy, long_len, short_len)) of the mask's minAreaRect.

    Returns angle 0 when the shape is too square to tell or the tilt is negligible/implausible.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 30:
        return 0.0, None
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    (cx, cy), (w, h), _ = cv2.minAreaRect(pts)
    box = cv2.boxPoints(((cx, cy), (w, h), _))
    e0, e1 = box[1] - box[0], box[2] - box[1]
    long_e = e0 if np.linalg.norm(e0) >= np.linalg.norm(e1) else e1
    long_len, short_len = max(w, h), min(w, h)
    ang = math.degrees(math.atan2(long_e[1], long_e[0]))
    while ang > 90:
        ang -= 180
    while ang <= -90:
        ang += 180
    if short_len <= 0 or long_len / short_len < min_aspect or abs(ang) > max_abs or abs(ang) < 2.5:
        return 0.0, (cx, cy, long_len, short_len)
    return float(ang), (cx, cy, long_len, short_len)


# ------------------------------------------------------------------ colours
def _lum(rgb) -> float:
    r, g, b = [float(v) for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _core_and_border(mask: np.ndarray):
    m = mask.astype(np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    if dist.max() <= 0:
        return mask, np.zeros_like(mask)
    half_stroke = float(np.percentile(dist[dist > 0], 90))
    if half_stroke < 4.0:  # thin strokes: no room for a separate outline
        return mask, np.zeros_like(mask)
    core = dist >= max(1.5, 0.45 * half_stroke)
    # band 1.8-2.8 px inside the glyph edge: skips the 1 px anti-aliased rim, so plain dark text
    # does not read as 'outlined'; real outlines are at least ~3 px wide
    border = mask & (dist >= 1.8) & (dist <= 2.8)
    return core, border


def enclosed_fill(crop: np.ndarray, mask: np.ndarray, min_pixels: int = 100):
    """Fill of outlined lettering whose FILL the cleaner could not tell from the background.

    When a thick outline is what the cleaner removed (cream rim around orange letters over a pink backdrop), the
    mask is a ring and the real fill sits in its holes. Hole pixels that differ clearly from every background
    colour seen just outside the lettering are the fill (letter counters look like the backdrop and are dropped).
    Returns (fill_selection, outline_rgb) or None."""
    m8 = mask.astype(np.uint8)
    n_mask = int(m8.sum())
    if n_mask < min_pixels:
        return None
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, contours, -1, 1, -1)
    holes = (filled > 0) & ~mask
    holes = cv2.erode(holes.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if int(holes.sum()) < 0.5 * n_mask:
        return None
    ring = cv2.dilate(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))).astype(bool) & ~cv2.dilate(
        filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    if int(ring.sum()) < 100:
        return None
    lab = cv2.cvtColor(np.clip(crop, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    rpx = lab[ring]
    sel = rpx[np.random.default_rng(0).choice(len(rpx), size=min(len(rpx), 4000), replace=False)]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, _, centers = cv2.kmeans(sel, 4, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    hy, hx = np.nonzero(holes)
    dmin = np.min(np.linalg.norm(lab[hy, hx][:, None, :] - centers[None], axis=2), axis=1)
    keep = dmin > 28.0
    if int(keep.sum()) < min_pixels:
        return None
    fill_sel = np.zeros(mask.shape, bool)
    fill_sel[hy[keep], hx[keep]] = True
    outline = tuple(int(v) for v in np.median(crop[mask], axis=0))
    return fill_sel, outline


def split_fill_outline(crop: np.ndarray, mask: np.ndarray, min_pixels: int = 150):
    """Separate lettering fill from its outline using colour clusters + depth inside the mask.

    The outline is the colour that dominates the glyph edge; the fill lies deeper inside. Works for thick
    outlines (cream rim around orange letters) where a plain 'median of the core' returns the outline colour.
    Returns (fill_selection bool mask, outline_rgb or None) or None when the mask is too small/degenerate."""
    m8 = mask.astype(np.uint8)
    if int(m8.sum()) < min_pixels:
        return None
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 3)
    ys, xs = np.nonzero(mask)
    px = crop[ys, xs]
    d = dist[ys, xs]
    lab = cv2.cvtColor(np.clip(px, 0, 255).astype(np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(lab), size=min(len(lab), 6000), replace=False)
    k = 4 if len(sel) >= 400 else 3
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    try:
        _, _, centers = cv2.kmeans(lab[sel], k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    except cv2.error:
        return None
    labels = np.argmin(((lab[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
    edge = d <= 2.5
    if int(edge.sum()) < 20:
        return None
    counts = np.bincount(labels[edge], minlength=k)
    out_c = int(counts.argmax())
    if counts[out_c] < 0.45 * edge.sum():
        return None
    depth90 = [float(np.percentile(d[labels == c], 90)) if (labels == c).sum() >= 8 else 0.0 for c in range(k)]
    # fill candidates: clusters (other than the edge colour) that reach deeper into the glyph than the edge colour
    deeper = [c for c in range(k) if c != out_c and (labels == c).sum() >= 0.04 * len(labels)
              and depth90[c] > depth90[out_c] + 1.5
              and float(np.linalg.norm(centers[c] - centers[out_c])) > 35]
    if not deeper:
        return np.logical_and(mask, dist >= 2.0) if int((dist >= 2.0)[mask].sum()) >= 40 else mask.copy(), None
    keep = np.isin(labels, deeper) & (d >= 2.5)
    if int(keep.sum()) < 40:
        return None
    fill_sel = np.zeros(mask.shape, bool)
    fill_sel[ys[keep], xs[keep]] = True
    out_px = px[(labels == out_c) & edge]
    return fill_sel, tuple(int(v) for v in np.median(out_px, axis=0))


def _fit_linear(coord: np.ndarray, colors: np.ndarray):
    """Per-channel linear fit colour = a + b*coord, refit once without outliers (residual > 2.5 MAD-sigma);
    returns (ends (2,3) at coord 0 and 1, R2 over all channels)."""
    X = np.stack([np.ones_like(coord), coord], axis=1)
    beta, *_ = np.linalg.lstsq(X, colors, rcond=None)
    res = np.linalg.norm(colors - X @ beta, axis=1)
    sigma = 1.4826 * float(np.median(np.abs(res - np.median(res)))) + 1e-6
    good = res <= np.median(res) + 2.5 * sigma
    if 0.5 * len(res) < int(good.sum()) < len(res):
        X, colors = X[good], colors[good]
        beta, *_ = np.linalg.lstsq(X, colors, rcond=None)
    pred = X @ beta
    ss_res = float(((colors - pred) ** 2).sum())
    ss_tot = float(((colors - colors.mean(axis=0)) ** 2).sum()) + 1e-6
    return np.stack([beta[0], beta[0] + beta[1]]), 1.0 - ss_res / ss_tot


def analyze_colors(orig: np.ndarray, mask: np.ndarray, offset: tuple[int, int], bg_rgb=None,
                   min_pixels: int = 80, grad_delta: float = 90.0, min_r2: float = 0.25) -> dict:
    """Describe the fill of the source lettering.

    Returns {"fill": (r,g,b) | None, "gradient": {"colors": [...], "vertical": bool} | None,
             "outline": (r,g,b) | None, "glow": (r,g,b) | None}.
    gradient needs a fitted colour change > grad_delta (sum of |dR|+|dG|+|dB|) along y or x
    that explains at least min_r2 of the colour variance, or two clearly different colour
    clusters that are spatially separated (two-tone letters).
    """
    out = {"fill": None, "gradient": None, "outline": None, "glow": None}
    if int(mask.sum()) < min_pixels:
        return out
    x0, y0 = offset
    h, w = mask.shape
    crop = orig[y0:y0 + h, x0:x0 + w].astype(np.float32)
    split = enclosed_fill(crop, mask) or split_fill_outline(crop, mask)
    if split is not None:
        core, outline_rgb = split
        border = np.zeros_like(mask)
        if int(core.sum()) < min_pixels * 0.5:
            core, outline_rgb = mask, None
    else:
        core, border = _core_and_border(mask)
        outline_rgb = None
        if int(core.sum()) < min_pixels * 0.5:
            core = mask
    ys, xs = np.nonzero(core)
    cols = crop[ys, xs]
    out["fill"] = tuple(int(v) for v in np.median(cols, axis=0))

    # outline: colour that dominates the glyph edge and differs clearly from the fill
    if outline_rgb is not None and float(np.abs(np.asarray(outline_rgb, np.float32) - np.median(cols, axis=0)).sum()) > 90:
        out["outline"] = outline_rgb
    elif int(border.sum()) >= 30 and core is not mask:
        bcol = np.median(crop[border], axis=0)
        if float(np.abs(bcol - np.median(cols, axis=0)).sum()) > 110:
            out["outline"] = tuple(int(v) for v in bcol)
    # glow: soft halo ring just outside the mask that differs from the background
    if bg_rgb is not None:
        ring = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & ~cv2.dilate(
            mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if int(ring.sum()) >= 40:
            rcol = np.median(crop[ring], axis=0)
            if float(np.abs(rcol - np.asarray(bg_rgb, dtype=np.float32)).sum()) > 70 and out["outline"] is None:
                out["glow"] = tuple(int(v) for v in rcol)

    if len(xs) < min_pixels:
        return out
    span_y, span_x = max(1.0, ys.max() - ys.min()), max(1.0, xs.max() - xs.min())
    ty, tx = (ys - ys.min()) / span_y, (xs - xs.min()) / span_x
    ends_y, r2_y = _fit_linear(ty, cols)
    ends_x, r2_x = _fit_linear(tx, cols)
    dy, dx = float(np.abs(ends_y[1] - ends_y[0]).sum()), float(np.abs(ends_x[1] - ends_x[0]).sum())
    best = None
    if dy >= grad_delta and r2_y >= min_r2 and (dy * r2_y >= dx * r2_x):
        best = (ends_y, True)
    elif dx >= grad_delta and r2_x >= min_r2 and span_x > 2 * span_y * 0.6:
        best = (ends_x, False)
    if best is not None:
        ends, vertical = best
        out["gradient"] = {"colors": [tuple(int(np.clip(v, 0, 255)) for v in e) for e in ends], "vertical": vertical}
        return out

    # two-tone: two colour clusters that occupy different parts of the glyph block
    if len(cols) >= 200:
        data = cols.astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        _, labels, centers = cv2.kmeans(data, 2, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        share = np.bincount(labels, minlength=2) / len(labels)
        sep = float(np.abs(centers[0] - centers[1]).sum())
        if sep > 150 and share.min() > 0.2:
            mean_ty = [float(ty[labels == k].mean()) for k in (0, 1)]
            mean_tx = [float(tx[labels == k].mean()) for k in (0, 1)]
            if abs(mean_ty[0] - mean_ty[1]) > 0.3 and abs(mean_ty[0] - mean_ty[1]) >= abs(mean_tx[0] - mean_tx[1]):
                order = np.argsort(mean_ty)
                out["gradient"] = {"colors": [tuple(int(v) for v in centers[k]) for k in order], "vertical": True}
            elif abs(mean_tx[0] - mean_tx[1]) > 0.3:
                order = np.argsort(mean_tx)
                out["gradient"] = {"colors": [tuple(int(v) for v in centers[k]) for k in order], "vertical": False}
    return out


def ensure_contrast(colors: list[tuple[int, int, int]], bg_rgb, min_delta: float = 70.0):
    """Returns (colors, needs_outline_colour or None). Outline is the lightness opposite of the fill average."""
    if bg_rgb is None:
        return colors, None
    bad = [c for c in colors if abs(_lum(c) - _lum(bg_rgb)) < min_delta]
    if not bad:
        return colors, None
    avg = sum(_lum(c) for c in colors) / len(colors)
    return colors, ((0, 0, 0) if avg > 140 else (255, 255, 255))


# ------------------------------------------------------------- merged blocks
def split_groups(mask: np.ndarray, cap_h: float, gap_factor: float = 1.1, min_area: int = 40):
    """Vertically separated text groups of a mask: [(x1, y1, x2, y2, ink_pixels), ...] (mask coordinates).

    OCR sometimes merges unrelated UI texts (menu bar, question, username) into one block. Rows of ink
    separated by more than gap_factor * cap_h of empty space belong to different groups.
    """
    if cap_h <= 0 or not mask.any():
        return []
    rows = mask.sum(axis=1) > 0
    ys = np.nonzero(rows)[0]
    groups, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y - prev > gap_factor * cap_h:
            groups.append((start, prev))
            start = y
        prev = y
    groups.append((start, prev))
    out = []
    for y1, y2 in groups:
        sub = mask[y1:y2 + 1]
        xs = np.nonzero(sub.any(axis=0))[0]
        if len(xs) and int(sub.sum()) >= min_area:
            out.append((int(xs.min()), int(y1), int(xs.max()) + 1, int(y2) + 1, int(sub.sum())))
    return out


def allocate_lines(lines: list[str], weights: list[float]) -> list[list[str]]:
    """Distribute translated lines over groups in reading order proportionally to the groups' ink."""
    n = len(weights)
    if n == 0 or len(lines) < n:
        return []
    total_w = float(sum(weights)) or 1.0
    total_c = float(sum(max(1, len(l)) for l in lines))
    cum_w, targets = 0.0, []
    for w in weights:
        cum_w += w
        targets.append(total_c * cum_w / total_w)
    segs: list[list[str]] = [[] for _ in range(n)]
    g, cum = 0, 0.0
    for i, line in enumerate(lines):
        segs[g].append(line)
        cum += max(1, len(line))
        remaining_lines, remaining_groups = len(lines) - i - 1, n - g - 1
        if g < n - 1 and (cum >= targets[g] - 0.5 * len(line) or remaining_lines <= remaining_groups):
            g += 1
    return segs


def median_letter_height(mask: np.ndarray) -> float:
    """Median component height (robust against one huge headline in a merged block)."""
    _, st, comps = components(mask, min_area=20)
    if not comps:
        return 0.0
    return float(np.median([st[i, cv2.CC_STAT_HEIGHT] for i in comps]))


# ------------------------------------------------------- text groups (merged OCR blocks)
def ink_mask_local(orig: np.ndarray, xyxy, kernel: int = 25, thr: int = 28) -> tuple[np.ndarray, tuple[int, int]]:
    """Lettering ink from the ORIGINAL page via top-hat / black-hat on lightness: strokes narrower than `kernel`
    stand out from any flat or smoothly shaded background (purple bar, light board), light or dark text alike."""
    H, W = orig.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return np.zeros((1, 1), bool), (x1, y1)
    lum = cv2.cvtColor(np.ascontiguousarray(orig[y1:y2, x1:x2]), cv2.COLOR_RGB2LAB)[..., 0]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    top = cv2.morphologyEx(lum, cv2.MORPH_TOPHAT, ker)
    black = cv2.morphologyEx(lum, cv2.MORPH_BLACKHAT, ker)
    return (top > thr) | (black > thr), (x1, y1)


def text_groups(orig: np.ndarray, xyxy, gap_ratio: float = 0.8) -> list[dict]:
    """Visually separate text groups inside a box, top to bottom.

    Rows come from the local ink mask; full-width thin rules (board separators) are removed and split groups;
    consecutive rows belong to one group while the empty gap is below gap_ratio * the taller row.
    Each group: {"box": (x1, y1, x2, y2) in page coordinates, "rows": [(y1, y2, width)], "aspect": sum(width/height)}."""
    mask, (ox, oy) = ink_mask_local(orig, xyxy)
    if not mask.any():
        return []
    h, w = mask.shape
    width_on = mask.sum(axis=1)
    rule_rows = width_on >= 0.8 * w
    rule_rows = cv2.morphologyEx(rule_rows.astype(np.uint8)[:, None], cv2.MORPH_OPEN, np.ones((1, 1), np.uint8))[:, 0].astype(bool)
    clean = mask.copy()
    clean[rule_rows] = False
    runs = row_runs(clean, min_height=5)
    runs = [(a, b) for a, b in runs if b - a >= 8 or int(clean[a:b].sum()) > 60]  # drop specks
    if not runs:
        return []
    rows = []
    for a, b in runs:
        xs = np.nonzero(clean[a:b].any(axis=0))[0]
        rows.append((a, b, float(xs.max() - xs.min() + 1), int(xs.min()), int(xs.max()) + 1))
    # a rule between two runs is a hard separator
    rule_idx = np.nonzero(rule_rows)[0]
    groups: list[list[tuple]] = [[rows[0]]]
    for prev, cur in zip(rows, rows[1:]):
        gap = cur[0] - prev[1]
        hard = bool(((rule_idx > prev[1] - 2) & (rule_idx < cur[0] + 2)).any())
        if hard or gap > gap_ratio * min(prev[1] - prev[0], cur[1] - cur[0]):
            groups.append([cur])
        else:
            groups[-1].append(cur)
    out = []
    for g in groups:
        out.append({
            "box": (ox + min(r[3] for r in g), oy + g[0][0], ox + max(r[4] for r in g), oy + g[-1][1]),
            "rows": [(r[0], r[1], r[2]) for r in g],
            "aspect": float(sum(r[2] / max(1.0, r[1] - r[0]) for r in g)),
        })
    return out


def source_groups(orig: np.ndarray, xyxy, text: str, min_groups: int = 3):
    """(groups, approximate source text per group) when a block holds >= min_groups separate text groups,
    else ([], []). The OCR text is split by each group's aspect (text length ~ width / height)."""
    groups = text_groups(orig, xyxy)
    if len(groups) < min_groups:
        return [], []
    parts = split_words_by_weights(text, [g["aspect"] for g in groups])
    return groups, (parts if len(parts) == len(groups) else [])


def row_cap_height(orig: np.ndarray, xyxy) -> float:
    """Median height of the text rows of a box (local ink on the original page; 0 when unmeasurable).

    A glow / smear that glues letters into one huge component inflates the component-based cap height;
    the row height cannot exceed the real line height, so it bounds it."""
    mask, _ = ink_mask_local(orig, xyxy)
    if not mask.any():
        return 0.0
    runs = [b - a for a, b in row_runs(mask, min_height=8)]
    return float(np.median(runs)) if runs else 0.0


def looks_merged(groups: list[dict], min_groups: int = 4) -> bool:
    """True when `groups` (from text_groups) look like several unrelated texts glued into one OCR block:
    at least min_groups of them and NOT evenly spaced (a paragraph / status list has a regular row pitch)."""
    if len(groups) < min_groups:
        return False
    rows = [r for g in groups for r in g["rows"]]
    centers = [(r[0] + r[1]) / 2 for r in rows]
    pitch = np.diff(centers)
    if len(pitch) >= 2 and (pitch.max() - pitch.min()) <= 0.35 * float(np.median(pitch)):
        return False
    return True
