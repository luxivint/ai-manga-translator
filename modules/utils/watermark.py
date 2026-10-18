"""Competitor scanlation watermark / banner handling (WATERMARK_POLICY=clean).

`match_watermark` decides WHAT is a watermark (never the site's own brand), `region_mask` decides WHERE to erase:
the whole banner, not just the glyphs, so LaMa repaints a clean area instead of smudging half a logo."""
from __future__ import annotations

import os
import re
import warnings

import cv2
import numpy as np

WATERMARK_PATTERNS = (
    r"vortex\s*scans",
    r"[a-z0-9-]+\.(?:com|net|org|io|me|gg|xyz|co|to)\b",
    r"read\s+(?:this\s+series\s+)?first\s+at",
    r"discord\.gg|t\.me/|www\.|https?://",
    r"\b(?:patreon|ko-?fi)\b",
    # scan-group credit strips without a domain ("THUNDERSCANS TEAM", "VIOLETSCANS", "PRODUCE BY X TEAM")
    r"\b[a-z0-9]{3,}scans?\b",
    r"\bscans?\s+(?:team|group)\b|\bscanlations?\b",
    r"\b(?:translated|cleaned|redrawn|typeset(?:ted)?|proofread|produced?)\s+by\b.*\b(?:team|scans?)\b",
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


def flat_fill(original: np.ndarray, cleaned: np.ndarray, wm_mask: np.ndarray, max_dev: float = 24.0,
              min_frac: float | None = None) -> np.ndarray:
    """Where the erased area is surrounded by an (almost) flat colour (e.g. a black banner page) paint it with that
    colour instead of trusting the inpainter, which invents textured grey on flat backgrounds."""
    if wm_mask is None or not wm_mask.any():
        return cleaned
    # 0.92: a plate half over a white gutter and half over a shirt is NOT flat (0.7 painted white boxes over art)
    min_frac = float(os.environ.get("WATERMARK_FLAT_MIN_FRAC", "0.92")) if min_frac is None else min_frac
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


# --------------------------------------------------------------------------------------- chapter plates (non-OCR)
# A scan group's plate (logo + text tag) is stamped identically on many pages of a chapter, but OCR reads it only on a
# few of them. Plates read once (seeds) become templates; every page is template-matched, and the plate's footprint is
# learned from the repetition itself: pixels that stay the same across all occurrences belong to the plate (logo, tag,
# arrow tip), the artwork behind it changes from page to page. Knobs: WATERMARK_PLATES (true), WATERMARK_PLATE_MATCH
# (normalised correlation, 0.72), WATERMARK_PLATE_MIN_HITS (occurrences needed to learn a footprint, 3).
_CHAPTER_PLATES: list = []


def plates_enabled() -> bool:
    return os.environ.get("WATERMARK_PLATES", "true").strip().lower() not in ("false", "0", "no", "off")


def _gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)


def find_template(gray: np.ndarray, tpl: np.ndarray, thr: float, max_hits: int = 8) -> list:
    """Top-left corners (x, y, score) of non-overlapping matches of `tpl` in `gray` scoring >= thr."""
    th, tw = tpl.shape[:2]
    if gray.shape[0] < th or gray.shape[1] < tw or tpl.std() < 8:
        return []
    res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
    hits = []
    for _ in range(max_hits):
        _mn, mx, _l, (x, y) = cv2.minMaxLoc(res)
        if mx < thr:
            break
        hits.append((int(x), int(y), float(mx)))
        res[max(0, y - th):y + th, max(0, x - tw):x + tw] = -1.0
    return hits


def _window(img: np.ndarray, x0: int, y0: int, ww: int, wh: int) -> np.ndarray:
    """RGB window (float, NaN outside the page)."""
    H, W = img.shape[:2]
    out = np.full((wh, ww, 3), np.nan, np.float32)
    sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + wh)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _drop_edge_runs(m: np.ndarray) -> np.ndarray:
    """Remove runs that reach the window edge along rows (left/right) and columns (top/bottom): a horizontal panel border
    or gutter the plate sits on repeats too, but it continues beyond the plate."""
    m = m.copy()
    for arr in (m, m.T):
        for row in arr:
            if row[0]:
                k = int(np.argmin(row)) if not row.all() else len(row)
                row[:k] = False
            if len(row) and row[-1]:
                rev = row[::-1]
                k = int(np.argmin(rev)) if not rev.all() else len(rev)
                row[len(row) - k:] = False
    return m


def _build_model(pages: list, grays: list, pi: int, box: tuple, thr: float, min_hits: int) -> dict | None:
    """Template = the plate crop `box` of page `pi`; matched on every page; footprint = what stays the same across the
    occurrences (logo, tag, arrow) inside a window around the template."""
    x1, y1, x2, y2 = box
    tpl = grays[pi][y1:y2, x1:x2].copy()
    if tpl.size == 0 or tpl.std() < 8:
        return None
    hits = [(qi, x, y) for qi, g in enumerate(grays) for x, y, _s in find_template(g, tpl, thr)]
    th, tw = tpl.shape
    wx, wy = 3 * th + tw // 4, th
    ww, wh = tw + 2 * wx, th + 2 * wy
    foot = None
    if len(hits) >= min_hits:
        stack = np.stack([_window(pages[qi][0], x - wx, y - wy, ww, wh) for qi, x, y in hits])
        with warnings.catch_warnings():  # all-NaN columns outside the page
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
            dev = np.nanmax(np.abs(stack - med[None]), axis=3)  # (N, wh, ww)
        valid = ~np.isnan(dev)
        same = (np.nan_to_num(dev, nan=999.0) <= 22.0) & valid
        share = same.sum(0) / np.maximum(valid.sum(0), 1)
        stable = (share >= 0.8) & (valid.sum(0) >= min_hits)
        stable = _drop_edge_runs(stable)  # panel borders / gutters running through the window are page structure
        core_px = stable.copy()
        stable = cv2.morphologyEx(stable.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        _n, lab = cv2.connectedComponents(stable)
        core = lab[wy:wy + th, wx:wx + tw]
        ids = np.unique(core[core > 0])
        f = np.isin(lab, ids).astype(np.uint8) if len(ids) else np.zeros_like(stable)
        ys_, xs_ = np.nonzero(f)
        f = np.zeros_like(f)
        if len(xs_):
            # convex hull: a translucent tag's bar changes with the art behind it (not 'stable'), but it spans the stable
            # logo and letters; the hull + a margin covers the whole plate
            cv2.fillConvexPoly(f, cv2.convexHull(np.stack([xs_, ys_], 1).astype(np.int32)), 1)
        k = max(7, th // 4) | 1
        f = cv2.dilate(f, np.ones((k, k), np.uint8))  # plate rim + anti-aliasing
        f[wy:wy + th, wx:wx + tw] = 1
        if f.sum() <= 6 * th * tw:  # a flat page area that happened to repeat is not a plate
            foot = f.astype(bool)
            core_px &= foot
            # textured plate pixels only (a flat white gutter matches anywhere): they verify a weak template hit
            gm = cv2.cvtColor(np.nan_to_num(med, nan=0.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            edges = cv2.dilate((cv2.Canny(gm, 40, 120) > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            core_px &= edges
            if core_px.sum() >= 80:
                return dict(tpl=tpl, wx=wx, wy=wy, foot=foot, hits=len(hits), thr=thr, ref=med, core=core_px)
    return dict(tpl=tpl, wx=wx, wy=wy, foot=foot, hits=len(hits), thr=thr)


def _verified(image: np.ndarray, m: dict, x: int, y: int) -> bool:
    """A (weak) template hit is the plate when its textured core pixels match the learned plate (logo / letters)."""
    fh, fw = m["core"].shape
    win = _window(image, x - m["wx"], y - m["wy"], fw, fh)
    core = m["core"] & ~np.isnan(win[..., 0])
    if core.sum() < 0.5 * m["core"].sum():
        return False
    d = np.abs(win[core] - m["ref"][core]).max(axis=1)
    return float((d <= 28.0).mean()) >= float(os.environ.get("WATERMARK_PLATE_VERIFY", "0.38"))


def _explained(models: list, gray: np.ndarray, box: tuple, thr: float) -> bool:
    x1, y1, x2, y2 = box
    win = gray[max(0, y1 - 8):y2 + 8, max(0, x1 - 8):x2 + 8]
    for m in models:
        th, tw = m["tpl"].shape
        if win.shape[0] >= th and win.shape[1] >= tw and find_template(win, m["tpl"], thr, 1):
            return True
        # the model's template lies inside the candidate box (candidate = bigger crop of the same plate)
        if m["hits"] and find_template(gray[max(0, y1 - th):y2 + th, max(0, x1 - tw):x2 + tw], m["tpl"], thr, 1):
            return True
    return False


def edge_candidate(box: tuple, W: int, edge: int = 60) -> bool:
    """A scan-group tag sits at the left / right page edge and is a compact one / two line strip."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return 60 <= bw <= 0.6 * W and 14 <= bh <= 90 and (x1 <= edge or x2 >= W - edge)


def learn_plates(pages: list, candidates=None) -> list:
    """pages = [(image RGB, blocks)]. Seeds = boxes OCR read as a watermark, plus OCR-free candidates
    `candidates(image) -> [box]` (text-like strips at the page edge, WATERMARK_PLATE_MINE) that must then really repeat
    over the chapter. Returns plate models dict(tpl, wx, wy, foot, hits) and stores them for `plate_mask`."""
    global _CHAPTER_PLATES
    _CHAPTER_PLATES = []
    if not plates_enabled() or not pages:
        return []
    thr = float(os.environ.get("WATERMARK_PLATE_MATCH", "0.72"))
    min_hits = int(os.environ.get("WATERMARK_PLATE_MIN_HITS", "3"))
    grays = [_gray(img) for img, _b in pages]
    seeds = []
    for pi, (img, blocks) in enumerate(pages):
        H, W = img.shape[:2]
        for blk in blocks or []:
            if not match_watermark(getattr(blk, "text", "") or ""):
                continue
            x1, y1, x2, y2 = [int(v) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
            x1, y1, x2, y2 = max(0, x1 + 2), max(0, y1 + 2), min(W, x2 - 2), min(H, y2 - 2)
            if x2 - x1 >= 30 and y2 - y1 >= 10 and (x2 - x1) * (y2 - y1) <= 0.05 * H * W:
                seeds.append(("ocr", pi, (x1, y1, x2, y2)))
    if candidates is not None and os.environ.get("WATERMARK_PLATE_MINE", "true").strip().lower() not in ("false", "0", "no", "off"):
        for pi, (img, _b) in enumerate(pages):
            try:
                boxes = candidates(img)
            except Exception as exc:  # optional detector: never break learning
                print(f"  [watermark] plate candidates skipped: {exc}", flush=True)
                break
            seeds += [("mined", pi, tuple(int(v) for v in b)) for b in boxes if edge_candidate(b, img.shape[1])]
    models: list = []
    for kind, pi, box in seeds:
        if _explained(models, grays[pi], box, thr):
            continue
        m = _build_model(pages, grays, pi, box, thr if kind == "ocr" else min(thr, float(os.environ.get(
            "WATERMARK_PLATE_MINE_MATCH", "0.66"))), min_hits)
        if m is None:
            continue
        if kind == "mined" and (m["hits"] < max(min_hits, 4) or m["foot"] is None):
            continue  # an OCR-free plate must really repeat over the chapter
        models.append(m)
        th, tw = m["tpl"].shape
        print(f"  [watermark] plate learned ({kind}) from page {pi + 1}: {tw}x{th} hits={m['hits']} "
              f"footprint={'yes' if m['foot'] is not None else 'no'}", flush=True)
    _CHAPTER_PLATES = models
    return models


def set_chapter_plates(models: list) -> None:
    global _CHAPTER_PLATES
    _CHAPTER_PLATES = list(models or [])


def _to_page_edge(one: np.ndarray, reach: int) -> None:
    """A plate stamped against the page edge (logo square there is translucent, so not learned): when the plate mask
    ends within `reach` px of the left / right edge, the rows it occupies there are filled up to the edge (in place)."""
    ys, xs = np.nonzero(one)
    if not len(xs):
        return
    W = one.shape[1]
    x0, x1 = int(xs.min()), int(xs.max())
    if 0 < x0 <= reach:
        rows = np.unique(ys[xs <= x0 + 20])
        one[rows, :x0 + 1] = 255
    if x1 < W - 1 and x1 >= W - 1 - reach:
        rows = np.unique(ys[xs >= x1 - 20])
        one[rows, x1:] = 255


def plate_mask(image: np.ndarray, models: list | None = None) -> np.ndarray:
    """Full-page uint8 mask (255) of every known chapter plate on this page (learned footprint, else the padded match
    united with the plate hull)."""
    models = _CHAPTER_PLATES if models is None else models
    H, W = image.shape[:2]
    out = np.zeros((H, W), np.uint8)
    if not models or not plates_enabled():
        return out
    thr = float(os.environ.get("WATERMARK_PLATE_MATCH", "0.72"))
    g = _gray(image)
    for m in models:
        th, tw = m["tpl"].shape
        if m.get("core") is not None:  # weak hits (translucent tag over other art) verified on the learned plate
            hits = [h for h in find_template(g, m["tpl"], 0.45, 12) if h[2] >= m.get("thr", thr) or _verified(image, m, h[0], h[1])]
        else:
            hits = find_template(g, m["tpl"], m.get("thr", thr))
        for x, y, _s in hits:
            if m["foot"] is not None:
                fh, fw = m["foot"].shape
                x0, y0 = x - m["wx"], y - m["wy"]
                sx0, sy0, sx1, sy1 = max(0, x0), max(0, y0), min(W, x0 + fw), min(H, y0 + fh)
                out[sy0:sy1, sx0:sx1][m["foot"][sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]] = 255
                # + the uniformly tinted tag around the match (a translucent bar is not 'stable' across pages)
                one = region_mask(image, (x, y, x + tw, y + th))
                one[sy0:sy1, sx0:sx1][m["foot"][sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]] = 255
                _to_page_edge(one, int(os.environ.get("WATERMARK_PLATE_EDGE", "40")))
                out |= one
            else:
                out |= region_mask(image, (x, y, x + tw, y + th))
    return out
