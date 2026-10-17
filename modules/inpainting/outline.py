"""Outline-aware glyph removal: letters that straddle a balloon outline.

A glyph that touches the balloon outline merges with it into one dark component, the bubble-outline protection mask
covers that whole component, and neither the first mask nor the residual / fringe passes may touch it, so the
fragments stay ("Uh..." -> "u", "But" -> "at"). Here the outline is *fitted* (robust cubic through the centre line of
the long thin dark curve, width from the distance transform), everything dark that is NOT on the fitted curve and
hugs the cleaned area is removed like any other leftover, and the outline is redrawn along the fitted curve over the
repaired stretch (same colour / width), so the outline stays intact and continuous.

Pure helpers (testable on synthetic pages): fit_outline, glyph_pieces, redraw.
Knobs: CLEAN_OUTLINE_GLYPHS (true), CLEAN_OUTLINE_REACH (px from the cleaned area, 24), CLEAN_FRINGE_CONTRAST (shared
with the fringe pass, Lab distance 40), CLEAN_FRINGE_MAX (longest piece side, 160).
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
    return os.environ.get("CLEAN_OUTLINE_GLYPHS", "true").strip().lower() not in ("false", "0", "no", "off")


def _ell(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _circle_from3(a, b, c):
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-6:
        return None
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.hypot(*(a - ctr)))


def _fit_circle(pts: np.ndarray, rng, tol: float) -> tuple[int, dict] | None:
    n = len(pts)
    best = None
    for _ in range(300):
        i, j, k = rng.choice(n, 3, replace=False)
        cr = _circle_from3(pts[i], pts[j], pts[k])
        if cr is None or not 25.0 <= cr[1] <= 4000.0:
            continue
        cnt = int((np.abs(np.hypot(*(pts - cr[0]).T) - cr[1]) < tol).sum())
        if best is None or cnt > best[0]:
            best = (cnt, cr[0], cr[1])
    if best is None:
        return None
    _, ctr, r = best
    inl = np.abs(np.hypot(*(pts - ctr).T) - r) < tol
    for _ in range(3):  # Kasa least squares on the inliers
        if inl.sum() < 12:
            return None
        q = pts[inl]
        A = np.column_stack([2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))])
        sol = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)[0]
        ctr = sol[:2]
        r = float(np.sqrt(max(sol[2] + ctr @ ctr, 1.0)))
        inl = np.abs(np.hypot(*(pts - ctr).T) - r) < tol
    if not 25.0 <= r <= 4000.0 or inl.sum() < 12:
        return None
    ang = np.arctan2(pts[inl, 1] - ctr[1], pts[inl, 0] - ctr[0])
    mid = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
    dev = (ang - mid + np.pi) % (2 * np.pi) - np.pi
    t = np.linspace(dev.min(), dev.max(), max(8, int(r * (dev.max() - dev.min()) * 2)))
    xy = ctr + r * np.column_stack([np.cos(mid + t), np.sin(mid + t)])
    return int(inl.sum()), dict(xy=xy, inl=inl)


def _fit_cubic(pts: np.ndarray, rng, tol: float) -> tuple[int, dict] | None:
    """Straight / gently bent stretch: chord of the densest line, then a trimmed cubic v(u) around it."""
    n = len(pts)
    best = None
    for _ in range(200):
        i, j = rng.choice(n, 2, replace=False)
        p, q = pts[i], pts[j]
        dv = q - p
        length = float(np.hypot(*dv))
        if length < 30:
            continue
        dv /= length
        nv = np.array([-dv[1], dv[0]])
        cnt = int((np.abs((pts - p) @ nv) < 4.0).sum())
        if best is None or cnt > best[0]:
            best = (cnt, p, dv)
    if best is None:
        return None
    _, p, d = best
    nrm = np.array([-d[1], d[0]])
    u = (pts - p) @ d
    v = (pts - p) @ nrm
    sel = np.abs(v) < 8.0
    coef = None
    for it in range(4):
        if sel.sum() < 20:
            return None
        span = u[sel].max() - u[sel].min()
        coef = np.polyfit(u[sel], v[sel], 3 if span >= 80 else 2)
        res = np.abs(v - np.polyval(coef, u))
        lo, hi = u[sel].min() - 15, u[sel].max() + 15
        sel = (res < tol) & (u >= lo) & (u <= hi)
    if sel.sum() < 12:
        return None
    umin, umax = float(u[sel].min()), float(u[sel].max())
    uu = np.linspace(umin, umax, int(umax - umin) * 2 + 2)
    xy = p + uu[:, None] * d + np.polyval(coef, uu)[:, None] * nrm
    return int(sel.sum()), dict(xy=xy, inl=sel)


def fit_outline(ink: np.ndarray, min_len: float = 40, min_pts: int = 30) -> dict | None:
    """Robust fit of the long thin dark curve in a window of `ink` (bool, glyph blobs allowed): the better of a circular
    arc (balloon seen over a long stretch, the arc continues through a gap on both sides) and a trimmed cubic (straight
    stretch). None when there is no clean arc. dict(xy=(M,2) centre line (window coords), width=px, pts=inlier xy)."""
    ink8 = ink.astype(np.uint8)
    if ink8.sum() < 60:
        return None
    dt = cv2.distanceTransform(ink8, cv2.DIST_L2, 3)
    ridge = (ink8 > 0) & (dt >= 1.0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    ys, xs = np.nonzero(ridge)
    if len(xs) < 30:
        return None
    rng = np.random.default_rng(0)
    if len(xs) > 900:
        keep = rng.choice(len(xs), 900, replace=False)
        ys, xs = ys[keep], xs[keep]
    pts = np.column_stack([xs, ys]).astype(np.float64)
    dts = dt[ys, xs]
    cands = [c for c in (_fit_circle(pts, rng, 2.0), _fit_cubic(pts, rng, 2.0)) if c is not None]
    if not cands:
        return None
    cnt, fit = max(cands, key=lambda c: c[0])
    xy, inl = fit["xy"], fit["inl"]
    length = float(np.hypot(*np.diff(xy, axis=0).T).sum())
    if cnt < min_pts or length < min_len:
        return None
    width = float(np.clip(2.0 * np.percentile(dts[inl], 75) - 1.0, 2.0, 14.0))
    return dict(xy=xy, width=width, pts=pts[inl])


def _draw(shape, xy: np.ndarray, px: float, aa: bool = False, offset=(0, 0)) -> np.ndarray:
    """Polyline `px` pixels wide (OpenCV draws thickness + 2 pixels)."""
    thickness = int(round(px)) - 2
    canvas = np.zeros(shape[:2], np.uint8)
    pts = np.round((xy + np.array(offset, np.float64)) * 16.0).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [pts], False, 255, thickness=max(1, int(thickness)),
                  lineType=cv2.LINE_AA if aa else cv2.LINE_8, shift=4)
    return canvas


def _reaches_border(xy: np.ndarray, shape, margin: int = 12) -> bool:
    hh, ww = shape[:2]
    for x, y in (xy[0], xy[-1]):
        if x < margin or y < margin or x > ww - 1 - margin or y > hh - 1 - margin:
            return True
    return False


def glyph_pieces(cleaned: np.ndarray, done: np.ndarray, exclude: np.ndarray, protected: np.ndarray | None,
                 debug: list | None = None) -> tuple[np.ndarray, list, int]:
    """Glyph leftovers glued to a balloon outline. `protected` = bubble outline protection mask. Returns (keep uint8
    mask of the glyph pixels to erase, bands [dict(xy=full-page centre line, width, color)] to redraw, piece count).
    A piece is accepted like a fringe fragment (letter sized, close to the cleaned area `done`, flat surrounding) but it
    may touch the outline: the outline is the fitted curve, only ink off that curve counts as glyph."""
    h, w = cleaned.shape[:2]
    keep = np.zeros((h, w), np.uint8)
    bands: list = []
    why = (lambda *a: debug.append(a)) if debug is not None else (lambda *a: None)
    if protected is None or not np.any(protected) or not (done > 0).any():
        return keep, bands, 0
    reach = int(_f("CLEAN_OUTLINE_REACH", 24))
    con = _f("CLEAN_FRINGE_CONTRAST", 40.0)
    max_side = int(_f("CLEAN_FRINGE_MAX", 160))
    neighbour = int(_f("CLEAN_OUTLINE_NEIGHBOUR", 60))
    # Candidates must lie within `reach` of `done` AND within `neighbour` of an outline: when the closest outline pixel is
    # further than both radii (+ ellipse rounding) from every cleaned pixel nothing can qualify, so skip the heavy work.
    dist_to_outline = cv2.distanceTransform((protected == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    if float(dist_to_outline[done > 0].min()) > reach + neighbour + 3:
        return keep, bands, 0
    near = cv2.dilate((done > 0).astype(np.uint8), _ell(2 * reach + 1)) > 0
    ys, xs = np.nonzero(near)
    pad = 60
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1), max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    sub = np.ascontiguousarray(cleaned[y0:y1, x0:x1])
    lab = cv2.cvtColor(sub, cv2.COLOR_RGB2LAB)
    labf = lab.astype(np.float32)
    bg = cv2.medianBlur(lab, 41).astype(np.float32)
    ink_all = np.linalg.norm(labf - bg, axis=2) > con
    # thick chunks of a big letter fill the median window (their 'background' is black): dark ink on paper is also what is
    # much darker than the local brightest surround
    paper = cv2.dilate(lab[..., 0], _ell(81))
    ink_all |= (paper.astype(np.float32) - labf[..., 0]) > max(con, 60.0)
    # ink still standing inside the cleaned area counts too: the protection restore puts the original pixels back there
    cand = ink_all & near[y0:y1, x0:x1] & ~(exclude[y0:y1, x0:x1] > 0)
    prot = protected[y0:y1, x0:x1] > 0
    ink_halo = cv2.dilate(ink_all.astype(np.uint8), _ell(5)) > 0
    count = 0
    sh, sw = sub.shape[:2]
    # only ink within reach of a protected outline is ours (it may be glued to the outline or hover right next to it);
    # leftovers elsewhere belong to the fringe pass
    cand &= cv2.dilate(prot.astype(np.uint8), _ell(2 * neighbour + 1)) > 0
    sh, sw = sub.shape[:2]
    nd, dlab, dst, _ = cv2.connectedComponentsWithStats(
        cv2.dilate((done[y0:y1, x0:x1] > 0).astype(np.uint8), _ell(9)), connectivity=8)
    for di in range(1, nd):
        rx, ry, rw_, rh_, _da = [int(v) for v in dst[di]]
        wy0, wy1 = max(0, ry - reach - 30), min(sh, ry + rh_ + reach + 30)
        wx0, wx1 = max(0, rx - reach - 30), min(sw, rx + rw_ + reach + 30)
        kc = cand[wy0:wy1, wx0:wx1] & ~(keep[y0:y1, x0:x1][wy0:wy1, wx0:wx1] > 0)
        why('win', (x0 + wx0, y0 + wy0, wx1 - wx0, wy1 - wy0), int(kc.sum()))
        if kc.sum() < 20:
            continue
        # the outline = the long thin dark curve(s) of the ink around the block (a letter cut loose from the outline still
        # belongs to the stretch the outline continues through)
        wink = ink_all[wy0:wy1, wx0:wx1].copy()
        arcs = []
        rest = wink.copy()
        for rank in range(4):
            fit = fit_outline(rest, min_len=40 if rank == 0 else 80, min_pts=30 if rank == 0 else 50)
            if fit is None:
                break
            if rank and not _reaches_border(fit["xy"], rest.shape):  # other arcs must run through the window; a chain of
                break  # glyph strokes is not an outline
            arcs.append(fit)
            rest &= ~(_draw(rest.shape, fit["xy"], int(np.ceil(fit["width"])) + 6) > 0)
        if not arcs:
            why((x0 + wx0, y0 + wy0), 'nofit')
            continue
        wink_d = cv2.dilate(wink.astype(np.uint8), _ell(5)) > 0
        pieces = np.zeros(wink.shape, bool)
        for _attempt in range(2):
            band = np.zeros(wink.shape, bool)
            for fit in arcs:
                band |= _draw(wink.shape, fit["xy"], int(np.ceil(fit["width"])) + 2) > 0
            pieces = kc & ~band
            opened = cv2.morphologyEx(pieces.astype(np.uint8), cv2.MORPH_OPEN, _ell(3))
            pieces &= cv2.dilate(opened, _ell(5)) > 0  # fit-error slivers along the outline are not glyphs
            if not pieces.any():
                break
            # the fitted curve must continue the visible outline on both sides of the gap; a curve that drifts away from the
            # drawn line next to the letters would erase / redraw the wrong thing
            pyx = np.nonzero(pieces)
            pc = np.array([pyx[1].mean(), pyx[0].mean()])
            far = cv2.distanceTransform((~pieces).astype(np.uint8), cv2.DIST_L2, 3)
            good = []
            for fit in arcs:
                xy = fit["xy"]
                ix = np.clip(np.round(xy[:, 0]).astype(int), 0, wink.shape[1] - 1)
                iy = np.clip(np.round(xy[:, 1]).astype(int), 0, wink.shape[0] - 1)
                sel = (np.hypot(*(xy - pc).T) <= 90) & (far[iy, ix] > 14)
                if sel.sum() < 12 or wink_d[iy[sel], ix[sel]].mean() >= 0.85:
                    good.append(fit)
                else:
                    why((x0 + wx0, y0 + wy0), 'arc-drift', round(float(wink_d[iy[sel], ix[sel]].mean()), 2))
            if len(good) == len(arcs):
                break
            arcs = good
            if not arcs:
                pieces = np.zeros(wink.shape, bool)
                break
        why('pieces', int(pieces.sum()), len(arcs))
        if pieces.sum() < 20:
            continue
        labf_win = labf[wy0:wy1, wx0:wx1]
        halo_win = ink_halo[wy0:wy1, wx0:wx1]
        shp = wink.shape
        ng, glab, gstats, _ = cv2.connectedComponentsWithStats(cv2.dilate(pieces.astype(np.uint8), _ell(5)), connectivity=8)
        took = 0
        for g in range(1, ng):
            gx, gy, gw, gh, _a = [int(v) for v in gstats[g]]
            core = (glab == g) & pieces
            at = (x0 + wx0 + gx, y0 + wy0 + gy, gw, gh)
            if core.sum() < 20:
                why(at, 'tiny')
                continue
            if max(gw, gh) > max_side or min(gw, gh) > 80:
                why(at, 'size')
                continue
            cys, cxs = np.nonzero(core)
            mid = np.array([cxs.mean(), cys.mean()])
            # enclosed by an arc: the nearest fitted point must not be an end of the arc (letters stranded past the end of
            # an outline are not "on" it)
            enclosed = False
            for fit in arcs:
                k = int(np.argmin(np.hypot(*(fit["xy"] - mid).T)))
                if 12 <= k <= len(fit["xy"]) - 12:
                    enclosed = True
                    break
            if not enclosed:
                why(at, 'off-arc')
                continue
            (_c, (rw2, rh2), _a2) = cv2.minAreaRect(np.column_stack([cxs, cys]).astype(np.float32))
            wmax = max(fit_w["width"] for fit_w in arcs)
            if max(rw2, rh2) >= 45 and min(rw2, rh2) <= 1.6 * wmax + 2:  # long thin curve = left-over of the outline itself
                why(at, 'arc-remnant', round(max(rw2, rh2)), round(min(rw2, rh2)))
                continue
            gcore = np.zeros((h, w), bool)
            gcore[y0 + wy0:y0 + wy0 + shp[0], x0 + wx0:x0 + wx0 + shp[1]] = core
            if exclude[gcore].mean() > 0.3:
                continue
            rp = 12
            by0, by1 = max(0, gy - rp), min(shp[0], gy + gh + rp)
            bx0, bx1 = max(0, gx - rp), min(shp[1], gx + gw + rp)
            full = core[by0:by1, bx0:bx1].astype(np.uint8)
            ring = (cv2.dilate(full, _ell(2 * rp + 1)) > 0) & (full == 0) & ~halo_win[by0:by1, bx0:bx1]
            rl = labf_win[by0:by1, bx0:bx1][ring]
            if len(rl) < 40:
                why(at, 'ring')
                continue
            flat = (np.linalg.norm(rl - np.median(rl, axis=0), axis=1) <= 14.0).mean()
            if flat < 0.6:
                why(at, 'busy', round(float(flat), 2))
                continue
            keep[gcore] = 255
            took += 1
        if took:
            # hairlines / dots of the same letters (too thin to survive the opening) next to what was accepted
            wk = keep[y0:y1, x0:x1][wy0:wy1, wx0:wx1] > 0
            near_acc = cv2.dilate(wk.astype(np.uint8), _ell(25)) > 0
            keep[y0 + wy0:y0 + wy1, x0 + wx0:x0 + wx1][kc & ~band & near_acc] = 255
            for fit in arcs:
                ix = np.clip(np.round(fit["pts"][:, 0]).astype(int) + wx0, 0, sw - 1)
                iy = np.clip(np.round(fit["pts"][:, 1]).astype(int) + wy0, 0, sh - 1)
                color = np.median(sub[iy, ix].reshape(-1, 3), axis=0)
                bands.append(dict(xy=fit["xy"] + np.array([x0 + wx0, y0 + wy0], np.float64), width=fit["width"], color=color))
            count += took
    return keep, bands, count


def gap_mask(shape, keep: np.ndarray, bands: list) -> np.ndarray:
    """The stretch of outline right next to the erased letters (uint8 0/255): erased with them (otherwise the inpainter
    copies the dark letter remains next to it into the fill) and redrawn afterwards."""
    out = np.zeros(shape[:2], np.uint8)
    if not bands or not keep.any():
        return out
    near = cv2.dilate((keep > 0).astype(np.uint8), _ell(31)) > 0
    for b in bands:
        out[(_draw(shape, b["xy"], int(np.ceil(b["width"])) + 6) > 0) & near] = 255
    return out


def redraw(img: np.ndarray, zone: np.ndarray, bands: list) -> np.ndarray:
    """Redraw each fitted outline (colour / width of the original) inside `zone` (the repaired area), anti-aliased."""
    if not bands:
        return img
    out = img.astype(np.float32)
    region = cv2.dilate((zone > 0).astype(np.uint8), _ell(5)) > 0
    for b in bands:
        alpha = _draw(img.shape, b["xy"], max(2.0, b["width"] - 1.0), aa=True).astype(np.float32) / 255.0
        alpha *= region
        out = out * (1 - alpha[..., None]) + np.asarray(b["color"], np.float32) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
