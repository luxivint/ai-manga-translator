"""Objective quality check + calmer fallback for inpainted (LaMa) fills, and the metrics used by scripts/replay_clean.py.

LaMa on textured / gradient / UI backgrounds sometimes leaves dithered noise (blue 'static', pink-green speckle) or a
visible tone step at the mask border. `qa_pass` measures every filled component against a ring of untouched pixels and,
only when the fill is objectively bad, replaces *just the masked pixels* by a calmer fill:

  candidates = plane + smooth ring residual | Telea | Navier-Stokes inpaint on a downscaled context
  chosen     = the candidate with the smallest border step against the ring, then
               + grain matched to the ring + membrane correction of the border offset (Poisson-lite seamless blend);
  chroma-only defects (pink/green speckle on grey) replace only Lab a/b and keep LaMa's luma; strong LaMa edges
  (line art continued through the hole) are never replaced.

Knobs (env, defaults ON): CLEAN_FILL_QA, CLEAN_FILL_QA_NOISE (luma band-pass MAD ratio fill/ring),
CLEAN_FILL_QA_NOISE_ABS, CLEAN_FILL_QA_CHROMA (chroma ratio), CLEAN_FILL_QA_CHROMA_ABS, CLEAN_FILL_QA_MIN_AREA,
CLEAN_FILL_QA_RING (ring px).
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
    return os.environ.get("CLEAN_FILL_QA", "true").strip().lower() not in ("false", "0", "no", "off")


def _ell(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _lab(img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100.0 / 255.0
    return lab


def _from_lab(lab: np.ndarray) -> np.ndarray:
    l8 = lab.copy()
    l8[..., 0] *= 255.0 / 100.0
    return cv2.cvtColor(np.clip(l8, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def _band(lab: np.ndarray) -> np.ndarray:
    """Band-pass (sigma 1 - sigma 8): speckle / dither / blotches, not the slow gradient and not the pixel grain."""
    return cv2.GaussianBlur(lab, (0, 0), 1.0) - cv2.GaussianBlur(lab, (0, 0), 8.0)


def _mad(v: np.ndarray) -> float:
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else 0.0


def _strong_edges(lab_l: np.ndarray, thr: float = 8.0, min_extent: int = 30) -> np.ndarray:
    """Line art / real structure: LONG runs of pixels with a clear luma step (>= ~30 levels), dilated. Excluded from the
    noise statistics. Short strong blobs (isolated dots) are not structure: they are exactly the speckle we look for."""
    g = cv2.GaussianBlur(lab_l, (0, 0), 1.5)
    gm = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    e = (gm > thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(e, connectivity=8)
    big = np.zeros(n, np.uint8)
    big[1:] = (np.maximum(st[1:, cv2.CC_STAT_WIDTH], st[1:, cv2.CC_STAT_HEIGHT]) >= min_extent).astype(np.uint8)
    return cv2.dilate(big[lab], _ell(5)) > 0


def blob_density(lab_l: np.ndarray, region: np.ndarray, dev: float = 7.0) -> float:
    """Compact dots / blotches (not lines) per 10k px of `region`: pixels that differ from a large median of the same
    image by > `dev` levels, grouped; lines (long / elongated groups) do not count."""
    l8 = np.clip(lab_l * (255.0 / 100.0), 0, 255).astype(np.uint8)
    d = np.abs(l8.astype(np.int16) - cv2.medianBlur(l8, 25).astype(np.int16))
    b = ((d > dev) & region).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    cnt = 0
    for i in range(1, n):
        a, w, h = int(st[i, 4]), int(st[i, 2]), int(st[i, 3])
        if 15 <= a <= 800 and max(w, h) <= 32 and max(w, h) / max(1, min(w, h)) <= 3.5:
            cnt += 1
    return cnt / max(1, int(region.sum())) * 1e4


# ------------------------------------------------------------------------------------------------ detection
def component_stats(orig: np.ndarray, out: np.ndarray, comp: np.ndarray, others: np.ndarray | None = None,
                    ring_px: int | None = None) -> dict | None:
    """How different is the fill `out` inside bool mask `comp` from the untouched ring of `orig` around it?

    luma_ratio / chroma_ratio = robust (MAD) band-pass spread of the fill divided by the ring's (structure pixels
    excluded); step = Lab distance of the fill border to the ring just outside. None when there is no usable ring."""
    ring_px = int(ring_px or _f("CLEAN_FILL_QA_RING", 40))
    c8 = comp.astype(np.uint8)
    inner = cv2.erode(c8, _ell(5)) > 0
    if inner.sum() < 30:
        inner = comp.copy()
    gap = cv2.dilate(c8, _ell(5)) > 0
    ring = (cv2.dilate(c8, _ell(2 * ring_px + 1)) > 0) & ~gap
    if others is not None:
        ring &= ~others
    if ring.sum() < 150 or inner.sum() < 30:
        return None
    lo, li = _lab(out), _lab(orig)
    bo, bi = _band(lo), _band(li)
    inner_ns = inner & ~_strong_edges(lo[..., 0])
    ring_ns = ring & ~_strong_edges(li[..., 0])
    if inner_ns.sum() < 30 or ring_ns.sum() < 100:
        return None
    fill_l = _mad(bo[..., 0][inner_ns])
    fill_p = float(np.percentile(np.abs(bo[..., 0][inner_ns]), 92))
    ring_p = float(np.percentile(np.abs(bi[..., 0][ring_ns]), 92))
    blot = blob_density(lo[..., 0], inner)
    ring_blot = blob_density(li[..., 0], ring)
    ring_l = _mad(bi[..., 0][ring_ns]) + 0.4
    fill_c = float(np.hypot(_mad(bo[..., 1][inner_ns]), _mad(bo[..., 2][inner_ns])))
    ring_c = float(np.hypot(_mad(bi[..., 1][ring_ns]), _mad(bi[..., 2][ring_ns]))) + 0.3
    band_in = comp & ~(cv2.erode(c8, _ell(7)) > 0)
    band_out = (cv2.dilate(c8, _ell(11)) > 0) & ~(cv2.dilate(c8, _ell(3)) > 0)
    if others is not None:
        band_out &= ~others
    step = 0.0
    if band_in.sum() >= 10 and band_out.sum() >= 10:
        step = float(np.linalg.norm(np.median(lo[band_in], axis=0) - np.median(li[band_out], axis=0)))
    return dict(area=int(comp.sum()), fill_p=fill_p, ring_p=ring_p, blot=blot, ring_blot=ring_blot, fill_l=fill_l, ring_l=ring_l, luma_ratio=fill_l / ring_l,
                fill_c=fill_c, ring_c=ring_c, chroma_ratio=fill_c / ring_c, step=step)


def bad_kind(st: dict | None) -> str:
    """'' (fine) | 'luma' (dithered static / blotches in luminance) | 'chroma' (coloured speckle on a calm background)
    | 'speck' (sparse compact dots / blotches, e.g. grey stipple on white)."""
    if st is None or st["area"] < _f("CLEAN_FILL_QA_MIN_AREA", 1200):
        return ""
    luma = st["luma_ratio"] >= _f("CLEAN_FILL_QA_NOISE", 2.5) and st["fill_l"] >= _f("CLEAN_FILL_QA_NOISE_ABS", 3.5)
    chroma = st["chroma_ratio"] >= _f("CLEAN_FILL_QA_CHROMA", 3.0) and st["fill_c"] >= _f("CLEAN_FILL_QA_CHROMA_ABS", 2.0)
    speck = st["blot"] >= _f("CLEAN_FILL_QA_BLOBS", 10.0) and st["blot"] >= _f("CLEAN_FILL_QA_BLOB_RATIO", 4.0) * (st["ring_blot"] + 1.0)
    return "luma" if luma else ("chroma" if chroma else ("speck" if speck else ""))


def is_bad(st: dict | None) -> bool:
    return bool(bad_kind(st))


# ------------------------------------------------------------------------------------------ calmer fills
def _plane_candidate(img: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    from modules.inpainting import patch_match as pm
    plane = pm._fit_plane(img.astype(np.float32), valid)
    if plane is None:
        return None
    resid = img.astype(np.float32) - plane
    rv = resid[valid]
    rstd = np.array([_mad(rv[:, c]) for c in range(3)])
    if rstd.max() > 14.0:
        return None  # not planar (stripes, art): the inpaint candidates handle it
    lf, den = pm._norm_blur(np.clip(resid, -3 * rstd - 2, 3 * rstd + 2), valid.astype(np.float32), 10.0)
    lf = lf * np.clip(den * 8.0, 0.0, 1.0)[..., None]
    return np.clip(plane + lf, 0, 255)


def _inpaint_candidate(img: np.ndarray, hole: np.ndarray, flag: int) -> np.ndarray:
    """cv2.inpaint on a downscaled context (<= ~320 px long side): fast and no pixel-level noise propagation."""
    h, w = hole.shape
    k = max(1, int(np.ceil(max(h, w) / 320.0)))
    small = cv2.resize(img, (max(2, w // k), max(2, h // k)), interpolation=cv2.INTER_AREA)
    # any coverage counts (thin outlier strokes must not vanish when the mask is downscaled)
    hm = (cv2.resize(hole.astype(np.float32), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 0.02).astype(np.uint8)
    hm = cv2.dilate(hm, np.ones((3, 3), np.uint8))
    res = cv2.inpaint(np.ascontiguousarray(small), hm, 5, flag)
    return cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _boundary_step(cand: np.ndarray, img: np.ndarray, hole: np.ndarray) -> float:
    c8 = hole.astype(np.uint8)
    b_in = hole & ~(cv2.erode(c8, _ell(7)) > 0)
    b_out = (cv2.dilate(c8, _ell(11)) > 0) & ~(cv2.dilate(c8, _ell(3)) > 0)
    if b_in.sum() < 10 or b_out.sum() < 10:
        return 1e9
    a = _lab(np.clip(cand, 0, 255).astype(np.uint8))[b_in]
    b = _lab(img)[b_out]
    return float(np.linalg.norm(np.median(a, axis=0) - np.median(b, axis=0)))


def _membrane_fix(cand: np.ndarray, img: np.ndarray, hole: np.ndarray, ok: np.ndarray | None = None,
                  sigma: float = 12.0) -> np.ndarray:
    """Poisson-lite seamless blend: the colour offset between the candidate's border band and the ring just outside
    the hole is interpolated smoothly into the hole (normalised blur) and fades with the distance from the border."""
    c8 = hole.astype(np.uint8)
    adj = (cv2.dilate(c8, _ell(7)) > 0) & ~hole
    if ok is not None:
        adj &= ok  # not the cut-off letters / outlines next to the hole
    b_in = hole & ~(cv2.erode(c8, _ell(5)) > 0)
    if adj.sum() < 10 or b_in.sum() < 10:
        return cand
    adjf = adj.astype(np.float32)
    near = cv2.GaussianBlur(img.astype(np.float32) * adjf[..., None], (0, 0), 3.0)
    near_w = cv2.GaussianBlur(adjf, (0, 0), 3.0)
    target = near / np.maximum(near_w, 1e-4)[..., None]
    w = (b_in & (near_w > 0.05)).astype(np.float32)  # only where the ring really is known (no 'black' targets)
    if w.sum() < 10:
        return cand
    num = cv2.GaussianBlur((target - cand) * w[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    field = num / np.maximum(den, 1e-4)[..., None]
    dist = cv2.distanceTransform(c8, cv2.DIST_L2, 3)
    fade = np.exp(-dist / (2.0 * sigma))[..., None]
    return np.clip(cand + field * fade * hole[..., None], 0, 255)


def _grain(shape, ring_l: float, seed: int = 11) -> np.ndarray:
    """Luma grain whose band-pass spread matches the ring's (only used on textured rings)."""
    rng = np.random.default_rng(seed)
    n = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), 0.8)
    b = _band(n[..., None])
    b = b[..., 0] if b.ndim == 3 else b
    return n * (max(ring_l - 0.4, 0.0) / max(_mad(b), 1e-6))


def calm_fill(orig: np.ndarray, out: np.ndarray, comp: np.ndarray, others: np.ndarray | None, kind: str,
              st: dict | None = None) -> tuple[np.ndarray, str] | None:
    """Replacement crop for `out` (masked pixels re-filled calmly) and the candidate name, or None."""
    c8 = comp.astype(np.uint8)
    hole = cv2.dilate(c8, _ell(3)) > 0
    if others is not None:  # neighbouring strokes of the same streak are holes too (no ring bleed from them)
        hole |= others & (cv2.dilate(c8, _ell(41)) > 0)
    # Thin dark/bright things next to the hole (cut-off letters of the source text, bubble outline) are not background:
    # they must not feed the fill (a dark smudge otherwise). Treat them as part of the hole for the candidates.
    l8 = np.clip(_lab(orig)[..., 0] * (255.0 / 100.0), 0, 255).astype(np.uint8)
    outl = np.abs(l8.astype(np.int16) - cv2.medianBlur(l8, 31).astype(np.int16)) > 12
    near_hole = cv2.dilate(hole.astype(np.uint8), _ell(61)) > 0
    outl = (cv2.dilate(outl.astype(np.uint8), _ell(11)) > 0) & near_hole
    cand_hole = hole | outl
    valid = ~(cv2.dilate(cand_hole.astype(np.uint8), _ell(5)) > 0)
    valid &= ~_strong_edges(_lab(orig)[..., 0])
    if valid.sum() < 150:
        return None
    lo = _lab(out)
    cands = []
    pl = _plane_candidate(orig, valid)
    if pl is not None:
        cands.append(("plane", pl))
    cands.append(("telea", _inpaint_candidate(orig, cand_hole, cv2.INPAINT_TELEA)))
    cands.append(("ns", _inpaint_candidate(orig, cand_hole, cv2.INPAINT_NS)))
    # LaMa's own low frequencies (denoised): the safe choice when the ring is not representative of the hole's
    # interior (dark bar inside a white page, ...). Ring candidates that disagree with it a lot are dropped.
    hw = hole.astype(np.float32)
    ys_, xs_ = np.nonzero(hole)
    sg = float(np.clip(min(np.ptp(ys_), np.ptp(xs_)) / 4.0, 7.0, 14.0))
    lp = cv2.GaussianBlur(lo * hw[..., None], (0, 0), sg) / np.maximum(cv2.GaussianBlur(hw, (0, 0), sg), 1e-3)[..., None]
    lp_rgb = _from_lab(np.where(hole[..., None], lp, lo)).astype(np.float32)
    scored = []
    for n_, c_ in cands:
        d = np.abs(_lab(np.clip(c_, 0, 255).astype(np.uint8)) - lp)[hole]
        if d[:, 0].mean() <= 12.0 and d[:, 1:].mean() <= 10.0:
            # border step + disagreement with LaMa's own low frequencies (a ring candidate that drags in a dark
            # neighbour region scores worse than LaMa's denoised fill)
            scored.append((_boundary_step(c_, orig, hole) + 0.6 * d[:, 0].mean() + 0.3 * d[:, 1:].mean(), n_, c_))
    scored.append((_boundary_step(lp_rgb, orig, hole), "lama_lp", lp_rgb))
    _, name, _best = min(scored, key=lambda t: t[0])
    best = _membrane_fix(_best, orig, hole, ~cand_hole)
    lc = _lab(np.clip(best, 0, 255).astype(np.uint8))
    new = lo.copy()
    if kind == "chroma" and st and st["luma_ratio"] >= 1.6 and st["fill_l"] >= 1.5:
        kind = "luma"  # luma is blotchy as well (grey mottling next to the colour speckle): replace everything
    if kind in ("luma", "speck"):
        new = lc.copy()
        ring_l = st["ring_l"] if st else 0.0
        if ring_l > 1.0:
            new[..., 0] += _grain(comp.shape, ring_l)
    else:  # chroma only: keep LaMa's luma structure, calm the colour
        new[..., 1:] = lc[..., 1:]
    strong = _strong_edges(lo[..., 0], 12.0)  # LaMa line art continued through the hole stays
    alpha = np.maximum(cv2.GaussianBlur(c8.astype(np.float32), (0, 0), 1.5), c8.astype(np.float32))
    if kind != "speck":
        alpha *= 1.0 - strong.astype(np.float32) * 0.9
    return _from_lab(lo * (1 - alpha[..., None]) + new * alpha[..., None]), name


def _groups(m: np.ndarray):
    """Streaks of nearby strokes are one component (they share one fill / one ring)."""
    n, glab = cv2.connectedComponents(cv2.dilate(m.astype(np.uint8), _ell(9)))
    H, W = m.shape
    for gi in range(1, n):
        gmask = (glab == gi) & m
        ys, xs = np.nonzero(gmask)
        if len(ys) < 30:
            continue
        y0, y1 = max(0, ys.min() - 48), min(H, ys.max() + 49)
        x0, x1 = max(0, xs.min() - 48), min(W, xs.max() + 49)
        comp = gmask[y0:y1, x0:x1]
        yield (y0, y1, x0, x1), comp, m[y0:y1, x0:x1] & ~comp


def flat_modes(lab: np.ndarray, ring: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Flat colours (Lab, <= 3 modes of >= 10 % share) of the ring and how much of the ring they explain, or None."""
    px = lab[ring]
    if len(px) < 150:
        return None
    q = np.round(px / 4.0).astype(np.int32)
    keys, inv, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(-cnt)[:3]
    modes = [np.median(px[inv.ravel() == o], axis=0) for o in order if cnt[o] >= 0.10 * len(px)]
    if not modes:
        return None
    modes = np.array(modes)
    d = np.min(np.linalg.norm(px[:, None, :] - modes[None], axis=2), axis=1)
    return modes, float((d <= 5.0).mean())


def _run_labels(comp: np.ndarray, lab: np.ndarray, modes: np.ndarray, hole: np.ndarray, reach: int = 40) -> np.ndarray:
    """Per masked pixel: index of the flat colour that BOTH ends of its horizontal run of masked pixels agree on (the
    pixels `skip` px outside the run; -1 when they disagree / are not one of the flat colours / the run touches the
    crop). This keeps a tone step (white band over grey) a straight line through the hole."""
    H, W = comp.shape
    d = np.linalg.norm(lab[:, :, None, :] - modes[None, None], axis=3)
    near = np.where(d.min(2) <= 6.0, d.argmin(2), -1)
    near[hole] = -1
    out = np.full((H, W), -1, np.int16)
    for y in range(H):
        row = comp[y]
        if not row.any():
            continue
        e = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(e == 1)[0], np.nonzero(e == -1)[0]):
            la = lb_ = -1
            for x in range(a - 1, max(-1, a - 1 - reach), -1):
                if near[y, x] >= 0:
                    la = near[y, x]
                    break
            for x in range(b, min(W, b + reach)):
                if near[y, x] >= 0:
                    lb_ = near[y, x]
                    break
            if la >= 0 and la == lb_:
                out[y, a:b] = la
    return out


def flat_snap(cleaned: np.ndarray, mask: np.ndarray, skip: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Ghost removal: a balloon / flat panel is one flat colour, a LaMa fill inside it must be too. When the ring around a
    filled group is flat (<= 3 colours explain >= CLEAN_FLAT_COVER of it) every filled pixel within CLEAN_FLAT_TOL (Lab
    distance, default 14) of one of these colours snaps to the nearest one. Pixels further away (outline, art) stay.
    Knobs: CLEAN_FLAT_SNAP (true), CLEAN_FLAT_TOL, CLEAN_FLAT_COVER (0.8)."""
    rep = dict(groups=0, snapped=0)
    if os.environ.get("CLEAN_FLAT_SNAP", "true").strip().lower() in ("false", "0", "no", "off") or mask is None or not (mask > 0).any():
        return cleaned, rep
    tol, cover = _f("CLEAN_FLAT_TOL", 14.0), _f("CLEAN_FLAT_COVER", 0.8)
    out = cleaned.copy()
    try:
        for (y0, y1, x0, x1), comp, others in _groups(mask > 0):
            if skip is not None and (skip[y0:y1, x0:x1] > 0)[comp].mean() > 0.3:
                continue
            c8 = comp.astype(np.uint8)
            ring = (cv2.dilate(c8, _ell(31)) > 0) & ~(cv2.dilate(c8, _ell(5)) > 0) & ~others
            lab = _lab(cleaned[y0:y1, x0:x1])
            ring &= ~_strong_edges(lab[..., 0])
            res = flat_modes(lab, ring)
            if res is None or res[1] < cover:
                continue
            modes, _cov = res
            rgb = np.array([_from_lab(m_.reshape(1, 1, 3).astype(np.float32))[0, 0] for m_ in modes], np.float32)
            # LaMa's feathered edge ends a few px outside the mask: the halo (small deviations only) is snapped as well
            reg = cv2.dilate(c8, _ell(2 * int(_f("CLEAN_FLAT_HALO", 8)) + 1)) > 0
            reg &= ~others | comp
            reg &= ~_strong_edges(lab[..., 0], 12.0) | comp
            # which flat colour a pixel belongs to is decided on the median-filtered fill (LaMa's mottling must not flip
            # a pixel across a tone step); undecided pixels take the nearest decided colour (run ends / flat ring)
            lab_med = _lab(cv2.medianBlur(np.ascontiguousarray(cleaned[y0:y1, x0:x1]), 25))
            dm_all = np.linalg.norm(lab_med[reg][:, None, :] - modes[None], axis=2)
            j, dmed = dm_all.argmin(1), dm_all.min(1)
            rl = _run_labels(reg, lab, modes, cv2.dilate(reg.astype(np.uint8), _ell(5)) > 0)
            rj = rl[reg]
            j = np.where(rj >= 0, rj, j)
            ok = (dmed <= tol) | (rj >= 0)
            lblmap = np.full(comp.shape, -1, np.int16)
            lblmap[reg] = np.where(ok, j, -1)
            dr = np.linalg.norm(lab[:, :, None, :] - modes[None, None], axis=3)
            ringlab = np.where(dr.min(2) <= 6.0, dr.argmin(2), -1)
            rg = ring & (ringlab >= 0)
            src = (lblmap >= 0) | rg
            lblmap = np.where(rg, ringlab, lblmap)
            if src.any() and not ok.all():
                _, nn = cv2.distanceTransformWithLabels((~src).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
                sy, sx = np.nonzero(src)
                lut = np.zeros(int(nn.max()) + 1, np.int16)
                lut[nn[sy, sx]] = lblmap[sy, sx]
                j = np.where(ok, j, lut[nn[reg]])
            dself = np.linalg.norm(lab[reg] - modes[j], axis=1)
            sel = np.where(comp[reg], dself <= 4.0 * tol, dself <= tol)  # line art / outline pixels far from every flat colour stay
            if not sel.any():
                continue
            crop = out[y0:y1, x0:x1]
            vals = crop[reg].astype(np.float32)
            vals[sel] = rgb[j[sel]]
            crop[reg] = np.clip(vals, 0, 255).astype(np.uint8)
            rep["groups"] += 1
            rep["snapped"] += int(sel.sum())
        return out, rep
    except Exception as exc:
        rep["error"] = str(exc)
        return cleaned, rep


def qa_pass(image: np.ndarray, cleaned: np.ndarray, mask: np.ndarray, skip: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Page-level QA: every filled component that is objectively bad gets the calmer fill. `skip` (e.g. whole-watermark
    regions, which are art continuations / flat fills) is never touched. Never raises."""
    rep = dict(comps=0, bad=0, fixed=0, kinds={})
    if not enabled() or mask is None or image.shape != cleaned.shape or not (mask > 0).any():
        return cleaned, rep
    try:
        out = cleaned.copy()
        for (y0, y1, x0, x1), comp, others in _groups(mask > 0):
            rep["comps"] += 1
            if skip is not None and (skip[y0:y1, x0:x1] > 0)[comp].mean() > 0.3:
                continue
            st = component_stats(image[y0:y1, x0:x1], cleaned[y0:y1, x0:x1], comp, others)
            kind = bad_kind(st)
            if not kind:
                continue
            rep["bad"] += 1
            # ring pixels come from the cleaned page (identical outside the mask, residual leftovers already gone)
            res = calm_fill(cleaned[y0:y1, x0:x1], cleaned[y0:y1, x0:x1], comp, others, kind, st)
            if res is None:
                continue
            crop, name = res
            new_st = component_stats(image[y0:y1, x0:x1], crop, comp, others)
            if new_st is not None and new_st["luma_ratio"] > st["luma_ratio"] * 0.9 \
                    and new_st["chroma_ratio"] > st["chroma_ratio"] * 0.9:
                continue  # not better: keep LaMa
            sel = cv2.dilate(comp.astype(np.uint8), _ell(7)) > 0
            out[y0:y1, x0:x1][sel] = crop[sel]
            rep["fixed"] += 1
            key = f"{kind}:{name}"
            rep["kinds"][key] = rep["kinds"].get(key, 0) + 1
        return out, rep
    except Exception as exc:  # QA must never break cleaning
        rep["error"] = str(exc)
        return cleaned, rep


# ---------------------------------------------------------------------------------------------- metrics
def page_metrics(image: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> dict:
    """Cleaning QA numbers for one page: groups, bad fills, worst ratios (groups >= 1200 px), border step."""
    res = dict(comps=0, bad=0, luma_max=0.0, chroma_max=0.0, step_mean=0.0, step_max=0.0)
    steps = []
    for (y0, y1, x0, x1), comp, others in _groups(mask > 0):
        st = component_stats(image[y0:y1, x0:x1], cleaned[y0:y1, x0:x1], comp, others)
        if st is None:
            continue
        res["comps"] += 1
        res["bad"] += int(is_bad(st))
        if st["area"] >= _f("CLEAN_FILL_QA_MIN_AREA", 1200):
            res["luma_max"] = max(res["luma_max"], st["luma_ratio"])
            res["chroma_max"] = max(res["chroma_max"], st["chroma_ratio"])
        steps.append(st["step"])
    if steps:
        res.update(step_mean=float(np.mean(steps)), step_max=max(steps))
    return res


def leftover_text_count(lb, cleaned: np.ndarray, mask: np.ndarray | None, near_px: int = 60, conf: float = 0.25) -> int:
    """Text-like segmenter components still present in the cleaned page within `near_px` of the cleaned mask."""
    try:
        _, fine = lb._tiled_segment(lb._get_segmenter(), cleaned, conf)
    except Exception:
        return -1
    if mask is None or not (mask > 0).any():
        return 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(fine, _ell(7)))
    near = cv2.dilate((mask > 0).astype(np.uint8), _ell(2 * near_px + 1)) > 0
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 40 and near[lab == i].any())
