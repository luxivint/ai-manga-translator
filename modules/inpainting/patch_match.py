"""Tone + grain matching for LaMa fills inside near-flat interiors (speech bubbles, caption boxes, sky).

LaMa paints a smooth fill: slightly different white than the paper around it and without the paper grain, so the
masked area shows up as a faint grey/blue rectangle on a phone. `refine_fill` repairs exactly that and nothing else:

* the low-frequency colour of the fill is replaced by a plane fitted to a ring of untouched pixels around the mask
  (plus a smooth residual interpolated from the ring), so region mean == ring mean without a new edge;
* the missing high-frequency grain is re-synthesised (Gaussian noise with the ring's std and lag-1 correlation);
* everything is skipped when the ring is not a flat, low-texture surface or when the LaMa fill itself carries line
  art (edge density guard), so drawn structure is never touched.
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def enabled() -> bool:
    return os.environ.get("CLEAN_PATCH_MATCH", "true").strip().lower() not in ("false", "0", "no", "off")


def _f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _fit_plane(img: np.ndarray, valid: np.ndarray) -> np.ndarray | None:
    """Per-channel plane a + b*x + c*y least-squares fitted on `valid` pixels (two robust passes). float32 HxWxC."""
    h, w = valid.shape
    ys, xs = np.nonzero(valid)
    if len(ys) < 60:
        return None
    step = max(1, len(ys) // 4000)
    ys, xs = ys[::step], xs[::step]
    sx, sy = max(w, 1), max(h, 1)
    A = np.stack([np.ones(len(ys), np.float32), xs / sx, ys / sy], axis=1).astype(np.float64)
    gx, gy = np.meshgrid(np.arange(w) / sx, np.arange(h) / sy)
    out = np.empty(img.shape, np.float32)
    for c in range(img.shape[2]):
        v = img[ys, xs, c].astype(np.float64)
        keep = np.ones(len(v), bool)
        coef = np.zeros(3)
        for _ in range(2):
            if keep.sum() < 30:
                return None
            coef = np.linalg.lstsq(A[keep], v[keep], rcond=None)[0]
            res = v - A @ coef
            sigma = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-3
            keep = np.abs(res) < 3.0 * sigma + 1.0
        out[..., c] = coef[0] + coef[1] * gx + coef[2] * gy
    return out


def _norm_blur(values: np.ndarray, weight: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    num = cv2.GaussianBlur(values * weight[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)[..., None], den


def refine_fill(orig: np.ndarray, out: np.ndarray, cm: np.ndarray, rng_seed: int = 7) -> np.ndarray:
    """Return `out` (LaMa result, uint8 HxWx3) with tone + grain matched to the ring of `orig` around mask `cm`.
    Unchanged `out` when the surroundings are not a flat interior (never raises on odd input)."""
    try:
        return _refine(orig, out, cm, rng_seed)
    except Exception:
        return out


def _refine(orig: np.ndarray, out: np.ndarray, cm: np.ndarray, rng_seed: int) -> np.ndarray:
    if not enabled() or orig.ndim != 3 or orig.shape[2] != 3 or orig.shape != out.shape:
        return out
    cm = cm.astype(bool)
    if cm.sum() < 20:
        return out
    ring_px = int(_f("CLEAN_PATCH_RING", 10))
    feather = _f("CLEAN_PATCH_FEATHER", 3.0)
    max_flat_std = _f("CLEAN_PATCH_MAX_STD", 12.0)
    max_ring_edge = _f("CLEAN_PATCH_MAX_RING_EDGE", 0.35)
    max_fill_edge = _f("CLEAN_PATCH_MAX_FILL_EDGE", 0.05)
    grain_gain = _f("CLEAN_PATCH_GRAIN", 1.0)

    cm8 = cm.astype(np.uint8)
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 3, 2 * ring_px + 3))
    k_gap = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ring = (cv2.dilate(cm8, k_out) > 0) & ~(cv2.dilate(cm8, k_gap) > 0)
    if ring.sum() < 120:
        return out

    gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_s = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gx = cv2.Sobel(gray_s, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_s, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy) / 4.0
    edge = cv2.dilate((grad > 14.0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    ring_ok = ring & ~edge
    if ring.sum() == 0 or ring_ok.sum() < 100 or (ring & edge).sum() / ring.sum() > max_ring_edge:
        return out  # ring is mostly line art / texture: not a flat interior

    plane = _fit_plane(orig.astype(np.float32), ring_ok)
    if plane is None:
        return out
    resid = orig.astype(np.float32) - plane
    rv = resid[ring_ok]
    rstd = np.array([1.4826 * np.median(np.abs(rv[:, c] - np.median(rv[:, c]))) for c in range(3)])
    if rstd.max() > max_flat_std:
        return out  # not planar/flat (textured art, strong gradients): leave LaMa alone

    # Edge-density guard on the LaMa fill: line art continued through the region must stay as LaMa drew it.
    out_gray = cv2.GaussianBlur(cv2.cvtColor(out, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    ograd = np.hypot(cv2.Sobel(out_gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(out_gray, cv2.CV_32F, 0, 1, ksize=3)) / 4.0
    inner = cv2.erode(cm8, k_gap) > 0
    if inner.sum() > 0 and (ograd[inner] > 14.0).mean() > max_fill_edge:
        return out

    # Low-frequency target: ring plane + smooth interpolated residual (captures soft blotches, no hard edges).
    wring = ring_ok.astype(np.float32)
    clip_r = np.clip(resid, -3 * rstd - 2, 3 * rstd + 2)
    lf_res, den = _norm_blur(clip_r, wring, max(4.0, ring_px * 1.2))
    lf_res = lf_res * np.clip(den * 8.0, 0.0, 1.0)[..., None]
    target = plane + lf_res

    out_f = out.astype(np.float32)
    # Low-pass of the fill measured over the fill region only (normalised), so ring pixels do not bias it.
    fill_w = cm.astype(np.float32)
    fill_lp, _ = _norm_blur(out_f, fill_w, 6.0)
    # Smooth the correction (normalised over the fill) and let it fade out just beyond the mask: no new edge.
    delta, den_f = _norm_blur(target - fill_lp, fill_w, feather)
    w_lf = np.maximum(fill_w, cv2.GaussianBlur(fill_w, (0, 0), feather))
    fixed = out_f + delta * w_lf[..., None]

    # Grain: whatever the ring has beyond the smooth target (fine speckle + small blotches), minus what the fill has.
    hp_ring = resid - lf_res
    std_ring = np.array([hp_ring[..., c][ring_ok].std() for c in range(3)])
    hp_fill = fixed - target
    std_fill = np.array([hp_fill[..., c][inner].std() if inner.any() else 0.0 for c in range(3)])
    need = np.sqrt(np.maximum(std_ring ** 2 - std_fill ** 2, 0.0)) * grain_gain
    if need.max() >= 0.3:
        lag = 2
        r0 = hp_ring[..., 1]
        a, b = r0[:, :-lag], r0[:, lag:]
        ok = ring_ok[:, :-lag] & ring_ok[:, lag:]
        if ok.sum() > 50 and a[ok].std() > 1e-3 and b[ok].std() > 1e-3:
            rho = float(np.clip(np.corrcoef(a[ok], b[ok])[0, 1], 0.03, 0.95))
        else:
            rho = 0.2
        sig = float(np.clip(lag / (2.0 * np.sqrt(-np.log(rho))), 0.4, 4.0))
        rng = np.random.default_rng(rng_seed)
        noise = rng.standard_normal(cm.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sig)
        noise /= max(float(noise.std()), 1e-6)
        grain = noise[..., None] * need[None, None, :].astype(np.float32)
        fixed = fixed + grain * fill_w[..., None]

    return np.clip(fixed, 0, 255).astype(np.uint8)
