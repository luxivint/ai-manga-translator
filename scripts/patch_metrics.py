"""Objective visibility metrics for cleaned (LaMa-filled) regions + before/after montages.

Per cleaned component (connected component of the dilated mask) on a near-flat surface:
  dE    mean CIE-Lab distance between the cleaned region and a 6-12 px ring around it
  gr    local grain std ratio (cleaned / ring), luma high-pass; 1.0 = same grain as the paper
  step  Lab distance across the mask boundary (2 px inside band vs 2 px outside band)
A component counts as a visible 'patch' when dE > 1.2 or gr < 0.5 or step > 1.5 (roughly the phone-visible limit).

Usage (from the repo root, GPU venv):
  python scripts/patch_metrics.py compare --dump output_patch_out/dump --out output_patch_test/compare [--limit N]
      replays the dumped pages (CLEAN_DUMP_DIR of local_batch.py) through block_inpaint with CLEAN_PATCH_MATCH
      off / on and prints the metrics table; --montages K writes K montages (original | before | after | mask,
      second row contrast-stretched 220-255 -> 0-255).
  python scripts/patch_metrics.py measure ORIGINAL CLEANED MASK.png     (one page, files on disk)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def stretch(img: np.ndarray, lo: int = 220, hi: int = 255) -> np.ndarray:
    return np.clip((img.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _lab(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32) * np.array([100 / 255, 1, 1], np.float32) - np.array([0, 128, 128], np.float32)


def _ell(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def measure(orig: np.ndarray, cleaned: np.ndarray, mask: np.ndarray, min_px: int = 150, flat_std: float = 12.0) -> list[dict]:
    """Per-component metrics (RGB uint8 inputs). Only components whose ring is a flat, low-texture surface count."""
    m = (mask > 0).astype(np.uint8)
    n, labels = cv2.connectedComponents(m)
    lab = _lab(cleaned)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hf = gray - cv2.GaussianBlur(gray, (0, 0), 2.0)
    ogray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gmag = np.hypot(cv2.Sobel(cv2.GaussianBlur(ogray, (0, 0), 1.0), cv2.CV_32F, 1, 0), cv2.Sobel(cv2.GaussianBlur(ogray, (0, 0), 1.0), cv2.CV_32F, 0, 1)) / 4
    edge = cv2.dilate((gmag > 14).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    allm = cv2.dilate(m, _ell(13)) > 0
    res = []
    h, w = m.shape
    for i in range(1, n):
        comp = (labels == i).astype(np.uint8)
        if comp.sum() < min_px:
            continue
        ys, xs = np.nonzero(comp)
        y1, y2, x1, x2 = max(0, ys.min() - 20), min(h, ys.max() + 21), max(0, xs.min() - 20), min(w, xs.max() + 21)
        c = comp[y1:y2, x1:x2] > 0
        ring = (cv2.dilate(c.astype(np.uint8), _ell(25)) > 0) & ~(cv2.dilate(c.astype(np.uint8), _ell(13)) > 0)
        ring &= ~(m[y1:y2, x1:x2] > 0)
        ring_ok = ring & ~edge[y1:y2, x1:x2]
        if ring.sum() < 100 or ring_ok.sum() < 100 or (ring & edge[y1:y2, x1:x2]).sum() / ring.sum() > 0.35:
            continue
        og = ogray[y1:y2, x1:x2]
        if np.std(og[ring_ok]) > flat_std * 2.5:
            continue  # textured surroundings: not a flat-interior patch case
        L = lab[y1:y2, x1:x2]
        inner = cv2.erode(c.astype(np.uint8), _ell(11)) > 0
        if inner.sum() < 30:
            inner = c
        dE = float(np.linalg.norm(L[inner].mean(0) - L[ring_ok].mean(0)))
        s_fill, s_ring = float(hf[y1:y2, x1:x2][inner].std()), float(hf[y1:y2, x1:x2][ring_ok].std())
        gr = s_fill / s_ring if s_ring >= 0.8 else 1.0  # paper without grain: nothing to match
        inb = c & ~(cv2.erode(c.astype(np.uint8), _ell(5)) > 0)
        outb = (cv2.dilate(c.astype(np.uint8), _ell(5)) > 0) & ~c & ~(m[y1:y2, x1:x2] > 0)
        step = float(np.linalg.norm(L[inb].mean(0) - L[outb].mean(0))) if inb.sum() > 10 and outb.sum() > 10 else 0.0
        res.append({"area": int(c.sum()), "dE": dE, "gr": gr, "step": step,
                    "visible": dE > 1.2 or gr < 0.5 or step > 1.5, "box": (int(x1), int(y1), int(x2), int(y2))})
    return res


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "dE": 0.0, "gr": 1.0, "step": 0.0, "visible": 0}
    return {"n": len(rows), "dE": float(np.mean([r["dE"] for r in rows])),
            "gr": float(np.mean([r["gr"] for r in rows])), "step": float(np.mean([r["step"] for r in rows])),
            "visible": int(sum(r["visible"] for r in rows))}


def _montage(orig, before, after, mask, rows_before, out_path: Path) -> None:
    h, w = mask.shape
    # window of ~900 px height holding most masked area
    colsum = (mask > 0).sum(1).astype(np.float64)
    win = min(h, 900)
    csum = np.concatenate([[0], np.cumsum(colsum)])
    best = max(range(0, h - win + 1, 20), key=lambda y: csum[y + win] - csum[y])
    sl = slice(best, best + win)
    ov = orig.copy()
    ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    tiles = [orig[sl], before[sl], after[sl], ov[sl]]
    row1 = np.hstack(tiles)
    row2 = np.hstack([stretch(t) for t in tiles[:3]] + [stretch(np.abs(before[sl].astype(np.int16) - after[sl].astype(np.int16)).astype(np.uint8) * 8 + 220)])
    img = np.vstack([row1, row2])
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def compare(dump: Path, out: Path, limit: int, montages: int) -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import local_batch as lb  # noqa: E402
    config = lb.Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
    files = sorted(dump.glob("*.npz"))[: limit or None]
    tot_b, tot_a, scored = [], [], []
    for f in files:
        d = np.load(f)
        img, mask = d["image"], d["mask"]
        os.environ["CLEAN_PATCH_MATCH"] = "false"
        os.environ["CLEAN_LAMA_FEATHER"] = "1.5"  # the previous fixed feather
        before = lb.block_inpaint(img, mask, config)
        os.environ["CLEAN_PATCH_MATCH"] = "true"
        os.environ.pop("CLEAN_LAMA_FEATHER", None)
        after = lb.block_inpaint(img, mask, config)
        rb, ra = measure(img, before, mask), measure(img, after, mask)
        sb, sa = summarize(rb), summarize(ra)
        tot_b += rb
        tot_a += ra
        changed = float(np.abs(before.astype(np.int16) - after.astype(np.int16)).mean())
        scored.append((sb["visible"] - sa["visible"] + (sb["dE"] - sa["dE"]), f.stem, rb, img, before, after, mask))
        print(f"{f.stem}: n={sb['n']}/{sa['n']} dE {sb['dE']:.2f}->{sa['dE']:.2f} gr {sb['gr']:.2f}->{sa['gr']:.2f} "
              f"step {sb['step']:.2f}->{sa['step']:.2f} visible {sb['visible']}->{sa['visible']} meanabsdiff {changed:.2f}", flush=True)
    sb, sa = summarize(tot_b), summarize(tot_a)
    improved = sum(1 for s in scored if s[0] > 0.2)
    print(f"TOTAL comps {sb['n']}/{sa['n']}: dE {sb['dE']:.2f}->{sa['dE']:.2f}  grain-ratio {sb['gr']:.2f}->{sa['gr']:.2f}  "
          f"step {sb['step']:.2f}->{sa['step']:.2f}  visible {sb['visible']}->{sa['visible']}  pages improved {improved}/{len(scored)}")
    scored.sort(key=lambda s: -s[0])
    for s in scored[:montages]:
        _montage(s[3], s[4], s[5], s[6], s[2], out / f"{s[1]}.png")
    return 0


def synthetic(paths: list[str], out: Path, montages: int) -> int:
    """Replay on pages that already carry bubble paper grain (e.g. downloaded live pages): rectangles are masked on
    bright flat windows and filled with LaMa off/on. Reproduces the faint-rectangle defect without needing raw pages."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import local_batch as lb  # noqa: E402
    config = lb.Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
    allb, alla, k = [], [], 0
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        flat = (cv2.blur(g, (9, 9)) > 238) & (cv2.blur(np.abs(g - cv2.blur(g, (5, 5))), (31, 31)) < 3)
        ok = cv2.erode(flat.astype(np.uint8), np.ones((60, 260), np.uint8)) > 0
        ys, xs = np.nonzero(ok)
        if len(ys) == 0:
            continue
        mask = np.zeros(g.shape, np.uint8)
        for j in range(min(3, len(ys))):
            q = (j * len(ys)) // 3 + len(ys) // 6
            cv2.rectangle(mask, (int(xs[q]) - 120, int(ys[q]) - 20), (int(xs[q]) + 120, int(ys[q]) + 20), 255, -1)
        os.environ["CLEAN_PATCH_MATCH"], os.environ["CLEAN_LAMA_FEATHER"] = "false", "1.5"
        before = lb.block_inpaint(img, mask, config)
        os.environ["CLEAN_PATCH_MATCH"] = "true"
        os.environ.pop("CLEAN_LAMA_FEATHER", None)
        after = lb.block_inpaint(img, mask, config)
        rb, ra = measure(img, before, mask, min_px=50), measure(img, after, mask, min_px=50)
        sb, sa = summarize(rb), summarize(ra)
        allb += rb
        alla += ra
        print(f"{Path(p).name}: n={sb['n']}/{sa['n']} dE {sb['dE']:.2f}->{sa['dE']:.2f} gr {sb['gr']:.2f}->{sa['gr']:.2f} "
              f"step {sb['step']:.2f}->{sa['step']:.2f} visible {sb['visible']}->{sa['visible']}", flush=True)
        if k < montages:
            _montage(img, before, after, mask, rb, out / f"synth_{Path(p).stem}.png")
            k += 1
    sb, sa = summarize(allb), summarize(alla)
    print(f"TOTAL comps {sb['n']}/{sa['n']}: dE {sb['dE']:.2f}->{sa['dE']:.2f} gr {sb['gr']:.2f}->{sa['gr']:.2f} "
          f"step {sb['step']:.2f}->{sa['step']:.2f} visible {sb['visible']}->{sa['visible']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--dump", required=True)
    c.add_argument("--out", default="output_patch_test/compare")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--montages", type=int, default=0)
    sy = sub.add_parser("synthetic")
    sy.add_argument("images", nargs="+")
    sy.add_argument("--out", default="output_patch_test/compare")
    sy.add_argument("--montages", type=int, default=0)
    m = sub.add_parser("measure")
    m.add_argument("original")
    m.add_argument("cleaned")
    m.add_argument("mask")
    a = ap.parse_args()
    if a.cmd == "compare":
        return compare(Path(a.dump), Path(a.out), a.limit, a.montages)
    if a.cmd == "synthetic":
        return synthetic(a.images, Path(a.out), a.montages)
    rd = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    rows = measure(rd(a.original), rd(a.cleaned), cv2.imread(a.mask, 0))
    for r in rows:
        print(r)
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
