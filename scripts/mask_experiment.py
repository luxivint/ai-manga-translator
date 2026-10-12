"""Mask experiment: python scripts/mask_experiment.py <before|after> [--inpaint] [--only SUBSTR]

Runs detection (+ OCR if available) and build_segmenter_mask on test images, then saves
red overlays (and optionally block_inpaint results) cropped to slices around text.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_batch as lb  # noqa: E402
import imkit as imk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT.parent / "manga-downloader/downloads/title-l2yrj-the-real-tycoon-all-by-myself-11475879-chapter-30/test"
OUT = ROOT / "output_mask_test"


def get_blocks(image, args):
    settings = lb.LocalSettings(args)
    main = lb.LocalMain(settings, args.source_lang, args.target_lang)
    blocks = lb.TextBlockDetector(settings).detect(image)
    try:
        ocr = lb.OCRProcessor()
        ocr.initialize(main, args.source_lang)
        ocr.process(image, blocks)
        blocks = lb.drop_non_english_ocr_blocks(blocks)
        blocks = lb.prune_ocr_blocks(blocks, image)
    except Exception as e:  # noqa: BLE001
        print("OCR failed, fallback text='x':", e)
    for b in blocks:
        if not (getattr(b, "text", "") or "").strip():
            b.text = "x"
    return blocks


def overlay(image, mask):
    o = image.copy()
    red = np.zeros_like(o)
    red[..., 0] = 255
    m = mask > 0
    o[m] = (0.35 * o[m] + 0.65 * red[m]).astype(np.uint8)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--inpaint", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--slice", type=int, default=1400)
    ns = ap.parse_args()
    sys.argv = [sys.argv[0]]
    args = lb.parse_args()
    out = OUT / ns.tag
    out.mkdir(parents=True, exist_ok=True)
    cache = OUT / "blocks_cache"
    cache.mkdir(exist_ok=True)
    import pickle

    for p in sorted(TEST_DIR.glob("*.webp")):
        if ns.only and ns.only not in p.name:
            continue
        image = imk.read_image(str(p))
        cp = cache / (p.stem + ".pkl")
        if cp.exists():
            blocks = pickle.loads(cp.read_bytes())
        else:
            blocks = get_blocks(image, args)
            cp.write_bytes(pickle.dumps(blocks))
        mask = lb.build_segmenter_mask(image, blocks)
        res = overlay(image, mask)
        if ns.inpaint and mask.sum():
            cleaned = lb.block_inpaint(image, mask, lb.Config(hd_strategy="Resize", hd_strategy_resize_limit=2048))
            res = np.concatenate([res, cleaned], axis=1)
        print(p.name, "blocks", len(blocks), "mask px", int((mask > 0).sum()))
        h = image.shape[0]
        ys = np.where(mask.any(axis=1))[0]
        if len(ys) == 0:
            continue
        # slices that contain mask, non-empty ones only
        k = 0
        for y in range(0, h, ns.slice):
            if mask[y:y + ns.slice].sum() == 0:
                continue
            sl = res[y:y + ns.slice]
            cv2.imwrite(str(out / f"{p.stem[:11]}_{p.stem[-6:]}_s{y:05d}.jpg"), sl[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
            k += 1


if __name__ == "__main__":
    main()
