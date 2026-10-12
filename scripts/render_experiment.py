"""Typesetting experiment harness.

python scripts/render_experiment.py acquire   # OCR + translate + clean (cached to render_test/)
python scripts/render_experiment.py render [--only SUBSTR]   # old vs new montage

Cached: render_test/translations.json (OCR text + Turkish + style), render_test/cleaned/*.png.
Nothing here changes detection/OCR/cleaning logic; it only calls it.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_batch as lb  # noqa: E402
import imkit as imk  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT.parent / "manga-downloader/downloads/title-l2yrj-the-real-tycoon-all-by-myself-11475879-chapter-30/test"
RT = ROOT / "render_test"
BLOCK_CACHE = ROOT / "output_mask_test" / "blocks_cache"
TR_JSON = RT / "translations.json"
CLEAN_DIR = RT / "cleaned"
MASK_DIR = RT / "masks"
OUT_DIR = RT / "out"


def log_progress(line: str) -> None:
    RT.mkdir(exist_ok=True)
    with open(RT / "progress.txt", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def make_args():
    sys.argv = [sys.argv[0]]
    args = lb.parse_args()
    return args


def acquire(args, only: str = "") -> None:
    """OCR (9router grid OCR) + GPT translate + LaMa clean, cached."""
    RT.mkdir(exist_ok=True)
    CLEAN_DIR.mkdir(exist_ok=True)
    MASK_DIR.mkdir(exist_ok=True)
    data = json.loads(TR_JSON.read_text(encoding="utf-8")) if TR_JSON.exists() else {}
    os.environ["RENDER_STYLE_TAGS"] = "true"
    settings = lb.LocalSettings(args)
    main = lb.LocalMain(settings, args.source_lang, args.target_lang)
    for p in sorted(TEST_DIR.glob("*.webp")):
        if only and only not in p.name:
            continue
        image = imk.read_image(str(p))
        blocks = pickle.loads((BLOCK_CACHE / (p.stem + ".pkl")).read_bytes())
        if p.name not in data:
            for b in blocks:
                b.text = ""
            ocr = lb.OCRProcessor()
            ocr.initialize(main, args.source_lang)
            ocr.process(image, blocks)
            lb.recover_empty_bubble_text_with_gpt(image, blocks, args)
            blocks = lb.drop_non_english_ocr_blocks(blocks)
            blocks = lb.prune_ocr_blocks(blocks, image)
            lb.strip_translation_garbage_letters(blocks)
            blocks = lb.sort_blk_list(blocks, right_to_left=False)
            os.environ["TRANSLATION_SOURCE_FILE"] = p.name
            tr = lb.Translator(main, args.source_lang, args.target_lang)
            tr.engine.timeout = 180
            for attempt in range(3):
                try:
                    tr.translate(blocks, image, args.extra_context)
                    break
                except Exception as exc:  # router can be slow; retry
                    print("translate retry", attempt, str(exc)[:120], flush=True)
            else:
                raise RuntimeError("translation failed 3x")
            lb.suppress_noisy_ocr_blocks(blocks, args.source_lang)
            lb.strip_translation_garbage_letters(blocks)
            (RT / "blocks").mkdir(exist_ok=True)
            (RT / "blocks" / (p.stem + ".pkl")).write_bytes(pickle.dumps(blocks))
            data[p.name] = {
                "source": "gpt",
                "blocks": [
                    {"text": b.text, "translation": b.translation, "style": getattr(b, "style", ""),
                     "class": b.text_class, "xyxy": [int(v) for v in b.xyxy]}
                    for b in blocks
                ],
            }
            TR_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            print("translated", p.name, len(blocks), flush=True)
        blocks = pickle.loads((RT / "blocks" / (p.stem + ".pkl")).read_bytes())
        cp = CLEAN_DIR / (p.stem + ".png")
        if not cp.exists():
            config = lb.Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
            mask = lb.build_segmenter_mask(image, blocks)
            cleaned = image.copy() if mask.sum() == 0 else lb.block_inpaint(image, mask, config)
            cleaned = lb.restore_untranslated_regions(image, cleaned, blocks)
            imk.write_image(str(cp), cleaned)
            imk.write_image(str(MASK_DIR / (p.stem + ".png")), mask)
            print("cleaned", p.name, flush=True)


def render_page(blocks, image, cleaned, args, main, v2: bool):
    """Run the shared render stage + Qt renderer; returns (rgb, seconds)."""
    import copy
    from app.ui.canvas.save_renderer import ImageSaveRenderer

    os.environ["RENDER_V2"] = "true" if v2 else "false"
    blks = copy.deepcopy(blocks)
    t = time.time()
    cl, state = lb.build_text_items_state(blks, image, cleaned.copy(), args, main)
    r = ImageSaveRenderer(cl)
    r.add_state_to_image({"text_items_state": state})
    out = r.render_to_image()
    return out, time.time() - t


def render_all(args, only: str = "", slice_h: int = 1300, col_w: int = 520) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    settings = lb.LocalSettings(args)
    main = lb.LocalMain(settings, args.source_lang, args.target_lang)
    timings = []
    for p in sorted(TEST_DIR.glob("*.webp")):
        if only and only not in p.name:
            continue
        image = imk.read_image(str(p))
        cleaned = imk.read_image(str(CLEAN_DIR / (p.stem + ".png")))
        blocks = pickle.loads((RT / "blocks" / (p.stem + ".pkl")).read_bytes())
        old, t_old = render_page(blocks, image, cleaned, args, main, False)
        new, t_new = render_page(blocks, image, cleaned, args, main, True)
        timings.append((p.name[:11], round(t_old, 2), round(t_new, 2)))
        ys = [int(b.xyxy[1]) for b in blocks] + [int(b.xyxy[3]) for b in blocks]
        y0, y1 = max(0, min(ys) - 80), min(image.shape[0], max(ys) + 80)
        k = 0
        for y in range(y0, y1, slice_h):
            ye = min(y + slice_h, image.shape[0])
            cols = []
            for im in (image, cleaned, old, new):
                c = im[y:ye]
                sc = col_w / c.shape[1]
                cols.append(cv2.resize(c, (col_w, max(1, int(c.shape[0] * sc))), interpolation=cv2.INTER_AREA))
            hh = max(c.shape[0] for c in cols)
            cols = [np.pad(c, ((0, hh - c.shape[0]), (4, 4), (0, 0)), constant_values=255) for c in cols]
            name = f"{p.stem[:11]}_{p.stem[-6:]}_s{k}.png"
            imk.write_image(str(OUT_DIR / name), np.concatenate(cols, axis=1))
            k += 1
    print("timings (page, old s, new s):", timings)
    log_progress(f"render done timings={timings}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acquire", "render"])
    ap.add_argument("--only", default="")
    ns = ap.parse_args()
    a = make_args()
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    a.resolved_font = lb.resolve_font(a)
    if ns.cmd == "acquire":
        t = time.time()
        acquire(a, ns.only)
        log_progress(f"acquire done {time.time()-t:.0f}s")
    else:
        render_all(a, ns.only)
