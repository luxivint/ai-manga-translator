"""Cleaning-side offline replay (NO network): recut raw pages exactly like the job, rebuild blocks from the worker's
job-log dump, run the REAL clean stage and compute cleaning QA metrics. Reuses scripts/replay_chapter.py read-only.

  recut   raw input dir -> recut-NNNNN.png pieces (same detector/recut as the live job)
  clean   replay clean (no render) for pages -> PNG per page + metrics CSV (--tag before/after via env knobs)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_chapter as rc  # noqa: E402


def cmd_recut(a):
    from modules.detection.factory import DetectionEngineFactory
    from scripts.safe_recut import prepare
    lb, args = rc.make_args()
    images = lb.iter_images(Path(a.input))
    engine = DetectionEngineFactory.create_engine(lb.LocalSettings(args), args.detector)
    engine.confidence_threshold = min(engine.confidence_threshold, 0.25)

    def detect_protected(image):
        bubbles, texts = engine._detect_single_image(image)
        return list(bubbles) + list(texts)
    plan = prepare(images, Path(a.out), detect_protected, target=args.recut_target_height,
                   minimum=min(800, args.recut_target_height // 2),
                   soft_max=max(args.recut_target_height, min(3600, args.recut_max_height)),
                   hard_max=args.recut_max_height)
    print("recut pieces:", len(plan.images))


def _load_entries(log: Path) -> dict:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    by_page: dict[str, list[dict]] = {}
    for e in rc.parse_dump_lines(lines):
        by_page.setdefault(e["page"], []).append(e)
    return by_page


def learn_plates(lb, args, input_dir: Path, names: list, by_page: dict, crops: dict | None = None) -> list:
    """Same chapter-level plate learning as the live batch (all pages + their dumped blocks) before cleaning."""
    from modules.utils import watermark as wl
    pages = []
    for n in names:
        image = rc.read_page(Path(input_dir) / n, crops)
        pages.append((image, rc.build_blocks(lb, args, image, by_page.get(n, []))))
    return wl.learn_plates(pages, candidates=lb.plate_candidates)


def clean_page(lb, args, path: Path, ents: list[dict], idx: int = 0, crops: dict | None = None):
    """The live _stage_clean on a recut page (blocks rebuilt from the dump). Returns (image, cleaned, mask, blocks)."""
    image = rc.read_page(path, crops)  # applies the live vision/banner crop so dump boxes line up
    blocks = rc.build_blocks(lb, args, image, ents)
    captured = {}
    orig_build = lb.build_segmenter_mask

    def spy(img, blks=None, require_translation=False):
        m = orig_build(img, blks, require_translation)
        captured["mask"] = m
        return m
    lb.build_segmenter_mask = spy
    try:
        _, _, _, blocks, cleaned = lb._stage_clean((idx, path, image, blocks, args))
    finally:
        lb.build_segmenter_mask = orig_build
    return image, cleaned, captured.get("mask"), blocks


def cmd_clean(a):
    import csv
    import numpy as np
    from PIL import Image
    from modules.inpainting import fill_qa
    lb, args = rc.make_args()
    by_page = _load_entries(Path(a.log))
    crops = rc.load_crops(a.log)
    names = sorted(p.name for p in Path(a.input).glob("recut-*.png"))
    if a.pages:
        want = {f"recut-{int(x):05d}.png" for x in a.pages.split(",")}
        names = [n for n in names if n in want]
    learn_plates(lb, args, Path(a.input), sorted(p.name for p in Path(a.input).glob("recut-*.png")), by_page, crops)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        ents = by_page.get(n, [])
        if not ents:
            continue
        image, cleaned, mask, blocks = clean_page(lb, args, Path(a.input) / n, ents, crops=crops)
        Image.fromarray(np.ascontiguousarray(cleaned)).save(out / n)
        if mask is not None:
            np.save(out / (n[:-4] + ".mask.npy"), np.packbits(mask > 0))
        m = fill_qa.page_metrics(image, cleaned, mask) if mask is not None else {}
        m["leftover_text"] = fill_qa.leftover_text_count(lb, cleaned, mask)
        m["page"] = n
        rows.append(m)
        print(n, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items() if k != "page"}, flush=True)
    if rows:
        keys = ["page"] + [k for k in rows[0] if k != "page"]
        with open(out / "metrics.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        tot = {k: sum(r.get(k, 0) for r in rows) for k in keys if k != "page"}
        print("TOTAL", {k: round(v, 2) for k, v in tot.items()})


def main(argv=None):
    rc.block_network()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("recut")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_recut)
    p = sub.add_parser("clean")
    p.add_argument("--input", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--pages", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_clean)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
