"""Offline per-stage profiler / equivalence runner for the clean + render pipeline (NO network, replay data only).

  python scripts/profile_pipeline.py --input <recut dir> --log <job log> --out <dir> [--pages 1,2] [--cprofile]

Per page: detect (local detector, blocks rebuilt from the job-log dump), clean (_stage_clean), render
(build_text_items_state + render_to_image) and webp encode. Writes final PNG (lossless, for pixel comparison),
timings.csv, spans.csv (+ cprofile txt with --cprofile) and peak RSS / VRAM.
  python scripts/profile_pipeline.py compare A_dir B_dir   -> pixel equivalence table
"""
from __future__ import annotations

import argparse
import csv
import cProfile
import io
import pstats
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import replay_chapter as rc  # noqa: E402
from scripts import replay_clean as rcl  # noqa: E402

SPANS: dict[str, list[float]] = defaultdict(list)


def _wrap(mod, name: str, label: str | None = None):
    fn = getattr(mod, name, None)
    if fn is None:
        return
    label = label or name

    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            SPANS[label].append(time.perf_counter() - t)
    setattr(mod, name, wrapper)


def install_spans(lb):
    for n in ("build_segmenter_mask", "block_inpaint", "clean_postprocess", "restore_untranslated_regions",
              "watermark_clear_mask", "_tiled_segment", "build_text_items_state", "_write_array",
              "fill_easy_light_text_regions", "clear_light_free_text_render_regions",
              "clear_light_bubble_text_render_regions", "clean_white_bubble_ghosts", "_tight_glyph_mask",
              "_seg_predict"):
        _wrap(lb, n)
    from modules.inpainting import residual, fill_qa, outline, patch_match
    for mod in (residual, fill_qa, outline, patch_match):
        for n in dir(mod):
            if n.startswith("__"):
                continue
            obj = getattr(mod, n)
            if callable(obj) and getattr(obj, "__module__", "") == mod.__name__ and not isinstance(obj, type):
                _wrap(mod, n, f"{mod.__name__.split('.')[-1]}.{n}")


def run(a):
    rc.block_network()
    import numpy as np
    import psutil
    from PIL import Image
    from app.ui.canvas.save_renderer import ImageSaveRenderer
    from PySide6 import QtWidgets

    lb, args = rc.make_args()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    main = lb.LocalMain(lb.LocalSettings(args), args.source_lang, args.target_lang)
    install_spans(lb)
    by_page = rcl._load_entries(Path(a.log))
    crops = rc.load_crops(a.log)
    names = sorted(p.name for p in Path(a.input).glob("recut-*.png"))
    if a.pages:
        want = {f"recut-{int(x):05d}.png" for x in a.pages.split(",")}
        names = [n for n in names if n in want]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        torch = None
    prof = cProfile.Profile() if a.cprofile else None
    rows = []
    t_all = time.perf_counter()
    for n in names:
        ents = by_page.get(n, [])
        if not ents:
            continue
        row = {"page": n}
        t0 = time.perf_counter()
        image = rc.read_page(Path(a.input) / n, crops)
        row["read_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        blocks = rc.build_blocks(lb, args, image, ents)
        row["detect_s"] = time.perf_counter() - t0
        if prof:
            prof.enable()
        t0 = time.perf_counter()
        _, _, _, blocks, cleaned = lb._stage_clean((0, Path(n), image, blocks, args))
        row["clean_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        cl, state = lb.build_text_items_state(blocks, image, cleaned, args, main)
        row["items_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        r = ImageSaveRenderer(cl)
        r.add_state_to_image({"text_items_state": state})
        final = r.render_to_image()
        row["render_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        lb._write_array(out / (n[:-4] + ".webp"), final)
        row["webp_s"] = time.perf_counter() - t0
        if prof:
            prof.disable()
        Image.fromarray(np.ascontiguousarray(final.astype(np.uint8))).save(out / n)
        row["total_s"] = sum(row[k] for k in ("clean_s", "items_s", "render_s", "webp_s"))
        row["rss_mb"] = proc.memory_info().rss / 2**20
        row["h"], row["w"] = image.shape[:2]
        rows.append(row)
        print(n, {k: round(v, 2) for k, v in row.items() if k.endswith("_s")}, flush=True)
    wall = time.perf_counter() - t_all
    keys = ["page", "read_s", "detect_s", "clean_s", "items_s", "render_s", "webp_s", "total_s", "rss_mb", "h", "w"]
    with open(out / "timings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "spans.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["span", "calls", "total_s"])
        for k, v in sorted(SPANS.items(), key=lambda kv: -sum(kv[1])):
            w.writerow([k, len(v), round(sum(v), 3)])
    tot = {k: round(sum(r[k] for r in rows), 1) for k in keys[1:7]}
    vram = ""
    try:
        vram = f"torch_peak_alloc_mb={torch.cuda.max_memory_allocated() / 2**20:.0f}"
    except Exception:
        pass
    print("TOTAL", tot, f"wall={wall:.1f}", "peak_rss_mb=%.0f" % max(r["rss_mb"] for r in rows), vram, flush=True)
    if prof:
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("tottime").print_stats(60)
        (out / "cprofile_tottime.txt").write_text(s.getvalue(), encoding="utf-8")
        s = io.StringIO()
        pstats.Stats(prof, stream=s).sort_stats("cumulative").print_stats(80)
        (out / "cprofile_cum.txt").write_text(s.getvalue(), encoding="utf-8")
        prof.dump_stats(str(out / "cprofile.prof"))
        s = io.StringIO()
        st = pstats.Stats(prof, stream=s)
        st.print_callers("dilate|medianBlur|GaussianBlur|cvtColor|erode|morphologyEx|distanceTransform")
        (out / "cprofile_callers.txt").write_text(s.getvalue(), encoding="utf-8")


def compare(a):
    import numpy as np
    from PIL import Image
    A, B = Path(a.a), Path(a.b)
    rows, differ = [], 0
    names = sorted(p.name for p in A.glob("recut-*.png"))
    for n in names:
        pb = B / n
        if not pb.exists():
            print("MISSING in B:", n)
            continue
        x = np.asarray(Image.open(A / n).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(pb).convert("RGB"), dtype=np.int16)
        if x.shape != y.shape:
            print("SHAPE", n, x.shape, y.shape)
            differ += 1
            continue
        d = np.abs(x - y)
        if d.any():
            differ += 1
            rows.append((n, float(d.mean()), int((d.max(axis=2) > 0).sum()), int(d.max())))
    print(f"pages compared={len(names)} differing={differ}")
    for r in sorted(rows, key=lambda r: -r[1])[:30]:
        print("  %s mean_abs=%.4f px_diff=%d max=%d" % r)
    return 0 if differ == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd_or_none", nargs="?")
    ap.add_argument("x", nargs="*")
    ap.add_argument("--input")
    ap.add_argument("--log")
    ap.add_argument("--out")
    ap.add_argument("--pages", default="")
    ap.add_argument("--cprofile", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd_or_none == "compare":
        a.a, a.b = a.x
        return compare(a)
    return run(a)


if __name__ == "__main__":
    raise SystemExit(main())
