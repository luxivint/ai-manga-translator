"""Offline replay of a finished chapter job: NO network, NO OCR/translate/vision calls.

Inputs
  * the raw source pages of the job (work/r2/<job>/input, read-only)
  * the worker log with the per-page block dumps (DEBUG_TRANSLATION_BLOCKS=true writes
    '--- DEBUG BLOCKS after-clean-translate ---' followed by one line per block)

The local detector (RT-DETR, GPU) is re-run per page (cached), the dumped OCR text / translation is matched back to the
detector blocks by box IoU, and the REAL clean (segmenter + LaMa) and render (render_v2 through
local_batch.build_text_items_state) code paths are executed. Nothing here touches 9router/OpenAI: every socket connect is
blocked while the tool runs.

Sub-commands (python scripts/replay_chapter.py <cmd> ...):
  recut     raw chapter -> recut-NNNNN.png pieces exactly like the job (local detector, no API)
  parse     log -> dump.json (page assignment by detector-box matching when the dump has no page= field)
  detect    run the detector on every raw page and cache the blocks (pickle)
  render    replay clean+render for pages -> PNGs (--mode before|after switches the new block clean-up on/off)
  defects   defect table per page (CSV)
  montage   original | live output | replay montages
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import pickle
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def block_network() -> None:
    """Hard guarantee for 'zero API calls': any socket connect (9router listens on loopback too) raises."""
    def _deny(self, *a, **k):
        raise RuntimeError("replay_chapter: network is disabled (offline replay)")
    socket.socket.connect = _deny
    socket.socket.connect_ex = _deny


# ---------------------------------------------------------------------------------------------- log parsing
_LINE_RE = re.compile(
    r"^(?:\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d+ )?(?P<idx>\d\d) cls=(?P<cls>\S*) bubble=(?P<bubble>True|False) "
    r"box=\[(?P<box>[^\]]*)\] text=(?P<text>.*?) trans=(?P<trans>'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")"
    r"(?P<extra>(?: \| .*)?)\s*$"
)
_EXTRA_RE = re.compile(r"(\w+)=(\[[^\]]*\]|'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|\S+)")


def _lit(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return s.strip("'\"")


def _unescape_nl(s):
    """The dump writer escapes real newlines to a literal backslash-n before repr(): undo it."""
    return s.replace("\\n", "\n") if isinstance(s, str) else s


def parse_dump_lines(lines, stage: str = "after-clean-translate") -> list[dict]:
    """Block dict list in log order. Old format (no page field) and new format (| page=.. bubble_xyxy=[..] ...)."""
    out, in_stage = [], False
    for ln_no, raw in enumerate(lines):
        raw = raw.rstrip("\n")
        if "--- DEBUG BLOCKS" in raw:
            in_stage = f"DEBUG BLOCKS {stage}" in raw
            continue
        if not in_stage:
            continue
        m = _LINE_RE.match(raw)
        if not m:
            continue
        ent = dict(
            line=ln_no, idx=int(m["idx"]), cls=m["cls"], bubble=m["bubble"] == "True",
            box=[int(v) for v in m["box"].split(",")], text=_unescape_nl(_lit(m["text"])),
            trans=_unescape_nl(_lit(m["trans"])),
        )
        for k, v in _EXTRA_RE.findall(m["extra"] or ""):
            ent[k] = _lit(v) if v[:1] in "[\"'" else v
        out.append(ent)
    return out


# ---------------------------------------------------------------------------------------------- page crops
_VCROP_RE = re.compile(r"\[vision-crop\] \[\d+\] (?P<page>\S+): .*?%=(?P<top>[\d.]+) alt%=(?P<bot>[\d.]+) (?P<verdict>kırpıldı|REDDEDİLDİ)")
_BCROP_RE = re.compile(r"\[banner-crop\] \[\d+\] (?P<page>\S+): üst=(?P<top>\d+)px alt=(?P<bot>\d+)px")


def parse_crops(lines) -> dict:
    """The live job crops promo strips off a page BEFORE detection/OCR, so every dumped box is in the CROPPED frame.
    Log lines: '[vision-crop] [i] page: üst%=T alt%=B kırpıldı|REDDEDİLDİ' and '[banner-crop] [i] page: üst=Npx alt=Npx'.
    -> {page: {'vision': (T, B) or None, 'banner': (top_px, bottom_px)}}"""
    out: dict = {}
    for ln in lines:
        m = _VCROP_RE.search(ln)
        if m:
            d = out.setdefault(m["page"], {"vision": None, "banner": (0, 0)})
            d["vision"] = (float(m["top"]), float(m["bot"])) if m["verdict"] == "kırpıldı" else None
            continue
        m = _BCROP_RE.search(ln)
        if m:
            d = out.setdefault(m["page"], {"vision": None, "banner": (0, 0)})
            d["banner"] = (int(m["top"]), int(m["bot"]))
    return out


def crop_rows(h: int, crop: dict | None) -> tuple[int, int]:
    """Same arithmetic as local_batch._stage_detect_ocr: (y1, y2) rows kept of a page h pixels tall."""
    if not crop:
        return 0, h
    y1, y2 = 0, h
    if crop.get("vision"):
        top, bottom = crop["vision"]
        vy1, vy2 = int(h * top / 100), int(h * (100 - bottom) / 100)
        if vy2 - vy1 >= h * 0.08:
            y1, y2 = vy1, vy2
    btop, bbot = crop.get("banner") or (0, 0)
    if btop or bbot:
        y1, y2 = max(y1, int(btop)), min(y2, h - int(bbot))
    if (y1, y2) != (0, h) and y2 - y1 >= h * 0.08:
        return y1, y2
    return 0, h


def read_page(path, crops: dict | None = None):
    """imk.read_image + the live crop (when the job log says the page was cropped)."""
    import imkit as imk
    image = imk.read_image(str(path))
    if crops:
        y1, y2 = crop_rows(image.shape[0], crops.get(Path(path).name))
        if (y1, y2) != (0, image.shape[0]):
            image = image[y1:y2].copy()
    return image


_CROPS_CACHE: dict = {}


def load_crops(log) -> dict:
    if not log:
        return {}
    if log not in _CROPS_CACHE:
        _CROPS_CACHE[log] = parse_crops(Path(log).read_text(encoding="utf-8", errors="replace").splitlines())
    return _CROPS_CACHE[log]


# ---------------------------------------------------------------------------------------------- geometry
def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------------------------------------- detector
def make_args():
    """Same argparse defaults as scripts/local_batch.py (.env is NOT loaded: it holds secrets and no render knobs)."""
    from scripts import local_batch as lb
    lb.load_env = lambda path: None
    old = sys.argv
    sys.argv = ["local_batch.py"]
    try:
        args = lb.parse_args()
    finally:
        sys.argv = old
    args.resolved_font = lb.resolve_font(args)
    return lb, args


def detect_pages(lb, args, input_dir: Path, cache: Path) -> dict:
    import imkit as imk
    from modules.detection.processor import TextBlockDetector

    if cache.exists():
        return pickle.loads(cache.read_bytes())
    settings = lb.LocalSettings(args)
    res = {}
    for p in lb.iter_images(input_dir):
        img = imk.read_image(str(p))
        blocks = TextBlockDetector(settings).detect(img)
        res[p.name] = [dict(xyxy=[int(v) for v in b.xyxy],
                            bubble=None if b.bubble_xyxy is None else [int(v) for v in b.bubble_xyxy],
                            cls=b.text_class, angle=float(b.angle or 0)) for b in blocks]
    cache.write_bytes(pickle.dumps(res))
    return res


# ---------------------------------------------------------------------------------------------- page assignment
def assign_pages(entries: list[dict], det: dict, size_of=None) -> list[dict]:
    """Give every dump entry a `page` (file name). The dump of a threaded job interleaves lines of different pages,
    so each line is matched to the page whose detector boxes explain it best (IoU), consistent with the 'idx'
    run structure: a run is a maximal sequence 00,01,02.. and is assigned as a whole."""
    have = all("page" in e for e in entries)
    if have:
        return entries
    pages = sorted(det)
    # split into runs (an idx 00 starts a run; interleaved runs are separated greedily by the next expected idx)
    runs: list[list[dict]] = []
    open_runs: list[list[dict]] = []
    for e in entries:
        if e["idx"] == 0:
            r = [e]
            runs.append(r)
            open_runs.append(r)
            continue
        cand = [r for r in open_runs if r[-1]["idx"] == e["idx"] - 1]
        if cand:
            r = max(cand, key=lambda r: -abs(r[-1]["line"] - e["line"]))  # nearest in the log
            r.append(e)
            if e["idx"] >= 2:
                pass
        else:  # orphan (should not happen): own run
            runs.append([e])

    def score(run, page):
        s = 0.0
        for e in run:
            best = max((iou(e["box"], d["xyxy"]) for d in det[page]), default=0.0)
            best = max(best, max((0.6 * iou(e["box"], d["bubble"]) for d in det[page] if d["bubble"]), default=0.0))
            s += best
        return s / max(1, len(run))

    # greedy one-to-one assignment run -> page by descending score (a page is dumped exactly once)
    cells = sorted(((score(r, p) + (0.02 if len(det[p]) >= len(r) else 0.0), i, p)
                    for i, r in enumerate(runs) for p in pages), reverse=True)
    used_r, used_p = set(), set()
    for sc, i, p in cells:
        if i in used_r or p in used_p:
            continue
        used_r.add(i)
        used_p.add(p)
        for e in runs[i]:
            e["page"] = p
            e["match"] = round(float(sc), 3)
    return entries


# ---------------------------------------------------------------------------------------------- replay
def guess_style(ent: dict) -> str:
    """The style tag was returned by GPT and is not in old dumps: use the logged one, else a conservative guess."""
    if ent.get("style"):
        return str(ent["style"])
    return "normal" if ent["cls"] == "text_bubble" else os.environ.get("REPLAY_FREE_STYLE", "narration")


def build_blocks(lb, args, image, ents: list[dict]):
    """Detector blocks of the page (same detector/settings as the job) filled with the dumped OCR text/translation,
    in dump order. bubble_xyxy comes from the detector (or from the dump when it logged one)."""
    from modules.detection.processor import TextBlockDetector
    import numpy as np

    settings = lb.LocalSettings(args)
    det = TextBlockDetector(settings).detect(image)
    out, used = [], set()
    for e in ents:
        best, bi = 0.0, None
        for k, b in enumerate(det):
            if k in used:
                continue
            v = iou(e["box"], [int(x) for x in b.xyxy])
            if v > best:
                best, bi = v, k
        if bi is None or best < 0.3:  # nearest by centre as the fallback
            cx, cy = (e["box"][0] + e["box"][2]) / 2, (e["box"][1] + e["box"][3]) / 2
            cand = [(abs(b.center[0] - cx) + abs(b.center[1] - cy), k) for k, b in enumerate(det) if k not in used]
            bi = min(cand)[1] if cand else None
        if bi is None:
            from modules.utils.textblock import TextBlock
            blk = TextBlock(text_bbox=np.array(e["box"]), text_class=e["cls"])
        else:
            used.add(bi)
            blk = det[bi]
        blk.xyxy = np.array(e["box"])
        if e.get("bubble_xyxy"):
            blk.bubble_xyxy = np.array(e["bubble_xyxy"])
        blk.text_class = e["cls"]
        blk.text = e["text"]
        blk.translation = e["trans"]
        blk.style = guess_style(e)
        out.append(blk)
    return out


def replay_page(lb, args, main, path: Path, ents: list[dict], out_png: Path, hooks=None, crops=None):
    """clean + render exactly like _stage_clean / _do_render_and_save (without stamp). Returns the final image."""
    import imkit as imk
    import numpy as np
    from app.ui.canvas.save_renderer import ImageSaveRenderer
    from modules.inpainting.schema import Config
    from modules.utils import watermark as watermark_lib

    image = read_page(path, crops)
    blocks = build_blocks(lb, args, image, ents)
    if hooks and hooks.get("pre_clean"):
        blocks = hooks["pre_clean"](blocks, image)
    config = Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
    mask = lb.build_segmenter_mask(image, blocks, require_translation=True)
    if mask.sum() == 0:
        cleaned = image.copy()
    else:
        cleaned = lb.block_inpaint(image, mask, config)
        cleaned = watermark_lib.flat_fill(image, cleaned, lb.watermark_clear_mask(image, blocks))
    cleaned = lb.restore_untranslated_regions(image, cleaned, blocks)
    cleaned, state = lb.build_text_items_state(blocks, image, cleaned, args, main)
    renderer = ImageSaveRenderer(cleaned)
    renderer.add_state_to_image({"text_items_state": state})
    final = renderer.render_to_image()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(final.astype(np.uint8))).save(out_png)
    return final, state, blocks


def _setup():
    from PySide6 import QtWidgets
    lb, args = make_args()
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    main = lb.LocalMain(lb.LocalSettings(args), args.source_lang, args.target_lang)
    return lb, args, main


def _cmd_render(a):
    ents = json.loads(Path(a.dump).read_text(encoding="utf-8"))
    by_page: dict[str, list[dict]] = {}
    for e in ents:
        by_page.setdefault(e["page"], []).append(e)
    lb, args, main = _setup()
    wanted = [p if p.endswith(".webp") else f"{int(p):03d}.webp" for p in a.pages.split(",")] if a.pages else sorted(by_page)
    for name in wanted:
        replay_page(lb, args, main, Path(a.input) / name, by_page.get(name, []), Path(a.out) / name.replace(".webp", ".png"), crops=load_crops(getattr(a, 'log', '')))
        print("rendered", name, flush=True)


def _clean_page(lb, image, blocks):
    from modules.inpainting.schema import Config
    from modules.utils import watermark as watermark_lib
    config = Config(hd_strategy="Resize", hd_strategy_resize_limit=2048)
    mask = lb.build_segmenter_mask(image, blocks, require_translation=True)
    if mask.sum() == 0:
        cleaned = image.copy()
    else:
        cleaned = lb.block_inpaint(image, mask, config)
        cleaned = watermark_lib.flat_fill(image, cleaned, lb.watermark_clear_mask(image, blocks))
    return lb.restore_untranslated_regions(image, cleaned, blocks)


def _render_only(lb, args, main, image, cleaned, blocks):
    import copy
    import numpy as np
    from app.ui.canvas.save_renderer import ImageSaveRenderer
    bl = copy.deepcopy(blocks)
    cl, state = lb.build_text_items_state(bl, image, cleaned.copy(), args, main)
    r = ImageSaveRenderer(cl)
    r.add_state_to_image({"text_items_state": state})
    return np.ascontiguousarray(r.render_to_image().astype(np.uint8)), state


STYLE_CANDIDATES = ["normal", "shout", "thought", "narration"]


def _cmd_styles(a):
    """The dump has no style tag (GPT returned it). Pick, per block, the style whose replay render is closest to the LIVE
    output image (local comparison only). Writes dump with style + style_diff."""
    import imkit as imk
    import numpy as np
    from PIL import Image

    ents = json.loads(Path(a.dump).read_text(encoding="utf-8"))
    by_page: dict[str, list[dict]] = {}
    for e in ents:
        by_page.setdefault(e["page"], []).append(e)
    lb, args, main = _setup()
    for name, pe in sorted(by_page.items()):
        live_p = Path(a.live) / name.replace(".webp", "_translated.webp")
        only = {x if x.endswith(".webp") else f"{int(x):03d}.webp" for x in a.pages.split(",") if x}
        if (only and name not in only) or not live_p.exists() or all(not e["trans"].strip() for e in pe):
            for e in pe:
                e.setdefault("style", guess_style(e))
            continue
        live = np.asarray(Image.open(live_p).convert("L")).astype(np.int16)
        image = read_page(Path(a.input) / name, load_crops(getattr(a, 'log', '')))
        blocks = build_blocks(lb, args, image, pe)
        cleaned = _clean_page(lb, image, blocks)
        for bi, e in enumerate(pe):
            if not e["trans"].strip():
                continue
            x1, y1, x2, y2 = e["box"]
            H, W = live.shape
            pad = 30
            sl = (slice(max(0, y1 - pad), min(H, y2 + pad)), slice(max(0, x1 - pad), min(W, x2 + pad)))
            best = None
            for st in STYLE_CANDIDATES:
                blocks[bi].style = st
                try:
                    img, _ = _render_only(lb, args, main, image, cleaned, blocks)
                except Exception:
                    continue
                d = float(np.abs(img[sl].mean(axis=2).astype(np.int16) - live[sl]).mean())
                if best is None or d < best[0] - 0.05:
                    best = (d, st)
            blocks[bi].style = best[1]
            e["style"], e["style_diff"] = best[1], round(best[0], 2)
        print("styles", name, [e.get("style") for e in pe], [e.get("style_diff") for e in pe], flush=True)
    Path(a.out).write_text(json.dumps(ents, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------------------------- defects
DEFECT_COLS = ["dup_ngram", "contained", "size_spread", "overflow", "overlap", "tiny", "narration_split"]
MODE_ENV = {
    "before": {"RENDER_DEDUPE_TRANSLATIONS": "false", "RENDER_MERGE_NARRATION": "false", "RENDER_ABSORB_STRAYS": "false",
               "RENDER_GROUP_ROBUST": "false", "RENDER_HARD_GUARD": "false", "BLOCK_CONTAINMENT_PRUNE": "false"},
    "after": {"RENDER_DEDUPE_TRANSLATIONS": "true", "RENDER_MERGE_NARRATION": "true", "RENDER_ABSORB_STRAYS": "true",
              "RENDER_GROUP_ROBUST": "true", "RENDER_HARD_GUARD": "true", "BLOCK_CONTAINMENT_PRUNE": "true"},
}


def _match_block(blocks, box):
    best, bi = 0.0, None
    for k, b in enumerate(blocks):
        v = iou([int(x) for x in b.xyxy], [int(x) for x in box])
        if v > best:
            best, bi = v, k
    return (bi, best)


def page_defects(lb, image, cleaned, final, blocks, items, min_px: float = 11.0) -> dict:
    """Per-page defect counts + details. Pure measurement on the replay result (blocks = what was rendered)."""
    import cv2
    import numpy as np
    from modules.rendering import balloon_layout as bl
    from modules.rendering import render_v2
    from modules.utils import block_cleanup as bcl

    H, W = image.shape[:2]
    out = {c: 0 for c in DEFECT_COLS}
    detail: list[str] = []
    # containment among the SOURCE blocks (what the detector/OCR delivered)
    for i, a in enumerate(blocks):
        for j, b in enumerate(blocks):
            if i == j or not (a.text or "").strip() or not (b.text or "").strip():
                continue
            if bcl.inside_fraction([float(v) for v in b.xyxy], [float(v) for v in a.xyxy]) >= 0.6 and \
                    bcl.covered_fraction(b.text, a.text) >= 0.8 and len(bcl.words(a.text)) >= len(bcl.words(b.text)) and i != j:
                if len(bcl.words(a.text)) == len(bcl.words(b.text)) and j < i:
                    continue
                out["contained"] += 1
                detail.append(f"contained[{j} in {i}]")
    # items
    sizes_px = [d["font_size"] * 800.0 / W for d in items]
    out["tiny"] = sum(1 for s in sizes_px if s < min_px)
    if out["tiny"]:
        detail.append("tiny=" + ",".join(f"{s:.1f}" for s in sizes_px if s < min_px))

    def poly(d):
        x, y = d["position"]
        w, h = float(d["width"]), float(d["height"])
        return bl.rotated_rect(x + 4, y + 4 + 0.06 * h, max(1.0, w - 8), max(1.0, h - 8 - 0.12 * h), float(d.get("rotation") or 0.0),
                               (w / 2 - 4, h / 2 - 4 - 0.06 * h))

    out["overlap"] = len(bl.overlapping_pairs([poly(d) for d in items])) if len(items) >= 2 else 0
    # duplicate n-grams between items that touch or share a balloon
    toks = [bcl.words(d["text"]) for d in items]
    grams = [{tuple(t[k:k + 2]) for k in range(len(t) - 1)} for t in toks]
    near = lambda a, b: (a["position"][0] < b["position"][0] + b["width"] and b["position"][0] < a["position"][0] + a["width"]
                         and abs((a["position"][1] + a["height"] / 2) - (b["position"][1] + b["height"] / 2)) < (a["height"] + b["height"]) * 0.75 + 60)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if grams[i] & grams[j] and near(items[i], items[j]):
                out["dup_ngram"] += 1
                detail.append(f"dup[{i},{j}]={sorted(grams[i] & grams[j])[:2]}")
    # size spread within a balloon + overflow
    item_blk = []
    for d in items:
        bi, v = _match_block(blocks, d["debug"]["obox"])
        item_blk.append(bi if v >= 0.3 else None)
    bubbles = [None if item_blk[k] is None or getattr(blocks[item_blk[k]], "bubble_xyxy", None) is None
               else tuple(float(v) for v in blocks[item_blk[k]].bubble_xyxy) for k in range(len(items))]
    boxes = [[float(v) for v in d["debug"]["obox"]] for d in items]
    for g in bl.group_by_balloon(bubbles, boxes):
        s = [items[i]["font_size"] for i in g]
        if max(s) / max(1, min(s)) > 1.4:
            out["size_spread"] += 1
            detail.append(f"spread{g}={s}")
    gray_f = final.astype(np.int16).mean(axis=2)
    gray_c = cleaned.astype(np.int16).mean(axis=2)
    for k, d in enumerate(items):
        if not d["debug"].get("bub") or item_blk[k] is None:
            continue
        blk = blocks[item_blk[k]]
        if getattr(blk, "bubble_xyxy", None) is None:
            continue
        x, y = [int(v) for v in d["position"]]
        w, h = int(d["width"]), int(d["height"])
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        ink = np.abs(gray_f[y0:y1, x0:x1] - gray_c[y0:y1, x0:x1]) > 60
        if ink.sum() < 30:
            continue
        # interior of the balloon measured on the CLEANED page (the original lettering is gone there, so the glyph
        # strokes of the English text do not punch holes into the interior); text pixels outside it cross the outline
        _, inner, off = render_v2.bubble_interior_ex(cleaned, blk)
        if inner is None:
            continue
        full = np.zeros((H, W), np.uint8)
        ox, oy = off
        full[oy:oy + inner.shape[0], ox:ox + inner.shape[1]] = inner.astype(np.uint8)
        full = cv2.dilate(full, np.ones((5, 5), np.uint8))
        outside = (ink & (full[y0:y1, x0:x1] == 0)).sum()
        if outside / max(1, ink.sum()) > 0.02 and outside > 25:
            out["overflow"] += 1
            detail.append(f"overflow[{k}]={outside}/{int(ink.sum())}px")
    # narration boxes drawn as >= 3 separate blocks
    styles = [d["debug"].get("style") for d in items]
    out["narration_split"] = sum(1 for g in bl.narration_groups(boxes, styles) if len(g) >= 3)
    out["detail"] = "; ".join(detail)
    out["n_items"] = len(items)
    return out


def _cmd_run(a):
    """Replay pages in the requested modes (before/after), write PNGs + defect CSV."""
    import imkit as imk
    import numpy as np
    from PIL import Image
    from app.ui.canvas.save_renderer import ImageSaveRenderer
    from modules.utils import block_cleanup as bcl

    ents = json.loads(Path(a.dump).read_text(encoding="utf-8"))
    by_page: dict[str, list[dict]] = {}
    for e in ents:
        by_page.setdefault(e["page"], []).append(e)
    lb, args, main = _setup()
    wanted = [p if p.endswith(".webp") else f"{int(p):03d}.webp" for p in a.pages.split(",")] if a.pages else sorted(by_page)
    modes = a.modes.split(",")
    rows = []
    out_dir = Path(a.out)
    for name in wanted:
        pe = by_page.get(name, [])
        image = read_page(Path(a.input) / name, load_crops(getattr(a, 'log', '')))
        if a.synth:
            pe = SYNTH[a.synth](pe, image) if a.synth in SYNTH else pe
        blocks0 = build_blocks(lb, args, image, pe)
        cleaned0 = _clean_page(lb, image, blocks0)
        for mode in modes:
            for k, v in MODE_ENV[mode].items():
                os.environ[k] = v
            blocks = blocks0
            dropped = 0
            if mode == "after" and bcl is not None:
                blocks, di = bcl.prune_contained_blocks(blocks0)
                dropped = len(di)
            import copy
            bl_ = copy.deepcopy(blocks)
            cl, state = lb.build_text_items_state(bl_, image, cleaned0.copy(), args, main)
            r = ImageSaveRenderer(cl)
            r.add_state_to_image({"text_items_state": state})
            final = np.ascontiguousarray(r.render_to_image().astype(np.uint8))
            (out_dir / mode).mkdir(parents=True, exist_ok=True)
            Image.fromarray(final).save(out_dir / mode / name.replace(".webp", ".png"))
            d = page_defects(lb, image, cl, final, blocks, state)
            d.update(page=name, mode=mode, n_blocks=len(blocks0), pruned_contained=dropped)
            rows.append(d)
            print("run", name, mode, {k: d[k] for k in DEFECT_COLS}, flush=True)
    with open(out_dir / "defects.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "mode", "n_blocks", "n_items", "pruned_contained"] + DEFECT_COLS + ["detail"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})
    for mode in modes:
        sel = [r for r in rows if r["mode"] == mode]
        print(mode, {c: sum(r[c] for r in sel) for c in DEFECT_COLS}, "pages with any:", sum(1 for r in sel if any(r[c] for c in DEFECT_COLS)), flush=True)


def _cmd_fit(a):
    """Text-fit QA: clean once per page (cached as pickle), then render only and measure the share of ink pixels that lies
    within --band px of a dark balloon outline pixel (per balloon, pooled), plus size statistics. NO network."""
    import copy
    import pickle
    import statistics
    import numpy as np
    from PIL import Image
    from modules.rendering.outline_metric import balloon_outline_overlap
    from scripts import replay_clean as rcl

    lb, args, main = _setup()
    by_page = rcl._load_entries(Path(a.log))
    crops = load_crops(a.log)
    cache, out = Path(a.cache), Path(a.out)
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in Path(a.input).glob("recut-*.png"))
    if a.pages:
        want = {f"recut-{int(x):05d}.png" for x in a.pages.split(",")}
        names = [n for n in names if n in want]
    tot_ink = tot_band = tot_near = n_bal = n_bad = 0
    sizes, spreads, tiny, rows = [], [], 0, []
    for n in names:
        ents = by_page.get(n, [])
        if not ents:
            continue
        cf = cache / (n[:-4] + ".pkl")
        if cf.exists():
            image, cleaned, blocks = pickle.loads(cf.read_bytes())
        else:
            image, cleaned, _, blocks = rcl.clean_page(lb, args, Path(a.input) / n, ents, crops=crops)
            cf.write_bytes(pickle.dumps((image, cleaned, blocks)))
        cl, state = lb.build_text_items_state(copy.deepcopy(blocks), image, cleaned.copy(), args, main)
        from app.ui.canvas.save_renderer import ImageSaveRenderer
        r = ImageSaveRenderer(cl)
        r.add_state_to_image({"text_items_state": state})
        final = np.ascontiguousarray(r.render_to_image().astype(np.uint8))
        Image.fromarray(final).save(out / (n[:-4] + ".png"))
        W = image.shape[1]
        px = [d["font_size"] * 800.0 / W for d in state]
        sizes += px
        tiny += sum(1 for s in px if s < 11.0)
        seen = set()
        for b in blocks:
            bb = getattr(b, "bubble_xyxy", None)
            if bb is None or not (b.translation or "").strip():
                continue
            key = tuple(int(v) for v in bb)
            if key in seen:
                continue
            seen.add(key)
            skip = [(d["position"][0], d["position"][1], d["position"][0] + d["width"], d["position"][1] + d["height"])
                    for d in state if not d["debug"]["bub"] or d["debug"]["style"] in ("sfx", "shout", "title")]
            m = balloon_outline_overlap(cl, final, bb, band_px=a.band, skip_rects=skip)
            if m is None:
                continue
            m8 = balloon_outline_overlap(cl, final, bb, band_px=a.band * 2, skip_rects=skip)
            n_bal += 1
            tot_ink += m[0]
            tot_band += m[1]
            tot_near += m8[1]
            if m8[1] / m[0] > 0.01:
                n_bad += 1
                rows.append((round(m[1] / m[0], 4), round(m8[1] / m[0], 4), n, key))
            mem = [d["font_size"] for d in state if key[0] <= d["position"][0] + d["width"] / 2 <= key[2] and key[1] <= d["position"][1] + d["height"] / 2 <= key[3]]
            if len(mem) >= 2:
                spreads.append(max(mem) / min(mem))
                if max(mem) / min(mem) > 1.5:
                    print("SPREAD", n, list(key), mem, flush=True)
        print(n, "items", len(state), flush=True)
    rows.sort(key=lambda r: -r[1])
    res = dict(balloons=n_bal, ink_in_band_pct=round(100.0 * tot_band / max(1, tot_ink), 3),
               ink_near_pct=round(100.0 * tot_near / max(1, tot_ink), 3), balloons_near_over_1pct=n_bad,
               items=len(sizes), tiny=tiny, size_median=round(statistics.median(sizes), 2) if sizes else 0,
               size_stdev=round(statistics.pstdev(sizes), 2) if sizes else 0,
               balloon_spread_mean=round(statistics.mean(spreads), 3) if spreads else 0,
               balloon_spread_max=round(max(spreads), 3) if spreads else 0)
    print("RESULT", json.dumps(res))
    print("WORST(touch,near,page,box)", json.dumps([(r[0], r[1], r[2], list(r[3])) for r in rows[:14]]))
    (out / "fit_result.json").write_text(json.dumps(dict(result=res, worst=[(r[0], r[1], r[2], list(r[3])) for r in rows]), indent=1), encoding="utf-8")


def _long_block(pe):
    """First text_bubble entry with >= 6 words of OCR text and a translation (index into pe) or None."""
    for k, e in enumerate(pe):
        if e["cls"] == "text_bubble" and len(e["text"].split()) >= 6 and len(e["trans"].split()) >= 4:
            return k
    return None


def _chunks(words, n):
    size = max(1, -(-len(words) // n))
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)]


def synth_echo(pe, image):
    """SYNTHETIC (derived from a real block): the detector reports the first half of a paragraph a second time as a
    smaller block inside it (the 'same phrase at two sizes' defect)."""
    k = _long_block(pe)
    if k is None:
        return pe
    e = pe[k]
    x1, y1, x2, y2 = e["box"]
    w, h = x2 - x1, y2 - y1
    tw, rw = e["text"].split(), e["trans"].split()
    echo = dict(e, box=[int(x1 + 0.05 * w), y1, int(x2 - 0.05 * w), int(y1 + 0.4 * h)],
                text=" ".join(tw[:max(2, len(tw) // 2)]), trans=" ".join(rw[:max(2, len(rw) // 2)]), synth="echo")
    return pe[:k + 1] + [echo] + pe[k + 1:]


def synth_narration(pe, image):
    """SYNTHETIC: one paragraph block becomes 4 stacked narration blocks (one per strip of the real text)."""
    k = _long_block(pe)
    if k is None:
        return pe
    e = pe[k]
    x1, y1, x2, y2 = e["box"]
    n = 4
    tw, rw = _chunks(e["text"].split(), n), _chunks(e["trans"].split(), n)
    m = min(len(tw), len(rw))
    if m < 2:
        return pe
    hh = (y2 - y1) / m
    parts = [dict(e, cls="text_free", bubble=False, bubble_xyxy=None, style="narration", synth="narration",
                  box=[x1, int(y1 + i * hh), x2, int(y1 + (i + 1) * hh)], text=tw[i], trans=rw[i]) for i in range(m)]
    return pe[:k] + parts + pe[k + 1:]


def synth_stray(pe, image):
    """SYNTHETIC: the last word of a paragraph is reported as a separate small block below the paragraph block."""
    k = _long_block(pe)
    if k is None:
        return pe
    e = pe[k]
    x1, y1, x2, y2 = e["box"]
    w, h = x2 - x1, y2 - y1
    tw, rw = e["text"].split(), e["trans"].split()
    main = dict(e, box=[x1, y1, x2, int(y1 + 0.72 * h)], text=" ".join(tw[:-1]), trans=" ".join(rw[:-1]))
    stray = dict(e, box=[int(x1 + 0.3 * w), int(y1 + 0.83 * h), int(x2 - 0.3 * w), int(y1 + 0.91 * h)], text=tw[-1], trans=rw[-1], synth="stray")
    return pe[:k] + [main, stray] + pe[k + 1:]


SYNTH = {"echo": synth_echo, "narration": synth_narration, "stray": synth_stray}


# ---------------------------------------------------------------------------------------------- montage
def make_montage(panels: list, crop_box, out: Path, labels=None, max_w: int = 2000, max_h: int = 1400) -> Path:
    """Side-by-side crop of the same page region from several images (PIL images), scaled to <= max_w x max_h."""
    from PIL import Image, ImageDraw

    x1, y1, x2, y2 = [int(v) for v in crop_box]
    crops = [im.crop((x1, y1, x2, y2)) for im in panels]
    w, h = crops[0].size
    gap = 6
    sheet = Image.new("RGB", (w * len(crops) + gap * (len(crops) - 1), h + (18 if labels else 0)), (255, 0, 255))
    for i, c in enumerate(crops):
        sheet.paste(c, (i * (w + gap), 18 if labels else 0))
        if labels:
            ImageDraw.Draw(sheet).text((i * (w + gap) + 4, 3), labels[i], fill=(255, 255, 255))
    k = min(1.0, max_w / sheet.width, max_h / sheet.height)
    if k < 1.0:
        sheet = sheet.resize((int(sheet.width * k), int(sheet.height * k)), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _cmd_montage(a):
    """original | live output | replay-before | replay-after crops. --box x1,y1,x2,y2 (default: whole page)."""
    from PIL import Image

    name = a.page if a.page.endswith(".webp") else f"{int(a.page):03d}.webp"
    stem = name[:3]
    panels, labels = [Image.open(Path(a.input) / name).convert("RGB")], ["original"]
    live = Path(a.live) / f"{stem}_translated.webp"
    if live.exists():
        panels.append(Image.open(live).convert("RGB"))
        labels.append("live")
    for d in a.replay:
        pth = Path(d) / f"{stem}.png"
        if pth.exists():
            panels.append(Image.open(pth).convert("RGB"))
            labels.append(Path(d).name)
    W, H = panels[0].size
    box = [int(v) for v in a.box.split(",")] if a.box else [0, 0, W, H]
    make_montage(panels, box, Path(a.out), labels)
    print("montage", a.out)


def _cmd_sheet(a):
    """Several pages stacked in one montage (<= 2000 x 1400): --items 086@x1,y1,x2,y2 015@... ; every row shows
    original | live | each --replay dir."""
    from PIL import Image

    rows = []
    for spec in a.items:
        name, box = spec.split("@")
        name = name if name.endswith(".webp") else f"{int(name):03d}.webp"
        stem = name[:3]
        panels, labels = [Image.open(Path(a.input) / name).convert("RGB")], [f"{stem} original"]
        live = Path(a.live) / f"{stem}_translated.webp"
        if live.exists():
            panels.append(Image.open(live).convert("RGB"))
            labels.append("live")
        for d in a.replay:
            pth = Path(d) / f"{stem}.png"
            if pth.exists():
                panels.append(Image.open(pth).convert("RGB"))
                labels.append(a.labels[a.replay.index(d)] if a.labels else Path(d).name)
        tmp = Path(a.out).with_suffix(f".row{len(rows)}.png")
        make_montage(panels, [int(v) for v in box.split(",")], tmp, labels, max_w=100000, max_h=100000)
        rows.append(Image.open(tmp).convert("RGB"))
        tmp.unlink()
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows)
    sheet = Image.new("RGB", (w, h), (255, 0, 255))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    k = min(1.0, 2000 / sheet.width, 1400 / sheet.height)
    if k < 1.0:
        sheet = sheet.resize((int(sheet.width * k), int(sheet.height * k)), Image.LANCZOS)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(a.out)
    print("sheet", a.out, sheet.size)


# ---------------------------------------------------------------------------------------------- CLI
def _cmd_recut(a):
    """Rebuild the safe-recut pieces of a raw chapter (local detector only, no API): the dump page names of a job that
    ran with --safe-recut are 'recut-NNNNN.png'. Writes the pieces into --out (must be empty)."""
    from modules.detection.factory import DetectionEngineFactory
    from scripts.safe_recut import prepare

    lb, args = make_args()
    engine = DetectionEngineFactory.create_engine(lb.LocalSettings(args), args.detector)
    engine.confidence_threshold = min(engine.confidence_threshold, 0.25)

    def detect(image):
        bubbles, texts = engine._detect_single_image(image)
        return list(bubbles) + list(texts)

    plan = prepare(lb.iter_images(Path(a.input)), Path(a.out), detect, target=args.recut_target_height,
                   minimum=min(800, args.recut_target_height // 2),
                   soft_max=max(args.recut_target_height, min(3600, args.recut_max_height)), hard_max=args.recut_max_height)
    print(f"{len(plan.images)} pieces -> {a.out}")


def _cmd_parse(a):
    lines = Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if a.marker in ln:
            start = i
    entries = parse_dump_lines(lines[start:], a.stage)
    lb, args = make_args()
    det = detect_pages(lb, args, Path(a.input), Path(a.cache))
    entries = assign_pages(entries, det)
    Path(a.out).write_text(json.dumps(entries, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(entries)} dump blocks -> {a.out}")


def main(argv=None) -> int:
    block_network()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("parse")
    p.add_argument("--log", required=True)
    p.add_argument("--marker", default="Faz 3/3")
    p.add_argument("--stage", default="after-clean-translate")
    p.add_argument("--input", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_parse)
    p = sub.add_parser("recut")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_recut)
    p = sub.add_parser("styles")
    p.add_argument("--dump", required=True)
    p.add_argument("--log", default="", help="job log: apply its vision/banner page crops so dump boxes line up")
    p.add_argument("--input", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--pages", default="", help="only infer these pages (others keep the default guess)")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_styles)
    p = sub.add_parser("run")
    p.add_argument("--dump", required=True)
    p.add_argument("--log", default="", help="job log: apply its vision/banner page crops so dump boxes line up")
    p.add_argument("--input", required=True)
    p.add_argument("--pages", default="")
    p.add_argument("--modes", default="before,after")
    p.add_argument("--synth", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_run)
    p = sub.add_parser("montage")
    p.add_argument("--page", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--replay", nargs="*", default=[])
    p.add_argument("--box", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_montage)
    p = sub.add_parser("sheet")
    p.add_argument("--items", nargs="+", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--live", required=True)
    p.add_argument("--replay", nargs="*", default=[])
    p.add_argument("--labels", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_sheet)
    p = sub.add_parser("fit")
    p.add_argument("--log", required=True)
    p.add_argument("--input", required=True, help="recut-NNNNN.png dir")
    p.add_argument("--cache", required=True)
    p.add_argument("--pages", default="")
    p.add_argument("--band", type=int, default=4)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_fit)
    p = sub.add_parser("render")
    p.add_argument("--dump", required=True)
    p.add_argument("--log", default="", help="job log: apply its vision/banner page crops so dump boxes line up")
    p.add_argument("--input", required=True)
    p.add_argument("--pages", default="")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=_cmd_render)
    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
