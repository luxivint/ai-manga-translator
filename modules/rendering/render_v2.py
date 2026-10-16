"""RENDER_V2: balloon-aware, style-aware Turkish typesetting.

Produces TextItemProperties dicts consumed by ImageSaveRenderer (same contract
as the old path in scripts/local_batch.py). Does not touch detection, OCR or
cleaning; it only reads the original and cleaned images.
"""
from __future__ import annotations

import json
import math
import statistics
import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QTextBlockFormat, QTextCursor, QTextDocument, QTextOption

from app.ui.canvas.text.text_item_properties import TextItemProperties
from app.ui.canvas.text_item import OutlineInfo, OutlineType
from modules.rendering import balloon_layout as bl
from modules.rendering import font_metrics as fm
from modules.rendering import orig_mask
from modules.rendering import font_profiles as fp
from modules.rendering import text_style as tst
from modules.rendering import typeset as ts
from modules.rendering.font_library import FALLBACK_FONT, load_style_map, pick_font, scan_fonts

ROOT = Path(__file__).resolve().parents[2]
_FAMILY_CACHE: dict[str, str] = {}
_COVERAGE: dict[str, bool] | None = None
FLOOR_SIZE = int(os.environ.get("RENDER_FLOOR_SIZE", "9"))
BAND_RATIO = float(os.environ.get("RENDER_BAND_RATIO", "1.45"))


def render_v2_enabled() -> bool:
    return os.environ.get("RENDER_V2", "true").lower() == "true"


def casing_policy(uppercase_flag: bool) -> str:
    env = os.environ.get("RENDER_CASING", "").lower()
    if env in {"upper", "keep", "auto"}:
        return env
    return "upper"  # product decision: all text in capitals (RENDER_CASING=auto|keep to override)


def _force_upper() -> bool:
    return os.environ.get("RENDER_FORCE_UPPERCASE", "false").lower() == "true"


def _coverage() -> dict[str, bool]:
    global _COVERAGE
    if _COVERAGE is None:
        cache = ROOT / "render_test" / "font_coverage.json"
        if cache.exists():
            _COVERAGE = json.loads(cache.read_text(encoding="utf-8"))
        else:
            _COVERAGE = scan_fonts(ROOT)
    return _COVERAGE


def family_for(rel_path: str) -> str | None:
    if rel_path in _FAMILY_CACHE:
        return _FAMILY_CACHE[rel_path]
    fid = QFontDatabase.addApplicationFont(str(ROOT / rel_path))
    fams = QFontDatabase.applicationFontFamilies(fid) if fid != -1 else []
    _FAMILY_CACHE[rel_path] = fams[0] if fams else None
    return _FAMILY_CACHE[rel_path]


def font_for_style(style: str, default_family: str) -> str:
    mapping = load_style_map(ROOT / "render_test" / "font_map.json")
    rel = pick_font(style, _coverage(), mapping, ROOT)
    return family_for(rel) or family_for(FALLBACK_FONT) or default_family


# ---------------------------------------------------------------- geometry
def _largest_rect(grid: np.ndarray) -> tuple[int, int, int, int]:
    """Largest all-True axis-aligned rectangle (x1, y1, x2, y2) in a boolean grid."""
    h, w = grid.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, (0, 0, 0, 0))
    for y in range(h):
        heights = np.where(grid[y], heights + 1, 0)
        stack: list[int] = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and heights[stack[-1]] >= cur:
                idx = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = int(heights[idx]) * (x - left)
                if area > best[0]:
                    best = (area, (left, y - int(heights[idx]) + 1, x, y + 1))
            stack.append(x)
    return best[1]


def bubble_interior_ex(cleaned: np.ndarray, blk):
    """Largest safe rectangle inside the balloon, from the cleaned page.

    Interior = pixels close to the balloon fill colour (sampled in the cleaned
    text box), connected to the text centre. Works for white, tinted and black
    narration boxes. Returns (rect, interior mask, (bx1, by1)); (None, None, None) when the balloon cannot be segmented.
    """
    H, W = cleaned.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in blk.bubble_xyxy]
    bx1, by1, bx2, by2 = max(0, bx1), max(0, by1), min(W, bx2), min(H, by2)
    if bx2 - bx1 < 20 or by2 - by1 < 20:
        return None, None, None
    tx1, ty1, tx2, ty2 = [int(v) for v in blk.xyxy]
    tx1, ty1, tx2, ty2 = max(bx1, tx1), max(by1, ty1), min(bx2, tx2), min(by2, ty2)
    if tx2 <= tx1 or ty2 <= ty1:
        return None, None, None
    patch = cleaned[ty1:ty2, tx1:tx2].reshape(-1, 3)
    fill = np.median(patch, axis=0)
    crop = cleaned[by1:by2, bx1:bx2].astype(np.int16)
    sim = (np.abs(crop - fill).sum(axis=2) < 60).astype(np.uint8)
    sim = cv2.morphologyEx(sim, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels = cv2.connectedComponents(sim, connectivity=4)
    cy, cx = (ty1 + ty2) // 2 - by1, (tx1 + tx2) // 2 - bx1
    lab = labels[cy, cx]
    if lab == 0:
        ys, xs = np.nonzero(labels[ty1 - by1:ty2 - by1, tx1 - bx1:tx2 - bx1])
        if len(ys) == 0:
            return None, None, None
        lab = np.bincount(labels[ty1 - by1:ty2 - by1, tx1 - bx1:tx2 - bx1][ys, xs]).argmax()
    comp = (labels == lab).astype(np.uint8)
    # fill holes (leftover glyph remnants) and keep a safety margin from the outline
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 1, -1)
    margin = max(3, int(0.022 * min(filled.shape)))
    inner = cv2.erode(filled, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    s = max(1, int(min(inner.shape) / 90))
    small = inner[::s, ::s].astype(bool)
    if not small.any():
        return None, None, None
    x1, y1, x2, y2 = _largest_rect(small)
    x1, y1, x2, y2 = x1 * s + bx1, y1 * s + by1, min(x2 * s, inner.shape[1]) + bx1, min(y2 * s, inner.shape[0]) + by1
    if (x2 - x1) * (y2 - y1) < 0.25 * (tx2 - tx1) * (ty2 - ty1):
        return None, None, None
    return (x1, y1, x2, y2), inner.astype(bool), (bx1, by1)


def bubble_interior(cleaned: np.ndarray, blk) -> tuple[int, int, int, int] | None:
    return bubble_interior_ex(cleaned, blk)[0]


def free_region(blk, W: int, H: int, orig_xyxy, tight: bool = False) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in orig_xyxy]
    w, h = x2 - x1, y2 - y1
    ex, ey = (int(w * 0.04), int(h * 0.02)) if tight else (int(w * 0.10), int(h * 0.10))
    return max(0, x1 - ex), max(0, y1 - ey), min(W, x2 + ex), min(H, y2 + ey)


def page_glyph_mask(image: np.ndarray, cleaned: np.ndarray, xyxy):
    """Glyph mask of a box. Geometry = pixels the cleaner changed (hugs the letters, also for stylised SFX where
    the segmenter mask is a blob), restricted to the cleaner's ORIGINAL-page text mask so repaint artefacts
    elsewhere in the box never count as lettering. No registered mask (standalone use): plain difference."""
    diff, off = tst.glyph_mask(image, cleaned, xyxy)
    got = orig_mask.glyph_mask(image, xyxy)
    if got is None or got[0].shape != diff.shape:
        return diff, off
    page = got[0]
    near = cv2.dilate(page.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    both = diff & near
    return (both, off) if int(both.sum()) >= 0.5 * int(diff.sum()) or not page.any() else (diff, off)


# ------------------------------------------------------------------ colour
def _lum(rgb) -> float:
    r, g, b = [float(v) for v in rgb]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sample_colors(image: np.ndarray, cleaned: np.ndarray, xyxy, fallback=None):
    """(text_rgb, bg_rgb, flat_bg) from original glyph pixels vs the cleaned background."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return fallback, None, False
    o = image[y1:y2, x1:x2].astype(np.int16)
    c = cleaned[y1:y2, x1:x2].astype(np.int16)
    bg = np.median(c.reshape(-1, 3), axis=0)
    flat = float(c.reshape(-1, 3).std(axis=0).mean()) < 18
    diff = np.abs(o - c).sum(axis=2)
    sel = diff > 90
    if int(sel.sum()) < 12:
        return fallback, tuple(int(v) for v in bg), flat
    thr = np.percentile(diff[sel], 50)
    px = o[diff >= thr]
    text = tuple(int(v) for v in np.median(px, axis=0))
    return text, tuple(int(v) for v in bg), flat


def choose_colors(text_rgb, bg_rgb, flat_bg: bool, size: float, default=(0, 0, 0)):
    """Text colour (kept light-on-dark), outline colour, outline width."""
    text = text_rgb or default
    if bg_rgb is not None and abs(_lum(text) - _lum(bg_rgb)) < 45:  # 45-90: keep the source colour, the outline carries contrast
        text = (0, 0, 0) if _lum(bg_rgb) > 128 else (255, 255, 255)
    outline_col = (0, 0, 0) if _lum(text) > 150 else (255, 255, 255)
    if flat_bg and bg_rgb is not None and abs(_lum(text) - _lum(bg_rgb)) >= 90:
        width = 0.0
    else:
        width = float(min(4, max(1, round(size * 0.07))))
    return text, outline_col, width


# ------------------------------------------------------------------ measure
def _qfont(family: str, size: float) -> QFont:
    f = QFont(family, int(size))
    return f


def make_measure(family: str, line_spacing: float):
    def measure(size: int):
        fm = QFontMetricsF(_qfont(family, size))
        return (lambda s: fm.horizontalAdvance(s)), fm.lineSpacing() * line_spacing
    return measure


def doc_size(text: str, family: str, size: int, line_spacing: float, alignment, direction) -> tuple[float, float]:
    doc = QTextDocument()
    doc.setDefaultFont(_qfont(family, size))
    doc.setPlainText(text)
    opt = QTextOption()
    opt.setTextDirection(direction)
    doc.setDefaultTextOption(opt)
    cur = QTextCursor(doc)
    cur.select(QTextCursor.SelectionType.Document)
    fmt = QTextBlockFormat()
    fmt.setLineHeight(line_spacing * 100, QTextBlockFormat.LineHeightTypes.ProportionalHeight.value)
    fmt.setAlignment(alignment)
    cur.mergeBlockFormat(fmt)
    s = doc.size()
    return s.width(), s.height()


# --------------------------------------------------------------------- main
VALID_STYLES = {"normal", "shout", "whisper", "thought", "narration", "sfx", "sign", "phone", "handwriting", "title"}
MIN_LINE_SPACING = float(os.environ.get("RENDER_MIN_LINE_SPACING", "1.12"))  # room for İ dots / Ğ breves in capitals
MAX_TARGET_SIZE = int(os.environ.get("RENDER_MAX_TARGET_SIZE", "400"))  # source-derived size replaces the old 58 cap
BUBBLE_MAX_GROW = 1.22  # balloons with room may set Turkish a little larger than the English
TINY_PX = int(os.environ.get("RENDER_TINY_PX", "14"))  # balloon text below this size (at 800 px width) retries on its original box
STRICT_FLOOR = int(os.environ.get("RENDER_STRICT_FLOOR", "14"))  # a strict-fit block below this size falls back to growing
TARGET_MAX_GROW = 1.08   # Turkish may be set up to 8 percent bigger than the English cap height
TARGET_MIN_KEEP = 0.82   # try to keep at least 82 percent of the source size before shrinking further


def size_for_cap(family: str, cap_px: float, rel: str | None = None) -> float:
    """Point size whose capital letters are cap_px tall.

    Uses the drawn cap height measured from the font file (font_traits capem) because the OS/2
    metric Qt reports is wrong in several comic fonts; falls back to Qt's capHeight.
    """
    capem = (fp.traits_of(rel) or {}).get("capem") if rel else None
    if capem:
        from PySide6.QtGui import QGuiApplication

        screen = QGuiApplication.primaryScreen()
        dpi = screen.logicalDotsPerInchY() if screen is not None else 96.0
        return cap_px / capem * 72.0 / dpi
    cap100 = QFontMetricsF(_qfont(family, 100)).capHeight()
    return cap_px / cap100 * 100.0 if cap100 > 0 else 0.0


def casing_text(raw: str, source: str, style: str, policy: str) -> str:
    """Casing of the translated text. Untranslated names/usernames/URLs keep the source letters
    (plain upper(), so 'Wild Grylls' never becomes 'WİLD'); translations use Turkish i/İ, ı/I rules."""
    src_key = "".join((source or "").casefold().split())
    if policy == "upper" and src_key and "".join(raw.casefold().split()) == src_key:
        return source.upper()
    return ts.apply_casing(raw, style, policy)


def panel_region(cleaned: np.ndarray, obox, W: int, H: int):
    """Flat panel/sign rectangle around a free text box (largest colour-similar rect), or None."""
    x1, y1, x2, y2 = [int(v) for v in obox]
    w, h = x2 - x1, y2 - y1
    ex, ey = int(w * 0.35) + 16, int(h * 0.60) + 10
    class _B:
        pass
    b = _B()
    b.bubble_xyxy = (max(0, x1 - ex), max(0, y1 - ey), min(W, x2 + ex), min(H, y2 + ey))
    b.xyxy = (x1, y1, x2, y2)
    r = bubble_interior(cleaned, b)
    if r is None:
        return None
    rx1, ry1, rx2, ry2 = r
    if rx1 > x1 + 0.1 * w or rx2 < x2 - 0.1 * w or ry1 > y1 + 0.2 * h or ry2 < y2 - 0.2 * h:
        return None  # does not contain the original text: the flood leaked or stopped early
    return r


def widen(region, fw: float, W: int, H: int, fh: float = 1.0):
    """Grow a region about its centre (fw sideways, fh vertically) but stay inside the page margins."""
    x1, y1, x2, y2 = region
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * fw, (y2 - y1) * fh
    mx, my = int(W * 0.03), int(H * 0.005)
    return (max(mx, int(cx - w / 2)), max(my, int(cy - h / 2)),
            min(W - mx, int(cx + w / 2)), min(H - my, int(cy + h / 2)))


def lines_inside(plan, text: str, size: int, region, line_spacing: float, frac: float = 0.97) -> bool:
    """True if every line of `text` (centred in `region`) lies inside the balloon interior mask."""
    inner, off = plan["inner"], plan["inner_off"]
    if inner is None:
        return True
    width, line_h = make_measure(plan["family"], line_spacing)(size)
    strict = bool(plan.get("strict"))  # members of a multi-block balloon: every line box (plus clearance) inside
    side = 0.02 * inner.shape[1] if strict else 0.0  # clearance to the outline on top of the eroded mask
    lines = text.split("\n")
    total = line_h * len(lines)
    cx, cy = (region[0] + region[2]) / 2, (region[1] + region[3]) / 2
    for i, ln in enumerate(lines):
        w = width(ln)
        y0, y1 = cy - total / 2 + i * line_h, cy - total / 2 + (i + 0.9) * line_h
        x0, x1 = cx - w / 2 - side, cx + w / 2 + side
        crop = inner[max(0, int(y0 - off[1])):max(0, int(y1 - off[1])), max(0, int(x0 - off[0])):max(0, int(x1 - off[0]))]
        if crop.size == 0 or int(y0 - off[1]) < 0 or int(x0 - off[0]) < 0 or float(crop.mean()) < frac:
            return False
        if strict:  # the four corners of the line box (plus clearance) must lie inside the balloon, not just 97 percent of its area
            ih, iw = inner.shape
            for px, py in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
                qx, qy = int(px - off[0]), int(py - off[1])
                if qx < 0 or qy < 0 or qx >= iw or qy >= ih or not inner[qy, qx]:
                    return False
    return True


def _merge_closest(groups: list[dict], target: int) -> list[dict]:
    """Merge the vertically closest neighbouring groups until only `target` groups remain."""
    groups = [dict(g) for g in groups]
    while len(groups) > target > 0:
        gaps = [groups[i + 1]["box"][1] - groups[i]["box"][3] for i in range(len(groups) - 1)]
        i = int(np.argmin(gaps))
        a, b = groups[i], groups[i + 1]
        groups[i:i + 2] = [{
            "box": (min(a["box"][0], b["box"][0]), a["box"][1], max(a["box"][2], b["box"][2]), b["box"][3]),
            "rows": a["rows"] + b["rows"], "aspect": a["aspect"] + b["aspect"]}]
    return groups


def untouched_by_cleaner(image: np.ndarray, cleaned: np.ndarray, xyxy, max_changed: float = 0.012) -> bool:
    """True when (almost) no pixel of the OCR box differs between the original and the cleaned page."""
    H, W = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return False
    d = np.abs(image[y1:y2, x1:x2].astype(np.int16) - cleaned[y1:y2, x1:x2].astype(np.int16)).sum(axis=2)
    return float((d > 30).mean()) < max_changed


def ink_box(mask: np.ndarray, off, obox) -> tuple[float, float, float, float]:
    """(left, top, right, bottom) of the original lettering in page coordinates (falls back to the OCR box)."""
    ox1, oy1, ox2, oy2 = [float(v) for v in obox]
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return ox1, oy1, ox2, oy2
    l, t, r, b = off[0] + xs.min(), off[1] + ys.min(), off[0] + xs.max() + 1, off[1] + ys.max() + 1
    # stay inside the OCR box (masks of stylised lettering can include halo/smear pixels)
    return max(l, ox1), max(t, oy1), min(r, ox2), min(b, oy2)


def caps_center_offset(family: str, size: float, n_lines: int, line_spacing: float) -> float:
    """Distance from the top of a text item to the vertical centre of its capital letters (QTextDocument adds a
    4 px margin, the first baseline sits at the font ascent, extra line spacing goes below each line)."""
    fm_ = QFontMetricsF(_qfont(family, size))
    cap_h = -fm_.tightBoundingRect("H").top() or fm_.capHeight()
    top = 4.0 + fm_.ascent() - cap_h
    bottom = 4.0 + fm_.ascent() + (max(1, n_lines) - 1) * fm_.lineSpacing() * line_spacing
    return (top + bottom) / 2.0


def needs_outline(cleaned: np.ndarray, xyxy, text_rgb, min_delta: float = 50.0, max_clash: float = 0.25) -> bool:
    """True when more than max_clash of the backdrop pixels behind the text are close in lightness to the text."""
    H, W = cleaned.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    crop = cleaned[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
    if crop.size == 0:
        return True
    lum = crop[..., 0] * 0.2126 + crop[..., 1] * 0.7152 + crop[..., 2] * 0.0722
    return float((np.abs(lum - _lum(text_rgb)) < min_delta).mean()) > max_clash


def split_merged_blocks(blocks, orig_boxes, image, cleaned):
    """OCR sometimes merges unrelated texts (menu bar, titles, question, username) into one block.

    Groups are found on the ORIGINAL page (local ink, independent of what the cleaner did): rows separated by
    more than ~0.8 row heights or by a rule line. A block is split when it has >= 3 groups or groups of clearly
    different letter size; an evenly spaced list (status window) stays one block. Translated lines go to the groups
    in reading order; fewer lines than groups merge the closest groups, more lines are shared by ink length."""
    import copy

    out_b, out_o = [], []
    for blk, obox in zip(blocks, orig_boxes):
        style = (getattr(blk, "style", "") or "normal").lower()
        raw = (blk.translation or "").strip()
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
        keep = (blk.text_class == "text_bubble" and style not in {"phone", "sign"}) or len(lines) < 2
        groups = [] if keep else tst.text_groups(image, obox)
        if len(groups) >= 2:
            caps = [float(np.median([r[1] - r[0] for r in g["rows"]])) for g in groups]
            varied = max(caps) > 1.6 * min(caps)
            rows = [r for g in groups for r in g["rows"]]
            centers = [(r[0] + r[1]) / 2 for r in rows]
            pitch = np.diff(centers) if len(centers) >= 3 else np.array([])
            evenly_spaced = len(pitch) >= 2 and (pitch.max() - pitch.min()) <= 0.35 * float(np.median(pitch))
            if evenly_spaced or (len(groups) < 3 and not varied):  # a status window / list stays one block
                groups = []
        else:
            groups = []
        if os.environ.get("RENDER_DEBUG"):
            print(f"[split] box={list(map(int, obox))} lines={len(lines)} groups={[g['box'] for g in groups]}", flush=True)
        if not groups:
            out_b.append(blk)
            out_o.append(obox)
            continue
        if len(lines) < len(groups):
            groups = _merge_closest(groups, len(lines))
        segs = [[ln] for ln in lines] if len(lines) == len(groups) else tst.allocate_lines(lines, [g["aspect"] for g in groups])
        if not segs:
            out_b.append(blk)
            out_o.append(obox)
            continue
        for g, seg in zip(groups, segs):
            sub = copy.copy(blk)
            box = np.array(g["box"])
            sub.xyxy = box.copy()
            sub.bubble_xyxy = None
            sub.text_class = "text_free"
            sub.split_origin = True
            one_row = len(g["rows"]) == 1  # a single text row: the pieces are menu items, not stacked lines
            sub.translation = " ".join(seg) if one_row else "\n".join(seg)
            sub.text = ""  # source text of a merged block cannot be split per group
            out_b.append(sub)
            out_o.append(box)
    return out_b, out_o


def _source_cap(image: np.ndarray, obox) -> float:
    """Letter height of the original lettering in a box (component based, so multi-row blocks do not inflate it)."""
    try:
        mask, _ = tst.ink_mask_original(image, obox)
        return float(tst.cap_height(mask)) if mask.any() else 0.0
    except Exception:
        return 0.0


def drop_split_duplicates(blocks, orig_boxes):
    """A title detected twice (alone + as the first line of a bigger merged UI block) leaves a split piece that repeats
    the standalone block: drop the piece (RENDER_STRIP_NEIGHBOR_LINES=false disables). Nothing is lost, the standalone
    block draws the same words."""
    from modules.utils import block_cleanup as bcl

    if not bcl.strip_line_enabled() or len(blocks) < 2:
        return blocks, orig_boxes
    boxes = [[float(v) for v in o] for o in orig_boxes]
    drop = set(bcl.split_piece_duplicates([(b.translation or "") for b in blocks], boxes,
                                          [bool(getattr(b, "split_origin", False)) for b in blocks]))
    if not drop:
        return blocks, orig_boxes
    if os.environ.get("RENDER_DEBUG"):
        print(f"[split-dup] dropped split pieces {sorted(drop)} that repeat a neighbour block", flush=True)
    keep = [i for i in range(len(blocks)) if i not in drop]
    return [blocks[i] for i in keep], [orig_boxes[i] for i in keep]


def dedupe_duplicate_blocks(blocks, orig_boxes):
    """Safety net after translation: blocks that sit in one balloon or overlap on the page and whose translation is a
    substring / near-duplicate of a longer neighbour are dropped (that text is already drawn by the neighbour); a
    phrase repeated back to back inside one translation is collapsed. RENDER_DEDUPE_TRANSLATIONS=false disables."""
    import copy

    from modules.utils import block_cleanup as bcl

    if not bcl.dedupe_enabled() or len(blocks) < 2:
        return blocks, orig_boxes
    boxes = [[float(v) for v in o] for o in orig_boxes]
    bubbles = [None if getattr(b, "bubble_xyxy", None) is None else tuple(float(v) for v in b.bubble_xyxy) for b in blocks]
    groups = bl.group_by_balloon(bubbles, boxes) + bcl.overlap_groups(boxes)
    if not groups:
        return blocks, orig_boxes
    drop, rewrite = bcl.drop_duplicate_translations(blocks, boxes, groups)
    if not drop and not rewrite:
        return blocks, orig_boxes
    blocks, orig_boxes = list(blocks), list(orig_boxes)
    for i, t in rewrite.items():
        blocks[i] = copy.copy(blocks[i])
        blocks[i].translation = t
    if os.environ.get("RENDER_DEBUG"):
        print(f"[dedupe] dropped duplicate translations of blocks {drop}, collapsed repeats in {sorted(rewrite)}", flush=True)
    dropped = set(drop)
    keep = [i for i in range(len(blocks)) if i not in dropped]
    return [blocks[i] for i in keep], [orig_boxes[i] for i in keep]


def prepare_balloon_blocks(blocks, orig_boxes, image):
    """Blocks that share a balloon: drop words one block repeats from the previous one (a sentence continued across
    blocks that the translator wrote out twice); absorb tiny stray 1-2 word blocks into the balloon's main text; and,
    when RENDER_MERGE_BALLOON is on and the blocks read as one continued speech (or stacked lines of one narration
    box), lay them out as ONE paragraph block (one font, one size, balanced wrapping)."""
    import copy

    from modules.translation import consistency

    n = len(blocks)
    if n < 2:
        return blocks, orig_boxes
    bubbles = [None if getattr(b, "bubble_xyxy", None) is None else tuple(float(v) for v in b.bubble_xyxy) for b in blocks]
    boxes = [[float(v) for v in o] for o in orig_boxes]
    blocks, orig_boxes = list(blocks), list(orig_boxes)
    drop: set[int] = set()

    def style_of(i):
        st = (getattr(blocks[i], "style", "") or "normal").lower()
        return st if st in VALID_STYLES else "normal"

    def merge(idx, base):
        """Lay blocks idx (reading order) out as ONE block built on `base`; the others are dropped."""
        merged = copy.copy(blocks[base])
        merged.translation = " ".join(blocks[i].translation.strip() for i in idx)
        merged.text = " ".join((getattr(blocks[i], "text", "") or "").strip() for i in idx).strip()
        box = np.array([min(boxes[i][0] for i in idx), min(boxes[i][1] for i in idx),
                        max(boxes[i][2] for i in idx), max(boxes[i][3] for i in idx)])
        merged.xyxy = box.copy()
        first = idx[0]
        base_bubble = bubbles[base]
        blocks[first], orig_boxes[first], boxes[first] = merged, box, [float(v) for v in box]
        bubbles[first] = base_bubble
        drop.update(idx[1:])
        return merged

    from modules.utils import block_cleanup as bcl

    if bcl.echo_merge_enabled():
        # OCR windows of ONE paragraph (each window re-read the previous window's last words): translated separately
        # they repeat words at several sizes. Join them into one paragraph, repeated leading words dropped.
        cand = [i for i in range(n) if (blocks[i].translation or "").strip() and not getattr(blocks[i], "is_ad", False)]
        srcs = [(getattr(blocks[i], "text", "") or "") for i in cand]
        for grp in bcl.overlap_chain_groups([boxes[i] for i in cand], srcs):
            idx = [cand[k] for k in grp]
            done, parts = "", []
            for pos, i in enumerate(idx):
                t = blocks[i].translation.strip()
                if pos:
                    ov = bcl.ocr_overlap_words(getattr(blocks[idx[pos - 1]], "text", "") or "", getattr(blocks[i], "text", "") or "")
                    t = bcl.dedupe_overlap_translation(done, t, max(ov, 1))
                parts.append(t)
                done = " ".join(parts)
            blocks[idx[0]] = copy.copy(blocks[idx[0]])
            for i, t in zip(idx, parts):
                if i != idx[0]:
                    blocks[i] = copy.copy(blocks[i])
                blocks[i].translation = t
            base = max(idx, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
            merged = merge(idx, base)
            merged.translation = bcl.collapse_repeats(merged.translation)
            if os.environ.get("RENDER_DEBUG"):
                print(f"[merge-echo] {len(idx)} OCR windows -> one paragraph: {merged.translation[:70]!r}", flush=True)

    for g in bl.group_by_balloon(bubbles, boxes):
        g = [i for i in g if (blocks[i].translation or "").strip() and not getattr(blocks[i], "is_ad", False) and i not in drop]
        if len(g) < 2:
            continue
        texts = consistency.dedupe_adjacent([blocks[i].translation.strip() for i in g])
        for i, t in zip(g, texts):
            if t != blocks[i].translation.strip():
                blocks[i] = copy.copy(blocks[i])
                blocks[i].translation = t
        if bl.absorb_enabled():
            owners = [bubbles[i] for i in g if bubbles[i] is not None]
            inside = []
            for i in g:
                cx, cy = (boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2
                inside.append(any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in owners))
            strays = bl.stray_fragments([blocks[i].translation for i in g], [getattr(blocks[i], "text", "") for i in g],
                                        [_source_cap(image, orig_boxes[i]) for i in g], inside)
            if strays:
                main = max(range(len(g)), key=lambda k: len((blocks[g[k]].translation or "").split()))
                idx = sorted([g[main]] + [g[k] for k in strays], key=lambda i: ((boxes[i][1] + boxes[i][3]) / 2, boxes[i][0]))
                merge(idx, g[main])
                if os.environ.get("RENDER_DEBUG"):
                    print(f"[absorb-stray] {len(strays)} tiny block(s) joined the balloon text: {blocks[idx[0]].translation[:70]!r}", flush=True)
                g = [i for i in g if i not in drop]
                if len(g) < 2:
                    continue
        if not bl.merge_enabled():
            continue
        styles = [style_of(i) for i in g]
        rows = [_source_cap(image, orig_boxes[i]) for i in g]
        gaps = [float(orig_boxes[g[k + 1]][1]) - float(orig_boxes[g[k]][3]) for k in range(len(g) - 1)]
        texts = [blocks[i].translation for i in g]
        bases = [i for i in g if bubbles[i] is not None and blocks[i].text_class == "text_bubble"]
        if os.environ.get("RENDER_DEBUG"):
            print(f"[merge-check] ids={g} styles={styles} rows={[round(float(r), 1) for r in rows]} gaps={[round(x) for x in gaps]} bases={bases}", flush=True)
        if not bases or not bl.mergeable(texts, styles, gaps, rows):
            continue
        base = max(bases, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
        merged = merge(g, base)
        if os.environ.get("RENDER_DEBUG"):
            print(f"[merge-balloon] {len(g)} blocks -> one paragraph: {merged.translation[:70]!r}", flush=True)

    # stacked lines of one narration / caption box (no balloon boxes to group by): one paragraph
    if bl.narration_merge_enabled() and bl.merge_enabled():
        live = [i for i in range(n) if i not in drop and (blocks[i].translation or "").strip() and not getattr(blocks[i], "is_ad", False)]
        sub_styles = [style_of(i) for i in live]
        sub_boxes = [boxes[i] for i in live]
        sub_rows = [_source_cap(image, orig_boxes[i]) if sub_styles[k] == "narration" else 0.0 for k, i in enumerate(live)]
        for g in bl.narration_groups(sub_boxes, sub_styles, sub_rows):
            g = [live[k] for k in g]
            rows = [_source_cap(image, orig_boxes[i]) for i in g]
            gaps = [float(boxes[g[k + 1]][1]) - float(boxes[g[k]][3]) for k in range(len(g) - 1)]
            texts = [blocks[i].translation for i in g]
            if not bl.mergeable(texts, ["narration"] * len(g), gaps, rows):
                continue
            base = max(g, key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]))
            merged = merge(g, base)
            if os.environ.get("RENDER_DEBUG"):
                print(f"[merge-narration] {len(g)} blocks -> one paragraph: {merged.translation[:70]!r}", flush=True)
    if drop:
        keep = [i for i in range(n) if i not in drop]
        blocks, orig_boxes = [blocks[i] for i in keep], [orig_boxes[i] for i in keep]
    return blocks, orig_boxes


def resolve_item_overlaps(items: list, plans: list, W: int, H: int, direction) -> int:
    """Rendered text boxes may not overlap (a later block's glow/outline or glyphs would hide an earlier block).
    Overlapping boxes are first pushed apart vertically (budget 60 percent of their own height for free text, 15
    percent inside a balloon), then both shrink one point at a time until clear or at the floor size.
    Returns the number of overlapping pairs that remain (0 = clean). RENDER_RESOLVE_OVERLAP=false disables."""
    if not bl.overlap_enabled() or len(items) < 2:
        return 0

    def poly(d):
        x, y = d["position"]
        w, h = float(d["width"]), float(d["height"])
        ix, iy = 4.0, 4.0 + 0.06 * h  # the document margin and line-spacing slack do not count as ink
        org = d.get("transform_origin")
        ox, oy = (org[0], org[1]) if org is not None and len(org) == 2 else (w / 2, h / 2)
        return bl.rotated_rect(x + ix, y + iy, max(1.0, w - 2 * ix), max(1.0, h - 2 * iy), float(d.get("rotation") or 0.0),
                               (ox - ix, oy - iy))

    remaining = 0
    for rnd in range(10):
        pairs = bl.overlapping_pairs([poly(d) for d in items])
        remaining = len(pairs)
        if not pairs:
            return 0
        idx = sorted({k for a, b, _ in pairs for k in (a, b)})
        flat = [k for k in idx if not float(items[k].get("rotation") or 0.0) and not plans[k].get("strict_fit")]
        if flat and rnd < 4:
            boxes = [[items[k]["position"][0], items[k]["position"][1], items[k]["width"], items[k]["height"]] for k in flat]
            shift = [(0.15 if plans[k]["bubble"] else 0.6) * items[k]["height"] for k in flat]
            for k, b in zip(flat, bl.separate_vertically(boxes, H, shift)):
                items[k]["position"] = (items[k]["position"][0], int(round(b[1])))
            continue
        for k in idx:  # still overlapping after the move budget: shrink about the centre
            d = items[k]
            if d["font_size"] <= FLOOR_SIZE:
                continue
            cx, cy = d["position"][0] + d["width"] / 2, d["position"][1] + d["height"] / 2
            size = d["font_size"] - 1
            w, h = doc_size(d["text"], d["font_family"], size, d["line_spacing"], d["alignment"], direction)
            d["font_size"], d["width"], d["height"] = size, w, h
            d["position"] = (int(cx - w / 2), int(cy - h / 2))
    return remaining


def build_items(blocks, image, cleaned, orig_boxes, render_settings, alignment, target_code: str):
    """Returns list of TextItemProperties dicts (V2 layout)."""
    H, W = image.shape[:2]
    line_spacing = max(float(render_settings.line_spacing), MIN_LINE_SPACING)
    policy = casing_policy(render_settings.upper_case)
    plans = []
    blocks, orig_boxes = split_merged_blocks(blocks, orig_boxes, image, cleaned)
    blocks, orig_boxes = drop_split_duplicates(blocks, orig_boxes)
    blocks, orig_boxes = dedupe_duplicate_blocks(blocks, orig_boxes)
    blocks, orig_boxes = prepare_balloon_blocks(blocks, orig_boxes, image)
    for blk, obox in zip(blocks, orig_boxes):
        raw = (blk.translation or "")
        if not raw.strip() or len(raw.strip()) == 1:
            continue
        style = (getattr(blk, "style", "") or "normal").lower()
        if style not in VALID_STYLES:
            style = "normal"
        if untouched_by_cleaner(image, cleaned, obox):
            # the cleaner left the original lettering in place (watermark / promo / logo kept on purpose):
            # a translation drawn on top would double the text, so keep the original
            if os.environ.get("RENDER_DEBUG"):
                print(f"[skip] original text left untouched at {[int(v) for v in obox]}", flush=True)
            continue
        mask, off = page_glyph_mask(image, cleaned, obox)
        src_key = "".join((getattr(blk, "text", "") or "").casefold().split())
        if src_key and src_key == "".join(raw.casefold().split()):
            ox1, oy1, ox2, oy2 = [int(v) for v in obox]
            if int(mask.sum()) < 0.02 * max(1, (ox2 - ox1) * (oy2 - oy1)):
                continue  # kept as-is (logo/URL/name) and the cleaner left it untouched: do not draw it twice
        cap_h = tst.cap_height(mask) if mask.any() else 0.0
        if cap_h >= 8:
            row_h = tst.row_cap_height(image, obox)
            if row_h >= 8 and cap_h > 1.35 * row_h:  # glued glow/smear: a letter cannot be taller than its text row
                cap_h = row_h * 0.92
            n_chars = len((getattr(blk, "text", "") or "").strip())
            if n_chars >= 3:
                # the English lettering has to fit its own box: ~0.84 cap^2 of area per character (glow/smear that
                # glues letters together can otherwise report a "cap height" several times too large)
                ox1_, oy1_, ox2_, oy2_ = [float(v) for v in obox]
                bound = 1.15 * math.sqrt(max(1.0, (ox2_ - ox1_) * (oy2_ - oy1_)) / (0.84 * n_chars))
                cap_h = min(cap_h, bound)
        in_bubble = blk.text_class == "text_bubble" and blk.bubble_xyxy is not None
        src_traits = fm.source_traits(mask) if cap_h >= 8 else None
        rel = fp.match_font(style, src_traits)
        family = family_for(rel) or family_for(FALLBACK_FONT) or render_settings.font_family
        italic = fp.wants_synthetic_italic(style, src_traits, rel)
        text = ts.normalize_spaces(casing_text(raw, getattr(blk, "text", ""), style, policy))
        text = fp.sanitize_for_font(text, rel)
        text_rgb, bg_rgb, flat = sample_colors(image, cleaned, obox, None)
        colors = tst.analyze_colors(image, mask, off, bg_rgb) if cap_h >= 12 else \
            {"fill": None, "gradient": None, "outline": None, "glow": None}
        angle, rect = (0.0, None)
        if not in_bubble or style in {"sign", "phone", "title", "sfx"}:
            angle, rect = tst.text_angle(mask) if cap_h >= 8 else (0.0, None)
            if (not angle and cap_h >= 8 and style in {"sign", "phone", "title", "sfx", "narration"}
                    and (not in_bubble or style in {"sign", "phone"}) and os.environ.get("RENDER_TILT_FALLBACK", "true").lower() == "true"):
                # the cleaned-page diff is ragged (leftover strokes) or too square for minAreaRect: read the tilt of the
                # lettering from the ORIGINAL ink instead, so diagonal source text stays diagonal
                rim_t, rim_toff = tst.ink_mask_original(image, obox)
                t_ang, t_rect = tst.estimate_tilt(rim_t)
                r_ang, _ = tst.text_angle(rim_t)
                # a rectangle fit that finds ANOTHER tilt contradicts the projection estimate (mixed horizontal + tilted
                # ink); pieces cut from a merged block are never re-tilted
                if t_ang and t_rect is not None and (not r_ang or abs(r_ang - t_ang) <= 6.0) and not getattr(blk, "split_origin", False):
                    angle = t_ang
                    rect = (t_rect[0] + rim_toff[0] - off[0], t_rect[1] + rim_toff[1] - off[1], t_rect[2], t_rect[3])
        region, is_bubble, inner, ioff = None, False, None, (0, 0)
        guard_inner, guard_off = None, (0, 0)
        if angle != 0.0 and rect is not None:
            cx, cy, long_len, short_len = rect
            cx, cy = cx + off[0], cy + off[1]
            region = (int(cx - long_len * 0.58), int(cy - short_len * 0.62), int(cx + long_len * 0.58), int(cy + short_len * 0.62))
        elif in_bubble:
            region, inner, ioff = bubble_interior_ex(cleaned, blk)
            is_bubble = region is not None
        if "\n" in text and blk.text_class != "text_bubble" and (getattr(blk, "text", "") or ""):
            rim_m, _ = tst.ink_mask_original(image, obox)
            if len(tst.row_runs(rim_m)) == 1:  # one text row in the source: the translation is one row too
                text = " ".join(text.split())
        list_like = ("\n" in text) and (style in {"phone", "sign"} or text.count("\n") >= 2)
        if region is None and not list_like:
            region = panel_region(cleaned, obox, W, H) if style in {"sign", "phone", "narration", "title"} else None
        if region is None:
            region = free_region(blk, W, H, obox, tight=style in {"sign", "phone"})
        shaped = False
        if is_bubble and inner is not None and style not in {"sfx", "shout", "title"}:
            cnts, _ = cv2.findContours(inner.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = np.concatenate(cnts) if cnts else None
            hull_area = float(cv2.contourArea(cv2.convexHull(pts))) if pts is not None else 0.0
            if hull_area > 0 and float(inner.sum()) / hull_area < 0.88 and bl.unify_enabled():
                # the cleaned page can be patchy (light-region wipes, repaint artefacts): the balloon outline is still
                # intact on the ORIGINAL page, so measure the interior there before giving up on the balloon shape
                alt_r, alt_inner, alt_off = bubble_interior_ex(image, blk)
                if alt_inner is not None:
                    acnts, _ = cv2.findContours(alt_inner.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    apts = np.concatenate(acnts) if acnts else None
                    a_hull = float(cv2.contourArea(cv2.convexHull(apts))) if apts is not None else 0.0
                    if a_hull > 0 and float(alt_inner.sum()) / a_hull >= 0.88:
                        region, inner, ioff, hull_area = alt_r, alt_inner, alt_off, 0.0
            if hull_area > 0 and float(inner.sum()) / hull_area < 0.88:
                # ragged flood fill (translucent hologram, textured panel): the balloon shape is not trustworthy, so
                # the text may only use the footprint of the original lettering (+12% sideways, +45% vertically)
                ox1, oy1, ox2, oy2 = [float(v) for v in obox]
                ow, oh = ox2 - ox1, oy2 - oy1
                region = (int(max(0, ox1 - 0.12 * ow)), int(max(0, oy1 - 0.45 * oh)),
                          int(min(W, ox2 + 0.12 * ow)), int(min(H, oy2 + 0.45 * oh)))
                bb_ = blk.bubble_xyxy
                if float(inner.sum()) >= 0.35 * max(1.0, float((bb_[2] - bb_[0]) * (bb_[3] - bb_[1]))):
                    guard_inner, guard_off = inner, ioff  # a spiky balloon is legitimately non-convex: keep it as the line guard
                inner, shaped = None, True
        if not is_bubble and not angle and not list_like and not flat and style not in {"sfx", "shout", "title"} and cap_h >= 8:
            # free text over artwork (hologram ellipses, signs on busy scenes): the backdrop shape is unknown, so the
            # translation may only use the footprint of the original lettering (+12% sideways, +45% vertically)
            ox1, oy1, ox2, oy2 = [float(v) for v in obox]
            ow, oh = ox2 - ox1, oy2 - oy1
            region = (int(max(0, ox1 - 0.12 * ow)), int(max(0, oy1 - 0.45 * oh)),
                      int(min(W, ox2 + 0.12 * ow)), int(min(H, oy2 + 0.45 * oh)))
            shaped = True
        n_src_lines = tst.line_pitch(mask)[1] if mask.any() else 1
        ink = ink_box(mask, off, obox)
        src_centers = []
        list_center = False
        if list_like:
            rim_mask, rim_off = tst.ink_mask_original(image, obox)  # rows measured on the original page only
            runs_ = tst.row_runs(rim_mask)
            src_centers = [rim_off[1] + (a + b) / 2 for a, b in runs_]
            ext = []
            for a, b in runs_:
                xs_ = np.nonzero(rim_mask[a:b].any(axis=0))[0]
                if len(xs_):
                    ext.append((float(xs_.min()), float(xs_.max())))
            if len(ext) >= 2:  # rows share a centre line but not a left edge -> a centred panel text, not a left list
                lefts, mids = [e[0] for e in ext], [(e[0] + e[1]) / 2 for e in ext]
                span = max(e[1] for e in ext) - min(e[0] for e in ext)
                list_center = (max(lefts) - min(lefts)) > 0.12 * span and (max(mids) - min(mids)) < 0.06 * span
            if int(rim_mask.sum()) >= 40:
                ink = ink_box(rim_mask, rim_off, obox)
                iw, ih = ink[2] - ink[0], ink[3] - ink[1]
                # a list never grows past the panel text column: original ink + a small margin
                region = (int(max(0, ink[0] - 0.02 * iw)), int(max(0, ink[1] - 0.04 * ih)),
                          int(min(W, ink[2] + 0.06 * iw)), int(min(H, ink[3] + 0.06 * ih)))
        plans.append(dict(guard_inner=guard_inner, guard_off=guard_off, list_center=list_center, shaped=shaped, italic=italic, ink=ink, src_centers=src_centers, blk=blk, style=style, text=text, family=family, font_rel=rel, region=region,
                          bubble=is_bubble, colors=(text_rgb, bg_rgb, flat), analysis=colors, angle=angle,
                          cap_h=cap_h, list_like=list_like, obox=[int(v) for v in obox], region_kind=("interior" if is_bubble else "other"), src_traits=src_traits, inner=inner, inner_off=ioff,
                          n_src_lines=n_src_lines))

    # balloon groups: blocks sharing one balloon (free text lying inside a balloon counts) get ONE font family and a
    # source cap-height compressed toward the balloon median (RENDER_BALLOON_UNIFY=false restores per-block sizing)
    balloon_groups: list[list[int]] = []
    if bl.unify_enabled() and len(plans) >= 2:
        speech = [p["style"] in bl.SPEECH_STYLES and not p["list_like"] for p in plans]
        cand = [i for i in range(len(plans)) if speech[i]]
        touch = 12.0 if os.environ.get("RENDER_GROUP_TOUCHING", "true").lower() == "true" else 0.0
        for g in bl.group_by_balloon([None if plans[i]["blk"].bubble_xyxy is None else tuple(float(v) for v in plans[i]["blk"].bubble_xyxy) for i in cand],
                                     [plans[i]["obox"] for i in cand], touch_gap=touch):
            balloon_groups.append([cand[k] for k in g])
        for g in balloon_groups:
            for i in g:
                plans[i]["strict"] = True
            lead = max(g, key=lambda i: len(plans[i]["text"]))
            own = [i for i in g if plans[i]["bubble"] and not plans[i]["angle"]]
            if len(own) >= 2:
                # several blocks of one balloon would all be centred in the SAME interior and print on top of each
                # other: each keeps to its own band of the balloon (its original rows +- 25 percent)
                for i in own:
                    rx1, ry1, rx2, ry2 = plans[i]["region"]
                    _, oy1, _, oy2 = [float(v) for v in plans[i]["obox"]]
                    h = oy2 - oy1
                    ny1, ny2 = max(ry1, int(oy1 - 0.25 * h)), min(ry2, int(oy2 + 0.25 * h))
                    if ny2 - ny1 >= 0.6 * h:
                        plans[i]["region"] = (rx1, ny1, rx2, ny2)
            shape = next((i for i in g if plans[i]["bubble"] and plans[i]["inner"] is not None), None)
            for i in g:  # free lettering inside the balloon is held to the same interior mask as its neighbours
                if shape is not None and plans[i]["inner"] is None and not plans[i]["bubble"] and not plans[i]["angle"]:
                    plans[i]["inner"], plans[i]["inner_off"] = plans[shape]["inner"], plans[shape]["inner_off"]
            caps = bl.compress_caps([plans[i]["cap_h"] for i in g])
            for i, c in zip(g, caps):
                plans[i]["cap_h"] = c
                if i != lead and (plans[i]["family"] != plans[lead]["family"] or plans[i]["italic"] != plans[lead]["italic"]):
                    plans[i].update(family=plans[lead]["family"], font_rel=plans[lead]["font_rel"], italic=plans[lead]["italic"])
                    plans[i]["text"] = fp.sanitize_for_font(plans[i]["text"], plans[lead]["font_rel"])
            if os.environ.get("RENDER_DEBUG"):
                print(f"[balloon-group] {g} family={plans[lead]['family']} caps={[round(plans[i]['cap_h'], 1) for i in g]}", flush=True)

    if os.environ.get("RENDER_STRICT_FIT", "true").lower() == "true":
        # crowded panel / UI blocks (touching another block, or cut from a merged OCR block): each translation must stay
        # inside its OWN original box (shrink only, no growth, no pushing into neighbours)
        for i, p in enumerate(plans):
            if p["bubble"] or p["style"] not in {"sign", "phone", "narration"}:
                continue
            ob = p["obox"]
            touching = any(j != i and not plans[j]["bubble"] and ob[0] - 10 < plans[j]["obox"][2] and plans[j]["obox"][0] < ob[2] + 10
                           and ob[1] - 10 < plans[j]["obox"][3] and plans[j]["obox"][1] < ob[3] + 10 for j in range(len(plans)))
            if touching or getattr(p["blk"], "split_origin", False):
                p["strict_fit"] = True
                p["region0"] = p["region"]
                if not p["angle"]:
                    bw, bh = ob[2] - ob[0], ob[3] - ob[1]
                    p["region"] = (int(max(0, ob[0] - 0.03 * bw)), int(max(0, ob[1] - 0.03 * bh)),
                                   int(min(W, ob[2] + 0.03 * bw)), int(min(H, ob[3] + 0.03 * bh)))
                if os.environ.get("RENDER_DEBUG"):
                    print(f"[strict-fit] block {ob} fits its own box", flush=True)

    def fit_list(plan, region, max_size):
        """List / status window: every entry stays on ONE row (never wraps); the size shrinks until the widest
        row fits the panel column. Returns None when even the floor size does not fit."""
        x1, y1, x2, y2 = region
        bw, bh = x2 - x1, y2 - y1
        rows_ = plan["text"].split("\n")
        for size in range(int(max_size), max(FLOOR_SIZE, int(render_settings.min_font_size)) - 1, -1):
            width, line_h = make_measure(plan["family"], 1.0)(size)
            if max(width(r) for r in rows_) + 8 <= bw and line_h * len(rows_) <= bh * 1.15:
                dw, dh = doc_size(plan["text"], plan["family"], size, 1.0, Qt.AlignmentFlag.AlignLeft, render_settings.direction)
                return plan["text"], size, dw, dh
        return None

    def fit_once(plan, region, max_size):
        if plan["list_like"] and plan["src_centers"] and len(plan["src_centers"]) == plan["text"].count("\n") + 1:
            got = fit_list(plan, region, max_size)
            if got is not None:
                return got
        x1, y1, x2, y2 = region
        measure = make_measure(plan["family"], line_spacing)
        pad = 2.0
        bw, bh = x2 - x1, y2 - y1
        mx = max_size
        out = size = dw = dh = None
        for _ in range(12):
            out, size, w, h, over = ts.fit_text(plan["text"], measure, bw, bh, max_size=mx,
                                                min_size=render_settings.min_font_size, floor=FLOOR_SIZE, pad=pad)
            dw, dh = doc_size(out, plan["family"], size, line_spacing, alignment, render_settings.direction)
            if (dw <= bw and dh <= bh) or size <= FLOOR_SIZE:
                break
            mx = size - 1
        return out, size, dw, dh

    def fit(plan, max_size):
        target = size_for_cap(plan["family"], plan["cap_h"], plan["font_rel"]) if plan["cap_h"] >= 8 else 0.0
        plan["target"] = target
        grow = BUBBLE_MAX_GROW if plan["bubble"] else TARGET_MAX_GROW
        if plan.get("strict_fit"):
            grow = 1.0
        cap = max_size if not target else max(FLOOR_SIZE, min(MAX_TARGET_SIZE, int(math.ceil(target * grow))))
        region = plan["region"]
        best = fit_once(plan, region, cap)
        plan["fit_region"] = region
        if plan.get("strict_fit"):
            if best[1] >= STRICT_FLOOR:
                return best
            # its own box is too small for a readable size: fall back to the normal (growing) fit
            plan["strict_fit"] = False
            region = plan["region"] = plan.get("region0", region)
            grow = BUBBLE_MAX_GROW if plan["bubble"] else TARGET_MAX_GROW
            cap = max_size if not target else max(FLOOR_SIZE, min(MAX_TARGET_SIZE, int(math.ceil(target * grow))))
            best = fit_once(plan, region, cap)
            plan["fit_region"] = region
        if plan["bubble"] and plan["inner"] is not None and best[1] < cap:
            # balloons are ellipses: the largest rectangle is narrow and tall. Try wider layouts and
            # keep one only if every laid-out line still lies inside the balloon interior.
            for fw, fh in ((1.2, 1.1), (1.4, 1.2), (1.6, 1.3)):
                wide = widen(region, fw, W, H, fh)
                cand = fit_once(plan, wide, cap)
                if cand[1] > best[1] and lines_inside(plan, cand[0], cand[1], wide, line_spacing):
                    best, plan["fit_region"] = cand, wide
                    if best[1] >= cap:
                        break
        if target and best[1] < TARGET_MIN_KEEP * target:
            # the detected region is too small for the translation (free text, signs, captions, holograms and
            # balloons whose interior could not be measured): grow around the ORIGINAL text box instead of
            # shrinking the font to a tiny size. The original lettering stood there, so it is safe ground.
            max_w = (region[2] - region[0]) if (plan["list_like"] or plan["shaped"]) else max(1.55 * (plan["obox"][2] - plan["obox"][0]), region[2] - region[0])
            bases = [region]
            if plan["bubble"] or plan["region_kind"] == "interior":
                bases.append(free_region(plan["blk"], W, H, plan["obox"]))
            steps = ((1.0, 1.1), (1.0, 1.2), (1.0, 1.3)) if plan["list_like"] else (
                (1.0, 1.0), (1.25, 1.0), (1.25, 1.8), (1.5, 1.0), (1.5, 2.0), (1.8, 1.0))
            for base in bases:
                for fw, fh in steps:
                    wide = widen(base, fw, W, H, fh) if (fw, fh) != (1.0, 1.0) else base
                    if wide[2] - wide[0] > max_w:  # never sprawl far beyond where the English text stood
                        mid = (wide[0] + wide[2]) / 2
                        wide = (int(mid - max_w / 2), wide[1], int(mid + max_w / 2), wide[3])
                    cand = fit_once(plan, wide, cap)
                    if plan["inner"] is not None and not lines_inside(plan, cand[0], cand[1], wide, line_spacing, 0.93):
                        continue  # a grown region may never push lines over the balloon / panel border
                    if cand[1] > best[1]:
                        best, plan["fit_region"] = cand, wide
                    if best[1] >= TARGET_MIN_KEEP * target:
                        break
                if best[1] >= TARGET_MIN_KEEP * target:
                    break
        return best

    max_size = render_settings.max_font_size
    fits = [fit(p, max_size) for p in plans]
    band_idx = [i for i, p in enumerate(plans) if p["bubble"] and p["style"] in {"normal", "thought", "whisper"}]
    if len(band_idx) > 2:
        capped = ts.apply_size_band([fits[i][1] for i in band_idx], BAND_RATIO)
        for i, cap in zip(band_idx, capped):
            if cap < fits[i][1]:
                fits[i] = fit_once(plans[i], plans[i]["fit_region"], cap)

    def balloon_cap(limit_fn):
        """Re-fit the members of every balloon group under the size limit limit_fn(sizes, caps) -> [max | None]."""
        for g in balloon_groups:
            limits = limit_fn([fits[i][1] for i in g], [plans[i]["cap_h"] for i in g])
            for i, lim in zip(g, limits):
                if lim is not None and fits[i][1] > lim:
                    fits[i] = fit_once(plans[i], plans[i]["fit_region"], max(FLOOR_SIZE, int(lim)))

    # one size band per balloon: nobody above median*(1+band) unless the source clearly had emphasis
    balloon_cap(bl.size_limits)

    def balloon_raise():
        """Free-text pieces inside a balloon that came out far smaller than their neighbours (their own tiny box
        limited them) may grow towards the balloon median inside the balloon interior of the lead block."""
        for g in balloon_groups:
            lead = next((i for i in g if plans[i]["bubble"] and plans[i]["inner"] is not None), None)
            if lead is None:
                continue
            sizes = [fits[i][1] for i in g]
            med = statistics.median(sizes)
            lo = int(med * (1.0 - bl.band()))
            lr = plans[lead]["fit_region"]
            for i in g:
                p = plans[i]
                if (fits[i][1] >= lo or p["bubble"] or p["angle"]
                        or bl.is_emphasis(p["cap_h"], [plans[j]["cap_h"] for j in g if j != i])):
                    continue
                ox1, oy1, ox2, oy2 = [float(v) for v in p["obox"]]
                cx, h = (ox1 + ox2) / 2, oy2 - oy1
                half = min(cx - lr[0], lr[2] - cx)
                if half <= (ox2 - ox1) / 2:
                    continue
                wide = (int(cx - half), int(max(0, oy1 - 0.3 * h)), int(cx + half), int(min(H, oy2 + 0.3 * h)))
                cand = fit_once(p, wide, int(med))
                probe = dict(p, inner=plans[lead]["inner"], inner_off=plans[lead]["inner_off"])
                if cand[1] > fits[i][1] and lines_inside(probe, cand[0], cand[1], wide, line_spacing, 0.93):
                    fits[i] = cand
                    p["fit_region"], p["inner"], p["inner_off"] = wide, plans[lead]["inner"], plans[lead]["inner_off"]

    balloon_raise()

    # last guard: never let a laid-out line cross the balloon / panel border (shrink until inside or at the floor)
    def guard_fit(hard: bool = False):
        """hard=True (RENDER_HARD_GUARD): every balloon block, not only members of a multi-block balloon, must keep every
        line box (plus 2 percent clearance, all four corners) inside the interior mask."""
        for i, plan in enumerate(plans):
            out, size = fits[i][0], fits[i][1]
            guard = 0
            was = plan.get("strict")
            probe, floor = plan, FLOOR_SIZE
            if (hard and plan["inner"] is None and plan.get("guard_inner") is not None and plan["bubble"] and not plan["angle"]
                    and os.environ.get("RENDER_SPIKY_GUARD", "true").lower() == "true"):
                # ragged / spiky balloon: the interior is trusted for the line check only (never shrunk below 70 percent)
                probe, floor = dict(plan, inner=plan["guard_inner"], inner_off=plan["guard_off"]), max(FLOOR_SIZE, int(0.7 * size))
            if hard and probe["inner"] is not None and plan["bubble"] and not plan["angle"]:
                plan["strict"] = probe["strict"] = True
            while (probe["inner"] is not None and size > floor and guard < (30 if hard else 14)
                   and not lines_inside(probe, out, size, plan["fit_region"], line_spacing, 0.97 if probe.get("strict") else 0.9)):
                fits[i] = fit_once(plan, plan["fit_region"], size - 1)
                out, size = fits[i][0], fits[i][1]
                guard += 1
            plan["strict"] = was

    before = [f[1] for f in fits]
    guard_fit()
    if bl.hard_guard_enabled():
        guard_fit(hard=True)
    if any(f[1] < b for f, b in zip(fits, before)):
        # a member had to shrink to stay inside the balloon: the whole balloon follows (balloon-wide size band)
        balloon_cap(lambda sizes, caps: [int(math.ceil(min(sizes) * (1.0 + bl.band()))) if not bl.is_emphasis(
            caps[k], [c for j, c in enumerate(caps) if j != k]) else None for k in range(len(sizes))])
        guard_fit()

    def tiny_retry():
        """Balloon text that ended unreadably small although the English lettering was big: the measured interior is a
        narrow sliver (hologram, textured balloon). The original text box stood on safe ground: retry on it, kept inside
        the balloon box (RENDER_TINY_PX, default 14 at 800 px width; 0 disables)."""
        for i, plan in enumerate(plans):
            target = plan.get("target") or 0.0
            if not (TINY_PX and target and plan["bubble"] and fits[i][1] < TINY_PX * W / 800.0 and fits[i][1] < 0.6 * target):
                continue
            ox1, oy1, ox2, oy2 = [float(v) for v in plan["obox"]]
            bb = getattr(plan["blk"], "bubble_xyxy", None)
            if bb is not None:
                bw_, bh_ = float(bb[2]) - float(bb[0]), float(bb[3]) - float(bb[1])
                ox1, oy1 = max(ox1, float(bb[0]) + 0.05 * bw_), max(oy1, float(bb[1]) + 0.05 * bh_)
                ox2, oy2 = min(ox2, float(bb[2]) - 0.05 * bw_), min(oy2, float(bb[3]) - 0.05 * bh_)
            region = (int(ox1), int(oy1), int(ox2), int(oy2))
            if region[2] - region[0] <= 40 or region[3] - region[1] <= 20:
                continue
            cand = fit_once(plan, region, int(math.ceil(target * TARGET_MAX_GROW)))
            if cand[1] > fits[i][1]:
                fits[i], plan["fit_region"], plan["inner"] = cand, region, None  # inner was the unreliable part
                if os.environ.get("RENDER_DEBUG"):
                    print(f"[tiny-retry] block {plan['obox']}: size -> {cand[1]}", flush=True)

    tiny_retry()

    items = []
    for plan, (out, size, dw, dh) in zip(plans, fits):
        blk = plan["blk"]
        x1, y1, x2, y2 = plan["fit_region"]
        text_rgb, bg_rgb, flat = plan["colors"]
        ana = plan["analysis"]
        fill = ana["fill"] or text_rgb
        rgb, ocol, owidth = choose_colors(fill, bg_rgb, flat and plan["bubble"], size)
        if not plan["bubble"] and not flat and ana["outline"] is None and needs_outline(cleaned, plan["obox"], rgb):
            owidth = max(owidth, float(min(4, max(1, round(size * 0.07)))))
        elif not plan["bubble"] and not flat and ana["outline"] is None and not ana["gradient"]:
            owidth = 0.0  # the lettering already contrasts with its backdrop (white text on a blue panel)
        grad, glow = None, None
        if ana["gradient"]:
            stops = ana["gradient"]["colors"]
            _, need_outline = tst.ensure_contrast(stops, bg_rgb)
            grad = {"colors": [tuple(c) for c in stops], "vertical": ana["gradient"]["vertical"]}
            rgb = stops[0]
            if need_outline is not None:
                ocol, owidth = need_outline, max(owidth, float(min(4, max(1, round(size * 0.07)))))
            elif ana["outline"] is None:
                owidth = owidth if not (flat and plan["bubble"]) else 0.0
        if ana["outline"] is not None:
            # the source had a real outline: keep fill + outline as drawn (legible through the outline),
            # unless both are close to the background luminance
            lo, lf = _lum(ana["outline"]), _lum(fill)
            lb_ = _lum(bg_rgb) if bg_rgb is not None else 128.0
            if grad is None and (abs(lo - lb_) >= 45 or abs(lf - lb_) >= 90):
                rgb = tuple(fill)
            ocol = ana["outline"]
            owidth = max(owidth, float(min(4, max(1, round(size * 0.07)))))
        if ana["glow"] is not None:
            glow = QColor(*ana["glow"])
        font_color = QColor(*rgb)
        outline_color = QColor(*ocol) if owidth > 0 else None
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        il, it, ir, ib = plan["ink"]
        free_text = not plan["bubble"] and not plan["angle"]
        if free_text:
            # anchor on the ORIGINAL lettering, not on the (possibly larger / shifted) search region
            cx, cy = (il + ir) / 2, (it + ib) / 2
        if plan["list_like"] and plan["list_center"]:
            pos_x = int(cx - dw / 2)
        elif plan["list_like"]:
            pos_x = int(max(il, x1) if il + dw <= x2 else x2 - dw)
        elif free_text and plan["style"] in {"phone", "sign"} and abs(cx - W / 2) > 0.06 * W and dw < (ir - il):
            pos_x = int(il)  # left-aligned UI text that got shorter keeps its left edge
        else:
            pos_x = int(cx - dw / 2)
        pos_y = int(cy - dh / 2)
        if free_text:  # centre the CAPITALS (not the line box) on the original lettering
            pos_y = int(cy - caps_center_offset(plan["family"], size, out.count("\n") + 1, line_spacing))
        if not plan["angle"]:
            pos_x = max(pos_x, x1 if dw <= x2 - x1 else pos_x)
            if not free_text:  # free text stays anchored on the original lettering (the box includes line-spacing padding)
                pos_y = max(pos_y, y1 if dh <= y2 - y1 else pos_y)
                if dh <= y2 - y1:
                    pos_y = min(pos_y, int(y2 - dh))
            mx = int(W * 0.015)
            if dw <= W - 2 * mx:
                pos_x = int(min(max(pos_x, mx), W - mx - dw))
        item_align = (Qt.AlignmentFlag.AlignCenter if plan["list_center"] else Qt.AlignmentFlag.AlignLeft) if plan["list_like"] else alignment
        item_ls = line_spacing
        n_out = out.count("\n") + 1
        centers = plan["src_centers"]
        if plan["list_like"] and n_out >= 2 and len(centers) == n_out:
            # list rows keep the ORIGINAL row pitch and first-row position (rows line up with the panel's grid)
            pitch = float(np.median(np.diff(centers)))
            fm_ = QFontMetricsF(_qfont(plan["family"], size))
            item_ls = min(max(pitch / max(1.0, fm_.lineSpacing()), 1.0), max(line_spacing, 1.25))
            cap_h_ = -fm_.tightBoundingRect("H").top() or fm_.capHeight()
            pos_y = int(centers[0] - (4.0 + fm_.ascent() - cap_h_ / 2))
            last_bottom = pos_y + 4 + fm_.ascent() + (n_out - 1) * fm_.lineSpacing() * item_ls + fm_.descent()
            if last_bottom > y2 + 0.04 * (y2 - y1):  # never touch the panel border below the original text
                pos_y -= int(last_bottom - y2)
        if plan["list_like"]:
            dw, dh = doc_size(out, plan["family"], size, item_ls, item_align, render_settings.direction)
        rotation = plan["angle"] if plan["angle"] else (blk.angle if plan["bubble"] else 0)
        origin = (dw / 2, dh / 2) if plan["angle"] else blk.tr_origin_point
        props = TextItemProperties(
            text=out, font_family=plan["family"], font_size=size, text_color=font_color,
            alignment=item_align, line_spacing=item_ls, outline_color=outline_color,
            outline_width=owidth, bold=False, italic=plan["italic"], underline=False,
            position=(pos_x, pos_y), rotation=rotation, scale=1.0,
            transform_origin=origin, width=dw, height=dh,
            direction=render_settings.direction, vertical=False,
            selection_outlines=[OutlineInfo(0, len(out), outline_color, owidth, OutlineType.Full_Document)]
            if owidth > 0 else [],
        )
        props.gradient = grad
        props.glow_color = glow
        d = props.to_dict()
        d["debug"] = dict(font=plan["font_rel"], style=plan["style"], target=round(plan.get("target", 0), 1), size=size, pos=(pos_x, pos_y), dwh=(round(dw), round(dh)), shaped=plan["shaped"], bub=plan["bubble"], flat=flat, region=[int(v) for v in plan["fit_region"]],
                          cap_h=round(plan["cap_h"], 1), src=plan["src_traits"], obox=plan["obox"], angle=round(plan["angle"], 1), gradient=grad,
                          outline=ana["outline"], glow=ana["glow"])
        if os.environ.get("RENDER_DEBUG"):
            print("[item]", {k: v for k, v in d["debug"].items() if k != "src"}, "text=", repr(out[:40]), flush=True)
        items.append(d)
    resolve_item_overlaps(items, plans, W, H, render_settings.direction)
    if bl.overlap_enabled():
        items.sort(key=lambda d: d.get("glow_color") is None)  # soft halos go under every other block's lettering (stable)
    return items


# ------------------------------------------------------- perspective (stage E)
def estimate_text_quad(image: np.ndarray, cleaned: np.ndarray, xyxy) -> np.ndarray | None:
    """4x2 float32 quad (tl, tr, br, bl) around the erased glyph pixels, or None."""
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    d = np.abs(image[y1:y2, x1:x2].astype(np.int16) - cleaned[y1:y2, x1:x2].astype(np.int16)).sum(axis=2) > 90
    ys, xs = np.nonzero(d)
    if len(xs) < 30:
        return None
    rect = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
    box = cv2.boxPoints(rect)
    box[:, 0] += x1
    box[:, 1] += y1
    s = box.sum(axis=1)
    diff = np.diff(box, axis=1).ravel()
    return np.array([box[s.argmin()], box[diff.argmin()], box[s.argmax()], box[diff.argmax()]], dtype=np.float32)


def warp_layer_to_quad(layer_rgba: np.ndarray, quad: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Perspective-warp a flat RGBA text layer onto `quad` (tl,tr,br,bl) in an out_hw canvas."""
    h, w = layer_rgba.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(src, np.asarray(quad, dtype=np.float32))
    return cv2.warpPerspective(layer_rgba, m, (out_hw[1], out_hw[0]), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def alpha_composite(base_rgb: np.ndarray, layer_rgba: np.ndarray) -> np.ndarray:
    a = layer_rgba[..., 3:4].astype(np.float32) / 255.0
    return (layer_rgba[..., :3] * a + base_rgb * (1 - a)).astype(np.uint8)
