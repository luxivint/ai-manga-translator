"""Pure block post-processing (no Qt, no network): containment/duplicate pruning of OCR blocks before translation and
duplicate-translation clean-up before rendering.

Why: the detector sometimes reports one text twice at different scales (a block around the whole paragraph plus a block
around one line of it); each is OCR'd and translated separately and the page shows the same phrase two or three times.

Env knobs (read at call time):
  BLOCK_CONTAINMENT_RATIO (0.6)   share of a block's area that must lie inside another block to count as 'contained'
  BLOCK_TEXT_COVER (0.8)          share of the contained block's words that must also be in the container's text
  RENDER_DEDUPE_TRANSLATIONS (true) drop a translation that repeats another block's translation / collapse repeats
  RENDER_SAFETY_NET (true)        after layout, a block with OCR text + translation that reached the page nowhere is
                                  logged loudly ('[render-miss]') and rendered anyway
"""
from __future__ import annotations

import difflib
import os
import re

_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("false", "0", "no", "off")


def containment_ratio() -> float:
    return float(os.environ.get("BLOCK_CONTAINMENT_RATIO", "0.6"))


def text_cover() -> float:
    return float(os.environ.get("BLOCK_TEXT_COVER", "0.8"))


def dedupe_enabled() -> bool:
    return _flag("RENDER_DEDUPE_TRANSLATIONS")


def fold(text: str) -> str:
    text = (text or "").replace("İ", "i").replace("I", "i").replace("ı", "i").lower()
    return text.translate(str.maketrans("şğüöçâîû", "sguocaiu"))


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


def covered_fraction(small: str, big: str) -> float:
    """Share (0..1) of the words of `small` that also occur in `big` (exact, or a near-identical OCR variant)."""
    sw, bw = words(small), words(big)
    if not sw:
        return 0.0
    if not bw:
        return 0.0
    bset = set(bw)
    hit = 0
    for w in sw:
        if w in bset or any(len(w) >= 4 and difflib.SequenceMatcher(None, w, b).ratio() >= 0.84 for b in bw):
            hit += 1
    return hit / len(sw)


def _area(b) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def inside_fraction(inner, outer) -> float:
    """Share of `inner`'s area that lies inside `outer`."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    a = _area(inner)
    return (ix * iy) / a if a > 0 else 0.0


def prune_contained_blocks(blocks) -> tuple[list, list[tuple[int, int, float]]]:
    """Drop a block whose box lies (>= BLOCK_CONTAINMENT_RATIO) inside another block's box AND whose words are
    (>= BLOCK_TEXT_COVER) already present in that block's text. Returns (kept, dropped[(index, container index, cover)]).

    Nothing is lost: the surviving block already contains every word of a dropped one. A contained block whose text is
    NOT covered is kept (a different caption inside a big balloon is real text)."""
    ratio, cover = containment_ratio(), text_cover()
    n = len(blocks)
    texts = [(getattr(b, "text", "") or "").strip() for b in blocks]
    boxes = [[float(v) for v in b.xyxy] for b in blocks]
    dropped: dict[int, tuple[int, float]] = {}
    # smaller / shorter blocks are examined first so a chain A > B > C collapses onto its largest member
    for j in sorted(range(n), key=lambda k: (len(words(texts[k])), _area(boxes[k]))):
        if not texts[j]:
            continue
        for i in range(n):
            if i == j or i in dropped:
                continue
            if inside_fraction(boxes[j], boxes[i]) < ratio:
                continue
            if len(words(texts[i])) < len(words(texts[j])):
                continue  # the container must carry at least as much text
            if len(words(texts[i])) == len(words(texts[j])) and _area(boxes[i]) < _area(boxes[j]):
                continue
            c = covered_fraction(texts[j], texts[i])
            if c >= cover:
                dropped[j] = (i, c)
                break
    kept = [b for k, b in enumerate(blocks) if k not in dropped]
    return kept, [(j, i, c) for j, (i, c) in sorted(dropped.items())]


# ---------------------------------------------------------------------------- translations
def collapse_repeats(text: str) -> str:
    """'KARŞILIK OLARAK KARŞILIK OLARAK' -> 'KARŞILIK OLARAK': a run of >= 2 words repeated back to back (case/accent
    insensitive) keeps one copy. Single repeated words ('çok çok') are left alone (emphasis)."""
    toks = text.split()
    f = [fold(re.sub(r"[^\w]+", "", t)) for t in toks]
    out_i: list[int] = []
    i = 0
    while i < len(toks):
        done = False
        for n in range(min(8, (len(toks) - i) // 2), 1, -1):
            if f[i:i + n] == f[i + n:i + 2 * n] and all(f[i:i + n]):
                out_i.extend(range(i, i + n))
                i += n
                while f[i:i + n] == f[i - n:i]:  # 3rd, 4th ... copy
                    i += n
                done = True
                break
        if not done:
            out_i.append(i)
            i += 1
    return " ".join(toks[k] for k in out_i) if out_i != list(range(len(toks))) else text


def is_duplicate_translation(small: str, big: str, ratio: float = 0.9) -> bool:
    """True when `small` repeats text that `big` already says: its normalised text is a substring of big's, or the two
    are near-identical."""
    a, b = "".join(words(small)), "".join(words(big))
    if not a or not b:
        return False
    if a in b:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= ratio


def drop_duplicate_translations(blocks, boxes, groups) -> tuple[list[int], dict[int, str]]:
    """Render-time safety net. `groups` = index groups of blocks that share a balloon or overlap on the page.
    Returns (indices whose translation must be dropped, {index: collapsed translation}).
    A block is dropped when its translation is a substring of / near-duplicate of a LONGER translation of another block
    in the same group (the longer one is kept). Words are never lost: the kept block says them already."""
    drop: set[int] = set()
    rewrite: dict[int, str] = {}
    for g in groups:
        tr = {i: (rewrite.get(i) or getattr(blocks[i], "translation", "") or "").strip() for i in g}
        for i in g:
            c = collapse_repeats(tr[i])
            if c != tr[i]:
                rewrite[i] = c
                tr[i] = c
        order = sorted(g, key=lambda k: (-len(words(tr[k])), -_area(boxes[k])))
        for pos, j in enumerate(order):
            if j in drop or not tr[j]:
                continue
            for i in order[:pos]:
                if i in drop or not tr[i]:
                    continue
                if is_duplicate_translation(tr[j], tr[i]):
                    drop.add(j)
                    break
    return sorted(drop), {i: t for i, t in rewrite.items() if i not in drop}


def overlap_groups(boxes, min_frac: float = 0.3) -> list[list[int]]:
    """Connected groups of boxes where one lies (>= min_frac of its area) inside / over another."""
    n = len(boxes)
    owner = list(range(n))

    def find(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if max(inside_fraction(boxes[i], boxes[j]), inside_fraction(boxes[j], boxes[i])) >= min_frac:
                owner[find(i)] = find(j)
    gs: dict[int, list[int]] = {}
    for i in range(n):
        gs.setdefault(find(i), []).append(i)
    return [g for g in gs.values() if len(g) >= 2]


# ---------------------------------------------------------------------------- OCR echo chains
def echo_merge_enabled() -> bool:
    return _flag("RENDER_MERGE_OCR_ECHO")


def _word_match(a: str, b: str) -> bool:
    return a == b or (len(a) >= 4 and len(b) >= 4 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.84)


def ocr_overlap_words(prev_text: str, next_text: str, max_words: int = 6) -> int:
    """How many words end `prev_text` AND start `next_text` (an OCR window that re-read the previous window's tail).
    A single shared word only counts when it is long (>= 5 letters)."""
    pw, nw = words(prev_text), words(next_text)
    for n in range(min(max_words, len(pw), len(nw)), 0, -1):
        if all(_word_match(a, b) for a, b in zip(pw[-n:], nw[:n])):
            if n >= 2 or len(nw[0]) >= 5:
                return n
    # the window cut a word of the previous text in half ('ARE YOU? YOU' -> 'U? YOU LOOK'): a leading 1-2 letter
    # fragment that is the tail of a previous word is skipped, the words after it must still repeat the previous tail
    if _flag("RENDER_OCR_FRAGMENT_SKIP") and len(nw) >= 2 and len(pw) >= 2 and len(nw[0]) <= 2             and any(len(w) > len(nw[0]) and w.endswith(nw[0]) for w in pw[-3:]):
        rest = nw[1:]
        for n in range(min(max_words, len(pw), len(rest)), 0, -1):
            if all(_word_match(a, b) for a, b in zip(pw[-n:], rest[:n])):
                return n + 1
    return 0


def overlap_chain_groups(boxes, texts, min_x_overlap: float = 0.5, gap_ratio: float = 0.8) -> list[list[int]]:
    """Index groups (top to bottom) of stacked blocks whose OCR texts overlap at the seam: the detector cut one
    paragraph into windows and each window's OCR repeated the neighbour's boundary words. Geometry must also fit:
    the boxes share a column and the vertical gap is small compared with their height."""
    n = len(boxes)
    order = sorted(range(n), key=lambda k: (boxes[k][1], boxes[k][0]))
    owner = list(range(n))

    def find(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    for a_pos, i in enumerate(order):
        for j in order[a_pos + 1:]:
            if boxes[j][1] > boxes[i][3] + 4 * max(boxes[i][3] - boxes[i][1], 1.0):
                break
            wi, wj = boxes[i][2] - boxes[i][0], boxes[j][2] - boxes[j][0]
            xo = max(0.0, min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])) / max(1.0, min(wi, wj))
            hmin = min(boxes[i][3] - boxes[i][1], boxes[j][3] - boxes[j][1])
            gap = boxes[j][1] - boxes[i][3]
            if xo < min_x_overlap or gap > gap_ratio * hmin + 8:
                continue
            if ocr_overlap_words(texts[i], texts[j]) > 0:
                owner[find(i)] = find(j)
    gs: dict[int, list[int]] = {}
    for k in order:
        gs.setdefault(find(k), []).append(k)
    return [g for g in gs.values() if len(g) >= 2]


def dedupe_overlap_translation(prev: str, nxt: str, max_drop: int) -> str:
    """`nxt` without up to `max_drop` leading words that `prev` already says (same stem: first 5 letters, words of >= 4
    letters only so connectives like 've' are never eaten). At least one word is always kept."""
    pw, nw = prev.split(), nxt.split()
    pstems = {fold(re.sub(r"[^\w]+", "", w))[:5] for w in pw if len(fold(re.sub(r"[^\w]+", "", w))) >= 4}
    drop = 0
    while drop < min(max_drop, len(nw) - 1):
        f = fold(re.sub(r"[^\w]+", "", nw[drop]))
        if len(f) >= 4 and f[:5] in pstems:
            drop += 1
        else:
            break
    return " ".join(nw[drop:])


# ---------------------------------------------------------------------------- piece of a split block that repeats a neighbour
def strip_line_enabled() -> bool:
    return _flag("RENDER_STRIP_NEIGHBOR_LINES")


def split_piece_duplicates(translations, boxes, is_piece, max_dy: float = 320.0) -> list[int]:
    """Indices of split pieces (is_piece[i]) whose whole translation repeats another block's translation (a title that
    was detected twice: alone, and again as the first line of a merged UI block). The other block still draws these
    words, so dropping the piece loses nothing. Only blocks within `max_dy` px vertically count as 'the same title'."""
    drop: list[int] = []
    norm = ["".join(words(t)) for t in translations]
    for i, piece in enumerate(is_piece):
        if not piece or len(norm[i]) < 6:
            continue
        for j in range(len(translations)):
            if j == i or j in drop or len(norm[j]) < 6 or abs(boxes[i][1] - boxes[j][1]) > max_dy:
                continue
            if norm[i] == norm[j] or (len(norm[j]) >= len(norm[i]) and norm[i] in norm[j]) or                     difflib.SequenceMatcher(None, norm[i], norm[j]).ratio() >= 0.92:
                drop.append(i)
                break
    return drop


# ---------------------------------------------------------------------------- OCR slivers of one text row
def sliver_guard_enabled() -> bool:
    return _flag("RENDER_SLIVER_GUARD")


def row_slivers(boxes, rows, ratio: float = 0.55) -> list[int]:
    """Indices of 'sliver' blocks to drop: a block whose box is much thinner than the ink row it sits in
    (height < `ratio` x row height) is a partial OCR read of that row ('USING SEMICONDUCTOR)' read from the top half of
    'YUSEONG SEMICONDUCTOR)'). When another block (sliver or full) already reads the same row (same row, horizontal
    overlap >= 50 percent), the sliver is a duplicate with no own source line and is dropped; the first block of a row
    is always kept, so no row loses its text.

    boxes = [[x1, y1, x2, y2]]; rows = [(row_y1, row_y2) | None] per block (the ink row around the box centre)."""
    n = len(boxes)

    def sliver(i):
        r = rows[i]
        return r is not None and (boxes[i][3] - boxes[i][1]) < ratio * max(1.0, r[1] - r[0])

    def same_row(i, j):
        ri, rj = rows[i], rows[j]
        if ri is None or rj is None:
            return False
        oy = min(ri[1], rj[1]) - max(ri[0], rj[0])
        if oy < 0.6 * min(ri[1] - ri[0], rj[1] - rj[0]):
            return False
        ox = min(boxes[i][2], boxes[j][2]) - max(boxes[i][0], boxes[j][0])
        return ox >= 0.5 * min(boxes[i][2] - boxes[i][0], boxes[j][2] - boxes[j][0])

    drop: list[int] = []
    for i in range(n):
        if not sliver(i):
            continue
        # a full-height block of the same row, or an earlier sliver of it, already reads this row
        if any(j != i and j not in drop and same_row(i, j) and (not sliver(j) or j < i) for j in range(n)):
            drop.append(i)
    return drop


# ---------------------------------------------------------------------------- render safety net
def safety_net_enabled() -> bool:
    return _flag("RENDER_SAFETY_NET")


def _rect_overlap(a, b) -> float:
    """Share of rect `a`'s area that lies inside rect `b`."""
    return inside_fraction([float(v) for v in a], [float(v) for v in b])


def coverage_misses(sources, items, deliberate=(), cover: float = 0.8) -> list[int]:
    """Indices of source blocks whose translation reached the page nowhere (the render dropped it silently).

    sources     [{'text','translation','box'}] as the render received them (box = original text box x1,y1,x2,y2)
    items       [{'text','box'}] what was rendered (box = the region the item stands on: its original box or position)
    deliberate  boxes the render skipped on purpose (original lettering kept by the cleaner / logo kept as-is)

    A source counts as covered when its centre lies in a deliberate box, when a rendered item stands on >= 30 percent
    of its box (or the other way round), or when a nearby item already says >= `cover` of its words (a duplicate that
    was folded into a neighbour). Only sources with OCR text AND a real translation can be 'missed'."""
    missed: list[int] = []
    for k, s in enumerate(sources):
        tr = (s.get("translation") or "").strip()
        if not (s.get("text") or "").strip() or len(tr) < 2 or not words(tr):
            continue
        box = [float(v) for v in s["box"]]
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        if any(d[0] <= cx <= d[2] and d[1] <= cy <= d[3] for d in deliberate):
            continue
        h = max(1.0, box[3] - box[1])
        done = False
        for it in items:
            ib = [float(v) for v in it["box"]]
            if _rect_overlap(box, ib) >= 0.3 or _rect_overlap(ib, box) >= 0.3:
                done = True
                break
            near = ib[0] < box[2] + 60 and box[0] < ib[2] + 60 and ib[1] < box[3] + 1.2 * h + 120 and box[1] < ib[3] + 1.2 * h + 120
            if near and covered_fraction(tr, it.get("text") or "") >= cover:
                done = True
                break
        if not done:
            missed.append(k)
    return missed
