"""Pure block post-processing (no Qt, no network): containment/duplicate pruning of OCR blocks before translation and
duplicate-translation clean-up before rendering.

Why: the detector sometimes reports one text twice at different scales (a block around the whole paragraph plus a block
around one line of it); each is OCR'd and translated separately and the page shows the same phrase two or three times.

Env knobs (read at call time):
  BLOCK_CONTAINMENT_RATIO (0.6)   share of a block's area that must lie inside another block to count as 'contained'
  BLOCK_TEXT_COVER (0.8)          share of the contained block's words that must also be in the container's text
  RENDER_DEDUPE_TRANSLATIONS (true) drop a translation that repeats another block's translation / collapse repeats
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
