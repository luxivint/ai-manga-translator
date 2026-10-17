"""Balloon-level layout helpers (pure geometry/text rules, no Qt): which blocks share a balloon, one size band per
balloon, merging a continued sentence into one paragraph, and resolving overlapping rendered boxes.

Env knobs (all read at call time):
  RENDER_BALLOON_UNIFY (true)  one font family + one size band per balloon (false = old per-block sizing)
  RENDER_MERGE_BALLOON (true)  lay a continued sentence split over several blocks of one balloon out as ONE paragraph
  RENDER_BALLOON_BAND  (0.12)  allowed size spread around the balloon median (+-12 percent)
  RENDER_EMPHASIS_RATIO (1.4)  source cap-height ratio above which a block is deliberate emphasis and left alone
  RENDER_RESOLVE_OVERLAP (true) separate rendered text boxes that overlap
  RENDER_MERGE_NARRATION (true) stacked narration/caption blocks of one box are laid out as ONE paragraph
  RENDER_GROUP_ROBUST    (true) tolerant balloon grouping (mutual centre containment, contained text boxes)
  RENDER_HARD_GUARD      (true) every balloon block keeps all line boxes (+2% clearance) inside the interior mask
  RENDER_ABSORB_STRAYS   (true) a tiny 1-2 word block inside a balloon joins the balloon's main text
  RENDER_DEDUPE_TRANSLATIONS (true) see modules/utils/block_cleanup.py
"""
from __future__ import annotations

import math
import os
import statistics

SPEECH_STYLES = {"normal", "thought", "whisper"}
TERMINAL = ".!?…"


def _flag(name: str, default: str = "true") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("false", "0", "no", "off")


def unify_enabled() -> bool:
    return _flag("RENDER_BALLOON_UNIFY")


def merge_enabled() -> bool:
    return _flag("RENDER_MERGE_BALLOON")


def overlap_enabled() -> bool:
    return _flag("RENDER_RESOLVE_OVERLAP")


def band() -> float:
    return float(os.environ.get("RENDER_BALLOON_BAND", "0.12"))


def narration_merge_enabled() -> bool:
    return _flag("RENDER_MERGE_NARRATION")


def hard_guard_enabled() -> bool:
    return _flag("RENDER_HARD_GUARD")


def absorb_enabled() -> bool:
    return _flag("RENDER_ABSORB_STRAYS")


def merge_shout_enabled() -> bool:
    """Blocks of one burst (shout) balloon may read as one paragraph like speech balloons do."""
    return _flag("RENDER_MERGE_SHOUT")


def absorb_rim_enabled() -> bool:
    return _flag("RENDER_ABSORB_RIM")


def absorb_tail_enabled() -> bool:
    return _flag("RENDER_ABSORB_TAIL")


def emphasis_ratio() -> float:
    return float(os.environ.get("RENDER_EMPHASIS_RATIO", "1.4"))


# ------------------------------------------------------------------ grouping
def _inside(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _contained_frac(a, b) -> float:
    """Share of box a's area that lies inside box b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1])
    return (ix * iy) / area if area > 0 else 0.0


def _touching(a, b, gap: float) -> bool:
    """Two balloon boxes stacked one above the other with at most `gap` px between them (or slightly overlapping) and
    sharing >= 50 percent of the narrower width: the lobes of one balloon outline."""
    top, low = (a, b) if a[1] <= b[1] else (b, a)
    dy = low[1] - top[3]
    if dy > gap or dy < -0.15 * min(a[3] - a[1], b[3] - b[1]):
        return False
    xo = min(a[2], b[2]) - max(a[0], b[0])
    return xo >= 0.5 * min(a[2] - a[0], b[2] - b[0])


def _nested_strip(a, b, ca, cb) -> bool:
    """One balloon box lies (>= 85 percent) inside the other, is at most 35 percent as tall, and its text centre is inside
    the bigger box: the detector reported a piece of the bigger balloon (its rim) as a balloon of its own."""
    for small, big, c in ((a, b, ca), (b, a, cb)):
        sh, bh = small[3] - small[1], big[3] - big[1]
        if bh > 0 and sh <= 0.35 * bh and _contained_frac(small, big) >= 0.85 and _inside(big, c):
            return True
    return False


def group_by_balloon(bubbles: list, boxes: list, touch_gap: float = 0.0) -> list[list[int]]:
    """Groups (>= 2 members, reading order) of blocks that sit in one balloon.

    bubbles[i] = balloon box of block i or None (free text); boxes[i] = text box. A block without a balloon box joins
    the smallest balloon (of another block) that contains its centre, because the detector often reports lettering
    inside a balloon as free text. Blocks whose balloon boxes (nearly) coincide share a balloon."""
    n = len(boxes)
    owner = list(range(n))

    def find(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    def union(i, j):
        owner[find(i)] = find(j)

    robust = _flag("RENDER_GROUP_ROBUST")
    nested = _flag("RENDER_GROUP_NESTED")

    def centre(k):
        return ((boxes[k][0] + boxes[k][2]) / 2, (boxes[k][1] + boxes[k][3]) / 2)

    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = bubbles[i], bubbles[j]
            if bi is not None and bj is not None:
                v = _iou(bi, bj)
                if v >= 0.8:
                    union(i, j)
                elif touch_gap > 0 and _touching(bi, bj, touch_gap):
                    union(i, j)  # two balloon boxes of one outline (lobes) that merely touch
                elif robust and v >= 0.3 and _inside(bi, centre(j)) and _inside(bj, centre(i)):
                    # the detector sized one balloon differently for two of its lettering blocks: each balloon box
                    # holds the other block's text, so it is the same balloon
                    union(i, j)
                elif nested and _nested_strip(bi, bj, centre(i), centre(j)):
                    union(i, j)  # a thin 'balloon' the detector cut out of another balloon's rim
            if robust and (_contained_frac(boxes[i], boxes[j]) >= 0.6 or _contained_frac(boxes[j], boxes[i]) >= 0.6):
                union(i, j)  # one lettering box lying inside another: the same text region (echo / piece of it)
    for i in range(n):
        if bubbles[i] is not None:
            continue
        c = ((boxes[i][0] + boxes[i][2]) / 2, (boxes[i][1] + boxes[i][3]) / 2)
        best, best_area = None, None
        for j in range(n):
            b = bubbles[j]
            if b is None or j == i or not _inside(b, c):
                continue
            area = (b[2] - b[0]) * (b[3] - b[1])
            if best is None or area < best_area:
                best, best_area = j, area
        if best is not None:
            union(i, best)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in groups.values():
        if len(g) >= 2:
            g.sort(key=lambda k: ((boxes[k][1] + boxes[k][3]) / 2, boxes[k][0]))
            out.append(g)
    return out


# ------------------------------------------------------------------ sizes
def is_emphasis(cap: float, others: list[float], ratio: float | None = None) -> bool:
    """True when a source cap height differs from the median of the other blocks by more than `ratio` (either way)."""
    ratio = ratio or emphasis_ratio()
    ref = [c for c in others if c and c > 0]
    if not cap or cap <= 0 or not ref:
        return False
    med = statistics.median(ref)
    r = cap / med
    return r > ratio or r < 1.0 / ratio


def compress_caps(caps: list[float], strength: float = 0.5) -> list[float]:
    """Pull source cap heights (0 = unknown) toward the balloon median: cap' = med * (cap/med) ** strength.
    Blocks that are deliberate emphasis (ratio beyond RENDER_EMPHASIS_RATIO) keep their own height."""
    known = [c for c in caps if c and c > 0]
    if len(known) < 2:
        return list(caps)
    med = statistics.median(known)
    out = []
    for i, c in enumerate(caps):
        if not c or c <= 0 or is_emphasis(c, [x for j, x in enumerate(caps) if j != i]):
            out.append(c)
        else:
            out.append(med * (c / med) ** strength)
    return out


def size_limits(sizes: list[int], caps: list[float], b: float | None = None) -> list[int | None]:
    """Largest size each member of a balloon may use: ceil(median * (1 + band)); None = emphasis, leave alone."""
    b = band() if b is None else b
    ref = [s for s in sizes if s]
    if not ref:
        return [None] * len(sizes)
    med = statistics.median(ref)
    if len(ref) == 2 and _flag("RENDER_PAIR_MIN"):
        med = min(ref)  # the median of two is their mean: a 35/22 pair would stay 32/22 (two sizes in one balloon)
    hi = int(math.ceil(med * (1.0 + b)))
    return [None if is_emphasis(caps[i], [c for j, c in enumerate(caps) if j != i]) else hi for i in range(len(sizes))]


# ------------------------------------------------------------------ merging
def ends_sentence(text: str) -> bool:
    t = (text or "").rstrip().rstrip("\"'”’)]»*")
    return bool(t) and t[-1] in TERMINAL


def mergeable(texts: list[str], styles: list[str], gaps: list[float], row_heights: list[float]) -> bool:
    """True when the blocks of one balloon read as ONE continued sentence/speech and may become one paragraph.

    Needs: every block a speech style (normal/thought/whisper); no emphasis (row heights within the emphasis ratio);
    and no hard break: a block of 3+ words that ends with . ! ? ... followed by a vertical source gap larger than 1.6 text
    rows is separate speech (second sentence after a pause / another speaker); a short fragment ('KARDEŞİM.') is a
    vocative / interjection and stays with the rest of the speech."""
    if len(texts) < 2 or len(texts) > (6 if styles and all(x == 'narration' for x in styles) else 4):
        return False
    narration = bool(styles) and all(s == "narration" for s in styles)
    if narration and not narration_merge_enabled():
        return False
    all_shout = merge_shout_enabled() and bool(styles) and all(s == "shout" for s in styles)  # one burst balloon only
    if not narration and not all_shout and any(s not in SPEECH_STYLES for s in styles):
        return False
    if any(not (t or "").strip() for t in texts):
        return False
    hs = [h for h in row_heights if h and h > 0]
    if len(hs) >= 2 and max(hs) > emphasis_ratio() * min(hs):
        return False
    row = statistics.median(hs) if hs else 0.0
    for i in range(len(texts) - 1):
        if ends_sentence(texts[i]) and len(texts[i].split()) >= 3 and row > 0 and gaps[i] > 1.6 * row:
            return False
    return True


def narration_groups(boxes: list, styles: list[str], rows: list[float] | None = None) -> list[list[int]]:
    """Groups (>= 2, reading order) of narration blocks that are lines of ONE caption box: vertically stacked, the
    horizontal extents overlap by >= 50 percent of the narrower block and the gap between them is at most 1.5 text
    rows (source row height, else the block height). Narration has no tails/speakers, so stacked lines are one text."""
    idx = [i for i, st in enumerate(styles) if st == "narration"]
    idx.sort(key=lambda k: ((boxes[k][1] + boxes[k][3]) / 2, boxes[k][0]))
    owner = {i: i for i in idx}

    def find(i):
        while owner[i] != i:
            owner[i] = owner[owner[i]]
            i = owner[i]
        return i

    for a_pos, a in enumerate(idx):
        for b in idx[a_pos + 1:]:
            ax1, ay1, ax2, ay2 = boxes[a]
            bx1, by1, bx2, by2 = boxes[b]
            ov = min(ax2, bx2) - max(ax1, bx1)
            if ov < 0.5 * min(ax2 - ax1, bx2 - bx1):
                continue
            row = max([(rows[k] if rows and rows[k] else 0.0) for k in (a, b)] + [0.0]) or min(ay2 - ay1, by2 - by1)
            gap = by1 - ay2
            if gap <= 1.5 * row and by1 >= ay1:
                owner[find(a)] = find(b)
    groups: dict[int, list[int]] = {}
    for i in idx:
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) >= 2]


def _word_count(text: str) -> int:
    return len([w for w in (text or "").split() if any(c.isalnum() for c in w)])


def stray_fragments(translations: list[str], sources: list[str], caps: list[float], inside: list[bool]) -> list[int]:
    """Positions of tiny stray blocks of one balloon that belong to the main text: a 1-2 word translation of a 1-2
    word source whose centre lies inside the balloon, in a balloon whose main (longest) block has >= 3 words and
    whose source letters are clearly smaller (< 0.8 x) than the main block's (an OCR/detector split, not a separate
    speaker: a different speaker has its own balloon)."""
    if len(translations) < 2:
        return []
    main = max(range(len(translations)), key=lambda k: _word_count(translations[k]))
    if _word_count(translations[main]) < 3:
        return []
    out = []
    for k, t in enumerate(translations):
        if k == main or not (t or "").strip() or not inside[k]:
            continue
        if _word_count(t) > 2 or _word_count(sources[k]) > 2:
            continue
        if caps[k] and caps[main] and caps[k] >= 0.8 * caps[main]:
            continue
        out.append(k)
    return out


def tail_fragments(translations: list[str], boxes: list, caps: list[float], inside: list[bool]) -> list[int]:
    """Positions of 1-2 word blocks that are the LAST WORDS of the balloon's main sentence (same letter size, so
    stray_fragments does not see them): the main text (>= 3 words) does not end its sentence, the fragment sits right
    below it (gap <= 1.5 text rows), inside the main block's column, and its centre lies inside the balloon."""
    if len(translations) < 2:
        return []
    main = max(range(len(translations)), key=lambda k: _word_count(translations[k]))
    mt = (translations[main] or "").strip()
    if _word_count(mt) < 3 or ends_sentence(mt):
        return []
    mb = boxes[main]
    out = []
    for k, t in enumerate(translations):
        if k == main or not (t or "").strip() or not inside[k] or _word_count(t) > 2:
            continue
        b = boxes[k]
        row = max(caps[k] or 0.0, caps[main] or 0.0) or (b[3] - b[1])
        gap = b[1] - mb[3]
        cx = (b[0] + b[2]) / 2
        if gap < -0.3 * row or gap > 1.5 * row or not (mb[0] - 0.1 * (mb[2] - mb[0]) <= cx <= mb[2] + 0.1 * (mb[2] - mb[0])):
            continue
        out.append(k)
    return out


# ------------------------------------------------------------------ overlap
def rotated_rect(x: float, y: float, w: float, h: float, angle: float = 0.0, origin=None):
    """Corner points (4x2 list) of a w x h item at (x, y) rotated by `angle` degrees about `origin` (item coords)."""
    ox, oy = origin if origin is not None else (w / 2.0, h / 2.0)
    a = math.radians(angle or 0.0)
    ca, sa = math.cos(a), math.sin(a)
    pts = []
    for px, py in ((0, 0), (w, 0), (w, h), (0, h)):
        dx, dy = px - ox, py - oy
        pts.append((x + ox + dx * ca - dy * sa, y + oy + dx * sa + dy * ca))
    return pts


def _poly_area(p) -> float:
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2.0


def overlap_area(p, q) -> float:
    """Intersection area of two convex polygons (rotated rectangles), Sutherland-Hodgman clipping."""
    def clip(subject, a, b):
        out = []
        def inside(pt):
            return (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0]) >= 0
        def cross(p1, p2):
            x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
            x3, y3, x4, y4 = a[0], a[1], b[0], b[1]
            den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(den) < 1e-12:
                return p2
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
        for i in range(len(subject)):
            cur, prev = subject[i], subject[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    # make q counter-clockwise relative to the clip test above
    def ccw(poly):
        s = sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly)))
        return poly if s > 0 else list(reversed(poly))

    subject, clipper = list(ccw(list(p))), ccw(list(q))
    for i in range(len(clipper)):
        if not subject:
            return 0.0
        subject = clip(subject, clipper[i], clipper[(i + 1) % len(clipper)])
    return _poly_area(subject) if len(subject) >= 3 else 0.0


def overlapping_pairs(polys: list, min_frac: float = 0.04) -> list[tuple[int, int, float]]:
    """(i, j, area) for polygon pairs whose intersection exceeds `min_frac` of the smaller polygon."""
    out = []
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            a = overlap_area(polys[i], polys[j])
            if a > 0 and a > min_frac * min(_poly_area(polys[i]), _poly_area(polys[j])):
                out.append((i, j, a))
    return out


def separate_vertically(boxes: list[list[float]], page_h: float, max_shift: list[float], rounds: int = 6) -> list[list[float]]:
    """Move axis-aligned boxes [x, y, w, h] apart along y (upper one up, lower one down, half the overlap each,
    never beyond max_shift[i] from where they started or off the page). Returns the new boxes; pairs that cannot be
    separated within the budget stay partly overlapped (the caller shrinks them)."""
    start = [b[1] for b in boxes]
    out = [list(b) for b in boxes]
    for _ in range(rounds):
        moved = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0]:
                    continue
                up, lo = (i, j) if (a[1] + a[3] / 2) <= (b[1] + b[3] / 2) else (j, i)
                ov = out[up][1] + out[up][3] - out[lo][1]
                if ov <= 0:
                    continue
                for k, d in ((up, -1), (lo, 1)):
                    room = max_shift[k] - abs(out[k][1] - start[k])
                    step = min(ov / 2.0 + 1.0, max(0.0, room))
                    ny = out[k][1] + d * step
                    ny = min(max(0.0, ny), max(0.0, page_h - out[k][3]))
                    if abs(ny - out[k][1]) > 0.01:
                        out[k][1] = ny
                        moved = True
        if not moved:
            break
    return out
