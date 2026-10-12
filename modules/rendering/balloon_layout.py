"""Balloon-level layout helpers (pure geometry/text rules, no Qt): which blocks share a balloon, one size band per
balloon, merging a continued sentence into one paragraph, and resolving overlapping rendered boxes.

Env knobs (all read at call time):
  RENDER_BALLOON_UNIFY (true)  one font family + one size band per balloon (false = old per-block sizing)
  RENDER_MERGE_BALLOON (true)  lay a continued sentence split over several blocks of one balloon out as ONE paragraph
  RENDER_BALLOON_BAND  (0.12)  allowed size spread around the balloon median (+-12 percent)
  RENDER_EMPHASIS_RATIO (1.4)  source cap-height ratio above which a block is deliberate emphasis and left alone
  RENDER_RESOLVE_OVERLAP (true) separate rendered text boxes that overlap
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


def group_by_balloon(bubbles: list, boxes: list) -> list[list[int]]:
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

    for i in range(n):
        for j in range(i + 1, n):
            if bubbles[i] is not None and bubbles[j] is not None and _iou(bubbles[i], bubbles[j]) >= 0.8:
                union(i, j)
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
    if len(texts) < 2 or len(texts) > 4:
        return False
    if any(s not in SPEECH_STYLES for s in styles) or any(not (t or "").strip() for t in texts):
        return False
    hs = [h for h in row_heights if h and h > 0]
    if len(hs) >= 2 and max(hs) > emphasis_ratio() * min(hs):
        return False
    row = statistics.median(hs) if hs else 0.0
    for i in range(len(texts) - 1):
        if ends_sentence(texts[i]) and len(texts[i].split()) >= 3 and row > 0 and gaps[i] > 1.6 * row:
            return False
    return True


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
