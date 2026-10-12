"""Typesetting helpers for translated manga text (RENDER_V2).

Pure functions, no Qt: text measurement is injected, so everything here is
unit-testable. Covers Turkish-aware casing, syllable hyphenation, balanced
(minimum raggedness) line breaking, fit-to-box with a hard floor, and the
page-level font size band.
"""
from __future__ import annotations

import re
from typing import Callable, Optional

TURKISH_LETTERS = "ğĞşŞıİçÇöÖüÜ"
_TR_UPPER = str.maketrans({"i": "İ", "ı": "I"})
_TR_LOWER = str.maketrans({"I": "ı", "İ": "i"})
_VOWELS = set("aeıioöuüAEIİOÖUÜ")
_SENT_END = ".!?…"

# styles that are rendered in capitals; everything else is sentence case
UPPER_STYLES = {"shout", "sfx"}


def tr_upper(text: str) -> str:
    return text.translate(_TR_UPPER).upper()


def tr_lower(text: str) -> str:
    return text.translate(_TR_LOWER).lower()


def sentence_case(text: str) -> str:
    """Lower-case with Turkish i/I rules, capitalise the first letter of each sentence."""
    low = tr_lower(text)
    out = []
    cap_next = True
    for ch in low:
        if cap_next and ch.isalpha():
            ch = tr_upper(ch)
            cap_next = False
        out.append(ch)
        if ch in _SENT_END:
            cap_next = True
    return "".join(out)


def apply_casing(text: str, style: str = "", policy: str = "auto") -> str:
    """policy: 'upper' (old behaviour), 'keep', or 'auto' (sentence case, UPPER for shout/sfx)."""
    if not text:
        return text
    if policy == "upper":
        return tr_upper(text)
    if policy == "keep":
        return text
    if (style or "").lower() in UPPER_STYLES:
        return tr_upper(text)
    letters = [c for c in text if c.isalpha()]
    # only rewrite text that is entirely capitals (OCR of caps lettering);
    # genuinely mixed-case text from the translator is left alone
    if letters and all(c == tr_upper(c) for c in letters):
        return sentence_case(text)
    return text


def turkish_syllables(word: str) -> list[str]:
    """Split a word into Turkish syllables (V-V, V-CV, VC-CV rule)."""
    vpos = [i for i, c in enumerate(word) if c in _VOWELS]
    if len(vpos) < 2:
        return [word]
    cuts = []
    for a, b in zip(vpos, vpos[1:]):
        k = b - a - 1
        if k == 0:
            cuts.append(b)
        elif k == 1:
            cuts.append(a + 1)
        else:
            cuts.append(a + 2)
    parts, prev = [], 0
    for c in cuts:
        parts.append(word[prev:c])
        prev = c
    parts.append(word[prev:])
    return parts


def split_long_word(word: str, width: Callable[[str], float], max_w: float) -> list[str]:
    """Split a word wider than max_w into hyphenated pieces (syllable boundaries, else hard split)."""
    if width(word) <= max_w:
        return [word]
    syl = turkish_syllables(word)
    pieces: list[str] = []
    cur = ""
    for s in syl:
        # a single syllable that is itself too wide is hard-split by characters
        while width(s + "-") > max_w and len(s) > 1:
            k = len(s) - 1
            while k > 1 and width(s[:k] + "-") > max_w:
                k -= 1
            head, s = s[:k], s[k:]
            if cur and width(cur + head + "-") <= max_w:
                pieces.append(cur + head + "-")
            else:
                if cur:
                    pieces.append(cur + "-")
                pieces.append(head + "-")
            cur = ""
        if not cur or width(cur + s + "-") <= max_w:
            cur += s
        else:
            pieces.append(cur + "-")
            cur = s
    if cur:
        pieces.append(cur)
    # a lone trailing letter looks bad: merge it back when it still fits
    if len(pieces) >= 2 and len(pieces[-1]) == 1 and width(pieces[-2][:-1] + pieces[-1]) <= max_w:
        pieces[-2] = pieces[-2][:-1] + pieces[-1]
        pieces.pop()
    return pieces


def _balanced_dp(widths: list[float], space: float, max_w: float, n_lines: int) -> Optional[list[int]]:
    """Break indices (exclusive line ends) for exactly n_lines minimising squared slack."""
    n = len(widths)
    if n_lines > n:
        return None
    INF = float("inf")
    prefix = [0.0]
    for w in widths:
        prefix.append(prefix[-1] + w)

    def line_w(i: int, j: int) -> float:  # words i..j-1
        return prefix[j] - prefix[i] + space * (j - i - 1)

    best = [[INF] * (n + 1) for _ in range(n_lines + 1)]
    back = [[0] * (n + 1) for _ in range(n_lines + 1)]
    best[0][0] = 0.0
    for k in range(1, n_lines + 1):
        for j in range(k, n + 1):
            for i in range(j - 1, k - 2, -1):
                w = line_w(i, j)
                if w > max_w + 1e-6:
                    break  # widths only grow as i decreases
                if best[k - 1][i] == INF:
                    continue
                cost = best[k - 1][i] + (max_w - w) ** 2
                if cost < best[k][j]:
                    best[k][j] = cost
                    back[k][j] = i
    if best[n_lines][n] == INF:
        return None
    ends, j = [], n
    for k in range(n_lines, 0, -1):
        ends.append(j)
        j = back[k][j]
    return list(reversed(ends)), best[n_lines][n]  # type: ignore[return-value]


def balanced_lines(text: str, width: Callable[[str], float], max_w: float,
                   extra_lines: int = 2, hyphenate: bool = True) -> list[tuple[list[str], float]]:
    """Candidate line layouts (lines, cost), balanced, fewest lines first.

    Words wider than max_w are hyphenated at syllable boundaries. Returns []
    if nothing fits horizontally (cannot happen after hyphenation unless max_w
    is smaller than a single glyph).
    """
    words: list[str] = []
    for w in text.split():
        words.extend(split_long_word(w, width, max_w) if hyphenate else [w])
    if not words:
        return []
    widths = [width(w) for w in words]
    space = width("a a") - width("aa")
    # minimum number of lines via greedy
    greedy, cur = 1, widths[0]
    for w in widths[1:]:
        if cur + space + w <= max_w:
            cur += space + w
        else:
            greedy += 1
            cur = w
    out = []
    if len(words) > 150:  # DP not worth it for huge inputs: single greedy layout
        extra_lines = 0
    for n_lines in range(greedy, min(len(words), greedy + extra_lines) + 1):
        res = _balanced_dp(widths, space, max_w, n_lines)
        if not res:
            continue
        ends, cost = res
        lines, prev = [], 0
        for e in ends:
            lines.append(" ".join(words[prev:e]))
            prev = e
        out.append((lines, cost))
    return out


def fit_text(text: str, measure: Callable[[int], tuple], box_w: float, box_h: float, *,
             max_size: int, min_size: int, floor: int = 9, pad: float = 0.0, hyphen_pref: float = 0.8):
    """Hyphenation is a last resort: a layout without it wins unless it needs a
    font smaller than hyphen_pref * the hyphenated size."""
    if "\n" in text.strip():
        return _fit_paragraphs(text, measure, box_w, box_h, max_size, floor, pad)
    plain = _fit(text, measure, box_w, box_h, max_size, min_size, floor, pad, False)
    hyph = _fit(text, measure, box_w, box_h, max_size, min_size, floor, pad, True)
    if not plain[4] and (hyph[4] or plain[1] >= hyphen_pref * hyph[1]):
        return plain
    return hyph


def _fit_paragraphs(text, measure, box_w, box_h, max_size, floor, pad):
    """Explicit line breaks (lists, status windows) are kept: every source line is its own paragraph,
    wrapped only when it is wider than the box. Same return contract as _fit."""
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    avail_w, avail_h = max(1.0, box_w - 2 * pad), max(1.0, box_h - 2 * pad)
    best = None
    for hyphenate in (False, True):
        for size in range(int(max_size), int(floor) - 1, -1):
            width, line_h = measure(size)
            lines: list[str] = []
            for para in paras:
                cands = balanced_lines(para, width, avail_w, hyphenate=hyphenate)
                lines.extend(cands[0][0] if cands else [para])
            w = max(width(l) for l in lines)
            h = line_h * len(lines)
            if w <= avail_w + 1e-6 and h <= avail_h + 1e-6:
                return "\n".join(lines), size, w + 2 * pad, h + 2 * pad, False
            if best is None or size == floor:
                best = (lines, size, w, h)
    lines, size, w, h = best
    return "\n".join(lines), size, w + 2 * pad, h + 2 * pad, True


def _fit(text, measure, box_w, box_h, max_size, min_size, floor, pad, hyphenate):
    """Largest size whose balanced wrap fits the box; falls to `floor` instead of overflowing.

    measure(size) -> (width_fn, line_height_px). Returns (text, size, w, h, overflow).
    `overflow` is True only if even `floor` does not fit (text is still the best effort).
    """
    avail_w = max(1.0, box_w - 2 * pad)
    avail_h = max(1.0, box_h - 2 * pad)
    best_effort = None
    for size in range(int(max_size), int(floor) - 1, -1):
        width, line_h = measure(size)
        cands = balanced_lines(text, width, avail_w, hyphenate=hyphenate)
        fitting = []
        for lines, cost in cands:
            w = max(width(l) for l in lines)
            h = line_h * len(lines)
            if w <= avail_w + 1e-6 and h <= avail_h + 1e-6:
                fitting.append((cost / (avail_w ** 2 * len(lines)), len(lines), lines, w, h))
            elif best_effort is None or size == floor:
                best_effort = (lines, size, w, h)
        if fitting:
            # prefer lower normalised raggedness, then fewer lines
            fitting.sort(key=lambda t: (round(t[0], 3), t[1]))
            _, _, lines, w, h = fitting[0]
            return "\n".join(lines), size, w + 2 * pad, h + 2 * pad, False
    if best_effort is None:
        return text, floor, box_w, box_h, True
    lines, size, w, h = best_effort
    return "\n".join(lines), size, w + 2 * pad, h + 2 * pad, True


def apply_size_band(sizes: list[int], ratio: float = 1.4, floor: int = 0) -> list[int]:
    """Cap each size at ratio * median so neighbouring bubbles stay consistent."""
    if not sizes:
        return []
    ordered = sorted(sizes)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 else (
        ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    cap = max(floor, int(median * ratio))
    return [min(s, cap) for s in sizes]


_WS = re.compile(r"[^\S\n]+")


def normalize_spaces(text: str) -> str:
    """Collapse runs of blanks but keep explicit line breaks (one per source line)."""
    lines = [_WS.sub(" ", ln).strip() for ln in text.replace("\r", "").split("\n")]
    return "\n".join(ln for ln in lines if ln)
