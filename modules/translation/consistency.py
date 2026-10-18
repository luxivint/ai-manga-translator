"""Cheap post-translation consistency guard (no GPT call unless something looks wrong).

An LLM answering a keyed batch can shift translations by one block (e.g. when a sentence continues across two
blocks) while every key is still present. `check_page` flags the symptoms; `guard_page` re-translates the flagged
page once and keeps the better result."""
from __future__ import annotations

import os
import re
from typing import Callable

ORDER_HINT = (
    "\nORDER RULE: translate this page on its own. Each key is ONE independent block; return exactly one translation "
    "per key, in the same order as the keys. Never merge two blocks, never move a sentence from one key to another, "
    "never drop or duplicate a key. If a sentence continues from one block into the next, split the translation "
    "naturally between those keys."
)

_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000}
# Turkish forms (accent-folded) of the number words above, used only when the target language is Turkish
_TR_WORDS = {1: ["bir"], 2: ["iki"], 3: ["uc"], 4: ["dort"], 5: ["bes"], 6: ["alti"], 7: ["yedi"], 8: ["sekiz"],
             9: ["dokuz"], 10: ["on"], 11: ["on bir"], 12: ["on iki"], 13: ["on uc"], 14: ["on dort"],
             15: ["on bes"], 16: ["on alti"], 17: ["on yedi"], 18: ["on sekiz"], 19: ["on dokuz"],
             20: ["yirmi"], 30: ["otuz"], 40: ["kirk"], 50: ["elli"], 60: ["altmis"], 70: ["yetmis"],
             80: ["seksen"], 90: ["doksan"], 100: ["yuz"], 1000: ["bin"]}

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*", re.UNICODE)


def enabled() -> bool:
    return os.environ.get("TRANSLATE_CHECK", "true").strip().lower() not in ("false", "0", "no", "off")


def _fold(text: str) -> str:
    """Lower-case, accent-folded (Turkish ı/İ/ş/ğ/ü/ö/ç -> ascii) for tolerant substring checks."""
    text = (text or "").replace("İ", "i").replace("I", "i").replace("ı", "i")
    text = text.lower()
    return text.translate(str.maketrans("şğüöçâîû", "sguocaiu"))


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", _fold(text))


def _letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text or "")


def _source_numbers(src: str) -> list[tuple[str, set[str]]]:
    """[(label, acceptable forms)]: digit groups and English number words (compounds: twenty-one -> 21)."""
    out: list[tuple[str, set[str]]] = []
    for m in re.finditer(r"\d[\d.,]*", src):
        digits = re.sub(r"\D", "", m.group(0))
        if digits:
            out.append((m.group(0).strip(".,"), {digits}))
    for m in re.finditer(r"\b(" + "|".join(_TENS) + r")(?:[- ](" + "|".join(_UNITS) + r"))?\b", src, re.IGNORECASE):
        tens = _TENS[m.group(1).lower()]
        unit = _UNITS[m.group(2).lower()] if m.group(2) else 0
        if unit and tens % 10:
            continue  # 'twelve one' is not a number
        n = tens + unit
        out.append((m.group(0), {str(n), f"tr:{tens}:{unit}"}))
    return out


def _has_number(translation: str, forms: set[str], target_lang: str) -> bool:
    folded = _fold(translation)
    digits_only = re.sub(r"\D", "", translation)
    plain = {f for f in forms if not f.startswith("tr:")}
    if any(f in re.sub(r"[^\d]+", " ", translation).split() or f in digits_only for f in plain):
        return True
    tr = [f for f in forms if f.startswith("tr:")]
    if not tr:
        return False
    if (target_lang or "").strip().lower() not in ("turkish", "türkçe", "tr", "tur"):
        return True  # number words in other languages are not checked, only digits (never flag them)
    _, tens, unit = tr[0].split(":")
    tens, unit = int(tens), int(unit)
    tens_ok = any(w in folded for w in _TR_WORDS.get(tens, []))
    unit_ok = (not unit) or any(w in folded for w in _TR_WORDS.get(unit, []))
    return tens_ok and unit_ok


# capitalised words that are translated rather than carried over (so their absence is not a misassignment)
_TRANSLATED_NAMES = {w for w in (
    "united states america american americans korea korean japan japanese china chinese england english london "
    "france french german germany europe asia africa russia russian paris tokyo seoul new york north south east west "
    "monday tuesday wednesday thursday friday saturday sunday january february march april june july august "
    "september october november december christmas earth heaven hell god lord mister miss president sir madam "
    "doctor professor company department school university hospital station street river mountain".split())}


def _proper_nouns(src: str) -> list[str]:
    """Capitalised words in the middle of a sentence (names); empty for ALL-CAPS text where case says nothing."""
    letters = [ch for ch in src if ch.isalpha()]
    if not letters or sum(ch.isupper() for ch in letters) > 0.7 * len(letters):
        return []
    names: list[str] = []
    prev_end = True  # start of text / after . ! ? counts as sentence start
    for m in re.finditer(r"[^\W\d_]+(?:[-'’][^\W\d_]+)*|[.!?…]+", src):
        tok = m.group(0)
        if tok[0] in ".!?…":
            prev_end = True
            continue
        if (not prev_end and tok[0].isupper() and len(tok) >= 4 and tok.upper() != "I"
                and tok.lower() not in _TRANSLATED_NAMES):
            names.append(re.split(r"['’]", tok)[0])
        prev_end = False
    return names


def _name_in(name: str, translation: str) -> bool:
    return _fold(name)[:4] in _fold(translation)


def _swap_tokens(src: str) -> set[str]:
    return {_fold(w)[:5] for w in _WORD_RE.findall(src) if len(w) >= 5}


_TERMINAL = ".!?…"


def _words(text: str) -> list[str]:
    return (text or "").split()


def dedupe_boundary(prev: str, nxt: str, max_words: int = 3) -> str:
    """`nxt` without a leading run of 1-3 words that repeat the end of `prev` (a sentence continued from block i into
    block i+1 that the translator wrote out in both). Words that end a sentence in `prev` ('NO! NO!') are kept, and
    nothing is dropped when it would leave `nxt` empty. Pure; returns `nxt` unchanged when there is no repeat."""
    pw, nw = _words(prev), _words(nxt)
    for n in range(min(max_words, len(pw), len(nw) - 1), 0, -1):
        tail, head = pw[-n:], nw[:n]
        if tail[-1][-1:] in _TERMINAL:
            continue
        if all(_norm(a) and _norm(a) == _norm(b) for a, b in zip(tail, head)):
            return " ".join(nw[n:])
    return nxt


def dedupe_adjacent(texts: list[str], max_words: int = 3) -> list[str]:
    """Apply dedupe_boundary to consecutive texts (reading order); the comparison uses the already cleaned previous."""
    out = list(texts)
    for i in range(1, len(out)):
        if out[i - 1].strip() and out[i].strip():
            out[i] = dedupe_boundary(out[i - 1], out[i], max_words)
    return out


def check_page(blocks, target_lang: str = "Turkish") -> list[str]:
    """Reasons a page's translations look suspicious (empty list = fine). Pure, no I/O."""
    reasons: list[str] = []
    items = [(i, (getattr(b, "text", "") or "").strip(), (getattr(b, "translation", "") or "").strip())
             for i, b in enumerate(blocks) if not getattr(b, "is_ad", False)]
    items = [(i, s, t) for i, s, t in items if _letters(s) >= 2 or re.search(r"\d", s)]

    seen: dict[str, tuple[int, str]] = {}
    for i, src, tr in items:
        if not tr:
            if _letters(src) >= 3:
                reasons.append(f"block {i} empty translation for source {src[:30]!r}")
            continue
        key = _norm(tr)
        if len(key) >= 4 and key in seen and _norm(seen[key][1]) != _norm(src):
            reasons.append(f"blocks {seen[key][0]} and {i} got an identical translation for different sources")
        seen.setdefault(key, (i, src))

        if len(src) >= 15 and len(src.split()) >= 3 and _norm(src) == key and "://" not in src:
            reasons.append(f"block {i} translation unchanged from source")
        if len(src) >= 12:
            ratio = len(tr) / len(src)
            if ratio < 0.25 or ratio > 4.0:
                reasons.append(f"block {i} length ratio {ratio:.2f} is an outlier")

        for label, forms in _source_numbers(src):
            if not _has_number(tr, forms, target_lang):
                reasons.append(f"block {i} number {label!r} missing in translation")
        for name in _proper_nouns(src):
            if not _name_in(name, tr):
                reasons.append(f"block {i} name {name!r} missing in translation")

    # shift evidence: a word of block i appears in another block's translation but not in its own
    for i, src, tr in items:
        if not tr:
            continue
        own = _fold(tr)
        for tok in _swap_tokens(src):
            if tok in own:
                continue
            for j, src_j, tr_j in items:
                if j != i and tok in _fold(tr_j) and tok not in _fold(src_j):
                    reasons.append(f"block {i} source word {tok!r} turned up in the translation of block {j}")
                    break
            else:
                continue
            break
    return reasons


def _snapshot(blocks):
    return [(getattr(b, "translation", ""), getattr(b, "style", ""), getattr(b, "is_ad", False)) for b in blocks]


def _restore(blocks, snap):
    for b, (tr, style, ad) in zip(blocks, snap):
        b.translation, b.style, b.is_ad = tr, style, ad


def guard_page(label, blocks, retry: Callable[[], None], target_lang: str = "Turkish") -> list[str]:
    """check_page; when flagged call `retry()` once (it rewrites blk.translation in place), keep the better
    result and warn if the page is still suspicious. Returns the final reasons ([] = fine)."""
    before = check_page(blocks, target_lang)
    if not before:
        return []
    print(f"[translate-check] page {label} flagged: {'; '.join(before[:4])} -> retrying page alone", flush=True)
    snap = _snapshot(blocks)
    try:
        retry()
    except Exception as exc:
        print(f"[translate-check] page {label} retry failed: {exc}", flush=True)
        _restore(blocks, snap)
        after = before
    else:
        after = check_page(blocks, target_lang)
        if len(after) >= len(before):  # not strictly better: keep what we had
            _restore(blocks, snap)
            after = before
    if after:
        print(f"[translate-check] page {label} suspicious: {'; '.join(after[:4])}", flush=True)
    return after


# ---------------------------------------------------------------------------- dropped 'I' (OCR) -> 3rd person verb
# Comic lettering draws the pronoun 'I' as a single stroke that OCR often drops: 'WANT TO FIND THE GOLD', "DIDN'T DO
# MUCH". The translator then reads a subject-less clause and conjugates for 'he' ('bulmak istiyor', 'yapmadı').
# Questions are left alone ('WILL YOU...?' inverts), and the verb must be followed by no subject pronoun.
_I_VERBS = {"want", "wanted", "will", "would", "didnt", "dont", "cant", "couldnt", "wont", "wasnt", "need", "hope",
            "think", "knew", "felt", "was", "am", "m"}
_PRONOUNS = {"you", "he", "she", "it", "they", "we", "i", "there", "this", "that"}
_CONJ = {"and", "but", "so", "then", "or"}
_TR_KEEP = {"şimdi", "kendi", "hadi", "haydi", "gibi", "dedi", "demek"}  # never conjugate these


def first_person_enabled() -> bool:
    return os.environ.get("RENDER_FIRST_PERSON_FIX", "true").strip().lower() not in ("false", "0", "no", "off")


def _en_words(src: str) -> list[str]:
    return [re.sub(r"[^a-z]", "", w.lower().replace("’", "'")) for w in (src or "").split() if re.sub(r"[^a-z]", "", w.lower())]


def _subjectless_i(src: str) -> bool:
    """True when the English clause starts with a verb that (with the pronoun 'I' dropped) needs a first-person subject."""
    s = (src or "").strip()
    if not s or "?" in s:
        return False
    w = _en_words(s)
    while w and w[0] in _CONJ:
        w = w[1:]
    if not w or w[0] not in _I_VERBS:
        return False
    return not (len(w) > 1 and w[1] in _PRONOUNS - {"it", "this", "that"})


def _tr_upper(s: str) -> str:
    return s.replace("i", "İ").replace("ı", "I").upper()


def to_first_person(word: str) -> str | None:
    """Turkish 3rd-person singular verb form -> 1st person singular ('istiyor' -> 'istiyorum', 'geçirecek' ->
    'geçireceğim', 'yapmadı' -> 'yapmadım'); None when the word is not such a form."""
    m = re.match(r"^(.*?)([.!…,]*)$", word)
    core, tail = m.group(1), m.group(2)
    upper = core.upper() == core and any(ch.isalpha() for ch in core)
    low = core.replace("İ", "i").replace("I", "ı").lower()
    if len(low) < 4 or low in _TR_KEEP:
        return None
    if low.endswith("yor"):
        new = low + "um"
    elif low.endswith(("ecek", "acak")):
        new = low[:-1] + "ğ" + ("im" if low.endswith("ecek") else "ım")
    elif low.endswith(("madı", "medi", "mıştı", "mişti")) or (len(low) >= 5 and low[-2:] in ("dı", "di", "du", "dü", "tı", "ti", "tu", "tü")):
        new = low + "m"
    else:
        return None
    return (_tr_upper(new) if upper else new) + tail


def first_person_fixes(sources: list[str], translations: list[str]) -> dict[int, str]:
    """{block index: corrected translation} for sentences whose English subject 'I' was dropped by OCR.

    A sentence may run over several blocks (no . ! at the end of a block's source): the verb is fixed in the block
    where the sentence ends, which is where Turkish puts it."""
    out: dict[int, str] = {}
    n = len(sources)
    k = 0
    while k < n:
        start = k
        while k < n - 1 and not re.search(r"[.!?…]\s*$", (sources[k] or "").strip()):
            k += 1  # the sentence continues in the next block
        if _subjectless_i(sources[start]) and "?" not in (sources[k] or ""):
            tr = (out.get(k) or translations[k] or "").rstrip()
            words = tr.split()
            if words:
                new = to_first_person(words[-1])
                if new and new != words[-1]:
                    out[k] = " ".join(words[:-1] + [new])
        k += 1
    return out
