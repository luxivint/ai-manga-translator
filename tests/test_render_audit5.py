"""Tests for the visual-audit render fixes: page style anchor (shout damping, one speech font, italic threshold),
hyphen preference, list rows, merged-block alignment, row slivers, tilted rows, dropped-'I' person fix.
All inputs are synthetic (neutral text, drawn shapes)."""
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rendering import font_profiles as fp  # noqa: E402
from modules.rendering import render_v2  # noqa: E402
from modules.rendering import text_style as tst  # noqa: E402
from modules.rendering import typeset as ts  # noqa: E402
from modules.translation import consistency  # noqa: E402
from modules.utils import block_cleanup as bcl  # noqa: E402


class _Env:
    def __init__(self, **kw):
        self.kw, self.old = kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = os.environ.get(k)
            os.environ[k] = v

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------------------------------------ style anchor
class ShoutDampingTests(unittest.TestCase):
    def test_calm_shout_without_exclamation_becomes_speech(self):
        e = [("normal", True, "THE TABLE IS READY.", 30.0), ("shout", True, "THE CHAIR IS BLUE.", 31.0)]
        self.assertEqual(render_v2.demote_shouts(e, 800), ["normal", "normal"])

    def test_exclamation_with_small_letters_becomes_speech(self):
        e = [("normal", True, "THE TABLE IS READY.", 30.0), ("shout", True, "OPEN THE DOOR!", 31.0)]
        self.assertEqual(render_v2.demote_shouts(e, 800)[1], "normal")

    def test_exclamation_with_big_letters_stays_shout(self):
        e = [("normal", True, "THE TABLE IS READY.", 30.0), ("shout", True, "OPEN THE DOOR!", 45.0)]
        self.assertEqual(render_v2.demote_shouts(e, 800)[1], "shout")

    def test_lone_shout_uses_page_width_threshold(self):
        self.assertEqual(render_v2.demote_shouts([("shout", True, "STOP!", 30.0)], 800), ["normal"])
        self.assertEqual(render_v2.demote_shouts([("shout", True, "STOP!", 60.0)], 800), ["shout"])

    def test_free_text_and_switch_off(self):
        self.assertEqual(render_v2.demote_shouts([("shout", False, "BOOM", 20.0)], 800), ["shout"])
        with _Env(RENDER_SHOUT_DEMOTE="false"):
            self.assertEqual(render_v2.demote_shouts([("shout", True, "A CHAIR.", 20.0)], 800), ["shout"])


class _Blk:
    def __init__(self, cls="text_bubble"):
        self.text_class = cls


def _plan(style, rel, text, cls="text_bubble"):
    return dict(blk=_Blk(cls), style=style, font_rel=rel, family=rel.split("/")[-1], text=text, angle=0.0, list_like=False)


class PageFontAnchorTests(unittest.TestCase):
    def test_speech_and_thought_share_the_longest_blocks_font(self):
        plans = [_plan("normal", "A/a.otf", "SHORT"), _plan("thought", "A/b.otf", "A MUCH LONGER LINE OF TEXT"),
                 _plan("normal", "A/c.otf", "MEDIUM TEXT")]
        changed = render_v2.anchor_page_fonts(plans)
        self.assertEqual(changed, 2)
        self.assertEqual({p["font_rel"] for p in plans}, {"A/b.otf"})

    def test_sfx_sign_and_free_text_are_left_alone(self):
        plans = [_plan("normal", "A/a.otf", "SPEECH TEXT HERE"), _plan("sfx", "A/s.otf", "BANG"),
                 _plan("normal", "A/f.otf", "FREE", cls="text_free")]
        render_v2.anchor_page_fonts(plans)
        self.assertEqual([p["font_rel"] for p in plans], ["A/a.otf", "A/s.otf", "A/f.otf"])

    def test_switch_off(self):
        plans = [_plan("normal", "A/a.otf", "ONE"), _plan("normal", "A/b.otf", "TWO TWO")]
        with _Env(RENDER_PAGE_FONT_ANCHOR="false"):
            self.assertEqual(render_v2.anchor_page_fonts(plans), 0)


class ItalicAndFontPoolTests(unittest.TestCase):
    def test_upright_noise_slant_is_not_italic(self):
        for slant in (0.10, 0.12, 0.16):
            self.assertFalse(fp.wants_synthetic_italic("normal", {"slant": slant, "height": 30.0}))
        self.assertTrue(fp.wants_synthetic_italic("normal", {"slant": 0.32, "height": 30.0}))

    def test_threshold_knob(self):
        with _Env(FONT_ITALIC_MIN_SLANT="0.12"):
            self.assertTrue(fp.wants_synthetic_italic("normal", {"slant": 0.13, "height": 30.0}))

    def test_thought_uses_the_speech_pool(self):
        pool = fp.speech_pool()
        if not pool:
            self.skipTest("font folder not present (gitignored)")
        self.assertEqual(fp.ranked_candidates("thought"), pool)
        with _Env(FONT_THOUGHT_SPEECH_POOL="false"):
            self.assertNotEqual(fp.ranked_candidates("thought"), pool)

    def test_slanted_shout_never_picks_an_sfx_script(self):
        rel = fp.match_font("shout", {"stem": 0.15, "slant": 0.4, "wid": 0.6, "height": 60.0})
        self.assertNotIn("/sfx/", rel)


# ------------------------------------------------------------------------------------------------ typesetting
def _measure(size):
    return (lambda s: float(len(s)) * size * 0.6), size * 1.2


class HyphenAndListTests(unittest.TestCase):
    def test_plain_layout_wins_at_the_source_size(self):
        text = "ALPHA BRAVOCHARLIEDELTA"
        hy = ts.fit_text(text, _measure, 200, 200, max_size=40, min_size=10)
        plain = ts.fit_text(text, _measure, 200, 200, max_size=40, min_size=10, plain_ok=12)
        self.assertIn("-\n", hy[0])
        self.assertNotIn("-", plain[0])

    def test_list_rows_stay_whole_when_affordable(self):
        text = "FIRST ENTRY ONE\nSECOND ENTRY IS LONGER"
        out = ts.fit_text(text, _measure, 400, 400, max_size=30, min_size=8)
        self.assertEqual(out[0].count("\n"), 1)

    def test_list_rows_wrap_when_one_row_is_too_small(self):
        text = "FIRST\nA VERY LONG SECOND ENTRY WITH MANY WORDS IN IT"
        out = ts.fit_text(text, _measure, 200, 600, max_size=30, min_size=8)
        self.assertGreater(out[0].count("\n"), 1)


# ------------------------------------------------------------------------------------------------ merged blocks
class AlignSegmentsTests(unittest.TestCase):
    def test_proportional_cut_prefers_line_breaks(self):
        segs = tst.align_segments(["one two three", "four five six seven eight nine ten"], [13.0, 30.0])
        self.assertEqual(segs, [["one two three"], ["four five six seven eight nine ten"]])

    def test_anchor_word_pulls_its_group(self):
        lines = ["Alpha bravo yorum. Charlie delta Zulu Kilo, echo foxtrot golf hotel india"]
        segs = tst.align_segments(lines, [20.0, 10.0, 40.0], ["ALPHA BRAVO", "WRITER ZULU KILO", "ECHO FOXTROT GOLF HOTEL INDIA"])
        self.assertEqual(len(segs), 3)
        self.assertIn("Zulu", " ".join(segs[1]))
        self.assertTrue(" ".join(segs[2]).startswith("echo"))

    def test_joined_item_is_not_cut(self):
        segs = tst.align_segments(["Home News World Science / Tech"], [22.0, 8.0])
        self.assertEqual(segs[1], ["Science / Tech"])

    def test_too_few_words(self):
        self.assertIsNone(tst.align_segments(["one"], [1.0, 1.0]))


def _letters(img, x0, y0, n, size, gap):
    for k in range(n):
        x = x0 + k * (size + gap)
        cv2.rectangle(img, (x, y0), (x + size // 2, y0 + size), (0, 0, 0), -1)


class GroupShapeTests(unittest.TestCase):
    def test_letter_counts(self):
        img = np.full((200, 400, 3), 255, np.uint8)
        _letters(img, 10, 10, 12, 16, 10)
        _letters(img, 10, 100, 4, 16, 10)
        a, b = tst.group_letter_counts(img, [(0, 0, 400, 40), (0, 90, 400, 130)])
        self.assertGreater(a, 2 * b)

    def test_tall_full_width_row_is_text_not_a_rule(self):
        img = np.full((200, 400, 3), 255, np.uint8)
        _letters(img, 2, 20, 30, 14, 2)                                  # dense menu row
        cv2.line(img, (0, 80), (399, 80), (0, 0, 0), 2)                  # a real rule
        _letters(img, 40, 120, 8, 14, 12)
        groups = tst.text_groups(img, (0, 0, 400, 200))
        self.assertTrue(any(g["box"][1] <= 22 for g in groups))

    def test_blob_split_separates_side_by_side_texts(self):
        img = np.full((300, 400, 3), 255, np.uint8)
        _letters(img, 10, 20, 8, 30, 6)
        _letters(img, 10, 200, 8, 30, 6)
        g = {"box": (0, 0, 400, 300), "rows": [(0, 300, 400.0)], "aspect": 1.0}
        self.assertEqual(len(tst.split_blob_group(img, g)), 2)

    def test_deskewed_rows_of_one_tilted_row(self):
        m = np.zeros((300, 400), np.uint8)
        cv2.line(m, (20, 250), (380, 60), 1, 24)
        ang, _ = tst.text_angle(m.astype(bool))
        self.assertEqual(len(tst.deskewed_rows(m.astype(bool), ang)), 1)


class RowSliverTests(unittest.TestCase):
    def test_second_read_of_a_row_is_dropped(self):
        boxes = [[10, 0, 300, 40], [10, 48, 300, 62], [12, 80, 298, 93]]
        rows = [(0, 40), (45, 95), (45, 95)]
        self.assertEqual(bcl.row_slivers(boxes, rows), [2])

    def test_full_row_block_wins_over_a_sliver(self):
        boxes = [[10, 45, 300, 95], [10, 48, 300, 62]]
        rows = [(45, 95), (45, 95)]
        self.assertEqual(bcl.row_slivers(boxes, rows), [1])

    def test_normal_rows_untouched(self):
        boxes = [[10, 0, 300, 40], [10, 50, 300, 90]]
        self.assertEqual(bcl.row_slivers(boxes, [(0, 40), (50, 90)]), [])


# ------------------------------------------------------------------------------------------------ person fix
class FirstPersonTests(unittest.TestCase):
    def test_verb_forms(self):
        self.assertEqual(consistency.to_first_person("istiyor."), "istiyorum.")
        self.assertEqual(consistency.to_first_person("AÇACAK."), "AÇACAĞIM.")
        self.assertEqual(consistency.to_first_person("gelecek"), "geleceğim")
        self.assertEqual(consistency.to_first_person("yapmadı..."), "yapmadım...")
        self.assertIsNone(consistency.to_first_person("şimdi"))
        self.assertIsNone(consistency.to_first_person("kapı"))

    def test_dropped_i_sentence_over_two_blocks(self):
        src = ["WANT TO OPEN THE DOOR", "AND CLOSE THE WINDOW.", "THE ROOM IS WARM."]
        tr = ["Kapıyı açmak", "ve pencereyi kapatmak istiyor.", "Oda sıcak."]
        self.assertEqual(consistency.first_person_fixes(src, tr), {1: "ve pencereyi kapatmak istiyorum."})

    def test_questions_and_other_subjects_untouched(self):
        self.assertEqual(consistency.first_person_fixes(["WILL YOU OPEN IT?"], ["Açacak mısın?"]), {})
        self.assertEqual(consistency.first_person_fixes(["HE WANTS TEA."], ["Çay istiyor."]), {})
        self.assertEqual(consistency.first_person_fixes(["WILL HE COME."], ["Gelecek."]), {})


if __name__ == "__main__":
    unittest.main()
