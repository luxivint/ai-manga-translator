"""Synthetic (neutral text) tests: OCR fragment echo, nested rim balloon, shout merge, tail fragment, repeat-colour unify."""
import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.rendering import balloon_layout as bl  # noqa: E402
from modules.rendering import render_v2  # noqa: E402
from modules.utils import block_cleanup as bcl  # noqa: E402


class FragmentEcho(unittest.TestCase):
    def test_cut_word_then_repeat(self):
        # window 2 starts with the cut tail of 'YOU' and then repeats the last word of window 1
        self.assertGreaterEqual(bcl.ocr_overlap_words("ARE YOU? YOU", "U? YOU LOOK A LITTLE"), 1)

    def test_unrelated_not_matched(self):
        self.assertEqual(bcl.ocr_overlap_words("WE WENT HOME", "U? THEY LEFT"), 0)

    def test_flag_off(self):
        os.environ["RENDER_OCR_FRAGMENT_SKIP"] = "false"
        try:
            self.assertEqual(bcl.ocr_overlap_words("ARE YOU? YOU", "U? YOU LOOK A LITTLE"), 0)
        finally:
            del os.environ["RENDER_OCR_FRAGMENT_SKIP"]


class NestedRim(unittest.TestCase):
    big = (60.0, 290.0, 630.0, 660.0)
    strip = (430.0, 590.0, 630.0, 640.0)

    def test_thin_strip_inside_big_balloon_is_grouped(self):
        boxes = [[160, 380, 530, 580], [430, 589, 634, 642]]
        self.assertEqual(bl.group_by_balloon([self.big, self.strip], boxes), [[0, 1]])

    def test_tall_nested_balloon_stays_separate(self):
        inner = (200.0, 350.0, 400.0, 600.0)
        boxes = [[160, 380, 530, 500], [210, 540, 390, 590]]
        self.assertEqual(bl.group_by_balloon([self.big, inner], boxes), [])


class ShoutAndTail(unittest.TestCase):
    def test_shout_blocks_merge(self):
        self.assertTrue(bl.mergeable(["AAA!", "BBB CCC! DDD EEE!"], ["shout", "shout"], [30.0], [21.0, 26.0]))

    def test_shout_merge_flag(self):
        os.environ["RENDER_MERGE_SHOUT"] = "false"
        try:
            self.assertFalse(bl.mergeable(["AAA!", "BBB CCC! DDD EEE!"], ["shout", "shout"], [30.0], [21.0, 26.0]))
        finally:
            del os.environ["RENDER_MERGE_SHOUT"]

    def test_tail_fragment_below_unfinished_text(self):
        tr = ["one two three four five,", "six."]
        boxes = [[100, 100, 400, 200], [200, 205, 300, 240]]
        self.assertEqual(bl.tail_fragments(tr, boxes, [30.0, 30.0], [True, True]), [1])

    def test_tail_not_taken_after_finished_sentence(self):
        tr = ["one two three four five.", "six."]
        boxes = [[100, 100, 400, 200], [200, 205, 300, 240]]
        self.assertEqual(bl.tail_fragments(tr, boxes, [30.0, 30.0], [True, True]), [])

    def test_tail_far_away_not_taken(self):
        tr = ["one two three four five,", "six."]
        boxes = [[100, 100, 400, 200], [200, 400, 300, 440]]
        self.assertEqual(bl.tail_fragments(tr, boxes, [30.0, 30.0], [True, True]), [])


class RepeatColours(unittest.TestCase):
    @staticmethod
    def _plan(fill, family="A"):
        return dict(text="ABC", style="normal", bubble=False, list_like=False, cap_h=40.0, family=family, font_rel="r" + family,
                    italic=False, analysis=dict(fill=fill, outline=None, gradient=None, glow=None),
                    colors=(fill, (120, 120, 130), False))

    def test_weak_copy_takes_strong_copy(self):
        strong, weak = self._plan((10, 10, 10)), self._plan((125, 125, 135), family="B")
        self.assertGreaterEqual(render_v2.unify_repeat_colors([strong, weak]), 1)
        self.assertEqual(weak["analysis"]["fill"], (10, 10, 10))
        self.assertEqual(weak["family"], "A")

    def test_different_text_untouched(self):
        a, b = self._plan((10, 10, 10)), self._plan((125, 125, 135))
        b["text"] = "XYZ"
        self.assertEqual(render_v2.unify_repeat_colors([a, b]), 0)


if __name__ == "__main__":
    unittest.main()
