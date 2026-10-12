"""Balloon-level layout: shared size band / font per balloon, interior fit, overlap resolution, duplicate words
across blocks, merging a continued sentence into one paragraph (modules/rendering/balloon_layout.py + render_v2)."""
import os
import sys
import types
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.rendering import balloon_layout as bl  # noqa: E402
from modules.translation import consistency  # noqa: E402
from modules.utils.textblock import TextBlock  # noqa: E402


class GroupingTests(unittest.TestCase):
    def test_free_text_inside_balloon_joins_it(self):
        bubbles = [None, None, (0, 0, 500, 500), (800, 0, 1000, 200)]
        boxes = [(200, 50, 300, 80), (200, 120, 300, 150), (150, 300, 350, 400), (850, 50, 950, 100)]
        self.assertEqual(bl.group_by_balloon(bubbles, boxes), [[0, 1, 2]])

    def test_same_balloon_box_groups_and_orders_by_reading_order(self):
        b = (10, 10, 400, 400)
        groups = bl.group_by_balloon([b, b, (12, 11, 402, 399)], [(50, 300, 90, 340), (50, 40, 90, 80), (50, 150, 90, 190)])
        self.assertEqual(groups, [[1, 2, 0]])

    def test_separate_balloons_are_not_grouped(self):
        self.assertEqual(bl.group_by_balloon([(0, 0, 100, 100), (300, 0, 400, 100)], [(10, 10, 90, 90), (310, 10, 390, 90)]), [])


class SizeBandTests(unittest.TestCase):
    def test_compress_pulls_caps_toward_median(self):
        out = bl.compress_caps([24.0, 30.0, 36.0])
        self.assertLess(max(out) - min(out), 36.0 - 24.0)
        self.assertAlmostEqual(out[1], 30.0, places=3)

    def test_emphasis_keeps_own_height(self):
        out = bl.compress_caps([20.0, 21.0, 60.0])
        self.assertEqual(out[2], 60.0)
        self.assertTrue(bl.is_emphasis(60.0, [20.0, 21.0]))
        self.assertFalse(bl.is_emphasis(26.0, [20.0, 21.0]))

    def test_size_limits_cap_everyone_at_median_plus_band(self):
        limits = bl.size_limits([20, 30, 60], [30.0, 30.0, 30.0], 0.12)
        self.assertEqual(limits, [34, 34, 34])  # ceil(30 * 1.12)

    def test_size_limits_leave_emphasis_alone(self):
        limits = bl.size_limits([20, 22, 80], [20.0, 21.0, 70.0], 0.12)
        self.assertIsNone(limits[2])
        self.assertEqual(limits[0], limits[1])


class DedupeTests(unittest.TestCase):
    def test_repeated_word_is_dropped_from_the_second_block(self):
        self.assertEqual(consistency.dedupe_boundary("BURMA, PAKİSTAN... UMARIM", "UMARIM ÜLKEMİZİN"), "ÜLKEMİZİN")

    def test_two_and_three_word_repeats(self):
        self.assertEqual(consistency.dedupe_boundary("gezip dolaşırken ipuçları", "dolaşırken ipuçları bulabilirim"), "bulabilirim")
        self.assertEqual(consistency.dedupe_boundary("a b c d", "b c d e f"), "e f")

    def test_turkish_casing_is_folded(self):
        self.assertEqual(consistency.dedupe_boundary("Bu ışık", "IŞIK söner"), "söner")

    def test_sentence_final_word_repeat_is_deliberate(self):
        self.assertEqual(consistency.dedupe_boundary("HAYIR!", "HAYIR bunu yapma"), "HAYIR bunu yapma")

    def test_never_empties_the_block_and_ignores_unrelated(self):
        self.assertEqual(consistency.dedupe_boundary("bir iki", "iki"), "iki")
        self.assertEqual(consistency.dedupe_boundary("bir iki", "üç dört"), "üç dört")

    def test_adjacent_chain(self):
        self.assertEqual(consistency.dedupe_adjacent(["a b", "b c", "c d"]), ["a b", "c", "d"])


class MergeRuleTests(unittest.TestCase):
    def test_continued_sentence_merges(self):
        self.assertTrue(bl.mergeable(["BURMA, PAKİSTAN... UMARIM", "ÜLKEMİZİN KALKINMASI İÇİN", "GEZİP DOLAŞIRKEN"],
                                     ["normal"] * 3, [4, 6], [30, 30, 31]))

    def test_sentence_end_with_a_large_gap_is_separate_speech(self):
        self.assertFalse(bl.mergeable(["BEN BURAYA GELDİM.", "PEKİ SEN NE DİYORSUN"], ["normal", "normal"], [80], [30, 30]))
        self.assertTrue(bl.mergeable(["BEN BURAYA GELDİM.", "PEKİ SEN NE DİYORSUN"], ["normal", "normal"], [10], [30, 30]))
        self.assertTrue(bl.mergeable(["KARDEŞİM.", "SANA BİR SİLAH VERİR"], ["normal", "normal"], [80], [30, 30]))

    def test_emphasis_shout_and_sfx_never_merge(self):
        self.assertFalse(bl.mergeable(["A", "B"], ["normal", "normal"], [3], [20, 60]))
        self.assertFalse(bl.mergeable(["A", "B"], ["normal", "shout"], [3], [30, 30]))
        self.assertFalse(bl.mergeable(["A", "B"], ["sfx", "normal"], [3], [30, 30]))
        self.assertFalse(bl.mergeable(["A", "B", "C", "D", "E"], ["normal"] * 5, [1] * 4, [30] * 5))
        self.assertFalse(bl.mergeable(["A", ""], ["normal"] * 2, [1], [30, 30]))


class OverlapGeometryTests(unittest.TestCase):
    def test_overlap_area_of_axis_aligned_and_rotated_rects(self):
        a = bl.rotated_rect(0, 0, 100, 40)
        self.assertAlmostEqual(bl.overlap_area(a, bl.rotated_rect(50, 0, 100, 40)), 50 * 40, places=3)
        self.assertEqual(bl.overlap_area(a, bl.rotated_rect(200, 0, 100, 40)), 0.0)
        r = bl.rotated_rect(0, 0, 100, 40, 90)  # same centre, turned: the cross of two bars
        self.assertAlmostEqual(bl.overlap_area(a, r), 40 * 40, places=2)

    def test_overlapping_pairs_ignore_slivers(self):
        polys = [bl.rotated_rect(0, 0, 100, 100), bl.rotated_rect(0, 99, 100, 100), bl.rotated_rect(0, 60, 100, 100)]
        self.assertEqual([(i, j) for i, j, _ in bl.overlapping_pairs(polys)], [(0, 2), (1, 2)])

    def test_separate_vertically_clears_the_overlap_within_budget(self):
        boxes = [[10, 100, 200, 60], [10, 130, 200, 60]]
        out = bl.separate_vertically(boxes, 1000, [60, 60])
        self.assertGreaterEqual(out[1][1], out[0][1] + out[0][3])

    def test_separate_vertically_respects_the_budget(self):
        out = bl.separate_vertically([[0, 100, 100, 50], [0, 110, 100, 50]], 1000, [2, 2])
        self.assertLessEqual(abs(out[0][1] - 100), 2.01)
        self.assertLess(out[0][1] + out[0][3], out[1][1] + 50)  # still overlapping: the caller has to shrink


# ------------------------------------------------------------------ render_v2 integration (Qt)
def _settings():
    from PySide6.QtCore import Qt

    return types.SimpleNamespace(line_spacing=1.0, upper_case=True, min_font_size=10, max_font_size=80,
                                 direction=Qt.LayoutDirection.LeftToRight, font_family="Arial")


def _balloon_page(rows, size=(700, 700), axes=(300, 250)):
    """(original, cleaned, ellipse mask): a white balloon on a grey page; `rows` = (text, y, scale) lettering rows."""
    page = np.full((size[1], size[0], 3), 120, np.uint8)
    c = (size[0] // 2, size[1] // 2)
    cv2.ellipse(page, c, axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(page, c, axes, 0, 0, 360, (0, 0, 0), 4)
    mask = np.zeros(page.shape[:2], np.uint8)
    cv2.ellipse(mask, c, axes, 0, 0, 360, 1, -1)
    orig = page.copy()
    for text, y, scale in rows:
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        cv2.putText(orig, text, (c[0] - w // 2, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 2, cv2.LINE_AA)
    return orig, page, mask


def _block(box, text, tr, bubble=None, free=False):
    b = TextBlock(text_bbox=np.array(box), bubble_bbox=None if free else np.array(bubble), text_class="text_free" if free else "text_bubble",
                  text=text, translation=tr)
    b.style = "normal"
    return b


def _render(items, cleaned):
    from app.ui.canvas.save_renderer import ImageSaveRenderer

    r = ImageSaveRenderer(cleaned.copy())
    r.add_state_to_image({"text_items_state": items})
    return r.render_to_image()


class RenderIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        from PySide6.QtCore import Qt

        cls.align = Qt.AlignmentFlag.AlignCenter

    def _three_block_case(self):
        rows = [("HEY BOSS", 235, 1.0), ("BROTHER", 300, 0.8), ("A SMALL POWER CAN GIVE YOU", 380, 1.0), ("A WEAPON TO FIGHT WITH", 440, 1.0)]
        orig, cleaned, mask = _balloon_page(rows)
        bub = (50, 100, 650, 600)
        blocks = [_block((250, 205, 450, 245), "HEY BOSS", "başkan", free=True),
                  _block((270, 275, 430, 310), "BROTHER", "kardeşim.", free=True),
                  _block((140, 350, 560, 455), "A SMALL POWER CAN GIVE YOU A WEAPON TO FIGHT WITH",
                         "küçük bir güç olabilir ama sana bir silah verir.", bubble=bub)]
        return orig, cleaned, mask, blocks

    def _items(self, blocks, orig, cleaned):
        from modules.rendering import render_v2

        return render_v2.build_items(blocks, orig, cleaned, [np.array(b.xyxy) for b in blocks], _settings(), self.align, "tr")

    def test_unmerged_balloon_blocks_share_one_font_and_one_size_band(self):
        orig, cleaned, mask, blocks = self._three_block_case()
        os.environ["RENDER_MERGE_BALLOON"] = "false"
        try:
            items = self._items(blocks, orig, cleaned)
        finally:
            del os.environ["RENDER_MERGE_BALLOON"]
        self.assertEqual(len(items), 3)
        self.assertEqual(len({d["font_family"] for d in items}), 1)
        sizes = sorted(d["font_size"] for d in items)
        self.assertLessEqual(sizes[-1], int(np.ceil(sizes[1] * 1.12)) + 1)
        self.assertLessEqual(sizes[-1] / sizes[0], 1.6)

    def test_old_behaviour_is_available_with_the_flag_off(self):
        orig, cleaned, mask, blocks = self._three_block_case()
        os.environ["RENDER_BALLOON_UNIFY"] = "false"
        os.environ["RENDER_MERGE_BALLOON"] = "false"
        try:
            items = self._items(blocks, orig, cleaned)
        finally:
            del os.environ["RENDER_BALLOON_UNIFY"], os.environ["RENDER_MERGE_BALLOON"]
        self.assertEqual(len(items), 3)

    def test_continued_sentence_becomes_one_paragraph_inside_the_balloon(self):
        orig, cleaned, mask, blocks = self._three_block_case()
        items = self._items(blocks, orig, cleaned)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["text"].replace("\n", " ").split(), "BAŞKAN KARDEŞİM. KÜÇÜK BİR GÜÇ OLABİLİR AMA SANA BİR SİLAH VERİR.".split())

    def test_no_text_pixel_crosses_the_balloon_interior(self):
        orig, cleaned, mask, blocks = self._three_block_case()
        for merge in ("true", "false"):
            os.environ["RENDER_MERGE_BALLOON"] = merge
            try:
                items = self._items(blocks, orig, cleaned)
            finally:
                del os.environ["RENDER_MERGE_BALLOON"]
            out = _render(items, cleaned)
            changed = np.abs(out.astype(int) - cleaned.astype(int)).sum(axis=2) > 60
            inner = cv2.erode(mask, np.ones((15, 15), np.uint8)).astype(bool)  # 7 px clearance to the outline
            self.assertTrue(changed.any())
            self.assertEqual(int((changed & ~inner).sum()), 0, f"text crosses the balloon outline (merge={merge})")

    def test_duplicate_word_across_blocks_is_removed_before_layout(self):
        orig, cleaned, mask, blocks = self._three_block_case()
        blocks[0].translation = "umarim"
        blocks[1].translation = "umarim ülkemizin"
        os.environ["RENDER_MERGE_BALLOON"] = "false"
        try:
            items = self._items(blocks, orig, cleaned)
        finally:
            del os.environ["RENDER_MERGE_BALLOON"]
        texts = [d["text"].replace("\n", " ").split() for d in items]
        flat = [w for t in texts for w in t]
        self.assertEqual(flat.count("UMARİM"), 1)

    def test_overlapping_free_blocks_are_pulled_apart(self):
        from modules.rendering import render_v2

        page = np.full((600, 700, 3), (110, 150, 230), np.uint8)
        orig = page.copy()
        cv2.putText(orig, "FIRST FOOTNOTE LINE ONE", (60, 300), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 20, 40), 2, cv2.LINE_AA)
        cv2.putText(orig, "SECOND FOOTNOTE", (60, 330), cv2.FONT_HERSHEY_DUPLEX, 0.9, (20, 20, 40), 2, cv2.LINE_AA)
        blocks = [_block((55, 270, 560, 310), "FIRST FOOTNOTE LINE ONE", "sık sık siyasi amaçlarla sürgün yerine kullanılan yurt dışı gezisi", free=True),
                  _block((55, 305, 420, 340), "SECOND FOOTNOTE", "siyasi bir sürgün aracı olarak kullanıldı.", free=True)]
        items = render_v2.build_items(blocks, orig, page, [np.array(b.xyxy) for b in blocks], _settings(), self.align, "tr")
        polys = []
        for d in items:
            x, y = d["position"]
            polys.append(bl.rotated_rect(x + 4, y + 4 + 0.06 * d["height"], d["width"] - 8, d["height"] * 0.88 - 8))
        self.assertEqual(bl.overlapping_pairs(polys), [])

    def test_resolver_can_be_switched_off(self):
        os.environ["RENDER_RESOLVE_OVERLAP"] = "false"
        try:
            self.assertFalse(bl.overlap_enabled())
        finally:
            del os.environ["RENDER_RESOLVE_OVERLAP"]
        self.assertTrue(bl.overlap_enabled())


if __name__ == "__main__":
    unittest.main()
