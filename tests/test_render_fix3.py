"""Render-side fixes found by replaying real chapters (synthetic neutral text only, no real chapter text):
gradient-aware free-text wipe, OCR echo chains, split-piece duplicates, tilt estimation, strict fit, tiny retry."""
import math
import os
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.rendering import text_style as tst  # noqa: E402
from modules.utils import block_cleanup as bc  # noqa: E402


def blk(box, text="", trans="", cls="text_free", bubble=None, style="normal"):
    return types.SimpleNamespace(xyxy=list(box), text=text, translation=trans, text_class=cls, bubble_xyxy=bubble,
                                 style=style, is_ad=False, angle=0, tr_origin_point=(0, 0), font_color=(0, 0, 0))


class EchoChainTests(unittest.TestCase):
    def test_overlap_words_counts_shared_seam(self):
        self.assertEqual(bc.ocr_overlap_words("ALPHA BETA GAMMA DELTA", "GAMMA DELTA EPSILON"), 2)
        self.assertEqual(bc.ocr_overlap_words("ALPHA BETA", "GAMMA DELTA"), 0)
        self.assertEqual(bc.ocr_overlap_words("ONE TWO", "TWO THREE"), 0)  # one short word is not evidence
        self.assertEqual(bc.ocr_overlap_words("ONE ALPHABETICAL", "ALPHABETICAL THREE"), 1)  # one long word is

    def test_chain_groups_need_overlap_and_geometry(self):
        boxes = [[100, 100, 400, 150], [105, 160, 395, 200], [100, 210, 400, 260], [100, 700, 400, 750]]
        texts = ["ONE TWO THREE FOUR", "THREE FOUR FIVE SIX", "FIVE SIX SEVEN", "SEVEN EIGHT"]
        groups = bc.overlap_chain_groups(boxes, texts)
        self.assertEqual([sorted(g) for g in groups], [[0, 1, 2]])  # block 3 is far away: not chained

    def test_chain_groups_ignore_overlap_in_other_column(self):
        boxes = [[10, 100, 200, 150], [500, 160, 700, 200]]
        self.assertEqual(bc.overlap_chain_groups(boxes, ["ONE TWO THREE", "TWO THREE FOUR"]), [])

    def test_translation_overlap_drops_only_repeated_leading_stems(self):
        out = bc.dedupe_overlap_translation("Kedi yatakta uyuyor", "Yatakta uyuyor ve sessiz", 2)
        self.assertEqual(out, "ve sessiz")
        self.assertEqual(bc.dedupe_overlap_translation("Kedi uyuyor", "ve uyuyor", 2), "ve uyuyor")  # 've' never eaten
        self.assertEqual(bc.dedupe_overlap_translation("kedi", "kedi", 3), "kedi")  # always keeps a word

    def test_split_piece_duplicates(self):
        tr = ["Haber Başlığı", "Haber Başlığı", "Başka bir satır"]
        boxes = [[0, 0, 100, 30], [0, 40, 100, 70], [0, 500, 100, 530]]
        self.assertEqual(bc.split_piece_duplicates(tr, boxes, [False, True, True]), [1])
        self.assertEqual(bc.split_piece_duplicates(tr, boxes, [False, False, True]), [])
        far = [[0, 0, 100, 30], [0, 900, 100, 930], [0, 500, 100, 530]]
        self.assertEqual(bc.split_piece_duplicates(tr, far, [False, True, True]), [])


class EchoMergeIntegration(unittest.TestCase):
    def test_four_windows_become_one_paragraph_without_repeats(self):
        from modules.rendering import render_v2

        img = np.full((400, 500, 3), 255, np.uint8)
        blocks = [
            blk([100, 100, 400, 150], "ALPHA BETA GAMMA", "Kedi yatakta uyuyor"),
            blk([105, 160, 395, 200], "BETA GAMMA DELTA EPSILON", "Yatakta uyuyor ve sessiz"),
            blk([100, 210, 400, 250], "DELTA EPSILON ZETA", "ve sessiz bir oda"),
        ]
        out, _ = render_v2.prepare_balloon_blocks(blocks, [b.xyxy for b in blocks], img)
        self.assertEqual(len(out), 1)
        words = out[0].translation.split()
        for w in ("Kedi", "yatakta", "uyuyor", "sessiz", "oda"):
            self.assertIn(w, words)
        self.assertEqual(len(words), len({w.lower() for w in words if w != "ve"}) + words.count("ve"))
        self.assertEqual([int(v) for v in out[0].xyxy], [100, 100, 400, 250])

    def test_switch_off_keeps_blocks(self):
        from modules.rendering import render_v2

        os.environ["RENDER_MERGE_OCR_ECHO"] = "false"
        try:
            img = np.full((400, 500, 3), 255, np.uint8)
            blocks = [blk([100, 100, 400, 150], "ALPHA BETA GAMMA", "Bir iki"),
                      blk([105, 160, 395, 200], "BETA GAMMA DELTA", "iki üç")]
            out, _ = render_v2.prepare_balloon_blocks(blocks, [b.xyxy for b in blocks], img)
            self.assertEqual(len(out), 2)
        finally:
            os.environ.pop("RENDER_MERGE_OCR_ECHO", None)


class TiltTests(unittest.TestCase):
    def _rows(self, angle):
        canvas = np.zeros((300, 600), np.uint8)
        for i, t in enumerate(["HELLO WORLD LINE", "SECOND LINE HERE"]):
            cv2.putText(canvas, t, (30, 110 + i * 60), cv2.FONT_HERSHEY_DUPLEX, 1.4, 255, 3, cv2.LINE_AA)
        m = cv2.getRotationMatrix2D((300, 150), -angle, 1.0)  # cv2 positive = counter-clockwise
        return cv2.warpAffine(canvas, m, (600, 300)) > 80

    def test_tilt_sign_and_size(self):
        for true in (-20.0, -9.0, 12.0):
            ang, rect = tst.estimate_tilt(self._rows(true))
            self.assertIsNotNone(rect)
            self.assertAlmostEqual(ang, true, delta=3.0)

    def test_horizontal_text_has_no_tilt(self):
        self.assertEqual(tst.estimate_tilt(self._rows(0.0))[0], 0.0)

    def test_empty_mask(self):
        self.assertEqual(tst.estimate_tilt(np.zeros((50, 50), bool)), (0.0, None))


class WipeTests(unittest.TestCase):
    def _page(self):
        h, w = 400, 700
        grad = np.linspace(250, 196, h, dtype=np.float32)[:, None, None] * np.ones((h, w, 3), np.float32)
        grad[..., 2] -= 6  # slightly warm
        page = np.clip(grad, 0, 255).astype(np.uint8)
        # ghost of an erased glyph run in the middle + a thin UI border line crossing the block bottom
        cv2.putText(page, "ghost", (260, 190), cv2.FONT_HERSHEY_DUPLEX, 1.2, (60, 60, 60), 6, cv2.LINE_AA)
        page[318:321, :] = (30, 30, 30)
        return page

    def _run(self, **env):
        import local_batch as lb

        old = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        try:
            page = self._page()
            block = blk([40, 120, 660, 330], "x", "y")
            return page, lb.clear_light_free_text_render_regions(page, [block])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_no_visible_rectangle_on_gradient(self):
        page, out = self._run()
        diff = np.abs(out.astype(np.int16) - page.astype(np.int16)).max(axis=2)
        clean_rows = np.r_[130:170, 215:300]
        self.assertLess(int(diff[clean_rows][:, 60:640].mean() * 100), 150)  # backdrop barely changes (< 1.5 level)
        inside = out[clean_rows][:, 60:640].astype(np.float32)
        step = np.abs(out[100:104, 100:600].astype(np.float32).mean() - out[126:130, 100:600].astype(np.float32).mean())
        self.assertLess(step, 6.0)
        self.assertGreater(inside.min(), 180)

    def test_ghost_glyph_is_wiped(self):
        page, out = self._run()
        ghost = page[150:200, 255:420]
        self.assertLess(int(ghost.min()), 100)
        self.assertGreater(int(out[150:200, 255:420].min()), 170)

    def test_border_line_survives(self):
        _, out = self._run()
        self.assertLess(int(out[319, 100:600].max()), 90)

    def test_legacy_mode_still_available(self):
        page, out = self._run(LIGHT_RENDER_CLEAR_LOCAL_BG="false")
        # one median fill: the gradient is flattened inside the box (what produced the visible grey rectangle)
        self.assertLess(float(out[140:150, 100:600].std(axis=(0, 1)).mean()), float(page[140:150, 100:600].std(axis=(0, 1)).mean()) + 1.0)


class FitRulesTests(unittest.TestCase):
    def test_strict_and_tiny_knobs_exist(self):
        from modules.rendering import render_v2

        self.assertGreaterEqual(render_v2.STRICT_FLOOR, 10)
        self.assertGreaterEqual(render_v2.TINY_PX, 0)

    def test_split_duplicates_drop_piece_only(self):
        from modules.rendering import render_v2

        a = blk([10, 10, 200, 40], "TITLE TEXT", "Başlık Metni")
        b = blk([10, 60, 200, 90], "", "Başlık Metni")
        b.split_origin = True
        c = blk([10, 120, 200, 400], "", "Gövde metni burada")
        c.split_origin = True
        blocks, boxes = render_v2.drop_split_duplicates([a, b, c], [x.xyxy for x in [a, b, c]])
        self.assertEqual([x.translation for x in blocks], ["Başlık Metni", "Gövde metni burada"])
        self.assertEqual(len(boxes), 2)


class BalloonLobeTests(unittest.TestCase):
    def test_touching_balloon_boxes_group_only_with_tolerance(self):
        from modules.rendering import balloon_layout as bl

        bubbles = [(30, 1263, 460, 1504), (146, 1509, 537, 1715)]
        boxes = [(90, 1347, 400, 1450), (213, 1549, 470, 1650)]
        self.assertEqual(bl.group_by_balloon(bubbles, boxes), [])
        self.assertEqual(bl.group_by_balloon(bubbles, boxes, touch_gap=12.0), [[0, 1]])
        far = [(30, 1263, 460, 1504), (146, 1600, 537, 1800)]
        self.assertEqual(bl.group_by_balloon(far, boxes, touch_gap=12.0), [])
        side = [(30, 1263, 460, 1504), (470, 1509, 700, 1715)]  # no shared column
        self.assertEqual(bl.group_by_balloon(side, boxes, touch_gap=12.0), [])

    def test_pair_limit_uses_the_smaller_size(self):
        from modules.rendering import balloon_layout as bl

        self.assertEqual(bl.size_limits([35, 22], [30.0, 30.0], 0.12), [25, 25])
        os.environ["RENDER_PAIR_MIN"] = "false"
        try:
            self.assertEqual(bl.size_limits([35, 22], [30.0, 30.0], 0.12), [32, 32])
        finally:
            os.environ.pop("RENDER_PAIR_MIN", None)


if __name__ == "__main__":
    unittest.main()
