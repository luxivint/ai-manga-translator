import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import fill_qa, residual  # noqa: E402


class FlatSnapTest(unittest.TestCase):
    def balloon(self):
        img = np.full((300, 400, 3), 235, np.uint8)
        img[:100] = 250  # tone step: lighter band on top
        mask = np.zeros((300, 400), np.uint8)
        mask[60:160, 100:300] = 255
        out = img.copy()
        rng = np.random.default_rng(1)
        ghost = cv2.GaussianBlur(rng.normal(0, 6, (300, 400)).astype(np.float32), (0, 0), 6)
        out = np.clip(out.astype(np.float32) + ghost[..., None], 0, 255).astype(np.uint8)
        return img, out, mask

    def test_ghost_removed_and_step_kept(self):
        img, ghosted, mask = self.balloon()
        res, rep = fill_qa.flat_snap(ghosted, mask)
        self.assertGreater(rep["snapped"], 0)
        inside = mask > 0
        self.assertLessEqual(int(np.abs(res.astype(int) - img.astype(int))[inside].max()), 2)
        self.assertGreater(int(res[70, 200, 0]), int(res[150, 200, 0]) + 10)  # step survives

    def test_textured_ring_untouched(self):
        rng = np.random.default_rng(2)
        img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), np.uint8)
        mask[80:120, 80:120] = 255
        res, rep = fill_qa.flat_snap(img, mask)
        self.assertEqual(rep["snapped"], 0)
        self.assertTrue((res == img).all())


class FringeTest(unittest.TestCase):
    def test_cut_glyph_next_to_cleaned_area_is_found_but_art_is_not(self):
        img = np.full((300, 500, 3), 250, np.uint8)
        cv2.rectangle(img, (110, 100), (125, 150), (20, 20, 20), -1)  # leftover piece on flat white
        done = np.zeros((300, 500), np.uint8)
        done[90:160, 126:300] = 255
        keep, n = residual.fringe_components(img, done, np.zeros_like(done))
        self.assertEqual(n, 1)
        self.assertTrue(keep[120, 115] > 0)
        # same piece on a busy background is art: rejected
        rng = np.random.default_rng(3)
        busy = rng.integers(0, 255, (300, 500, 3), dtype=np.uint8)
        busy[100:150, 110:126] = 20
        _, n2 = residual.fringe_components(busy, done, np.zeros_like(done))
        self.assertEqual(n2, 0)

    def test_untouched_box_and_far_ink_ignored(self):
        img = np.full((300, 500, 3), 250, np.uint8)
        cv2.rectangle(img, (110, 100), (125, 150), (20, 20, 20), -1)
        cv2.rectangle(img, (440, 100), (455, 150), (20, 20, 20), -1)
        done = np.zeros((300, 500), np.uint8)
        done[90:160, 126:300] = 255
        excl = np.zeros_like(done)
        excl[90:160, 100:130] = 255
        _, n = residual.fringe_components(img, done, excl)
        self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main()


class BoxLeftoverAndRestoreTest(unittest.TestCase):
    def test_box_mode_takes_speck_next_to_cleaned_area_only(self):
        img = np.full((300, 500, 3), 245, np.uint8)
        cv2.rectangle(img, (150, 100), (160, 130), (90, 90, 90), -1)  # low-contrast speck inside a translated box
        cv2.rectangle(img, (400, 100), (410, 130), (90, 90, 90), -1)  # same speck, far from anything cleaned
        done = np.zeros((300, 500), np.uint8)
        done[90:140, 170:300] = 255
        region = np.zeros_like(done)
        region[80:150, 100:450] = 255
        region &= (cv2.dilate(done, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81)))).astype(np.uint8)
        keep, n = residual.fringe_components(img, done, np.zeros_like(done), region=region, contrast=30.0)
        self.assertEqual(n, 1)
        self.assertTrue(keep[115, 155] > 0)
        self.assertEqual(int(keep[:, 380:].sum()), 0)

    def test_translated_boxes_skip_ads_untranslated_and_huge(self):
        from types import SimpleNamespace as NS
        blocks = [NS(translation="ok", xyxy=(10, 10, 60, 40), is_ad=False),
                  NS(translation="", xyxy=(100, 10, 160, 40), is_ad=False),
                  NS(translation="ok", xyxy=(200, 10, 260, 40), is_ad=True),
                  NS(translation="ok", xyxy=(0, 0, 400, 280), is_ad=False)]
        m = residual.translated_boxes(blocks, 300, 500)
        self.assertTrue(m[20, 30] > 0)
        self.assertEqual(int(m[20, 120]), 0)
        self.assertEqual(int(m[20, 220]), 0)
        self.assertEqual(int(m[200, 300]), 0)

    def test_untranslated_block_gets_its_letters_back(self):
        from types import SimpleNamespace as NS
        lbm = NS(watermark_match=lambda b: None)
        img = np.full((100, 200, 3), 250, np.uint8)
        img[30:50, 20:60] = 10
        cleaned = img.copy()
        cleaned[30:50, 20:40] = 250  # half erased by a neighbour's mask
        mask = np.zeros((100, 200), np.uint8)
        mask[30:50, 20:40] = 255
        blocks = [NS(translation="", text="Abc", xyxy=(15, 25, 65, 55), is_ad=False)]
        out, n = residual.restore_untouched(lbm, img, cleaned, mask, blocks)
        self.assertEqual(n, 1)
        self.assertTrue((out[30:50, 20:60] == 10).all())
        blocks[0].translation = "Tamam"  # translated block: the cleaning stays
        out2, n2 = residual.restore_untouched(lbm, img, cleaned, mask, blocks)
        self.assertEqual(n2, 0)
