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
