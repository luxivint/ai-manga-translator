"""Speed-ups must not change a single pixel: compare each fast path with the straightforward original."""
import unittest

import cv2
import numpy as np

from modules.inpainting import residual


class NearMaskEquivalence(unittest.TestCase):
    def _check(self, h, w, radius, seed, n_pts):
        rng = np.random.default_rng(seed)
        fm = np.zeros((h, w), np.uint8)
        for _ in range(n_pts):
            y, x = int(rng.integers(0, h)), int(rng.integers(0, w))
            fm[y:y + int(rng.integers(1, 6)), x:x + int(rng.integers(1, 9))] = 255
        near_full = cv2.dilate((fm > 0).astype(np.uint8), residual._ell(2 * radius + 1)) > 0
        oracle = residual._NearMask(fm, radius)
        seen = 0
        for _ in range(400):
            cw, ch = int(rng.integers(1, 12)), int(rng.integers(1, 12))
            x, y = int(rng.integers(0, w - cw)), int(rng.integers(0, h - ch))
            comp = rng.random((ch, cw)) > 0.4
            if not comp.any():
                continue
            seen += 1
            self.assertEqual(oracle.any_near(x, y, cw, ch, comp), bool(near_full[y:y + ch, x:x + cw][comp].any()),
                             (x, y, cw, ch))
        self.assertGreater(seen, 100)

    def test_small_radius(self):
        for seed in range(6):
            self._check(260, 200, 20, seed, 4)

    def test_default_radius(self):
        for seed in range(3):
            self._check(900, 700, 150, seed, 3)


if __name__ == "__main__":
    unittest.main()
