import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.inpainting import patch_match  # noqa: E402
from patch_metrics import measure  # noqa: E402


def paper(h=200, w=400, seed=1):
    rng = np.random.default_rng(seed)
    base = np.array([251.5, 254.3, 252.5], np.float32)
    noise = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 1.0) * 2.2
    return np.clip(base + noise[..., None] * np.array([0.8, 0.6, 1.0], np.float32), 0, 255).astype(np.uint8)


def lama_like(img, mask, shift=-4.0):
    out = img.copy().astype(np.float32)
    flat = np.median(img[~mask], axis=0) + shift
    out[mask] = flat
    return np.clip(out, 0, 255).astype(np.uint8)


class PatchMatchTests(unittest.TestCase):
    def setUp(self):
        os.environ.pop("CLEAN_PATCH_MATCH", None)
        self.img = paper()
        self.mask = np.zeros(self.img.shape[:2], bool)
        self.mask[70:130, 100:300] = True

    def test_tone_and_step_improve(self):
        out = lama_like(self.img, self.mask)
        fixed = patch_match.refine_fill(self.img, out, self.mask)
        m8 = self.mask.astype(np.uint8) * 255
        before, after = measure(self.img, out, m8, min_px=50)[0], measure(self.img, fixed, m8, min_px=50)[0]
        self.assertGreater(before["dE"], 1.0)
        self.assertLess(after["dE"], 0.5)
        self.assertLess(after["step"], before["step"])

    def test_grain_resynthesised(self):
        out = lama_like(self.img, self.mask)
        fixed = patch_match.refine_fill(self.img, out, self.mask)
        inner = np.zeros_like(self.mask)
        inner[85:115, 120:280] = True
        self.assertLess(out[inner].reshape(-1, 3).std(0).max(), 0.2)
        ring_std = self.img[~self.mask].reshape(-1, 3).std(0)
        self.assertTrue((fixed[inner].reshape(-1, 3).std(0) > 0.5 * ring_std).all())

    def test_pixels_outside_mask_barely_change(self):
        out = lama_like(self.img, self.mask)
        fixed = patch_match.refine_fill(self.img, out, self.mask)
        far = ~cv2.dilate(self.mask.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
        self.assertTrue(np.array_equal(fixed[far], out[far]))

    def test_line_art_in_fill_is_kept(self):
        out = lama_like(self.img, self.mask)
        cv2.line(out, (100, 100), (299, 100), (10, 10, 10), 2)  # LaMa continued an outline through the region
        fixed = patch_match.refine_fill(self.img, out, self.mask)
        self.assertTrue(np.array_equal(fixed, out))  # guarded: structure inside the fill -> untouched

    def test_textured_surroundings_untouched(self):
        rng = np.random.default_rng(3)
        tex = rng.integers(0, 255, self.img.shape, dtype=np.uint8)
        out = lama_like(tex, self.mask)
        self.assertTrue(np.array_equal(patch_match.refine_fill(tex, out, self.mask), out))

    def test_ring_with_art_line_ignores_line(self):
        img = self.img.copy()
        cv2.line(img, (0, 40), (399, 40), (0, 0, 0), 3)  # bubble outline near the region
        out = lama_like(img, self.mask)
        fixed = patch_match.refine_fill(img, out, self.mask)
        self.assertLess(abs(float(fixed[self.mask].mean()) - float(img[~self.mask & (np.arange(200)[:, None] > 60)].mean())), 2.0)

    def test_gradient_sky_follows_plane(self):
        h, w = 200, 400
        grad = np.linspace(150, 190, h, dtype=np.float32)[:, None, None] * np.array([0.6, 0.85, 1.0], np.float32)
        img = np.repeat(np.repeat(grad, w, 1), 1, 2).astype(np.uint8)
        out = lama_like(img, self.mask, shift=6.0)
        fixed = patch_match.refine_fill(img, out, self.mask)
        err_before = np.abs(out[self.mask].astype(float) - img[self.mask]).mean()
        err_after = np.abs(fixed[self.mask].astype(float) - img[self.mask]).mean()
        self.assertLess(err_after, err_before * 0.35)

    def test_switch_off_and_bad_input(self):
        out = lama_like(self.img, self.mask)
        os.environ["CLEAN_PATCH_MATCH"] = "false"
        try:
            self.assertTrue(np.array_equal(patch_match.refine_fill(self.img, out, self.mask), out))
        finally:
            os.environ.pop("CLEAN_PATCH_MATCH", None)
        self.assertTrue(np.array_equal(patch_match.refine_fill(self.img[..., 0], out[..., 0], self.mask), out[..., 0]))
        empty = np.zeros_like(self.mask)
        self.assertTrue(np.array_equal(patch_match.refine_fill(self.img, out, empty), out))


if __name__ == "__main__":
    unittest.main()
