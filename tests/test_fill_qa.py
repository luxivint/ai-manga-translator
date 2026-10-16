import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import fill_qa  # noqa: E402


def gradient_page(h=260, w=420):
    """Smooth grey gradient 'paper'."""
    x = np.linspace(200, 232, w, dtype=np.float32)[None, :].repeat(h, 0)
    y = np.linspace(0, 10, h, dtype=np.float32)[:, None].repeat(w, 1)
    g = x + y
    return np.clip(np.stack([g, g, g + 2], -1), 0, 255).astype(np.uint8)


def stripes_page(h=260, w=420):
    """Blue UI-like panel with vertical light stripes."""
    img = np.zeros((h, w, 3), np.float32)
    img[...] = (120, 190, 235)
    img[:, ::40] += 18
    return np.clip(img, 0, 255).astype(np.uint8)


def hole_mask(h=260, w=420):
    m = np.zeros((h, w), bool)
    m[100:160, 70:350] = True
    return m


def noisy_fill(img, mask, kind, seed=3):
    """What a bad LaMa fill looks like: blotchy luma ('luma'), coloured speckle ('chroma') or sparse dots ('speck')."""
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    if kind == "luma":
        n = cv2.GaussianBlur(rng.standard_normal(mask.shape).astype(np.float32), (0, 0), 3.0)
        n = n / n.std() * 9.0
        out += n[..., None]
    elif kind == "chroma":
        for c, sgn in ((0, 1), (1, -1)):
            n = cv2.GaussianBlur(rng.standard_normal(mask.shape).astype(np.float32), (0, 0), 4.0)
            out[..., c] += sgn * n / n.std() * 9.0
    else:  # speck: sparse compact dark dots
        pts = rng.integers(0, 1, 1)
        ys, xs = np.nonzero(mask)
        for i in rng.choice(len(ys), 60, replace=False):
            cv2.circle(out, (int(xs[i]), int(ys[i])), 3, (150, 150, 150), -1)
    res = img.copy()
    res[mask] = np.clip(out, 0, 255).astype(np.uint8)[mask]
    return res


class FillQaTests(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith("CLEAN_FILL_QA"):
                del os.environ[k]

    def run_case(self, img, kind):
        mask = hole_mask(*img.shape[:2])
        bad = noisy_fill(img, mask, kind)
        before = fill_qa.page_metrics(img, bad, mask.astype(np.uint8) * 255)
        out, rep = fill_qa.qa_pass(img, bad, mask.astype(np.uint8) * 255)
        after = fill_qa.page_metrics(img, out, mask.astype(np.uint8) * 255)
        return mask, bad, out, rep, before, after

    def test_noisy_luma_on_gradient_is_calmed(self):
        mask, bad, out, rep, before, after = self.run_case(gradient_page(), "luma")
        self.assertGreaterEqual(rep["bad"], 1)
        self.assertGreaterEqual(rep["fixed"], 1)
        self.assertEqual(before["bad"], rep["bad"])
        self.assertLess(after["luma_max"], before["luma_max"] * 0.5)
        self.assertEqual(after["bad"], 0)
        # fill stays close to the surrounding gradient (no tone jump)
        ref = gradient_page().astype(int)
        self.assertLess(np.abs(out.astype(int) - ref)[mask].mean(), 6.0)

    def test_chroma_speckle_is_calmed(self):
        mask, bad, out, rep, before, after = self.run_case(gradient_page(), "chroma")
        self.assertGreaterEqual(rep["fixed"], 1)
        self.assertLess(after["chroma_max"], before["chroma_max"] * 0.4)

    def test_ui_stripes_background(self):
        mask, bad, out, rep, before, after = self.run_case(stripes_page(), "luma")
        self.assertGreaterEqual(rep["fixed"], 1)
        self.assertLess(after["luma_max"], before["luma_max"] * 0.5)

    def test_sparse_dots_are_removed(self):
        img = np.full((260, 420, 3), 250, np.uint8)
        mask, bad, out, rep, before, after = self.run_case(img, "speck")
        self.assertGreaterEqual(rep["fixed"], 1)
        self.assertGreater(out[mask].mean(), bad[mask].mean())  # dark dots gone

    def test_pixels_outside_mask_untouched(self):
        img = gradient_page()
        mask, bad, out, rep, *_ = self.run_case(img, "luma")
        far = ~(cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
        self.assertTrue(np.array_equal(out[far], bad[far]))

    def test_good_fill_is_left_alone(self):
        img = gradient_page()
        mask = hole_mask()
        out, rep = fill_qa.qa_pass(img, img.copy(), mask.astype(np.uint8) * 255)
        self.assertEqual(rep["fixed"], 0)
        self.assertTrue(np.array_equal(out, img))

    def test_line_art_in_fill_survives(self):
        img = gradient_page()
        mask = hole_mask()
        bad = noisy_fill(img, mask, "luma")
        cv2.line(bad, (40, 130), (380, 135), (20, 20, 20), 3)  # a long dark line LaMa continued through the hole
        out, rep = fill_qa.qa_pass(img, bad, mask.astype(np.uint8) * 255)
        line_before = bad[130:136, 120:300].astype(int).mean()
        line_after = out[130:136, 120:300].astype(int).mean()
        self.assertLess(line_after, 120)  # still dark: not smeared away
        self.assertLess(abs(line_after - line_before), 60)

    def test_env_knob_disables(self):
        img = gradient_page()
        mask = hole_mask()
        bad = noisy_fill(img, mask, "luma")
        os.environ["CLEAN_FILL_QA"] = "false"
        try:
            out, rep = fill_qa.qa_pass(img, bad, mask.astype(np.uint8) * 255)
        finally:
            del os.environ["CLEAN_FILL_QA"]
        self.assertTrue(np.array_equal(out, bad))
        self.assertEqual(rep["fixed"], 0)

    def test_degenerate_inputs_do_not_raise(self):
        img = gradient_page()
        out, rep = fill_qa.qa_pass(img, img.copy(), np.zeros(img.shape[:2], np.uint8))
        self.assertTrue(np.array_equal(out, img))
        out, rep = fill_qa.qa_pass(img, img[:10], np.ones(img.shape[:2], np.uint8))  # shape mismatch
        self.assertEqual(rep["fixed"], 0)
        tiny = np.zeros((20, 20), np.uint8)
        tiny[5:8, 5:8] = 255
        fill_qa.qa_pass(img[:20, :20], img[:20, :20].copy(), tiny)


if __name__ == "__main__":
    unittest.main()
