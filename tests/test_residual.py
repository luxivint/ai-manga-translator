import os
import sys
import types
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import residual  # noqa: E402


def page_with_text(h=400, w=600):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.putText(img, "NEWS HOME ECONOMY", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
    return img


def seg_for(img, box):
    seg = np.zeros(img.shape[:2], np.uint8)
    x1, y1, x2, y2 = box
    seg[y1:y2, x1:x2] = 255
    return seg


class FakeBlock:
    def __init__(self, box, translation="x merhaba"):
        self.xyxy = np.array(box)
        self.translation = translation
        self.is_ad = False


class FakeLb:
    """Minimal stand-in for scripts.local_batch: a fake segmenter that 'finds' a fixed region, a trivial inpaint."""

    def __init__(self, seg_box):
        self.seg_box = seg_box
        self.inpaint_calls = 0

    def bubble_outline_protection_mask(self, image, blocks):
        return np.zeros(image.shape[:2], np.uint8)

    def _get_segmenter(self):
        return object()

    def _tiled_segment(self, segmenter, image, conf):
        seg = seg_for(image, self.seg_box)
        return seg, seg

    def _tight_glyph_mask(self, image, seg, close_k, dilate_k, fine, text_boxes):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return (((gray < 120) & (seg > 0)).astype(np.uint8) * 255)

    def block_inpaint(self, image, mask, config):
        self.inpaint_calls += 1
        out = image.copy()
        out[mask > 0] = 245
        return out


class ResidualTests(unittest.TestCase):
    def setUp(self):
        for k in list(os.environ):
            if k.startswith("CLEAN_RESIDUAL"):
                del os.environ[k]
        self.img = page_with_text()
        self.first = np.zeros(self.img.shape[:2], np.uint8)
        self.first[60:90, 40:300] = 255  # an earlier cleaned block, ~110 px above the leftover text
        self.text_box = (20, 170, 330, 215)

    def test_select_accepts_near_textlike_component(self):
        keep, n = residual.select_components(seg_for(self.img, self.text_box), self.img, self.first,
                                             np.zeros_like(self.first))
        self.assertEqual(n, 1)
        self.assertTrue(keep[190, 100] > 0)

    def test_select_rejects_far_component(self):
        far = np.zeros_like(self.first)
        far[0:5, 0:5] = 255
        keep, n = residual.select_components(seg_for(self.img, self.text_box), self.img, far, np.zeros_like(self.first))
        self.assertEqual(n, 0)

    def test_select_rejects_untouched_block_region(self):
        ex = residual.untouched_boxes([FakeBlock((10, 160, 340, 230), translation="")], *self.img.shape[:2])
        keep, n = residual.select_components(seg_for(self.img, self.text_box), self.img, self.first, ex)
        self.assertEqual(n, 0)

    def test_select_rejects_blank_background_and_blobs(self):
        blank = np.full_like(self.img, 245)
        _, n = residual.select_components(seg_for(blank, self.text_box), blank, self.first, np.zeros_like(self.first))
        self.assertEqual(n, 0)  # segmenter hit but no ink
        solid = blank.copy()
        solid[170:215, 20:330] = 10
        _, n = residual.select_components(seg_for(solid, self.text_box), solid, self.first, np.zeros_like(self.first))
        self.assertEqual(n, 0)  # solid dark slab = art, not text

    def test_rotated_text_uses_min_area_rect(self):
        img = np.full((500, 500, 3), 245, np.uint8)
        cv2.putText(img, "CONDUCTOR", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (20, 20, 20), 3)
        m = cv2.getRotationMatrix2D((250, 250), 35, 1.0)
        rot = cv2.warpAffine(img, m, (500, 500), borderValue=(245, 245, 245))
        gray = cv2.cvtColor(rot, cv2.COLOR_RGB2GRAY)
        seg = (cv2.dilate((gray < 120).astype(np.uint8) * 255, np.ones((9, 9), np.uint8)))
        first = np.zeros((500, 500), np.uint8)
        first[200:260, 200:300] = 255
        keep, n = residual.select_components(seg, rot, first, np.zeros_like(first))
        self.assertEqual(n, 1)

    def test_residual_pass_cleans_and_reports(self):
        lb = FakeLb(self.text_box)
        blocks = [FakeBlock((40, 60, 300, 90))]
        out, rep = residual.residual_pass(lb, self.img, self.img.copy(), self.first, blocks, None)
        self.assertEqual(rep["added"], 1)
        self.assertGreaterEqual(rep["passes"], 1)
        self.assertLessEqual(rep["passes"], 2)
        self.assertGreater(out[170:215, 20:330].mean(), self.img[170:215, 20:330].mean())
        self.assertIsNotNone(rep["mask"])

    def test_residual_pass_disabled_and_error_safe(self):
        os.environ["CLEAN_RESIDUAL_PASS"] = "false"
        lb = FakeLb(self.text_box)
        out, rep = residual.residual_pass(lb, self.img, self.img.copy(), self.first, [], None)
        self.assertEqual(rep["added"], 0)
        self.assertEqual(lb.inpaint_calls, 0)
        del os.environ["CLEAN_RESIDUAL_PASS"]
        broken = types.SimpleNamespace()  # no segmenter at all -> must not raise
        out, rep = residual.residual_pass(broken, self.img, self.img.copy(), self.first, [], None)
        self.assertTrue(np.array_equal(out, self.img))
        self.assertIn("error", rep)

    def test_empty_first_mask_noop(self):
        lb = FakeLb(self.text_box)
        out, rep = residual.residual_pass(lb, self.img, self.img.copy(), np.zeros_like(self.first), [], None)
        self.assertEqual(rep["added"], 0)


if __name__ == "__main__":
    unittest.main()
