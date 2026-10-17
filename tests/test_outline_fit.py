import os
import unittest

import numpy as np

from modules.rendering import render_v2 as r
from modules.rendering.outline_metric import balloon_outline_overlap


class Blk:
    text_class = "text_bubble"

    def __init__(self, bubble, xyxy):
        self.bubble_xyxy, self.xyxy = bubble, xyxy


def dashed_balloon():
    """Synthetic white page, balloon outline open in its lower half (the fill leaks to the box edge)."""
    import cv2
    img = np.full((300, 320, 3), 255, np.uint8)
    img[:140] = 90  # dark artwork around the upper half
    cv2.ellipse(img, (160, 150), (150, 140), 0, 180, 360, (255, 255, 255), -1)
    cv2.ellipse(img, (160, 150), (150, 140), 0, 180, 360, (0, 0, 0), 2)  # top arch only: the lower half is open
    return img


class OutlineFitTest(unittest.TestCase):
    def test_leaky_balloon_uses_ellipse(self):
        img = dashed_balloon()
        blk = Blk([5, 5, 315, 295], [100, 120, 220, 180])
        rect, inner, _ = r.bubble_interior_ex(img, blk)
        self.assertIsNotNone(inner)
        self.assertFalse(inner[0, 0] or inner[-1, -1] or inner[-1, 0])  # box corners are not interior
        self.assertLess(rect[2] - rect[0], 300)

    def test_margin_knob_off(self):
        self.assertEqual(r.outline_margin_px(400), 20.0)
        os.environ["RENDER_OUTLINE_MARGIN"] = "0"
        try:
            self.assertEqual(r.outline_margin_px(400), 0.0)
        finally:
            del os.environ["RENDER_OUTLINE_MARGIN"]

    def test_metric_counts_ink_on_outline(self):
        c = np.full((100, 100, 3), 255, np.uint8)
        c[:, 0:3] = 0
        f = c.copy()
        f[40:50, 3:30] = 0   # ink touching the outline
        f[40:50, 60:80] = 0  # ink far away
        n, band = balloon_outline_overlap(c, f, [0, 0, 100, 100], band_px=4)
        self.assertEqual(n, 10 * 27 + 10 * 20)
        self.assertEqual(band, 10 * 4)


if __name__ == "__main__":
    unittest.main()
