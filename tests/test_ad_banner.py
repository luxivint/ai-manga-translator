import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.utils.ad_banner import detect_edge_banner  # noqa: E402


def story(h=1500, w=800):
    img = np.full((h, w, 3), 253, np.uint8)
    cv2.circle(img, (400, 300), 200, (20, 20, 20), 4)
    return img


def add_banner(img, rows, purple=True, bottom=True):
    h, w = img.shape[:2]
    y0 = h - rows if bottom else 0
    band = np.full((rows, w, 3), 5, np.uint8)
    cv2.putText(band, "READ ONLY AT", (40, min(rows - 10, 60)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
    if purple and rows > 90:
        band[80:min(rows, 140), 60:700] = (110, 70, 180)
    img[y0:y0 + rows] = band if bottom else band
    return img


class BannerTests(unittest.TestCase):
    def test_bottom_banner_found_with_edge(self):
        img = add_banner(story(), 120)
        self.assertEqual(detect_edge_banner(img), (0, 122))

    def test_top_banner_found(self):
        img = add_banner(story(), 200, bottom=False)
        top, bottom = detect_edge_banner(img)
        self.assertEqual(bottom, 0)
        self.assertAlmostEqual(top, 202, delta=2)

    def test_dark_story_panel_without_purple_is_kept(self):
        self.assertEqual(detect_edge_banner(add_banner(story(), 400, purple=False)), (0, 0))

    def test_dark_panel_without_hard_step_is_kept(self):
        img = story()
        img[:] = 120
        img[1100:] = 5
        img[1100:1140, 100:600] = (110, 70, 180)
        self.assertEqual(detect_edge_banner(img), (0, 0))

    def test_plain_pages_and_odd_input(self):
        self.assertEqual(detect_edge_banner(story()), (0, 0))
        self.assertEqual(detect_edge_banner(np.zeros((1500, 800, 3), np.uint8)), (0, 0))
        self.assertEqual(detect_edge_banner(np.zeros((50, 50, 3), np.uint8)), (0, 0))
        self.assertEqual(detect_edge_banner(np.zeros((300, 300), np.uint8)), (0, 0))

    def test_banner_taking_most_of_page_is_not_cut(self):
        self.assertEqual(detect_edge_banner(add_banner(story(), 1200)), (0, 0))

    def test_thin_strip_only_in_relaxed_mode(self):
        img = story()
        img[-32:] = 5
        self.assertEqual(detect_edge_banner(img), (0, 0))
        self.assertEqual(detect_edge_banner(img, strict=False), (0, 34))

    def test_switch_off(self):
        os.environ["AD_BANNER_DETECT"] = "false"
        try:
            self.assertEqual(detect_edge_banner(add_banner(story(), 120)), (0, 0))
        finally:
            os.environ.pop("AD_BANNER_DETECT", None)


if __name__ == "__main__":
    unittest.main()
