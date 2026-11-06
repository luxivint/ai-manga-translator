"""Geometry regression for two OCR texts sharing ONE bubble_xyxy."""
import unittest

import cv2
import numpy as np

from modules.rendering.linked_lobes import (
    separated_text_boxes, split_lobe_interiors, fallback_lobe_regions,
)


class LinkedLobesTests(unittest.TestCase):
    def test_joined_two_lobes_have_separate_safe_masks(self):
        mask = np.zeros((700, 900), np.uint8)
        cv2.ellipse(mask, (320, 250), (225, 190), 0, 0, 360, 1, -1)
        cv2.ellipse(mask, (590, 450), (225, 190), 0, 0, 360, 1, -1)
        mask = cv2.erode(mask, np.ones((11, 11), np.uint8)).astype(bool)
        original_text_boxes = [[195, 160, 405, 330], [485, 370, 750, 555]]
        pieces = split_lobe_interiors(mask, (0, 0), original_text_boxes, [90, 60, 815, 640])
        self.assertIsNotNone(pieces)
        self.assertEqual(len(pieces), 2)
        self.assertFalse(np.any(pieces[0][1] & pieces[1][1]))
        for rect, interior, offset in pieces:
            x1, y1, x2, y2 = rect
            self.assertGreater((x2-x1) * (y2-y1), 10000)
            self.assertGreater(interior[y1:y2, x1:x2].mean(), 0.995)

    def test_one_round_balloon_not_falsely_partitioned(self):
        mask = np.zeros((700, 900), np.uint8)
        cv2.ellipse(mask, (450, 350), (330, 260), 0, 0, 360, 1, -1)
        self.assertIsNone(split_lobe_interiors(mask.astype(bool), (0, 0),
                          [[180, 140, 400, 310], [500, 370, 720, 540]],
                          [120, 90, 780, 610]))

    def test_two_nearby_text_lines_are_one_balloon(self):
        self.assertFalse(separated_text_boxes(
            [[150, 230, 330, 280], [160, 290, 345, 344]],
            [60, 80, 520, 520]))

    def test_unmeasurable_outline_fallback_does_not_overlap(self):
        r = fallback_lobe_regions([[200, 100, 450, 260], [310, 350, 560, 570]], (800, 900, 3))
        self.assertLessEqual(r[0][3], r[1][1])


if __name__ == "__main__":
    unittest.main()
