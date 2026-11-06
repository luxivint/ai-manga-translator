"""Regression tests for dialogue spanning two touching speech balloons."""
import unittest
import numpy as np
from unittest.mock import patch

from modules.detection.utils.balloon_regions import assign_balloon_regions
from modules.detection.api import APIDetection


class DoubleBalloonTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((800, 950, 3), 255, np.uint8)

    def test_two_vertical_lobes_split_before_ocr(self):
        bubbles = [[155, 35, 620, 415], [365, 285, 780, 615]]
        box = [250, 85, 650, 545]
        self.image[120:285:14, 310:570] = 0
        self.image[405:510:14, 440:625] = 0
        parts = assign_balloon_regions(self.image, [box], bubbles)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(p[2] for p in parts))
        self.assertEqual([p[1].tolist() for p in parts], bubbles)
        self.assertLessEqual(parts[0][0][3], parts[1][0][1])
        self.assertGreater(parts[0][0][3], 260)
        self.assertLess(parts[0][0][3], 410)

    def test_two_horizontal_lobes_split(self):
        bubbles = [[70, 100, 440, 525], [335, 110, 820, 540]]
        parts = assign_balloon_regions(self.image, [[110, 160, 780, 485]], bubbles)
        self.assertEqual(len(parts), 2)
        self.assertLessEqual(parts[0][0][2], parts[1][0][0])

    def test_normal_single_balloon_stays_one(self):
        b = [[90, 80, 580, 540]]
        self.assertEqual(len(assign_balloon_regions(self.image, [[200, 150, 420, 400]], b)), 1)

    def test_overlapping_neighbor_does_not_split_one_bubble_text(self):
        bubbles = [[50, 50, 450, 425], [340, 305, 800, 700]]
        parts = assign_balloon_regions(self.image, [[145, 100, 405, 360]], bubbles)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][1].tolist(), bubbles[0])

    def test_best_overlap_wins_even_when_second(self):
        bubbles = [[5, 5, 125, 150], [90, 80, 500, 440]]
        parts = assign_balloon_regions(self.image, [[102, 105, 400, 300]], bubbles)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0][1].tolist(), bubbles[1])

    def test_duplicate_balloon_boxes_do_not_split(self):
        bubbles = [[40, 40, 550, 590], [44, 42, 548, 586]]
        self.assertEqual(len(assign_balloon_regions(
            self.image, [[140, 180, 450, 500]], bubbles)), 1)

    def test_detection_preserves_two_ocr_blocks_and_distinct_owners(self):
        # Integration point: both child boxes must be visible to the OCR engine.
        detector = APIDetection()
        mock_font = type("FontEngine", (), {
            "process_batch": lambda self, crops: [{} for _ in crops]
        })()
        with patch("modules.detection.base.FontEngineFactory.create_engine", return_value=mock_font):
            blocks = detector.create_text_blocks(
                self.image, np.array([[250, 85, 650, 545]]),
                np.array([[155, 35, 620, 415], [365, 285, 780, 615]]))
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(block.split_from_double_balloon for block in blocks))
        self.assertFalse(np.array_equal(blocks[0].bubble_xyxy, blocks[1].bubble_xyxy))
        self.assertLessEqual(blocks[0].xyxy[3], blocks[1].xyxy[1])


if __name__ == "__main__":
    unittest.main()
