"""Regression checks for missing or duplicated OCR crop IDs."""
import unittest

from modules.ocr.qwen_grid_ocr import QwenGridOCR


class OCRCompletenessTests(unittest.TestCase):
    def check(self, items, expected=2):
        return QwenGridOCR._looks_like_failed_batch(items, expected, set(range(1, expected + 1)))

    def test_two_valid_balloons(self):
        self.assertFalse(self.check([{"id": 1, "text": "Hello"}, {"id": 2, "text": "World"}]))

    def test_missing_second_balloon_is_failure(self):
        self.assertTrue(self.check([{"id": 1, "text": "Hello"}]))

    def test_missing_first_balloon_is_failure(self):
        self.assertTrue(self.check([{"id": 2, "text": "World"}]))

    def test_duplicate_first_id_is_failure(self):
        self.assertTrue(self.check([{"id": 1, "text": "One"}, {"id": 1, "text": "Two"}]))

    def test_explicit_blank_crop_is_valid(self):
        self.assertFalse(self.check([{"id": 1, "text": "Hello"}, {"id": 2, "text": ""}]))

    def test_missing_text_field_is_failure(self):
        self.assertTrue(self.check([{"id": 1, "text": "Hi"}, {"id": 2}]))

    def test_unexpected_id_is_failure(self):
        self.assertTrue(self.check([{"id": 1, "text": "Hi"}, {"id": 3, "text": "Test"}]))

    def test_single_crop_missing_is_failure(self):
        self.assertTrue(self.check([], 1))

    def test_many_explicit_empty_crops_are_suspicious(self):
        self.assertTrue(self.check([{"id": i, "text": ""} for i in range(1, 5)], 4))


if __name__ == "__main__":
    unittest.main()
