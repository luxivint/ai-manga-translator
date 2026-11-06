"""Nonblocking page QA metrics regression."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from modules.rendering.quality_audit import audit_cleaning, audit_layout, write_report


class QualityAuditTests(unittest.TestCase):
    def setUp(self):
        self.original = np.full((260, 370, 3), 255, dtype=np.uint8)
        cv2.putText(self.original, "HELLO", (75, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 0), 3, cv2.LINE_AA)
        self.block = SimpleNamespace(text="HELLO", translation="MERHABA",
                                     xyxy=[65, 65, 220, 135],
                                     text_class="text_bubble", is_ad=False)

    def test_cleaned_white_balloon_has_no_residual_warning(self):
        cleaned = self.original.copy()
        cleaned[65:135, 65:220] = 255
        result = audit_cleaning(self.original, cleaned, [self.block])
        self.assertEqual(result["translated_blocks"], 1)
        self.assertEqual(result["warnings"], [])

    def test_unmodified_source_is_flagged(self):
        result = audit_cleaning(self.original, self.original, [self.block])
        self.assertIn("possible_residual_ink", [e["kind"] for e in result["warnings"]])

    def test_missing_translation_warns_not_crashes(self):
        self.block.translation = ""
        result = audit_cleaning(self.original, self.original, [self.block])
        self.assertIn("missing_translation", [e["kind"] for e in result["warnings"]])

    def test_render_overflow(self):
        items = [{"text": "hello", "position": (330, 95), "width": 90, "height": 20, "rotation": 0}]
        self.assertIn("outside_page", [w["kind"] for w in audit_layout(items, self.original.shape)])

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = Path(temp) / "chapter" / "page.json"
            report = write_report(dest, self.original, self.original, [self.block], [])
            self.assertTrue(dest.exists())
            self.assertIn("no_render_items", [w["kind"] for w in report["warnings"]])


if __name__ == "__main__":
    unittest.main()
