"""Real-image regressions from the six user-reported layout defects."""
from pathlib import Path
import sys
import unittest
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from modules.detection.utils.balloon_regions import assign_balloon_regions
from modules.rendering.text_style import row_cap_height


def source(number):
    with Image.open(ROOT / f'tests/fixtures/layout_review/page-{number:03d}.png') as image:
        return np.asarray(image.convert('RGB')).copy()


class ReportedLayoutTests(unittest.TestCase):
    def test_source_graphic_is_restored_without_reviving_overlapping_caption(self):
        from unittest.mock import patch
        import local_batch as lb
        from modules.utils.textblock import TextBlock
        original = np.full((40, 40, 3), 30, dtype=np.uint8)
        cleaned = np.full_like(original, 240)
        graphic = TextBlock(text_bbox=np.array([0, 0, 30, 30]))
        graphic.preserve_original = True
        caption = TextBlock(text_bbox=np.array([10, 0, 25, 10]))
        caption.translation = 'Gerçek konuşma'
        mask = np.zeros((40, 40), dtype=np.uint8); mask[:30, :30] = 255
        with patch.object(lb, 'generate_mask', return_value=mask):
            result = lb.restore_untranslated_regions(original, cleaned, [graphic, caption])
        np.testing.assert_array_equal(result[10:30, :30], original[10:30, :30])
        np.testing.assert_array_equal(result[:10, 10:25], cleaned[:10, 10:25])
        np.testing.assert_array_equal(result[30:], cleaned[30:])

    def test_helper_code_change_invalidates_checkpoint_identity(self):
        import tempfile
        from unittest.mock import patch
        import local_batch as lb
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); helper = root / 'modules/rendering/linked_lobes.py'
            helper.parent.mkdir(parents=True); helper.write_text('before', encoding='utf8')
            with patch.object(lb, 'ROOT', root):
                try:
                    lb._pipeline_code_fingerprint.cache_clear()
                    before = lb._pipeline_code_fingerprint()
                    helper.write_text('after', encoding='utf8')
                    lb._pipeline_code_fingerprint.cache_clear()
                    self.assertNotEqual(before, lb._pipeline_code_fingerprint())
                finally:
                    lb._pipeline_code_fingerprint.cache_clear()

    def test_real_letter_height_cannot_collapse_to_nine_pixels(self):
        height = row_cap_height(source(91), [76, 289, 531, 519])
        self.assertGreater(height, 25)
        self.assertLess(height, 40)

    def test_complete_owner_beats_partial_overlap_with_neighbour_balloon(self):
        text = [366, 580, 723, 677]
        bubbles = [[32, 104, 556, 761], [336, 450, 754, 886]]
        rows = assign_balloon_regions(source(96), [text], bubbles)
        self.assertEqual(len(rows), 1)
        np.testing.assert_array_equal(rows[0][0], text)
        np.testing.assert_array_equal(rows[0][1], bubbles[1])
        self.assertFalse(rows[0][2])

    def test_one_detection_covering_two_real_lobes_splits_before_ocr(self):
        rows = assign_balloon_regions(source(95), [[74, 1502, 756, 2029]], [[40, 1278, 783, 2164]])
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r[2] for r in rows))
        self.assertLessEqual(rows[0][0][3], rows[1][0][1])

    def test_graphic_below_balloon_is_not_owned_by_small_overlap(self):
        rows = assign_balloon_regions(source(17), [[41, 540, 388, 1243]], [[32, 74, 786, 784]])
        self.assertIsNone(rows[0][1])


if __name__ == '__main__':
    unittest.main()
