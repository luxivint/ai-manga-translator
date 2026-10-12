"""Cleaning-side invariants: watermark policy, gating == render set, output size, vision-crop validation."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402


def blk(xyxy, text="", translation="", is_ad=False):
    return SimpleNamespace(xyxy=np.array(xyxy), text=text, translation=translation,
                           is_ad=is_ad, text_class="text_free")


class WatermarkRuleTests(unittest.TestCase):
    def test_is_ad_block_is_kept(self):
        self.assertEqual(lb.watermark_rule(blk((0, 0, 10, 10), "x", is_ad=True), 800, 800), "is_ad")

    def test_scan_group_texts_match(self):
        for t in ("VORTEXSCANS.COM", "READ THIS SERIES FIRST AT:", "visit www.foo", "mangahub.net"):
            self.assertIsNotNone(lb.watermark_rule(blk((0, 0, 10, 10), t), 800, 800), t)

    def test_dialogue_not_matched(self):
        for t in ("SHE SCANS THE ROOM.", "I READ IT AT NIGHT", "", "...ISN'T THE MONSTERS."):
            self.assertIsNone(lb.watermark_rule(blk((0, 0, 10, 10), t), 800, 800), t)

    def test_policy_clean_disables_rule(self):
        with mock.patch.dict(os.environ, {"WATERMARK_POLICY": "clean"}):
            self.assertIsNone(lb.watermark_rule(blk((0, 0, 10, 10), "VORTEXSCANS.COM", is_ad=True), 800, 800))

    def test_extra_env_pattern(self):
        with mock.patch.dict(os.environ, {"WATERMARK_PATTERNS": r"my\s*group"}):
            self.assertIsNotNone(lb.watermark_rule(blk((0, 0, 10, 10), "MY GROUP"), 800, 800))


class VisionCropTests(unittest.TestCase):
    def test_no_evidence_rejects_crop(self):
        blocks = [blk((10, 100, 300, 200), "HELLO THERE", "MERHABA")]
        self.assertFalse(lb.crop_has_ad_evidence(blocks, 0, 900, 800, 1000))

    def test_url_in_cropped_strip_accepts(self):
        blocks = [blk((10, 950, 300, 990), "VORTEXSCANS.COM")]
        self.assertTrue(lb.crop_has_ad_evidence(blocks, 0, 900, 800, 1000))

    def test_story_text_in_kept_area_not_evidence(self):
        blocks = [blk((10, 100, 300, 200), "VORTEXSCANS.COM")]
        self.assertFalse(lb.crop_has_ad_evidence(blocks, 0, 900, 800, 1000))

    def test_validation_can_be_disabled(self):
        with mock.patch.dict(os.environ, {"AD_VISION_CROP_REQUIRE_TEXT": "false"}):
            self.assertTrue(lb.crop_has_ad_evidence([], 0, 900, 800, 1000))


class OutputSizeTests(unittest.TestCase):
    def _save(self, d, name, w, h):
        p = Path(d) / name
        Image.new("RGB", (w, h)).save(p)
        return p

    def test_equal_size_passes_and_parts_sum(self):
        img = np.zeros((300, 100, 3), np.uint8)
        with tempfile.TemporaryDirectory() as d:
            lb.verify_output_dims([self._save(d, "a.png", 100, 300)], img)
            lb.verify_output_dims([self._save(d, "b1.png", 100, 120), self._save(d, "b2.png", 100, 180)], img)

    def test_truncated_output_raises(self):
        img = np.zeros((5000, 689, 3), np.uint8)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                lb.verify_output_dims([self._save(d, "a.png", 689, 4450)], img)


class MaskGatingTests(unittest.TestCase):
    def setUp(self):
        self.img = np.full((400, 600, 3), 20, np.uint8)
        for (x, y) in ((40, 80), (40, 250)):
            cv2.putText(self.img, "HELLO", (x, y), cv2.FONT_HERSHEY_DUPLEX, 2.0, (240, 240, 240), 5)
        self.seg = np.zeros(self.img.shape[:2], np.uint8)  # segmenter finds nothing

    def _mask(self, blocks, **kw):
        with mock.patch.object(lb, "_get_segmenter", lambda: None), \
             mock.patch.object(lb, "_tiled_segment", lambda s, i, c: (self.seg.copy(), self.seg.copy())):
            self.seg[:] = 0
            return lb.build_segmenter_mask(self.img, blocks, **kw)

    def test_only_translated_blocks_cleaned_after_translation(self):
        a = blk((30, 30, 330, 100), "HELLO", "MERHABA")
        b = blk((30, 200, 330, 270), "HELLO", "")  # no translation -> must stay untouched
        m = self._mask([a, b], require_translation=True)
        self.assertGreater((m[30:100, 30:330] > 0).sum(), 200)
        self.assertEqual(int((m[200:270, 30:330] > 0).sum()), 0)

    def test_segmenter_miss_is_recovered_from_block(self):
        a = blk((30, 30, 330, 100), "HELLO", "MERHABA")
        m = self._mask([a], require_translation=True)
        glyph = (self.img[30:100, 30:330].mean(axis=2) > 128)
        covered = ((m[30:100, 30:330] > 0) & glyph).sum() / max(1, glyph.sum())
        self.assertGreater(covered, 0.8)

    def test_watermark_block_never_masked(self):
        wm = blk((30, 200, 330, 270), "VORTEXSCANS.COM", "", is_ad=False)
        a = blk((30, 30, 330, 100), "HELLO", "MERHABA")
        m = self._mask([a, wm], require_translation=True)
        self.assertEqual(int((m[190:280, 20:340] > 0).sum()), 0)
        self.assertGreater(int((m[30:100, 30:330] > 0).sum()), 0)


if __name__ == "__main__":
    unittest.main()
