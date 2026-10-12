import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import local_batch as lb  # noqa: E402
from modules.utils import watermark as wm  # noqa: E402


def blk(text, box=(10, 10, 100, 40), is_ad=False):
    return SimpleNamespace(text=text, xyxy=np.array(box), is_ad=is_ad, translation="")


class MatchTests(unittest.TestCase):
    def test_competitor_marks_match(self):
        for t in ("VORTEXSCANS.COM", "EN-THUNDERSCANS.COM", "READ THIS SERIES FIRST AT:", "discord.gg/abc"):
            self.assertIsNotNone(wm.match_watermark(t), t)

    def test_dialogue_does_not_match(self):
        for t in ("HELLO THERE", "I WILL FIND YOU.", ""):
            self.assertIsNone(wm.match_watermark(t))

    def test_own_brand_is_never_a_watermark(self):
        for t in ("TrendManga", "TRENDMANGA.COM", "Trend Manga .com"):
            self.assertIsNone(wm.match_watermark(t), t)
        with mock.patch.dict(os.environ, {"SEO_SITE_NAME": "MyBrand"}):
            self.assertIsNone(wm.match_watermark("MYBRAND.NET"))


class PolicyTests(unittest.TestCase):
    def test_default_is_clean(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("WATERMARK_POLICY", None)
            self.assertEqual(lb.watermark_policy(), "clean")
            self.assertIsNone(lb.watermark_rule(blk("VORTEXSCANS.COM"), 800, 800))

    def test_keep_policy_keeps_marks(self):
        with mock.patch.dict(os.environ, {"WATERMARK_POLICY": "keep"}):
            self.assertEqual(lb.watermark_policy(), "keep")
            self.assertTrue(lb.watermark_rule(blk("VORTEXSCANS.COM"), 800, 800))
            self.assertEqual(lb.watermark_rule(blk("x", is_ad=True), 800, 800), "is_ad")
            self.assertEqual(lb.watermark_clear_mask(np.zeros((100, 100, 3), np.uint8), [blk("VORTEXSCANS.COM")]).sum(), 0)

    def test_clean_mask_covers_competitor_not_own_brand(self):
        img = np.zeros((200, 300, 3), np.uint8)
        with mock.patch.dict(os.environ, {"WATERMARK_POLICY": "clean"}):
            mask = lb.watermark_clear_mask(img, [blk("VORTEXSCANS.COM", (50, 50, 150, 80)),
                                                 blk("TRENDMANGA.COM", (50, 120, 150, 150))])
        self.assertTrue(mask[60, 100] > 0)
        self.assertEqual(mask[135, 100], 0)


class RegionTests(unittest.TestCase):
    def banner_image(self):
        rng = np.random.default_rng(0)
        img = rng.integers(120, 255, (300, 500, 3), dtype=np.uint8)           # noisy artwork
        img[100:160, 40:400] = (70, 70, 75)                                    # flat grey tag
        for x in range(120, 300, 12):
            img[112:148, x:x + 3] = (240, 240, 240)                            # thin glyph strokes on the tag
        return img

    def test_whole_tag_is_covered_not_only_the_text_box(self):
        img = self.banner_image()
        mask = wm.region_mask(img, (120, 112, 300, 148), pad=6)
        self.assertTrue((mask[100:160, 40:400] > 0).all())  # entire tag incl. the part outside the text box
        self.assertEqual(mask[10, 10], 0)                   # artwork far away untouched

    def test_text_on_artwork_falls_back_to_padded_box(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 255, (200, 300, 3), dtype=np.uint8)
        mask = wm.region_mask(img, (100, 80, 200, 110), pad=10)
        self.assertTrue((mask[70:120, 90:210] > 0).all())
        self.assertLess((mask > 0).sum(), 60 * 140)

    def test_flat_fill_paints_flat_surround_and_skips_busy_one(self):
        flat = np.zeros((100, 100, 3), np.uint8)
        flat[40:60, 40:60] = 200
        m = np.zeros((100, 100), np.uint8); m[35:65, 35:65] = 255
        out = wm.flat_fill(flat, np.full_like(flat, 99), m)
        self.assertEqual(int(out[50, 50, 0]), 0)
        busy = np.random.default_rng(2).integers(0, 255, (100, 100, 3), dtype=np.uint8)
        cleaned = np.full_like(busy, 99)
        self.assertTrue((wm.flat_fill(busy, cleaned, m) == cleaned).all())


if __name__ == "__main__":
    unittest.main()
