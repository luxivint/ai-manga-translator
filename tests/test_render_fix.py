"""Typesetting upgrade: casing, size derivation, gradient detection, font profiles, line breaks."""
import os
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from modules.rendering import font_metrics as fm  # noqa: E402
from modules.rendering import font_profiles as fp  # noqa: E402
from modules.rendering import text_style as tst  # noqa: E402
from modules.rendering import typeset as ts  # noqa: E402

ARIAL = "C:/Windows/Fonts/arial.ttf"


def _text_image(text, size=60, fill=(255, 255, 255), bg=(0, 0, 0), angle=0.0):
    font = ImageFont.truetype(ARIAL if Path(ARIAL).exists() else str(ROOT / "assets" / "Comic Geek.ttf"), size)
    im = Image.new("RGB", (700, 300), bg)
    ImageDraw.Draw(im).text((40, 90), text, font=font, fill=fill)
    if angle:
        im = im.rotate(-angle, resample=Image.BICUBIC, fillcolor=bg)
    return np.asarray(im).copy()


class CasingTests(unittest.TestCase):
    def test_turkish_upper_dotted_and_dotless(self):
        self.assertEqual(ts.tr_upper("ışık iyi"), "IŞIK İYİ")
        self.assertEqual(ts.tr_upper("istanbul"), "İSTANBUL")

    def test_upper_never_yields_dotless_small_i(self):
        out = ts.tr_upper("Shin hiçbir şey ıslak")
        self.assertNotIn("ı", out)
        self.assertEqual(out, "SHİN HİÇBİR ŞEY ISLAK")

    def test_lower_roundtrip_i(self):
        self.assertEqual(ts.tr_lower("IŞIK İYİ"), "ışık iyi")

    def test_untranslated_name_keeps_source_letters(self):
        from modules.rendering.render_v2 import casing_text

        self.assertEqual(casing_text("Wild Grylls", "WILD GRYLLS", "phone", "upper"), "WILD GRYLLS")
        self.assertEqual(casing_text("Shin", "Shin", "normal", "upper"), "SHIN")
        self.assertEqual(casing_text("Ölme, Ricky!", "DON'T DIE, RICKY!", "shout", "upper"), "ÖLME, RİCKY!")

    def test_policy_keep_and_auto(self):
        self.assertEqual(ts.apply_casing("Shin geldi", "normal", "keep"), "Shin geldi")
        self.assertEqual(ts.apply_casing("Shin geldi", "normal", "auto"), "Shin geldi")
        self.assertEqual(ts.apply_casing("HAYIR", "normal", "auto"), "Hayır")


class LineBreakTests(unittest.TestCase):
    @staticmethod
    def measure(size):
        return (lambda s: 0.6 * size * len(s)), 1.2 * size

    def test_normalize_keeps_line_breaks(self):
        self.assertEqual(ts.normalize_spaces("  SEVİYE   - 1 \r\n\nCAN - 71 \nMP"), "SEVİYE - 1\nCAN - 71\nMP")

    def test_fit_paragraphs_one_line_per_source_line(self):
        out, size, w, h, over = ts.fit_text("SEVİYE - 1\nCAN - 71\nMP - 0", self.measure, 400, 300,
                                            max_size=40, min_size=8, floor=8)
        self.assertEqual(out.split("\n"), ["SEVİYE - 1", "CAN - 71", "MP - 0"])
        self.assertFalse(over)

    def test_long_paragraph_wraps_only_when_needed(self):
        out, size, w, h, over = ts.fit_text("KISA\nBU SATIR ÇOK UZUN BİR SATIR OLDUĞU İÇİN BÖLÜNÜR", self.measure, 200, 300,
                                            max_size=20, min_size=8, floor=8)
        lines = out.split("\n")
        self.assertEqual(lines[0], "KISA")
        self.assertGreater(len(lines), 2)


class SizeDerivationTests(unittest.TestCase):
    def test_cap_height_of_rendered_text(self):
        img = _text_image("HELLO WORLD", size=60)
        mask = img.sum(axis=2) > 300
        cap = tst.cap_height(mask)
        # Arial cap height is 0.716 em
        self.assertAlmostEqual(cap, 60 * 0.716, delta=4)

    def test_size_for_cap_roundtrip(self):
        from PySide6.QtWidgets import QApplication
        from modules.rendering import render_v2

        QApplication.instance() or QApplication([])
        fam = render_v2.family_for(fp.STYLE_RANKS["normal"][0])
        size = render_v2.size_for_cap(fam, 30)
        from PySide6.QtGui import QFontMetricsF
        cap = QFontMetricsF(render_v2._qfont(fam, round(size))).capHeight()
        self.assertAlmostEqual(cap, 30, delta=3)

    def test_line_pitch_counts_lines(self):
        img = _text_image("AAAA\nBBBB", size=50)
        mask = img.sum(axis=2) > 300
        pitch, n = tst.line_pitch(mask)
        self.assertEqual(n, 2)
        self.assertGreater(pitch, 40)

    def test_angle_of_rotated_text(self):
        img = _text_image("TILTED SIGN TEXT", size=60, angle=-12)
        mask = img.sum(axis=2) > 300
        ang, rect = tst.text_angle(mask)
        self.assertAlmostEqual(ang, -12, delta=3)
        flat = tst.text_angle(_text_image("TILTED SIGN TEXT", size=60).sum(axis=2) > 300)[0]
        self.assertEqual(flat, 0.0)


class MergedBlockTests(unittest.TestCase):
    def test_split_groups_and_allocation(self):
        mask = np.zeros((400, 300), bool)
        mask[10:40, 10:200] = True
        mask[200:230, 10:120] = True
        mask[330:360, 10:80] = True
        groups = tst.split_groups(mask, 30.0)
        self.assertEqual(len(groups), 3)
        segs = tst.allocate_lines(["BIRINCI SATIR UZUN", "IKINCI", "UCUNCU"], [g[4] for g in groups])
        self.assertEqual([len(x) for x in segs], [1, 1, 1])

    def test_allocation_needs_enough_lines(self):
        self.assertEqual(tst.allocate_lines(["A"], [1.0, 2.0]), [])


class GradientTests(unittest.TestCase):
    def _glyph_block(self, painter):
        h, w = 160, 480
        base = np.full((h, w, 3), 30, np.uint8)
        mask = np.zeros((h, w), np.uint8)
        im = Image.fromarray(mask)
        ImageDraw.Draw(im).text((20, 10), "GRADIENT", font=ImageFont.truetype(
            ARIAL if Path(ARIAL).exists() else str(ROOT / "assets" / "Comic Geek.ttf"), 120), fill=255)
        m = np.asarray(im) > 127
        img = base.copy()
        painter(img, m)
        return img, m

    def test_vertical_gradient_detected(self):
        def paint(img, m):
            ys = np.nonzero(m.any(axis=1))[0]
            t = np.clip((np.arange(img.shape[0]) - ys.min()) / (ys.max() - ys.min()), 0, 1)
            top, bot = np.array([40, 200, 60]), np.array([230, 240, 60])
            col = (top[None] * (1 - t[:, None]) + bot[None] * t[:, None]).astype(np.uint8)
            img[m] = col[np.nonzero(m)[0]]
        img, m = self._glyph_block(paint)
        res = tst.analyze_colors(img, m, (0, 0), bg_rgb=(30, 30, 30))
        self.assertIsNotNone(res["gradient"])
        self.assertTrue(res["gradient"]["vertical"])
        top, bot = res["gradient"]["colors"]
        self.assertLess(top[0], bot[0] - 80)

    def test_flat_colour_is_not_gradient(self):
        def paint(img, m):
            img[m] = (250, 250, 250)
        img, m = self._glyph_block(paint)
        res = tst.analyze_colors(img, m, (0, 0), bg_rgb=(30, 30, 30))
        self.assertIsNone(res["gradient"])
        self.assertEqual(res["fill"], (250, 250, 250))

    def test_two_tone_top_bottom(self):
        def paint(img, m):
            ys = np.nonzero(m.any(axis=1))[0]
            mid = (ys.min() + ys.max()) // 2
            rows = np.arange(img.shape[0])[:, None] < mid
            img[m & rows] = (255, 215, 0)
            img[m & ~rows] = (255, 255, 255)
        img, m = self._glyph_block(paint)
        res = tst.analyze_colors(img, m, (0, 0), bg_rgb=(30, 30, 30))
        self.assertIsNotNone(res["gradient"])
        self.assertTrue(res["gradient"]["vertical"])

    def test_contrast_helper_adds_outline_for_low_contrast(self):
        _, outline = tst.ensure_contrast([(200, 200, 200), (220, 220, 220)], (210, 210, 210))
        self.assertEqual(outline, (0, 0, 0))
        _, none = tst.ensure_contrast([(255, 255, 255)], (10, 10, 10))
        self.assertIsNone(none)


class FontProfileTests(unittest.TestCase):
    def test_every_style_has_usable_candidates(self):
        for style in fp.STYLES:
            cands = fp.ranked_candidates(style)
            self.assertGreaterEqual(len(cands), 2, style)
            for rel in cands:
                self.assertTrue((ROOT / rel).exists(), rel)

    def test_candidates_cover_turkish_and_not_blocklisted(self):
        for style in fp.STYLES:
            for rel in fp.ranked_candidates(style):
                t = fp.traits_of(rel)
                self.assertTrue(t["turkish_ok"], rel)
                self.assertNotIn(rel, fp.BLOCKLIST)

    def test_font_without_ghi_is_blocklisted(self):
        moritat = "FONT KLASÖR/Süslü balonlar için/CCMoritatW00Regular-Regular(2).ttf"
        self.assertFalse(fp.is_usable(moritat))

    def test_caps_detection_matches_specimens(self):
        self.assertTrue(fp.caps_only("FONT KLASÖR/konuşma/CCVictorySpeech.otf"))
        self.assertTrue(fp.caps_only("FONT KLASÖR/elektronik alet/Bebas Neue.ttf"))
        self.assertFalse(fp.caps_only("FONT KLASÖR/Kare/CCTALLTALES-REGULAR.OTF"))
        self.assertFalse(fp.caps_only("FONT KLASÖR/Kare/Chinacat.ttf"))

    def test_default_casing_policy_is_upper(self):
        self.assertEqual(fp.casing_for("FONT KLASÖR/Kare/Chinacat.ttf", "normal"), "upper")
        self.assertEqual(fp.casing_for("FONT KLASÖR/Kare/Chinacat.ttf", "normal", "auto"), "natural")
        self.assertEqual(fp.casing_for("FONT KLASÖR/Kare/Chinacat.ttf", "sfx", "auto"), "upper")

    def test_sanitize_replaces_missing_punctuation(self):
        rel = "FONT KLASÖR/resim üstü/Orange_Fizz.ttf"  # lacks … – —
        out = fp.sanitize_for_font("EVET… HAYIR – BELKİ", rel)
        self.assertNotIn("…", out)
        self.assertNotIn("–", out)
        self.assertEqual(fp.sanitize_for_font("EVET…", "FONT KLASÖR/konuşma/CCVictorySpeech.otf"), "EVET…")

    def test_match_font_prefers_closest_weight(self):
        heavy = {"stem": 0.20, "slant": 0.0, "wid": 0.6, "height": 40.0, "caps": True}
        light = {"stem": 0.07, "slant": 0.0, "wid": 0.6, "height": 40.0, "caps": True}
        a, b = fp.match_font("normal", heavy), fp.match_font("normal", light)
        self.assertGreaterEqual(fp.traits_of(a)["stem"], fp.traits_of(b)["stem"])
        self.assertIn(b, fp.speech_pool())  # even a hairline source never gets a thin font for speech
        self.assertEqual(fp.match_font("normal", None), fp.ranked_candidates("normal")[0])

    def test_source_traits_measure_stem_and_slant(self):
        img = _text_image("HHHH EEEE", size=80)
        t = fm.source_traits(img.sum(axis=2) > 300)
        self.assertTrue(t["caps"])
        self.assertAlmostEqual(t["slant"], 0.0, delta=0.06)
        self.assertGreater(t["stem"], 0.08)


if __name__ == "__main__":
    unittest.main()
