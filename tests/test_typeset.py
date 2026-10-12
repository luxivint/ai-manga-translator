import json
import unittest
from pathlib import Path

from modules.rendering import typeset as ts
from modules.rendering.font_library import covers_turkish, scan_fonts, pick_font


def mono(size):
    """Fake monospace font: every char is size*0.5 wide, line height size*1.2."""
    return (lambda s: len(s) * size * 0.5), size * 1.2


class CasingTests(unittest.TestCase):
    def test_turkish_dotted_dotless(self):
        self.assertEqual(ts.tr_upper("ışık iyi"), "IŞIK İYİ")
        self.assertEqual(ts.tr_lower("IŞIK İYİ"), "ışık iyi")

    def test_sentence_case(self):
        self.assertEqual(ts.sentence_case("İLK ŞARKI NASIL YAZILDI? ÇOK İYİ."), "İlk şarkı nasıl yazıldı? Çok iyi.")

    def test_auto_policy(self):
        self.assertEqual(ts.apply_casing("IŞIK YOK!", "normal"), "Işık yok!")
        self.assertEqual(ts.apply_casing("ışık yok!", "shout"), "IŞIK YOK!")
        self.assertEqual(ts.apply_casing("Kıdemli Ayşe", "normal"), "Kıdemli Ayşe")  # mixed case untouched
        self.assertEqual(ts.apply_casing("iyi", "normal", "upper"), "İYİ")
        self.assertEqual(ts.apply_casing("", "normal"), "")
        self.assertEqual(ts.apply_casing("...!!", "normal"), "...!!")

    def test_unicode_and_emoji_safe(self):
        self.assertEqual(ts.apply_casing("ÇOK İYİ 🙂", "normal"), "Çok iyi 🙂")


class HyphenTests(unittest.TestCase):
    def test_syllables(self):
        self.assertEqual(ts.turkish_syllables("kalem"), ["ka", "lem"])
        self.assertEqual(ts.turkish_syllables("istatistik"), ["is", "ta", "tis", "tik"])
        self.assertEqual(ts.turkish_syllables("bu"), ["bu"])
        self.assertEqual("".join(ts.turkish_syllables("kahretsin")), "kahretsin")

    def test_split_only_when_too_wide(self):
        width, _ = mono(10)  # 5px per char
        self.assertEqual(ts.split_long_word("merhaba", width, 100), ["merhaba"])
        parts = ts.split_long_word("istatistikler", width, 40)  # 8 chars max
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(width(p) <= 40 for p in parts))
        self.assertEqual("".join(p.rstrip("-") for p in parts), "istatistikler")
        self.assertTrue(all(p.endswith("-") for p in parts[:-1]))


class WrapTests(unittest.TestCase):
    def test_balanced_not_greedy(self):
        width, _ = mono(10)
        cands = ts.balanced_lines("aa bb cc dd ee ff", width, 5 * 11)  # 11 chars/line
        lines = cands[0][0]
        self.assertEqual(len(lines), 2)
        self.assertEqual(len(lines[0]), len(lines[1]))  # 8 / 8 not 11 / 5

    def test_fit_never_overflows(self):
        text = "Bu bölümü sizlere Vortex Scans ekibi sunuyor. Desteğiniz için teşekkürler"
        for bw, bh in [(300, 120), (120, 300), (200, 60), (80, 80), (40, 400)]:
            out, size, w, h, overflow = ts.fit_text(text, mono, bw, bh, max_size=58, min_size=14, floor=9)
            if not overflow:
                self.assertLessEqual(w, bw + 1e-6)
                self.assertLessEqual(h, bh + 1e-6)
            self.assertGreaterEqual(size, 9)

    def test_floor_instead_of_overflow(self):
        out, size, w, h, overflow = ts.fit_text("kelime " * 12, mono, 60, 30, max_size=40, min_size=14, floor=9)
        self.assertLessEqual(size, 14)

    def test_long_word_hyphenated_to_fit(self):
        out, size, w, h, overflow = ts.fit_text("elektromanyetizma", mono, 70, 200, max_size=20, min_size=14, floor=9)
        self.assertFalse(overflow)
        self.assertLessEqual(w, 70)

    def test_largest_size_wins(self):
        out, size, *_ = ts.fit_text("Merhaba", mono, 500, 500, max_size=40, min_size=14, floor=9)
        self.assertEqual(size, 40)
        self.assertEqual(out, "Merhaba")

    def test_empty_and_whitespace(self):
        out, size, w, h, overflow = ts.fit_text("   ", mono, 100, 100, max_size=30, min_size=14, floor=9)
        self.assertTrue(overflow or out.strip() == "")

    def test_large_input_fast(self):
        text = " ".join(["kelime"] * 600)
        out, size, w, h, overflow = ts.fit_text(text, mono, 1500, 1500, max_size=30, min_size=14, floor=9)
        self.assertGreaterEqual(size, 9)


class BandTests(unittest.TestCase):
    def test_band_caps_outliers(self):
        self.assertEqual(ts.apply_size_band([20, 20, 22, 60], 1.4), [20, 20, 22, 29])
        self.assertEqual(ts.apply_size_band([], 1.4), [])
        self.assertEqual(ts.apply_size_band([30], 1.4), [30])


class FontCoverageTests(unittest.TestCase):
    def test_covers_turkish(self):
        full = {ord(c) for c in ts.TURKISH_LETTERS}
        self.assertTrue(covers_turkish(full))
        self.assertFalse(covers_turkish(full - {ord("ğ")}))
        self.assertFalse(covers_turkish(set()))

    def test_scan_and_pick_exclude_missing(self):
        root = Path(__file__).resolve().parents[1]
        cov = scan_fonts(root)
        self.assertTrue(cov)
        fallback = [p for p, ok in cov.items() if p.endswith("Comic Geek.ttf")]
        self.assertTrue(fallback and cov[fallback[0]])
        bad = [p for p, ok in cov.items() if not ok]
        chosen = pick_font("shout", cov, {"shout": ["bağırma"]}, root)
        self.assertNotIn(chosen, bad)


class PerspectiveTests(unittest.TestCase):
    def test_warp_and_composite_synthetic(self):
        import numpy as np
        from modules.rendering.render_v2 import warp_layer_to_quad, alpha_composite, estimate_text_quad
        layer = np.zeros((40, 100, 4), np.uint8)
        layer[..., :] = (255, 0, 0, 255)
        quad = np.array([[20, 30], [120, 20], [125, 70], [22, 62]], np.float32)
        out = warp_layer_to_quad(layer, quad, (100, 150))
        self.assertGreater(int((out[..., 3] > 0).sum()), 2000)
        self.assertEqual(int(out[5, 5, 3]), 0)
        base = np.full((100, 150, 3), 255, np.uint8)
        comp = alpha_composite(base, out)
        self.assertTrue((comp[45, 70] == (255, 0, 0)).all())
        self.assertTrue((comp[2, 2] == 255).all())
        # quad estimation from erased glyph pixels
        img = np.full((100, 150, 3), 255, np.uint8)
        cl = img.copy()
        img[40:60, 30:120] = 0
        q = estimate_text_quad(img, cl, (0, 0, 150, 100))
        self.assertEqual(q.shape, (4, 2))
        self.assertIsNone(estimate_text_quad(cl, cl, (0, 0, 150, 100)))


if __name__ == "__main__":
    unittest.main()
