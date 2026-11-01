"""Synthetic tests for the visual-audit cleaning fixes: gradient / texture-aware flat snap, QA calm-fill guards,
outline-pass guards, half-erased SFX revert and the local watermark flat-fill guard."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import outline  # noqa: E402
from modules.utils import watermark as wm  # noqa: E402


H, W = 420, 600
INK = (25, 25, 25)


def _balloon():
    img = np.full((H, W, 3), 250, np.uint8)
    cv2.ellipse(img, (300, 330), (250, 190), 0, 0, 360, INK, 4, cv2.LINE_AA)
    arc = img.copy()
    for x in (270, 300, 330):
        cv2.rectangle(img, (x - 3, 110), (x + 3, 185), INK, -1)
    cv2.rectangle(img, (265, 178), (336, 185), INK, -1)
    protected = cv2.dilate((arc.mean(axis=2) < 140).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    done = np.zeros((H, W), np.uint8)
    done[100:230, 200:400] = 255
    done[135:190, 262:340] = 0
    return img, protected, done


class OutlineGuardTest(unittest.TestCase):
    def test_letters_outside_every_translated_box_are_left_alone(self):
        img, protected, done = _balloon()
        far_box = np.zeros((H, W), np.uint8)
        far_box[300:380, 20:120] = 255
        _k, _b, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), protected, boxes=far_box)
        self.assertEqual(n, 0)
        box = np.zeros((H, W), np.uint8)
        box[95:200, 250:350] = 255
        _k, _b, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), protected, boxes=box)
        self.assertGreaterEqual(n, 1)

    def test_spike_ring_is_not_a_clean_line(self):
        ink = np.zeros((200, 300), bool)
        xs = np.arange(20, 280)
        ys = (120 + 0.0015 * (xs - 150) ** 2).astype(int)
        for x, y in zip(xs, ys):
            ink[y - 2:y + 2, x] = True
            if x % 6 == 0:
                ink[y - 30:y, x - 1:x + 1] = True  # spikes on the outer side
        fit = outline.fit_outline(ink)
        self.assertIsNotNone(fit)
        self.assertFalse(outline.is_clean_line(fit, ink, np.zeros_like(ink)))
        plain = np.zeros_like(ink)
        for x, y in zip(xs, ys):
            plain[y - 2:y + 2, x] = True
        fit2 = outline.fit_outline(plain)
        self.assertTrue(outline.is_clean_line(fit2, plain, np.zeros_like(plain)))

    def test_stroke_fraction(self):
        core = np.zeros((80, 80), bool)
        for x in range(5, 75, 8):
            core[10:60, x:x + 3] = True
        self.assertGreater(outline.stroke_fraction(core), 0.9)
        blob = np.zeros((80, 80), bool)
        blob[20:60, 20:60] = True
        self.assertEqual(outline.stroke_fraction(blob), 0.0)


class FlatFillGuardTest(unittest.TestCase):
    def test_flat_fill_needs_a_really_flat_surround(self):
        img = np.full((200, 300, 3), 250, np.uint8)
        img[:, 200:] = 60  # a third of the ring is dark art
        m = np.zeros((200, 300), np.uint8)
        m[80:120, 150:230] = 255
        cleaned = np.full_like(img, 99)
        self.assertTrue((wm.flat_fill(img, cleaned, m) == cleaned).all())


if __name__ == "__main__":
    unittest.main()
