import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.inpainting import outline  # noqa: E402

H, W = 420, 600
INK = (25, 25, 25)


def balloon(with_glyph: bool = True):
    """White page, a thick elliptical balloon outline; synthetic bar-shaped 'letters' straddle its top arc."""
    img = np.full((H, W, 3), 250, np.uint8)
    cv2.ellipse(img, (300, 330), (250, 190), 0, 0, 360, INK, 4, cv2.LINE_AA)
    arc_only = img.copy()
    if with_glyph:
        # top of the ellipse is at y = 140: three strokes cross it, one long bar hangs below it
        for x in (270, 300, 330):
            cv2.rectangle(img, (x - 3, 110), (x + 3, 185), INK, -1)
        cv2.rectangle(img, (265, 178), (336, 185), INK, -1)
    return img, arc_only


def masks(img, arc_only):
    dark = (arc_only.mean(axis=2) < 140).astype(np.uint8)
    protected = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    done = np.zeros((H, W), np.uint8)
    done[100:230, 200:400] = 255  # first mask took most of the line; the letters on the outline stayed
    done[135:190, 262:340] = 0
    return protected, done


class OutlineGlyphTest(unittest.TestCase):
    def test_fit_finds_arc_through_glyph(self):
        img, arc = balloon()
        ink = img.mean(axis=2) < 140
        fit = outline.fit_outline(ink[60:260, 120:480])
        self.assertIsNotNone(fit)
        self.assertGreater(float(np.hypot(*np.diff(fit["xy"], axis=0).T).sum()), 150)
        self.assertTrue(3.0 <= fit["width"] <= 6.0, fit["width"])

    def test_glyph_off_the_arc_removed_and_outline_redrawn(self):
        img, arc = balloon()
        protected, done = masks(img, arc)
        keep, bands, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), protected)
        self.assertGreaterEqual(n, 1)
        self.assertGreaterEqual(len(bands), 1)
        # glyph strokes above and next to the arc are marked, the arc far from the letters is not
        self.assertTrue(keep[130, 300] > 0 and keep[150, 300] > 0)
        self.assertEqual(int(keep[:, :200].sum()), 0)
        self.assertEqual(int(keep[:, 420:].sum()), 0)
        # stand-in for the inpainter: erase the grown mask with paper white, then redraw the outline
        add = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        cur = img.copy()
        cur[add > 0] = 250
        out = outline.redraw(cur, add, bands)
        xs = np.arange(260, 345)
        ys = np.round(330 - 190 * np.sqrt(1 - ((xs - 300) / 250.0) ** 2)).astype(int)
        col = np.array([out[max(y - 2, 0):y + 3, x].mean(axis=1).min() for x, y in zip(xs, ys)])
        self.assertLess(float(col.max()), 120.0, "outline stays dark and continuous under the repaired letters")
        # no letter ink left next to the arc (everything darker than paper off the arc band)
        band = cv2.dilate((arc[..., 0] < 140).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        left = (out.mean(axis=2) < 140) & ~band
        self.assertEqual(int(left[100:165, 250:350].sum()), 0)

    def test_plain_outline_is_not_touched(self):
        img, arc = balloon(with_glyph=False)
        protected, done = masks(img, arc)
        keep, bands, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), protected)
        self.assertEqual(n, 0)
        self.assertEqual(int(keep.sum()), 0)

    def test_busy_art_next_to_outline_is_kept(self):
        img, arc = balloon()
        rng = np.random.default_rng(5)
        busy = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        ink = img.mean(axis=2) < 140
        busy[ink] = INK  # same outline + letters, but on a noisy (art-like) background
        protected, done = masks(img, arc)
        img = busy
        _, _, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), protected)
        self.assertEqual(n, 0)

    def test_no_protection_no_work_and_knob(self):
        img, arc = balloon()
        _, done = masks(img, arc)
        _, _, n = outline.glyph_pieces(img, done, np.zeros((H, W), np.uint8), None)
        self.assertEqual(n, 0)
        self.assertTrue(outline.enabled())
        with mock.patch.dict(os.environ, {"CLEAN_OUTLINE_GLYPHS": "false"}):
            self.assertFalse(outline.enabled())


if __name__ == "__main__":
    unittest.main()
