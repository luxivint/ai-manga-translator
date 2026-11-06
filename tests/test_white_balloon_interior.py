"""Regression tests for whole-interior white speech balloon cleanup."""
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

from modules.inpainting.white_bubble import clean_white_balloons


def block(box, bubble, translation="MERHABA"):
    return SimpleNamespace(xyxy=np.asarray(box), bubble_xyxy=np.asarray(bubble),
                           translation=translation, text_class="text_bubble")


class WhiteBalloonInteriorTests(unittest.TestCase):
    def page(self, compound=False):
        im = np.full((640, 900, 3), 220, np.uint8)
        cv2.rectangle(im, (40, 415), (840, 610), (80, 80, 80), 2)
        if compound:
            cv2.ellipse(im, (315, 220), (225, 175), 0, 0, 360, (255,)*3, -1)
            cv2.ellipse(im, (545, 330), (220, 166), 0, 0, 360, (255,)*3, -1)
            m = np.zeros(im.shape[:2], np.uint8)
            cv2.ellipse(m, (315, 220), (225, 175), 0, 0, 360, 255, -1)
            cv2.ellipse(m, (545, 330), (220, 166), 0, 0, 360, 255, -1)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnts, -1, (4,)*3, 5)
            boxes = [[220, 140, 405, 290], [470, 260, 680, 392]]
            balloon = [85, 38, 774, 505]
            cv2.putText(im, "HELLO WORLD", (230, 205), cv2.FONT_HERSHEY_SIMPLEX,
                        .9, (0,)*3, 3, cv2.LINE_AA)
            cv2.putText(im, "HOW ARE YOU", (467, 327), cv2.FONT_HERSHEY_SIMPLEX,
                        .8, (0,)*3, 3, cv2.LINE_AA)
        else:
            cv2.ellipse(im, (450, 300), (300, 205), 0, 0, 360, (255,)*3, -1)
            cv2.ellipse(im, (450, 300), (300, 205), 0, 0, 360, (0,)*3, 5)
            boxes = [[280, 200, 620, 390]]
            balloon = [145, 92, 755, 510]
            cv2.putText(im, "HELLO WORLD", (302, 263),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,)*3, 3, cv2.LINE_AA)
        return im, boxes, balloon

    def test_double_lobe_and_outline_protection(self):
        image, boxes, bubble = self.page(True)
        out, count = clean_white_balloons(image, image,
                                         [block(b, bubble) for b in boxes])
        self.assertEqual(count, 1)
        self.assertGreater(out[203, 285].min(), 240)
        self.assertGreater(out[325, 500].min(), 240)
        changed = np.any(out != image, axis=2)
        shape = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(shape, (315, 220), (225, 175), 0, 0, 360, 255, -1)
        cv2.ellipse(shape, (545, 330), (220, 166), 0, 0, 360, 255, -1)
        cnts, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        border = np.zeros_like(shape)
        cv2.drawContours(border, cnts, -1, 255, 5)
        self.assertFalse(np.any(changed & (border > 0)))
        np.testing.assert_array_equal(out[415, 40:845], image[415, 40:845])

    def test_single_balloon_is_cleaned(self):
        image, boxes, bubble = self.page()
        out, count = clean_white_balloons(image, image, [block(boxes[0], bubble)])
        self.assertEqual(count, 1)
        self.assertGreater(out[257, 318].min(), 230)

    def test_untranslated_source_is_preserved(self):
        image, boxes, bubble = self.page(True)
        out, count = clean_white_balloons(image, image,
            [block(boxes[0], bubble), block(boxes[1], bubble, "")])
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(out, image)

    def test_open_white_page_is_not_wiped(self):
        image = np.full((400, 500, 3), 255, np.uint8)
        cv2.putText(image, "HELLO", (145, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,)*3, 3)
        out, count = clean_white_balloons(
            image, image, [block([130, 160, 300, 240], [70, 50, 430, 345])])
        self.assertEqual(count, 0)
        np.testing.assert_array_equal(out, image)

    def test_coloured_balloon_is_not_wiped(self):
        image, boxes, bubble = self.page()
        image[np.all(image > 240, axis=2)] = (170, 180, 240)
        _, count = clean_white_balloons(image, image, [block(boxes[0], bubble)])
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
