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


    def test_open_white_surrounding_page_uses_text_box_fallback(self):
        # Touching double balloons on white paper can make the bright connected
        # component touch the detector box edge. The old full-contour cleaner
        # silently rejected it, leaving all missed YOLO glyphs in place.
        image = np.full((420, 550, 3), 255, np.uint8)
        image[282:, :] = (195, 202, 216)
        cv2.rectangle(image, (65, 280), (510, 411), (38, 38, 38), 3)
        shape = np.zeros(image.shape[:2], np.uint8)
        cv2.ellipse(shape, (210, 160), (143, 104), 0, 0, 360, 255, -1)
        cv2.ellipse(shape, (362, 240), (128, 98), 0, 0, 360, 255, -1)
        image[shape > 0] = 255
        contours, _ = cv2.findContours(shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 0), 4)
        cv2.putText(image, "WHAT IS", (133, 134), cv2.FONT_HERSHEY_SIMPLEX,
                    .78, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "HAPPENING", (115, 167), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "ANOTHER", (300, 224), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, "SENTENCE", (294, 257), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (0, 0, 0), 2, cv2.LINE_AA)
        bubble = [65, 55, 490, 345]
        blocks = [block([110, 108, 293, 180], bubble),
                  block([286, 196, 443, 267], bubble)]
        output, count = clean_white_balloons(image, image, blocks)
        self.assertGreater(count, 0)
        self.assertGreater(int(output[134, 148].min()), 238)
        self.assertGreater(int(output[225, 322].min()), 238)
        np.testing.assert_array_equal(output[385:400, 80:500], image[385:400, 80:500])
        from modules.inpainting.white_bubble import _outline_guard
        outline = _outline_guard(image, bubble)
        self.assertFalse(np.any(np.any(output != image, axis=2) & outline))

    def test_full_interior_is_reported_as_safe_skip_candidate(self):
        image, boxes, bubble = self.page()
        details = {}
        _, count = clean_white_balloons(image, image, [block(boxes[0], bubble)], details=details)
        self.assertEqual(count, 1)
        self.assertIn(tuple(bubble), details.get("full_balloons", set()))

    def test_fallback_does_not_claim_safe_full_interior(self):
        image = np.full((420, 550, 3), 255, np.uint8)
        cv2.ellipse(image, (220, 200), (180, 130), 0, 0, 360, (0, 0, 0), 4)
        cv2.putText(image, "SPEECH", (130, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        details = {}
        clean_white_balloons(image, image, [block([100, 150, 330, 230], [40, 30, 450, 345])], details=details)
        self.assertNotIn((40, 30, 450, 345), details.get("full_balloons", set()))

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
