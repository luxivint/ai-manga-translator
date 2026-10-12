import unittest

import numpy as np

from modules.inpainting.mask import constrain_mask_to_allowed_regions


class InpaintingMaskTests(unittest.TestCase):
    def test_mask_never_escapes_allowed_regions_or_protected_edges(self):
        mask = np.full((3, 4), 255, dtype=np.uint8)
        allowed = np.array(
            [
                [255, 255, 0, 0],
                [255, 255, 255, 0],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        protected = np.zeros_like(mask)
        protected[1, 1] = 255

        result = constrain_mask_to_allowed_regions(mask, allowed, protected)

        expected = np.array(
            [
                [255, 255, 0, 0],
                [255, 0, 255, 0],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        )
        np.testing.assert_array_equal(result, expected)

    def test_allows_no_protected_mask(self):
        mask = np.array([[0, 255], [255, 255]], dtype=np.uint8)
        allowed = np.array([[255, 255], [0, 255]], dtype=np.uint8)

        result = constrain_mask_to_allowed_regions(mask, allowed)

        np.testing.assert_array_equal(result, np.array([[0, 255], [0, 255]], dtype=np.uint8))

    def test_rejects_mismatched_mask_shapes(self):
        with self.assertRaises(ValueError):
            constrain_mask_to_allowed_regions(
                np.zeros((2, 2), dtype=np.uint8),
                np.zeros((2, 3), dtype=np.uint8),
            )


if __name__ == "__main__":
    unittest.main()
