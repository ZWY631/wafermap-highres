from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.robustness import LEVEL_NAMES, ROBUSTNESS_LEVELS, apply_corruption


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((64, 64), dtype=np.uint8)
        self.image[8:56, 8:56] = 1
        self.image[28:36, 28:36] = 2

    def test_protocol_has_four_corruptions_and_three_levels(self):
        self.assertEqual(len(ROBUSTNESS_LEVELS), 4)
        self.assertEqual(LEVEL_NAMES, ("light", "medium", "heavy"))
        for levels in ROBUSTNESS_LEVELS.values():
            self.assertEqual(len(levels), 3)

    def test_every_corruption_is_deterministic_and_discrete(self):
        for corruption in ROBUSTNESS_LEVELS:
            for level_index in range(3):
                first = apply_corruption(
                    self.image, 12345, corruption, level_index
                )
                second = apply_corruption(
                    self.image, 12345, corruption, level_index
                )
                np.testing.assert_array_equal(first, second)
                self.assertEqual(first.shape, (64, 64))
                self.assertEqual(first.dtype, np.uint8)
                self.assertTrue(np.isin(first, (0, 1, 2)).all())

    def test_bin_flip_levels_are_nested(self):
        changed = []
        for level_index in range(3):
            corrupted = apply_corruption(
                self.image, 88, "bin_flip", level_index
            )
            changed.append(set(np.flatnonzero(corrupted != self.image)))
        self.assertTrue(changed[0] < changed[1] < changed[2])

    def test_translation_does_not_wrap_pixels(self):
        image = np.zeros((64, 64), dtype=np.uint8)
        image[0, 0] = 2
        corrupted = apply_corruption(image, 1, "translation", 2)
        self.assertNotEqual(corrupted[-1, -1], 2)

    def test_local_missing_removes_more_at_higher_levels(self):
        nonzero_counts = [
            np.count_nonzero(
                apply_corruption(self.image, 99, "local_missing", level)
            )
            for level in range(3)
        ]
        self.assertGreater(nonzero_counts[0], nonzero_counts[1])
        self.assertGreater(nonzero_counts[1], nonzero_counts[2])


if __name__ == "__main__":
    unittest.main()
