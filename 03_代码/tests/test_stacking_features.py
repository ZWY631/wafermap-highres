from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import interpolate, stats
from skimage import measure
from skimage.transform import radon


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.stacking_features import (
    FEATURE_NAMES,
    NUM_HANDCRAFTED_FEATURES,
    extract_handcrafted_features,
    extract_region_densities,
    failed_die_map,
    pad_for_thirteen_regions,
)


def author_reference(wafer_map):
    x = wafer_map.copy()
    rows, columns = x.shape
    row_indices = np.arange(0, rows, rows // 5)
    column_indices = np.arange(0, columns, columns // 5)
    regions = (
        x[row_indices[0] : row_indices[1], :],
        x[:, column_indices[4] :],
        x[row_indices[4] :, :],
        x[:, column_indices[0] : column_indices[1]],
        x[row_indices[1] : row_indices[2], column_indices[1] : column_indices[2]],
        x[row_indices[1] : row_indices[2], column_indices[2] : column_indices[3]],
        x[row_indices[1] : row_indices[2], column_indices[3] : column_indices[4]],
        x[row_indices[2] : row_indices[3], column_indices[1] : column_indices[2]],
        x[row_indices[2] : row_indices[3], column_indices[2] : column_indices[3]],
        x[row_indices[2] : row_indices[3], column_indices[3] : column_indices[4]],
        x[row_indices[3] : row_indices[4], column_indices[1] : column_indices[2]],
        x[row_indices[3] : row_indices[4], column_indices[2] : column_indices[3]],
        x[row_indices[3] : row_indices[4], column_indices[3] : column_indices[4]],
    )
    density = np.asarray([100 * np.sum(item == 2) / item.size for item in regions])
    x[x == 1] = 0
    theta = np.linspace(0.0, 180.0, max(x.shape), endpoint=False)
    sinogram = radon(x, theta=theta, preserve_range=True)

    def interpolate_twenty(values):
        source = np.linspace(1, values.size, values.size)
        target = np.linspace(1, values.size, 20)
        return interpolate.interp1d(source, values, kind="cubic")(target) / 100

    radon_mean = interpolate_twenty(sinogram.mean(axis=1))
    radon_std = interpolate_twenty(sinogram.std(axis=1))
    labels = measure.label(x, connectivity=1, background=0)
    if labels.max() == 0:
        labels[labels == 0] = 1
        region_index = 0
    else:
        region_index = int(
            stats.mode(labels[labels > 0], axis=None, keepdims=False).mode
        ) - 1
    salient = measure.regionprops(labels)[region_index]
    norm_area = x.shape[0] * x.shape[1]
    norm_perimeter = np.hypot(*x.shape)
    geometry = np.asarray(
        [
            salient.area / norm_area,
            salient.perimeter / norm_perimeter,
            salient.axis_major_length / norm_perimeter,
            salient.axis_minor_length / norm_perimeter,
            salient.eccentricity,
            salient.solidity,
        ]
    )
    return np.concatenate((density, radon_mean, radon_std, geometry)).astype(np.float32)


class StackingFeatureTests(unittest.TestCase):
    def test_feature_schema_is_exactly_59(self):
        self.assertEqual(NUM_HANDCRAFTED_FEATURES, 59)
        self.assertEqual(len(FEATURE_NAMES), 59)
        self.assertEqual(len(set(FEATURE_NAMES)), 59)

    def test_optimized_implementation_matches_author_reference(self):
        rng = np.random.default_rng(42)
        for shape in ((25, 27), (39, 31), (64, 64)):
            image = np.zeros(shape, dtype=np.uint8)
            image[rng.random(shape) < 0.55] = 1
            image[rng.random(shape) < 0.08] = 2
            expected = author_reference(image)
            actual = extract_handcrafted_features(image)
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)

    def test_none_map_is_finite_and_input_is_not_mutated(self):
        image = np.ones((25, 25), dtype=np.uint8)
        original = image.copy()
        features = extract_handcrafted_features(image)
        np.testing.assert_array_equal(image, original)
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(features.shape, (59,))

    def test_failed_die_map_removes_normal_dies_only(self):
        image = np.asarray(
            [
                [0, 1, 2, 0, 1],
                [2, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 2, 0, 0, 1],
                [0, 1, 2, 1, 0],
            ]
        )
        converted = failed_die_map(image)
        np.testing.assert_array_equal(converted, np.where(image == 1, 0, image))
        self.assertEqual(int(np.count_nonzero(converted == 2)), 4)

    def test_region_order_matches_author_definition(self):
        image = np.zeros((25, 25), dtype=np.uint8)
        image[:5, :] = 2
        density = extract_region_densities(image)
        self.assertEqual(density[0], 100.0)
        self.assertEqual(density[2], 0.0)
        self.assertEqual(density[4], 0.0)

    def test_invalid_values_are_rejected(self):
        image = np.zeros((10, 10), dtype=np.uint8)
        image[0, 0] = 3
        with self.assertRaises(ValueError):
            extract_handcrafted_features(image)

    def test_author_excluded_tiny_map_is_retained_by_documented_padding(self):
        image = np.ones((3, 4), dtype=np.uint8)
        padded = pad_for_thirteen_regions(image)
        self.assertEqual(padded.shape, (5, 5))
        features = extract_handcrafted_features(image)
        self.assertEqual(features.shape, (59,))
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:13], np.zeros(13))


if __name__ == "__main__":
    unittest.main()
