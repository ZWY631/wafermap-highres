from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES
from wafermap.stacking_reproduction import (
    HandcraftedFNN,
    VGG16WaferCNN,
    WaferBinaryDataset,
    fit_meta_ridge,
    make_lot_grouped_oof_folds,
)


class StackingReproductionTests(unittest.TestCase):
    def test_author_base_model_shapes(self):
        mfe = HandcraftedFNN()
        cnn = VGG16WaferCNN()
        with torch.inference_mode():
            self.assertEqual(tuple(mfe(torch.zeros(2, 59)).shape), (2, NUM_CLASSES))
            self.assertEqual(
                tuple(cnn(torch.zeros(2, 1, 64, 64)).shape),
                (2, NUM_CLASSES),
            )
        self.assertEqual(cnn.network.features[0].in_channels, 1)
        self.assertEqual(cnn.network.features[0].stride, (1, 1))
        self.assertEqual(cnn.network.classifier[-1].in_features, 512)
        self.assertEqual(cnn.network.classifier[-1].out_features, NUM_CLASSES)

    def test_cnn_dataset_uses_failed_dies_and_author_centering(self):
        images = np.asarray(
            [
                [[0, 1], [2, 0]],
                [[2, 2], [1, 0]],
            ],
            dtype=np.uint8,
        )
        dataset = WaferBinaryDataset(
            images,
            np.asarray([1, 0]),
            np.asarray([3, 4]),
        )
        image, label = dataset[0]
        np.testing.assert_array_equal(
            image.numpy(),
            np.asarray([[[0.5, 0.5], [-0.5, -0.5]]], dtype=np.float32),
        )
        self.assertEqual(int(label), 3)

    def test_grouped_oof_has_exact_coverage_and_no_lot_leakage(self):
        labels = np.tile(np.arange(NUM_CLASSES), 12)
        groups = np.asarray([f"lot_{index // NUM_CLASSES}" for index in range(len(labels))])
        folds = make_lot_grouped_oof_folds(labels, groups, seed=42, n_splits=3)
        coverage = np.zeros(len(labels), dtype=np.int64)
        for train_indices, holdout_indices in folds:
            coverage[holdout_indices] += 1
            self.assertFalse(
                set(groups[train_indices]).intersection(groups[holdout_indices])
            )
        np.testing.assert_array_equal(coverage, np.ones(len(labels), dtype=np.int64))

    def test_meta_ridge_accepts_two_nine_class_probability_vectors(self):
        rng = np.random.default_rng(42)
        labels = np.tile(np.arange(NUM_CLASSES), 20)
        first = rng.dirichlet(np.ones(NUM_CLASSES), size=len(labels))
        second = rng.dirichlet(np.ones(NUM_CLASSES), size=len(labels))
        model = fit_meta_ridge(np.concatenate((first, second), axis=1), labels)
        self.assertEqual(model.coefficient.shape, (NUM_CLASSES, NUM_CLASSES * 2))
        self.assertEqual(model.intercept.shape, (NUM_CLASSES,))
        probabilities = model.predict_probabilities(np.concatenate((first, second), axis=1))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()

