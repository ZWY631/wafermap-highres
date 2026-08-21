from __future__ import annotations

import sys
import importlib.util
import unittest
from pathlib import Path

import numpy as np
from torch.utils.data import RandomSampler, WeightedRandomSampler


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap import imbalance_training as training
from wafermap.constants import NUM_CLASSES


def load_runner():
    path = CODE_DIR / "scripts" / "run_imbalance_training.py"
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class ImbalanceTrainingTests(unittest.TestCase):
    def test_architecture_freeze_selects_s1n(self):
        payload, actual_hash = training.validate_selected_architecture()
        self.assertEqual(payload["selected_architecture_id"], "S1N")
        self.assertFalse(payload["test_inputs_used"])
        self.assertEqual(actual_hash, training.EXPECTED_FREEZE_SHA256)

    def test_train_class_counts_match_frozen_dataset(self):
        dataset = training.WaferMapDataset("train")
        counts = training.calculate_class_counts(dataset.labels)
        self.assertEqual(
            counts.tolist(),
            [3006, 389, 3633, 6776, 2515, 105, 606, 835, 103204],
        )
        self.assertEqual(int(counts.sum()), 121_069)

    def test_inverse_frequency_formula_equalizes_class_contribution(self):
        counts = np.asarray(
            [3006, 389, 3633, 6776, 2515, 105, 606, 835, 103204],
            dtype=np.int64,
        )
        weights = training.inverse_frequency_weights(counts)
        expected = counts.sum() / (NUM_CLASSES * counts)
        np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-12)
        contributions = counts * weights
        np.testing.assert_allclose(
            contributions,
            np.full(NUM_CLASSES, counts.sum() / NUM_CLASSES),
            rtol=0,
            atol=1e-9,
        )
        self.assertAlmostEqual(float(weights.min()), 0.13034486174093166)
        self.assertAlmostEqual(float(weights.max()), 128.1153439153439)

    def test_weighted_ce_uses_shuffle_without_weighted_sampler(self):
        train_loader, _, counts, weights = training.build_dataloaders(
            "weighted_ce",
            42,
        )
        self.assertIsInstance(train_loader.sampler, RandomSampler)
        self.assertNotIsInstance(train_loader.sampler, WeightedRandomSampler)
        self.assertEqual(len(train_loader.dataset), int(counts.sum()))
        self.assertEqual(weights.shape, (NUM_CLASSES,))

    def test_balanced_sampler_has_replacement_and_fixed_epoch_size(self):
        train_loader, _, counts, _ = training.build_dataloaders(
            "balanced_sampler",
            42,
        )
        sampler = train_loader.sampler
        self.assertIsInstance(sampler, WeightedRandomSampler)
        self.assertTrue(sampler.replacement)
        self.assertEqual(sampler.num_samples, int(counts.sum()))
        self.assertEqual(len(train_loader.dataset), int(counts.sum()))

    def test_sampler_sequence_is_reproducible_for_same_seed(self):
        first = training.build_dataloaders("balanced_sampler", 123)[0]
        second = training.build_dataloaders("balanced_sampler", 123)[0]
        first_indices = list(iter(first.sampler))[:100]
        second_indices = list(iter(second.sampler))[:100]
        self.assertEqual(first_indices, second_indices)

    def test_run_names_are_unique(self):
        names = {
            training.run_name_for_seed(strategy, seed)
            for strategy in training.SUPPORTED_STRATEGIES
            for seed in training.SUPPORTED_SEEDS
        }
        self.assertEqual(len(names), 6)
        self.assertIn(
            "shufflenet_v2_highres_weighted_ce_full_seed2026",
            names,
        )
        self.assertIn(
            "shufflenet_v2_highres_balanced_sampler_full_seed123",
            names,
        )

    def test_validation_criterion_is_unweighted_for_both_strategies(self):
        import torch

        counts = np.asarray(
            [3006, 389, 3633, 6776, 2515, 105, 606, 835, 103204],
            dtype=np.int64,
        )
        weights = training.inverse_frequency_weights(counts)
        for strategy in training.SUPPORTED_STRATEGIES:
            train_criterion, validation_criterion = training.build_criteria(
                strategy,
                weights,
                torch.device("cpu"),
            )
            if strategy == "weighted_ce":
                self.assertIsNotNone(train_criterion.weight)
            else:
                self.assertIsNone(train_criterion.weight)
            self.assertIsNone(validation_criterion.weight)

    def test_runner_can_plan_the_three_session_groups(self):
        runner = load_runner()
        first_group = runner.planned_commands(
            python_executable="python-for-test",
            strategies=("weighted_ce",),
            seeds=(42, 123, 2026),
        )
        second_group = runner.planned_commands(
            python_executable="python-for-test",
            strategies=("balanced_sampler",),
            seeds=(42,),
        )
        third_group = runner.planned_commands(
            python_executable="python-for-test",
            strategies=("balanced_sampler",),
            seeds=(123, 2026),
        )
        self.assertEqual(len(first_group), 3)
        self.assertEqual(len(second_group), 1)
        self.assertEqual(len(third_group), 2)


if __name__ == "__main__":
    unittest.main()
