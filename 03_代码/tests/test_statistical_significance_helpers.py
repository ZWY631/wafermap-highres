from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_final_vs_resnet18_significance.py"
)


def load_analysis_module():
    specification = importlib.util.spec_from_file_location(
        "statistical_significance_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class StatisticalSignificanceHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis_module()

    def test_confusion_metrics(self):
        true_labels = np.array([0, 0, 1, 1])
        predicted_labels = np.array([0, 1, 1, 1])
        confusion = self.analysis.confusion_matrix_from_labels(
            true_labels, predicted_labels, n_classes=2
        )

        accuracy, macro_f1 = self.analysis.metrics_from_confusion(confusion)

        self.assertAlmostEqual(float(accuracy), 0.75)
        self.assertAlmostEqual(float(macro_f1), (2 / 3 + 0.8) / 2)

    def test_mcnemar_exact_counts_discordant_pairs(self):
        baseline_correct = np.array([True, True, False, False, False])
        final_correct = np.array([True, False, True, True, False])

        result = self.analysis.mcnemar_exact(
            baseline_correct, final_correct
        )

        self.assertEqual(result["both_correct"], 1)
        self.assertEqual(result["baseline_only_correct"], 1)
        self.assertEqual(result["final_only_correct"], 2)
        self.assertEqual(result["both_wrong"], 1)
        self.assertEqual(result["discordant_pairs"], 3)
        self.assertAlmostEqual(result["exact_two_sided_p_value"], 1.0)

    def test_holm_adjustment_preserves_original_order(self):
        adjusted = self.analysis.holm_adjust(np.array([0.04, 0.01, 0.03]))
        np.testing.assert_allclose(adjusted, [0.06, 0.03, 0.06])

    def test_identical_predictions_have_zero_bootstrap_difference(self):
        true_labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0])

        accuracy_delta, macro_f1_delta = (
            self.analysis.paired_stratified_bootstrap(
                true_labels,
                predictions,
                predictions,
                replicates=50,
                random_seed=123,
                n_classes=2,
            )
        )

        np.testing.assert_array_equal(accuracy_delta, np.zeros(50))
        np.testing.assert_array_equal(macro_f1_delta, np.zeros(50))

    def test_identical_predictions_have_randomization_p_one(self):
        true_labels = np.array([0, 0, 1, 1])
        predictions = np.array([0, 1, 1, 0])

        deltas, p_value, extreme_count = (
            self.analysis.paired_macro_f1_randomization(
                true_labels,
                predictions,
                predictions,
                replicates=50,
                random_seed=123,
                n_classes=2,
            )
        )

        np.testing.assert_array_equal(deltas, np.zeros(50))
        self.assertEqual(extreme_count, 50)
        self.assertEqual(p_value, 1.0)


if __name__ == "__main__":
    unittest.main()
