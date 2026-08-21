from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_final_errors.py"
)


def load_analysis_module():
    specification = importlib.util.spec_from_file_location(
        "error_analysis_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class ErrorAnalysisHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis_module()

    def test_consensus_counts_three_seed_errors(self):
        frame = pd.DataFrame(
            {
                "source_index": [100, 200],
                "label": ["Loc", "none"],
                "correct_seed42": [False, True],
                "correct_seed123": [False, True],
                "correct_seed2026": [False, False],
                "predicted_label_seed42": ["none", "none"],
                "predicted_label_seed123": ["none", "none"],
                "predicted_label_seed2026": ["none", "Edge-Loc"],
                "confidence_seed42": [0.8, 0.9],
                "confidence_seed123": [0.7, 0.8],
                "confidence_seed2026": [0.9, 0.6],
            }
        )

        result = self.analysis.build_sample_consensus(frame)
        first = result.loc[result["source_index"] == 100].iloc[0]
        second = result.loc[result["source_index"] == 200].iloc[0]

        self.assertEqual(first["wrong_seed_count"], 3)
        self.assertTrue(first["all_seeds_wrong"])
        self.assertTrue(first["unanimous_wrong"])
        self.assertAlmostEqual(first["mean_prediction_confidence"], 0.8)
        self.assertEqual(second["wrong_seed_count"], 1)
        self.assertFalse(second["unanimous_prediction"])

    def test_representative_selection_is_deterministic(self):
        rows = []
        source_index = 1
        for true_class, predicted_class in self.analysis.REPRESENTATIVE_PAIRS:
            for confidence in (0.1, 0.2, 0.3, 0.4, 0.5):
                rows.append(
                    {
                        "source_index": source_index,
                        "label": true_class,
                        "unanimous_predicted_label": predicted_class,
                        "unanimous_wrong": True,
                        "mean_prediction_confidence": confidence,
                    }
                )
                source_index += 1

        selected = self.analysis.select_representative_samples(
            pd.DataFrame(rows)
        )

        self.assertEqual(len(selected), 12)
        first_pair = selected.loc[selected["pair_order"] == 0]
        self.assertEqual(
            first_pair["mean_prediction_confidence"].tolist(),
            [0.2, 0.3, 0.4],
        )

    def test_source_index_is_not_used_as_image_row(self):
        combined = pd.DataFrame(
            {
                "source_index": [1000, 2000],
                "label": ["Loc", "none"],
            }
        )
        metadata = pd.DataFrame(
            {
                "source_index": [1000, 2000],
                "image_row": [4, 9],
            }
        )
        first_seed = self.analysis.EXPECTED_SEEDS[0]
        predictions = {}
        for seed in self.analysis.EXPECTED_SEEDS:
            predictions[seed] = pd.DataFrame(
                {
                    "source_index": [1000, 2000],
                    "predicted_label": ["none", "none"],
                    "confidence": [0.8, 0.9],
                    "correct": [False, True],
                }
            )
        for column in self.analysis.KEY_COLUMNS:
            if column not in combined:
                combined[column] = 0
        predictions[first_seed] = combined.merge(
            predictions[first_seed], on="source_index"
        )
        for seed in self.analysis.EXPECTED_SEEDS[1:]:
            for column in self.analysis.KEY_COLUMNS:
                if column not in predictions[seed]:
                    predictions[seed][column] = combined[column]

        result = self.analysis.combine_predictions(predictions, metadata)
        self.assertEqual(result["image_row"].tolist(), [4, 9])


if __name__ == "__main__":
    unittest.main()
