from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "plot_final_multiseed_training_curves.py"
)


def load_plotter_module():
    specification = importlib.util.spec_from_file_location(
        "final_training_curve_plotter_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class FinalTrainingCurveHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plotter = load_plotter_module()

    def test_epoch_aggregate_uses_sample_standard_deviation(self):
        histories = {}
        for seed, values in zip(
            self.plotter.EXPECTED_SEEDS,
            ([1.0, 2.0], [2.0, 3.0], [3.0, 4.0]),
        ):
            frame = pd.DataFrame(
                {
                    "epoch": [1, 2],
                    **{
                        column: values
                        for column in self.plotter.AGGREGATE_COLUMNS
                    },
                }
            )
            histories[seed] = frame

        aggregate = self.plotter.aggregate_histories(histories)

        self.assertAlmostEqual(aggregate.loc[0, "val_macro_f1_mean"], 2.0)
        self.assertAlmostEqual(
            aggregate.loc[0, "val_macro_f1_sample_std"], 1.0
        )

    def test_paper_table_formats_percent_values(self):
        best = pd.DataFrame(
            {
                "best_epoch": [26, 27, 28],
                "best_val_loss": [0.05, 0.06, 0.07],
                "best_val_accuracy": [0.97, 0.98, 0.99],
                "best_val_macro_f1": [0.90, 0.91, 0.92],
                "best_val_balanced_accuracy": [0.88, 0.89, 0.90],
                "total_training_minutes": [210.0, 215.0, 220.0],
            }
        )

        table = self.plotter.build_paper_table(best)

        self.assertEqual(table.loc[0, "best_epoch_mean_sd"], "27.00 +/- 1.00")
        self.assertEqual(
            table.loc[0, "best_val_macro_f1_percent_mean_sd"],
            "91.0000 +/- 1.0000",
        )

    def test_mean_sample_std_uses_ddof_one(self):
        mean, sample_std = self.plotter.mean_sample_std(
            pd.Series([1.0, 2.0, 3.0])
        )
        self.assertEqual(mean, 2.0)
        self.assertTrue(np.isclose(sample_std, 1.0))


if __name__ == "__main__":
    unittest.main()
