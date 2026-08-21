from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analyze_kang_vs_highres_significance_corrected.py"
)
PROJECT_ROOT = SCRIPT_FILE.parents[2]


def load_analysis_module():
    specification = importlib.util.spec_from_file_location(
        "kang_highres_significance_corrected_for_tests", SCRIPT_FILE
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class KangHighResSignificanceCorrectedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = load_analysis_module()

    def test_canonical_seed42_has_no_tolerated_typo(self):
        path = (
            PROJECT_ROOT
            / "04_实验/metrics/20260729_highres_ce_multiseed_final_test/predictions"
            / "predictions_seed42.csv"
        )
        frame, tolerated = self.analysis.load_prediction(path, "HighRes")
        self.assertEqual(len(frame), self.analysis.SAMPLES)
        self.assertEqual(tolerated, 0)

    def test_published_seed42_typo_is_recorded_without_rewriting(self):
        path = (
            PROJECT_ROOT
            / "05_结果/predictions/20260729_highres_ce_multiseed_final_test"
            / "predictions_seed42.csv"
        )
        frame, tolerated = self.analysis.load_prediction(path, "HighRes")
        self.assertEqual(len(frame), self.analysis.SAMPLES)
        self.assertEqual(tolerated, 1)

    def test_corrected_table_uses_unambiguous_count_names(self):
        path = PROJECT_ROOT / (
            "05_结果/tables/table_kang_vs_highres_paired_significance_canonical.csv"
        )
        table = pd.read_csv(path)
        self.assertIn("mcnemar_discordant_pairs", table.columns)
        self.assertIn("macro_f1_randomization_extreme_count", table.columns)
        self.assertNotIn("macro_f1_randomization_discordant_predictions", table.columns)
        self.assertEqual(table["mcnemar_discordant_pairs"].tolist(), [331, 298, 304])


if __name__ == "__main__":
    unittest.main()
