from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "synthesize_final_evidence.py"
)


def load_synthesis_module():
    specification = importlib.util.spec_from_file_location(
        "final_evidence_synthesis_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class FinalEvidenceSynthesisHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthesis = load_synthesis_module()

    def test_percentage_change(self):
        self.assertAlmostEqual(
            self.synthesis.percentage_change(80.0, 100.0), -20.0
        )
        self.assertAlmostEqual(
            self.synthesis.percentage_change(125.0, 100.0), 25.0
        )

    def test_percentage_change_rejects_zero_baseline(self):
        with self.assertRaises(ValueError):
            self.synthesis.percentage_change(1.0, 0.0)

    def test_claim_table_contains_supported_and_prohibited_claims(self):
        evidence = {
            "selection": {
                "no_eca_validation_macro_f1_percent_mean": 90.4,
                "eca_validation_macro_f1_percent_mean": 90.3,
                "no_eca_minus_eca_macro_f1_percentage_points": 0.1,
            },
            "final_test": {
                "accuracy_percent_mean": 98.0,
                "accuracy_percent_sample_std": 0.1,
                "macro_f1_percent_mean": 90.0,
                "macro_f1_percent_sample_std": 0.2,
            },
            "final_minus_baseline": {
                "accuracy_percentage_points": 0.2,
                "macro_f1_percentage_points": 1.2,
                "accuracy_largest_holm_adjusted_p": 0.02,
                "macro_f1_largest_holm_adjusted_p": 0.03,
            },
            "complexity_and_mps": {
                "parameter_reduction_percent": 88.7,
                "final_parameter_count": 1_262_397,
                "resnet18_parameter_count": 11_174_857,
                "state_dict_footprint_reduction_percent": 88.6,
                "flops_change_percent": 25.8,
                "mps_batch1_latency_change_percent": 178.7,
            },
            "error_analysis": {
                "samples_wrong_all_three_seeds": 381,
                "samples_unanimously_wrong_same_class": 359,
            },
        }

        claims = self.synthesis.build_claims_table(evidence)

        self.assertIn("supported", set(claims["status"]))
        self.assertIn("not_supported", set(claims["status"]))
        speed_claim = claims.loc[claims["claim_id"].eq("C09")].iloc[0]
        self.assertEqual(speed_claim["status"], "not_supported")


if __name__ == "__main__":
    unittest.main()
