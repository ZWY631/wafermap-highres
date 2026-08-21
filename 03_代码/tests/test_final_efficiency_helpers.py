from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_final_model_efficiency.py"
)


def load_benchmark_module():
    specification = importlib.util.spec_from_file_location(
        "final_efficiency_benchmark_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class FinalEfficiencyHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark_module()

    def test_round_aggregation_uses_sample_standard_deviation(self):
        rounds = pd.DataFrame(
            {
                "model_id": ["model"] * 3,
                "model_name": ["Model"] * 3,
                "device": ["mps"] * 3,
                "batch_size": [1] * 3,
                "mean_batch_seconds": [0.001, 0.002, 0.003],
                "batch_latency_ms": [1.0, 2.0, 3.0],
                "latency_ms_per_sample": [1.0, 2.0, 3.0],
                "samples_per_second": [1000.0, 500.0, 1000.0 / 3.0],
            }
        )

        summary = self.benchmark.aggregate_benchmark(rounds)

        self.assertAlmostEqual(summary.loc[0, "batch_latency_ms_mean"], 2.0)
        self.assertAlmostEqual(
            summary.loc[0, "batch_latency_ms_sample_std"], 1.0
        )

    def test_relative_complexity_uses_resnet_as_baseline(self):
        complexity = pd.DataFrame(
            {
                "model_id": ["resnet18_baseline", "highres_shufflenetv2_ce"],
                "parameter_count": [100, 10],
                "conv_linear_flops": [1000, 1200],
                "state_dict_memory_mib": [40.0, 4.0],
            }
        )

        result = self.benchmark.add_relative_complexity(complexity)
        final = result.loc[
            result["model_id"] == "highres_shufflenetv2_ce"
        ].iloc[0]

        self.assertTrue(
            np.isclose(final["parameter_change_percent_vs_baseline"], -90.0)
        )
        self.assertTrue(
            np.isclose(final["flops_change_percent_vs_baseline"], 20.0)
        )

    def test_mean_sd_format(self):
        self.assertEqual(
            self.benchmark.format_mean_sd(1.23456, 0.0789, 3),
            "1.235 +/- 0.079",
        )

    def test_bar_annotation_accounts_for_error_bar(self):
        figure, axis = plt.subplots()
        bars = axis.bar(["A", "B"], [2.0, 4.0], yerr=[0.5, 2.0])

        self.benchmark.annotate_bars(axis, bars, errors=[0.5, 2.0])

        self.assertGreater(axis.get_ylim()[1], 6.0)
        self.assertGreater(axis.texts[1].get_position()[1], 6.0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
