from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT_FILE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "evaluate_final_highres_ce_multiseed.py"
)


def load_evaluator_module():
    specification = importlib.util.spec_from_file_location(
        "final_evaluator_for_tests",
        SCRIPT_FILE,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class FinalEvaluatorHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator_module()

    def test_perfect_predictions_produce_finite_metrics(self):
        y_true = np.repeat(np.arange(9, dtype=np.int64), 2)
        probabilities = np.eye(9, dtype=np.float32)[y_true]
        results = {
            "test_loss": 0.0,
            "y_true": y_true,
            "y_pred": y_true.copy(),
            "probabilities": probabilities,
            "confidences": np.ones(len(y_true), dtype=np.float32),
            "elapsed_seconds": 1.0,
        }
        checkpoint_info = {
            "seed": 42,
            "run_name": "synthetic",
        }
        checkpoint = {
            "epoch": 1,
            "parameter_count": 1262397,
        }

        summary, per_class, matrix = self.evaluator.calculate_metrics(
            results,
            checkpoint_info,
            checkpoint,
            torch.device("cpu"),
        )

        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["macro_f1"], 1.0)
        self.assertEqual(len(per_class), 9)
        np.testing.assert_array_equal(matrix, np.eye(9, dtype=np.int64) * 2)

    def test_canonical_bundle_commit_and_exclusive_publication(self):
        evaluator = self.evaluator
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluator.RAW_OUTPUT_DIR = root / "canonical"
            evaluator.PREDICTIONS_DIR = root / "published" / "predictions"
            evaluator.FIGURE_PNG = (
                root
                / "published"
                / "figures"
                / "final_highres_ce_multiseed_confusion_matrix_mean.png"
            )
            evaluator.FIGURE_PDF = evaluator.FIGURE_PNG.with_suffix(".pdf")
            evaluator.PAPER_SUMMARY_TABLE = (
                root
                / "published"
                / "tables"
                / "table_final_highres_ce_multiseed_test.csv"
            )
            evaluator.PAPER_PER_CLASS_TABLE = (
                root
                / "published"
                / "tables"
                / "table_final_highres_ce_per_class_test.csv"
            )
            evaluator.COMPLETION_RECORD = (
                root / "published" / "records" / "final_test_completed.md"
            )

            staged_bundle = root / "staged_bundle"
            staged_bundle.mkdir()
            expected_paths = evaluator.expected_bundle_paths()
            for relative_path in expected_paths:
                path = staged_bundle / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"synthetic:{relative_path}\n", encoding="utf-8")

            evaluator.write_artifact_manifest(staged_bundle)
            manifest_hash = evaluator.commit_outputs(staged_bundle)

            self.assertEqual(len(manifest_hash), 64)
            self.assertTrue(evaluator.RAW_OUTPUT_DIR.is_dir())
            self.assertFalse(staged_bundle.exists())
            self.assertTrue(evaluator.PREDICTIONS_DIR.is_dir())
            self.assertTrue(evaluator.FIGURE_PNG.is_file())
            self.assertTrue(evaluator.FIGURE_PDF.is_file())
            self.assertTrue(evaluator.PAPER_SUMMARY_TABLE.is_file())
            self.assertTrue(evaluator.PAPER_PER_CLASS_TABLE.is_file())
            self.assertTrue(evaluator.COMPLETION_RECORD.is_file())


if __name__ == "__main__":
    unittest.main()
