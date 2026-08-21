from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_selection_module():
    path = SCRIPTS_DIR / "select_stem_architecture.py"
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


selection = load_selection_module()


class StemSelectionTests(unittest.TestCase):
    def test_lower_flop_model_wins_when_practically_equivalent(self):
        scores = {
            "S2P": {42: 0.80, 123: 0.80, 2026: 0.80},
            "S2N": {42: 0.899, 123: 0.900, 2026: 0.901},
            "S1P": {42: 0.898, 123: 0.899, 2026: 0.900},
            "S1N": {42: 0.900, 123: 0.901, 2026: 0.902},
        }
        leader, winner, _, comparisons, eligible = (
            selection.choose_architecture(scores)
        )
        self.assertEqual(leader, "S1N")
        self.assertIn("S2N", eligible)
        self.assertTrue(comparisons["S2N"]["practically_equivalent"])
        self.assertEqual(winner, "S2N")

    def test_accuracy_leader_wins_when_others_are_not_equivalent(self):
        scores = {
            "S2P": {42: 0.80, 123: 0.80, 2026: 0.80},
            "S2N": {42: 0.85, 123: 0.85, 2026: 0.85},
            "S1P": {42: 0.86, 123: 0.86, 2026: 0.86},
            "S1N": {42: 0.90, 123: 0.90, 2026: 0.90},
        }
        leader, winner, _, _, eligible = selection.choose_architecture(scores)
        self.assertEqual(leader, "S1N")
        self.assertEqual(eligible, ["S1N"])
        self.assertEqual(winner, "S1N")

    def test_incomplete_seed_set_is_rejected(self):
        scores = {
            "S2P": {42: 0.8, 123: 0.8},
            "S2N": {42: 0.8, 123: 0.8, 2026: 0.8},
            "S1P": {42: 0.8, 123: 0.8, 2026: 0.8},
            "S1N": {42: 0.8, 123: 0.8, 2026: 0.8},
        }
        with self.assertRaises(ValueError):
            selection.choose_architecture(scores)

    def test_history_requires_exactly_thirty_ordered_epochs(self):
        fields = (
            "epoch",
            "val_accuracy",
            "val_macro_f1",
            "val_balanced_accuracy",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                for epoch in range(1, 30):
                    writer.writerow(
                        {
                            "epoch": epoch,
                            "val_accuracy": 0.9,
                            "val_macro_f1": 0.8,
                            "val_balanced_accuracy": 0.8,
                        }
                    )
            with self.assertRaises(ValueError):
                selection.read_best_validation_row(path)


if __name__ == "__main__":
    unittest.main()
