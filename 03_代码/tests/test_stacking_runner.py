from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


CODE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CODE_DIR / "scripts"


def load_script(filename):
    path = SCRIPTS_DIR / filename
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class StackingRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script("run_kang_stacking_multiseed.py")
        cls.training = load_script("train_kang_stacking_seed.py")
        cls.evaluator = load_script("evaluate_kang_stacking_test.py")

    def test_multiseed_plan_contains_three_explicit_seeds(self):
        commands = self.runner.planned_commands("python-for-test")
        self.assertEqual(len(commands), 3)
        self.assertEqual(
            [command[command.index("--seed") + 1] for command in commands],
            ["42", "123", "2026"],
        )
        self.assertTrue(all("--preflight" in command for command in commands))

    def test_default_multiseed_mode_never_executes_training(self):
        commands = self.runner.planned_commands("python-for-test", execute=False)
        self.assertTrue(all("--execute" not in command for command in commands))

    def test_execute_mode_is_explicit(self):
        commands = self.runner.planned_commands(
            "python-for-test", seeds=(123,), execute=True
        )
        self.assertEqual(len(commands), 1)
        self.assertIn("--execute", commands[0])

    def test_training_script_does_not_import_or_name_test_dataset(self):
        source = (SCRIPTS_DIR / "train_kang_stacking_seed.py").read_text(encoding="utf-8")
        self.assertNotIn('eq("test")', source)
        self.assertNotIn("WaferMapDataset(\"test\")", source)

    def test_preflight_dispatch_cannot_call_execute(self):
        fake_metadata = mock.MagicMock()
        with (
            mock.patch.object(
                self.training,
                "load_and_validate_inputs",
                return_value=(fake_metadata, object(), object(), {"artifact_sha256": {}}),
            ),
            mock.patch.object(self.training, "preflight") as preflight,
            mock.patch.object(self.training, "execute") as execute,
        ):
            self.training.main(["--seed", "42", "--preflight"])
        preflight.assert_called_once()
        execute.assert_not_called()

    def test_stop_after_requires_execute(self):
        with self.assertRaises(SystemExit):
            self.training.parse_args(
                ["--seed", "42", "--preflight", "--stop-after", "fold0_cnn"]
            )

    def test_execute_accepts_a_frozen_stage_boundary(self):
        args = self.training.parse_args(
            ["--seed", "42", "--execute", "--stop-after", "fold1_cnn"]
        )
        self.assertTrue(args.execute)
        self.assertEqual(args.stop_after, "fold1_cnn")

    def test_pause_after_stage_only_matches_requested_boundary(self):
        self.assertFalse(self.training.pause_after_stage("fold0_mfe", "fold0_cnn"))
        with mock.patch("builtins.print") as output:
            self.assertTrue(
                self.training.pause_after_stage("fold0_cnn", "fold0_cnn")
            )
        self.assertGreaterEqual(output.call_count, 3)

    def test_test_evaluator_preflight_does_not_load_test_metadata(self):
        with (
            mock.patch.object(
                self.evaluator,
                "validate_completed_runs",
                return_value={42: "missing", 123: "missing", 2026: "missing"},
            ),
            mock.patch.object(self.evaluator.pd, "read_csv") as read_csv,
            mock.patch.object(self.evaluator, "execute") as execute,
        ):
            self.evaluator.main(["--preflight"])
        read_csv.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
