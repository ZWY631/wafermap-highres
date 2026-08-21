from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES
from wafermap.stem_ablation_models import (
    ShuffleNetV2Stride1MaxPool,
    ShuffleNetV2Stride2NoPool,
)


def load_script(filename: str):
    path = SCRIPTS_DIR / filename
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


class StemAblationModelTests(unittest.TestCase):
    def test_stride_two_no_pool_definition(self):
        model = ShuffleNetV2Stride2NoPool()
        self.assertEqual(model.network.conv1[0].stride, (2, 2))
        self.assertIsInstance(model.network.maxpool, nn.Identity)
        self.assertEqual(model.network.fc.out_features, NUM_CLASSES)

    def test_stride_one_max_pool_definition(self):
        model = ShuffleNetV2Stride1MaxPool()
        self.assertEqual(model.network.conv1[0].stride, (1, 1))
        self.assertIsInstance(model.network.maxpool, nn.MaxPool2d)
        self.assertEqual(model.network.fc.out_features, NUM_CLASSES)

    def test_intermediate_stems_enter_stage_two_at_32_by_32(self):
        for model in (
            ShuffleNetV2Stride2NoPool(),
            ShuffleNetV2Stride1MaxPool(),
        ):
            features = model.network.conv1(torch.zeros(1, 1, 64, 64))
            features = model.network.maxpool(features)
            self.assertEqual(tuple(features.shape[-2:]), (32, 32))

    def test_parameter_counts_and_output_shapes_match(self):
        models = (
            ShuffleNetV2Stride2NoPool(),
            ShuffleNetV2Stride1MaxPool(),
        )
        for model in models:
            self.assertEqual(
                sum(parameter.numel() for parameter in model.parameters()),
                1_262_397,
            )
            with torch.inference_mode():
                output = model(torch.zeros(2, 1, 64, 64))
            self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))


class StemAblationTrainingPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s2_script = load_script(
            "train_shufflenet_stem_s2_nopool_ce_multiseed.py"
        )
        cls.s1_script = load_script(
            "train_shufflenet_stem_s1_pool_ce_multiseed.py"
        )
        cls.runner = load_script("run_stem_ablation_training.py")

    def test_run_names_are_unique_and_explicit(self):
        self.assertEqual(
            self.s2_script.EXPERIMENT.base_run_name,
            "shufflenet_v2_stem_s2_nopool_ce_full",
        )
        self.assertEqual(
            self.s1_script.EXPERIMENT.base_run_name,
            "shufflenet_v2_stem_s1_pool_ce_full",
        )

    def test_plan_contains_two_configurations_and_three_seeds(self):
        commands = self.runner.planned_commands("python-for-test")
        self.assertEqual(len(commands), 6)
        self.assertEqual(
            {(item[0].configuration_id, item[1]) for item in commands},
            {
                (configuration, seed)
                for configuration in ("s2_nopool", "s1_pool")
                for seed in (42, 123, 2026)
            },
        )

    def test_default_runner_mode_never_starts_training(self):
        with (
            mock.patch.object(self.runner, "preflight", return_value=[]),
            mock.patch.object(self.runner.subprocess, "run") as run,
        ):
            self.runner.main([])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
