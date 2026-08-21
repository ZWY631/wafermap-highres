from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap import controlled_ce_training as training
from wafermap import lightweight_baselines as baselines
from wafermap.constants import NUM_CLASSES
from wafermap.lightweight_baselines import (
    EfficientNetB0Baseline,
    MobileNetV3SmallBaseline,
)


def load_script(filename: str):
    path = SCRIPTS_DIR / filename
    specification = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


class LightweightBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mobile_script = load_script(
            "train_mobilenet_v3_small_ce_multiseed.py"
        )
        cls.efficient_script = load_script(
            "train_efficientnet_b0_ce_multiseed.py"
        )

    def test_mobilenet_v3_small_input_and_output(self):
        model = MobileNetV3SmallBaseline().eval()
        self.assertEqual(model.network.features[0][0].in_channels, 1)
        self.assertEqual(model.network.classifier[-1].out_features, NUM_CLASSES)
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))

    def test_efficientnet_b0_input_and_output(self):
        model = EfficientNetB0Baseline().eval()
        self.assertEqual(model.network.features[0][0].in_channels, 1)
        self.assertEqual(model.network.classifier[-1].out_features, NUM_CLASSES)
        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))

    def test_models_are_randomly_initialized(self):
        with mock.patch.object(
            baselines,
            "mobilenet_v3_small",
            wraps=baselines.mobilenet_v3_small,
        ) as mobile_builder:
            MobileNetV3SmallBaseline()
            mobile_builder.assert_called_once_with(weights=None)

        with mock.patch.object(
            baselines,
            "efficientnet_b0",
            wraps=baselines.efficientnet_b0,
        ) as efficient_builder:
            EfficientNetB0Baseline()
            efficient_builder.assert_called_once_with(weights=None)

    def test_run_names_are_stable(self):
        self.assertEqual(
            training.run_name_for_seed(
                self.mobile_script.EXPERIMENT.base_run_name,
                42,
            ),
            "mobilenet_v3_small_ce_full",
        )
        self.assertEqual(
            training.run_name_for_seed(
                self.mobile_script.EXPERIMENT.base_run_name,
                123,
            ),
            "mobilenet_v3_small_ce_full_seed123",
        )
        self.assertEqual(
            training.run_name_for_seed(
                self.efficient_script.EXPERIMENT.base_run_name,
                2026,
            ),
            "efficientnet_b0_ce_full_seed2026",
        )


if __name__ == "__main__":
    unittest.main()
