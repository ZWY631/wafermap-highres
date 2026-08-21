from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
SCRIPTS_DIR = CODE_DIR / "scripts"
PROJECT_ROOT = CODE_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap import controlled_ce_training as training
from wafermap.constants import NUM_CLASSES
from wafermap.models import ResNet18Baseline, ShuffleNetV2Baseline


def load_script(filename: str):
    path = SCRIPTS_DIR / filename
    specification = importlib.util.spec_from_file_location(
        path.stem,
        path,
    )
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def reference_protocol_constants():
    reference = SCRIPTS_DIR / "train_shufflenet_highres_ce.py"
    tree = ast.parse(reference.read_text(encoding="utf-8"))
    names = {
        "TRAIN_BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "NUM_EPOCHS",
        "MAX_TRAIN_BATCHES",
        "LEARNING_RATE",
        "WEIGHT_DECAY",
        "NUM_WORKERS",
    }
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in names:
            values[target.id] = ast.literal_eval(node.value)
    return values


class ControlledTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shufflenet_script = load_script(
            "train_shufflenet_standard_ce_multiseed.py"
        )
        cls.resnet_script = load_script(
            "train_resnet18_ce_multiseed.py"
        )

    def test_training_constants_match_highres_ce_reference(self):
        expected = reference_protocol_constants()
        actual = {
            name: getattr(training, name)
            for name in expected
        }
        self.assertEqual(actual, expected)

    def test_data_protocol_matches_highres_ce_reference(self):
        dataset_calls = []
        loader_calls = []
        train_transform = object()

        class FakeDataset:
            def __init__(self, split_name, transform=None):
                self.split_name = split_name
                self.transform = transform
                dataset_calls.append((split_name, transform))

        class FakeLoader:
            def __init__(self, dataset, **kwargs):
                self.dataset = dataset
                self.kwargs = kwargs
                loader_calls.append(self)

        with (
            mock.patch.object(
                training,
                "WaferMapDataset",
                FakeDataset,
            ),
            mock.patch.object(
                training,
                "build_train_transform",
                return_value=train_transform,
            ),
            mock.patch.object(training, "DataLoader", FakeLoader),
        ):
            training.build_dataloaders(seed=123)

        self.assertEqual(
            dataset_calls,
            [("train", train_transform), ("val", None)],
        )
        self.assertEqual(loader_calls[0].kwargs["batch_size"], 128)
        self.assertTrue(loader_calls[0].kwargs["shuffle"])
        self.assertEqual(
            loader_calls[0].kwargs["generator"].initial_seed(),
            123,
        )
        self.assertEqual(loader_calls[1].kwargs["batch_size"], 256)
        self.assertFalse(loader_calls[1].kwargs["shuffle"])
        for loader in loader_calls:
            self.assertEqual(loader.kwargs["num_workers"], 0)
            self.assertFalse(loader.kwargs["pin_memory"])
            self.assertFalse(loader.kwargs["drop_last"])

    def test_resnet_log_protocol_matches_existing_seed42_run(self):
        log_file = (
            PROJECT_ROOT
            / "04_实验"
            / "logs"
            / "resnet18_baseline_full.txt"
        )
        lines = log_file.read_text(encoding="utf-8").splitlines()
        header_lines = [line for line in lines if not line.startswith("epoch=")]
        old_protocol = dict(
            line.split("=", maxsplit=1)
            for line in header_lines[1:]
        )

        self.assertEqual(
            lines[0],
            self.resnet_script.EXPERIMENT.log_title,
        )
        self.assertEqual(old_protocol["seed"], "42")
        self.assertEqual(
            int(old_protocol["train_batch_size"]),
            training.TRAIN_BATCH_SIZE,
        )
        self.assertEqual(
            int(old_protocol["eval_batch_size"]),
            training.EVAL_BATCH_SIZE,
        )
        self.assertEqual(
            int(old_protocol["num_epochs"]),
            training.NUM_EPOCHS,
        )
        self.assertEqual(
            float(old_protocol["learning_rate"]),
            training.LEARNING_RATE,
        )
        self.assertEqual(
            float(old_protocol["weight_decay"]),
            training.WEIGHT_DECAY,
        )
        self.assertEqual(old_protocol["loss"], "CrossEntropyLoss")
        self.assertEqual(old_protocol["class_weight"], "None")

    def test_standard_shufflenet_preserves_standard_downsampling(self):
        model = ShuffleNetV2Baseline()
        conv1 = model.network.conv1[0]

        self.assertEqual(conv1.in_channels, 1)
        self.assertEqual(conv1.stride, (2, 2))
        self.assertIsInstance(model.network.maxpool, nn.MaxPool2d)
        self.assertEqual(model.network.fc.out_features, NUM_CLASSES)

        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))

    def test_resnet18_matches_baseline_architecture(self):
        model = ResNet18Baseline()
        self.assertEqual(model.network.conv1.in_channels, 1)
        self.assertEqual(model.network.conv1.stride, (2, 2))
        self.assertEqual(model.network.fc.out_features, NUM_CLASSES)

        with torch.inference_mode():
            output = model(torch.zeros(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, NUM_CLASSES))

    def test_run_names_follow_existing_multiseed_protocol(self):
        self.assertEqual(
            training.run_name_for_seed(
                self.shufflenet_script.EXPERIMENT.base_run_name,
                42,
            ),
            "shufflenet_v2_standard_ce_full",
        )
        self.assertEqual(
            training.run_name_for_seed(
                self.shufflenet_script.EXPERIMENT.base_run_name,
                123,
            ),
            "shufflenet_v2_standard_ce_full_seed123",
        )
        self.assertEqual(
            training.run_name_for_seed(
                self.resnet_script.EXPERIMENT.base_run_name,
                2026,
            ),
            "resnet18_baseline_full_seed2026",
        )

    def test_preflight_checks_paths_without_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = training.build_run_paths("synthetic", root)
            training.ensure_output_paths_are_available(paths)

            self.assertFalse(paths.checkpoint_dir.exists())
            self.assertFalse(paths.metrics_file.exists())
            self.assertFalse(paths.log_file.exists())

            paths.metrics_file.parent.mkdir(parents=True)
            paths.metrics_file.write_text("existing\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                training.ensure_output_paths_are_available(paths)

    def test_only_planned_seeds_are_accepted(self):
        for seed in training.SUPPORTED_SEEDS:
            args = training.parse_args(["--seed", str(seed), "--preflight"])
            self.assertEqual(args.seed, seed)

        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            training.parse_args(["--seed", "7", "--preflight"])


if __name__ == "__main__":
    unittest.main()
