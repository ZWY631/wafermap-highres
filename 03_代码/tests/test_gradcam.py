from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


CODE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = CODE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.gradcam import compute_gradcam


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(4, 3)

    def forward(self, image):
        features = torch.relu(self.features(image))
        return self.classifier(self.pool(features).flatten(1))


class GradCamTests(unittest.TestCase):
    def test_gradcam_has_input_shape_and_unit_range(self):
        torch.manual_seed(3)
        model = TinyModel().eval()
        image = torch.rand(1, 1, 16, 16)
        cam, logits = compute_gradcam(model, model.features, image, 1)
        self.assertEqual(tuple(cam.shape), (16, 16))
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertGreaterEqual(float(cam.min()), 0.0)
        self.assertLessEqual(float(cam.max()), 1.0)

    def test_hook_is_removed_after_each_call(self):
        model = TinyModel().eval()
        image = torch.rand(1, 1, 16, 16)
        before = len(model.features._forward_hooks)
        compute_gradcam(model, model.features, image, 0)
        self.assertEqual(len(model.features._forward_hooks), before)

    def test_invalid_batch_is_rejected(self):
        model = TinyModel().eval()
        with self.assertRaises(ValueError):
            compute_gradcam(model, model.features, torch.rand(2, 1, 16, 16), 0)


if __name__ == "__main__":
    unittest.main()
