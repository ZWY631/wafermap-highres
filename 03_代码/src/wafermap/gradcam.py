from __future__ import annotations

import torch
from torch.nn import functional as functional


def compute_gradcam(model, target_layer, image, target_class: int):
    """Return normalized Grad-CAM and logits for a single input image."""
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single NCHW image.")

    captured = {}

    def forward_hook(_module, _inputs, output):
        captured["activation"] = output
        output.retain_grad()

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if not 0 <= int(target_class) < logits.shape[1]:
            raise ValueError(f"Invalid target class: {target_class}")
        logits[0, int(target_class)].backward()
        activation = captured["activation"]
        gradient = activation.grad
        if gradient is None:
            raise RuntimeError("Target-layer gradients were not captured.")
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        cam = cam[0, 0]
        minimum = cam.min()
        maximum = cam.max()
        if float((maximum - minimum).detach()) > 0.0:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        return cam.detach().cpu(), logits.detach().cpu()
    finally:
        handle.remove()
