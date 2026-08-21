from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transform_functional


ROBUSTNESS_LEVELS = {
    "bin_flip": (0.01, 0.05, 0.10),
    "translation": (2, 5, 8),
    "local_missing": (8, 16, 24),
    "rotation": (2, 5, 10),
}

LEVEL_NAMES = ("light", "medium", "heavy")
BASE_RANDOM_SEED = 20260801


def _validate_image(image: np.ndarray):
    if image.shape != (64, 64) or image.dtype != np.uint8:
        raise ValueError("Expected one uint8 wafer map with shape 64 x 64.")
    if not np.isin(image, (0, 1, 2)).all():
        raise ValueError("Wafer map values must be in {0, 1, 2}.")


def _rng(source_index: int, corruption: str) -> np.random.Generator:
    corruption_index = tuple(ROBUSTNESS_LEVELS).index(corruption)
    seed_sequence = np.random.SeedSequence(
        [BASE_RANDOM_SEED, int(source_index), corruption_index]
    )
    return np.random.default_rng(seed_sequence)


def _translate(image: np.ndarray, dy: int, dx: int) -> np.ndarray:
    height, width = image.shape
    output = np.zeros_like(image)
    source_y_start = max(0, -dy)
    source_y_end = min(height, height - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(width, width - dx)
    target_y_start = max(0, dy)
    target_y_end = min(height, height + dy)
    target_x_start = max(0, dx)
    target_x_end = min(width, width + dx)
    output[target_y_start:target_y_end, target_x_start:target_x_end] = image[
        source_y_start:source_y_end,
        source_x_start:source_x_end,
    ]
    return output


def apply_corruption(
    image: np.ndarray,
    source_index: int,
    corruption: str,
    level_index: int,
) -> np.ndarray:
    _validate_image(image)
    if corruption not in ROBUSTNESS_LEVELS:
        raise ValueError(f"Unknown corruption: {corruption}")
    if level_index not in range(len(LEVEL_NAMES)):
        raise ValueError(f"Unknown level index: {level_index}")

    severity = ROBUSTNESS_LEVELS[corruption][level_index]
    rng = _rng(source_index, corruption)
    output = image.copy()

    if corruption == "bin_flip":
        active_positions = np.flatnonzero(output.reshape(-1) > 0)
        if len(active_positions) == 0:
            return output
        count = max(1, int(round(float(severity) * len(active_positions))))
        selected = rng.permutation(active_positions)[:count]
        flat = output.reshape(-1)
        flat[selected] = np.where(flat[selected] == 2, 1, 2)
        return output

    if corruption == "translation":
        directions = (
            (-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1),
        )
        direction_y, direction_x = directions[int(rng.integers(len(directions)))]
        magnitude = int(severity)
        return _translate(
            output,
            dy=direction_y * magnitude,
            dx=direction_x * magnitude,
        )

    if corruption == "local_missing":
        active_coordinates = np.argwhere(output > 0)
        if len(active_coordinates) == 0:
            return output
        center_y, center_x = active_coordinates[
            int(rng.integers(len(active_coordinates)))
        ]
        side = int(severity)
        y_start = max(0, int(center_y) - side // 2)
        x_start = max(0, int(center_x) - side // 2)
        y_end = min(output.shape[0], y_start + side)
        x_end = min(output.shape[1], x_start + side)
        y_start = max(0, y_end - side)
        x_start = max(0, x_end - side)
        output[y_start:y_end, x_start:x_end] = 0
        return output

    signed_angle = float(severity) * (-1.0 if rng.integers(2) == 0 else 1.0)
    tensor = torch.from_numpy(output).unsqueeze(0)
    rotated = transform_functional.rotate(
        tensor,
        angle=signed_angle,
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    return rotated.squeeze(0).numpy().astype(np.uint8, copy=False)
