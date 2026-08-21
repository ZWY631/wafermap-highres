from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pandas.compat import pickle_compat
from scipy import interpolate
from scipy.stats import mode
from skimage import measure
from skimage.transform import radon


NUM_HANDCRAFTED_FEATURES = 59
REGION_FEATURES = 13
RADON_FEATURES_PER_STATISTIC = 20
GEOMETRY_FEATURES = 6

FEATURE_NAMES = tuple(
    [f"region_density_{index:02d}" for index in range(1, 14)]
    + [f"radon_mean_{index:02d}" for index in range(1, 21)]
    + [f"radon_std_{index:02d}" for index in range(1, 21)]
    + [
        "salient_area",
        "salient_perimeter",
        "salient_major_axis",
        "salient_minor_axis",
        "salient_eccentricity",
        "salient_solidity",
    ]
)


def load_legacy_pandas_pickle(path: Path) -> Any:
    """Load the pandas 0.x WM-811K pickle with current pandas."""
    module_aliases = {
        "pandas.indexes.base": "pandas.core.indexes.base",
        "pandas.indexes.range": "pandas.core.indexes.range",
    }

    class WM811KUnpickler(pickle_compat.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            return super().find_class(module_aliases.get(module, module), name)

    with path.open("rb") as file_handle:
        return WM811KUnpickler(file_handle, encoding="latin1").load()


def _validate_wafer_map(wafer_map: np.ndarray) -> np.ndarray:
    image = np.asarray(wafer_map)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D wafer map, got {image.shape}")
    if not np.isin(image, (0, 1, 2)).all():
        raise ValueError("Wafer map contains values outside {0, 1, 2}.")
    return image.astype(np.uint8, copy=False)


def pad_for_thirteen_regions(wafer_map: np.ndarray) -> np.ndarray:
    """Retain author-excluded tiny maps by zero-padding each side to 5."""
    image = _validate_wafer_map(wafer_map)
    target_rows = max(5, image.shape[0])
    target_columns = max(5, image.shape[1])
    if image.shape == (target_rows, target_columns):
        return image
    row_padding = target_rows - image.shape[0]
    column_padding = target_columns - image.shape[1]
    return np.pad(
        image,
        (
            (row_padding // 2, row_padding - row_padding // 2),
            (column_padding // 2, column_padding - column_padding // 2),
        ),
        mode="constant",
        constant_values=0,
    )


def _failed_die_density(region: np.ndarray) -> float:
    return 100.0 * float(np.count_nonzero(region == 2)) / float(region.size)


def extract_region_densities(wafer_map: np.ndarray) -> np.ndarray:
    image = pad_for_thirteen_regions(wafer_map)
    rows, columns = image.shape
    row_step = rows // 5
    column_step = columns // 5
    row_indices = np.arange(0, rows, row_step)
    column_indices = np.arange(0, columns, column_step)

    regions = (
        image[row_indices[0] : row_indices[1], :],
        image[:, column_indices[4] :],
        image[row_indices[4] :, :],
        image[:, column_indices[0] : column_indices[1]],
        image[row_indices[1] : row_indices[2], column_indices[1] : column_indices[2]],
        image[row_indices[1] : row_indices[2], column_indices[2] : column_indices[3]],
        image[row_indices[1] : row_indices[2], column_indices[3] : column_indices[4]],
        image[row_indices[2] : row_indices[3], column_indices[1] : column_indices[2]],
        image[row_indices[2] : row_indices[3], column_indices[2] : column_indices[3]],
        image[row_indices[2] : row_indices[3], column_indices[3] : column_indices[4]],
        image[row_indices[3] : row_indices[4], column_indices[1] : column_indices[2]],
        image[row_indices[3] : row_indices[4], column_indices[2] : column_indices[3]],
        image[row_indices[3] : row_indices[4], column_indices[3] : column_indices[4]],
    )
    return np.asarray([_failed_die_density(region) for region in regions], dtype=np.float64)


def failed_die_map(wafer_map: np.ndarray) -> np.ndarray:
    """Match the author code's change_val operation without mutating input."""
    image = pad_for_thirteen_regions(wafer_map).copy()
    image[image == 1] = 0
    return image


def _interpolate_to_twenty(values: np.ndarray) -> np.ndarray:
    source_x = np.linspace(1, values.size, values.size)
    target_x = np.linspace(1, values.size, RADON_FEATURES_PER_STATISTIC)
    function = interpolate.interp1d(source_x, values, kind="cubic")
    return np.asarray(function(target_x) / 100.0, dtype=np.float64)


def extract_radon_features(wafer_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = failed_die_map(wafer_map)
    theta = np.linspace(0.0, 180.0, max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, preserve_range=True)
    return (
        _interpolate_to_twenty(np.mean(sinogram, axis=1)),
        _interpolate_to_twenty(np.std(sinogram, axis=1)),
    )


def extract_geometry_features(wafer_map: np.ndarray) -> np.ndarray:
    image = failed_die_map(wafer_map)
    normalized_area = float(image.shape[0] * image.shape[1])
    normalized_perimeter = float(np.hypot(image.shape[0], image.shape[1]))
    labels = measure.label(image, connectivity=1, background=0)

    if labels.max() == 0:
        labels = np.ones_like(labels, dtype=np.int32)
        region_index = 0
    else:
        positive_labels = labels[labels > 0]
        largest_label = int(mode(positive_labels, axis=None, keepdims=False).mode)
        region_index = largest_label - 1

    properties = measure.regionprops(labels)
    salient = properties[region_index]
    return np.asarray(
        [
            salient.area / normalized_area,
            salient.perimeter / normalized_perimeter,
            salient.axis_major_length / normalized_perimeter,
            salient.axis_minor_length / normalized_perimeter,
            salient.eccentricity,
            salient.solidity,
        ],
        dtype=np.float64,
    )


def extract_handcrafted_features(wafer_map: np.ndarray) -> np.ndarray:
    density = extract_region_densities(wafer_map)
    radon_mean, radon_std = extract_radon_features(wafer_map)
    geometry = extract_geometry_features(wafer_map)
    features = np.concatenate((density, radon_mean, radon_std, geometry))
    if features.shape != (NUM_HANDCRAFTED_FEATURES,):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Handcrafted features contain non-finite values.")
    return features.astype(np.float32)


def extract_many(wafer_maps: Iterable[np.ndarray]) -> np.ndarray:
    rows = [extract_handcrafted_features(wafer_map) for wafer_map in wafer_maps]
    result = np.asarray(rows, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != NUM_HANDCRAFTED_FEATURES:
        raise RuntimeError(f"Unexpected feature matrix shape: {result.shape}")
    return result
