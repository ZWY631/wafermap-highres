#!/usr/bin/env python3
"""Generate the frozen HighRes ShuffleNetV2 Grad-CAM audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "03_代码" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wafermap.constants import NUM_CLASSES, WM811K_CLASS_NAMES
from wafermap.gradcam import compute_gradcam
from wafermap.models_improved import ShuffleNetV2HighRes


ANALYSIS_ID = "20260801_highres_gradcam"
PROTOCOL_FILE = PROJECT_ROOT / "00_项目管理" / "20260801_GradCAM可解释性冻结协议.md"
FREEZE_MANIFEST = PROJECT_ROOT / "00_项目管理" / "20260801_五模型统一评估冻结清单.json"
CHECKPOINT_FILE = PROJECT_ROOT / "04_实验" / "checkpoints" / "shufflenet_v2_highres_ce_full" / "best.pt"
PREDICTION_FILE = PROJECT_ROOT / "04_实验" / "metrics" / "20260729_highres_ce_multiseed_final_test" / "predictions" / "predictions_seed42.csv"
ERROR_SAMPLE_FILE = PROJECT_ROOT / "04_实验" / "metrics" / "20260729_highres_ce_multiseed_error_analysis" / "representative_error_samples.csv"
IMAGE_FILE = PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64" / "images_uint8.npy"
METADATA_FILE = PROJECT_ROOT / "02_数据" / "processed" / "wm811k_labeled_64x64" / "metadata.csv"

OUTPUT_DIR = PROJECT_ROOT / "04_实验" / "metrics" / ANALYSIS_ID
ARRAY_DIR = OUTPUT_DIR / "cam_arrays"
FIGURE_DIR = PROJECT_ROOT / "05_结果" / "figures" / "model_results"
INDIVIDUAL_DIR = FIGURE_DIR / "highres_gradcam_samples"
COMPOSITE_PNG = FIGURE_DIR / "highres_gradcam_representative_samples.png"
COMPOSITE_PDF = FIGURE_DIR / "highres_gradcam_representative_samples.pdf"

EXPECTED_PROTOCOL_SHA256 = "f1b1a38e6289482924dd75093a6bc54cc1768078e7389801ee299f514f250d12"
EXPECTED_FREEZE_SHA256 = "2a97b07434239589da586aaa5b836d3b1aaba8d957c40cc8f5d5e54fd1139d5f"
EXPECTED_CHECKPOINT_SHA256 = "cafebb6a76ef566ea05fa47e7745677c35cb3abbec87b1de468c0898b02c67bc"
EXPECTED_PREDICTION_SHA256 = "da5e5f43339e538efd40797de3e2cd6abbacb7c5fec07cac956d72aa32cdfc21"
EXPECTED_ERROR_SAMPLE_SHA256 = "f0ea6f7f1c70994956afc2e8b68c2ddf23700119cd82c2c9f0ca1e445ea5d465"
EXPECTED_IMAGE_SHA256 = "ec496dd8a5cdb2f83c54a048497d1f6d77e64ae74efd495c36456629c9cf9aa3"
EXPECTED_METADATA_SHA256 = "b6b1e0c2d819c60aa879e50015408161324616cbe5d928f4001d6609a7c9e681"

TARGET_CLASSES = ("Edge-Loc", "Loc", "Scratch")
NONE_CLASS_ID = WM811K_CLASS_NAMES.index("none")
EXPECTED_TEST_SAMPLES = 25943

INPUT_HASHES = {
    PROTOCOL_FILE: EXPECTED_PROTOCOL_SHA256,
    FREEZE_MANIFEST: EXPECTED_FREEZE_SHA256,
    CHECKPOINT_FILE: EXPECTED_CHECKPOINT_SHA256,
    PREDICTION_FILE: EXPECTED_PREDICTION_SHA256,
    ERROR_SAMPLE_FILE: EXPECTED_ERROR_SAMPLE_SHA256,
    IMAGE_FILE: EXPECTED_IMAGE_SHA256,
    METADATA_FILE: EXPECTED_METADATA_SHA256,
}


def parse_args():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs():
    for path, expected_hash in INPUT_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen input: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen input hash mismatch: {path}")

    manifest = json.loads(FREEZE_MANIFEST.read_text(encoding="utf-8"))
    highres = next(
        item for item in manifest["models"]
        if item["model_id"] == "highres_shufflenet_v2"
    )
    seed42 = next(
        item for item in highres["checkpoints"] if int(item["seed"]) == 42
    )
    if seed42["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Seed 42 checkpoint is inconsistent with the freeze manifest.")
    if seed42["reused_predictions_sha256"] != EXPECTED_PREDICTION_SHA256:
        raise ValueError("Seed 42 predictions are inconsistent with the freeze manifest.")
    return highres, seed42


def ensure_outputs_absent():
    candidates = (OUTPUT_DIR, INDIVIDUAL_DIR, COMPOSITE_PNG, COMPOSITE_PDF)
    existing = [path for path in candidates if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing Grad-CAM outputs:\n"
            + "\n".join(f"- {path}" for path in existing)
        )


def load_model(highres: dict, seed42: dict):
    checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=False)
    model = ShuffleNetV2HighRes(num_classes=NUM_CLASSES)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != int(
        highres["parameter_count"]
    ):
        raise ValueError("Frozen model parameter count changed.")
    if int(checkpoint["random_seed"]) != 42:
        raise ValueError("Checkpoint random seed is not 42.")
    if int(checkpoint["epoch"]) != int(seed42["best_epoch"]):
        raise ValueError("Checkpoint best epoch is inconsistent with the freeze manifest.")
    model.eval()
    return model


def normalize_correct(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    result = series.astype(str).str.lower().map({"true": True, "false": False})
    if result.isna().any():
        raise ValueError("Invalid values in predictions correct column.")
    return result.astype(bool)


def load_and_validate_data():
    predictions = pd.read_csv(PREDICTION_FILE)
    representatives = pd.read_csv(ERROR_SAMPLE_FILE)
    metadata = pd.read_csv(METADATA_FILE)
    images = np.load(IMAGE_FILE, mmap_mode="r")

    required_prediction_columns = {
        "source_index", "label", "label_id", "predicted_label",
        "predicted_label_id", "confidence", "correct", "split",
    }
    if not required_prediction_columns.issubset(predictions.columns):
        raise ValueError("Seed 42 prediction columns changed.")
    if len(predictions) != EXPECTED_TEST_SAMPLES:
        raise ValueError("Unexpected fixed test prediction count.")
    predictions["correct"] = normalize_correct(predictions["correct"])
    if predictions["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index in seed 42 predictions.")
    if not predictions["split"].eq("test").all():
        raise ValueError("Seed 42 predictions are not exclusively from the test split.")

    if images.shape != (len(metadata), 64, 64) or images.dtype != np.uint8:
        raise ValueError("Processed image array shape or dtype changed.")
    if metadata["source_index"].duplicated().any():
        raise ValueError("Duplicate source_index in metadata.")
    test_metadata = metadata.loc[metadata["split"].eq("test")].copy()
    test_metadata = test_metadata.reset_index(names="image_row")
    if len(test_metadata) != EXPECTED_TEST_SAMPLES:
        raise ValueError("Unexpected fixed test metadata count.")

    expected_keys = test_metadata[
        ["source_index", "lotName", "waferIndex", "label", "split", "label_id"]
    ].reset_index(drop=True)
    observed_keys = predictions[
        ["source_index", "lotName", "waferIndex", "label", "split", "label_id"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected_keys, observed_keys, check_dtype=False, check_exact=True
    )
    return predictions, representatives, metadata, images


def choose_samples(predictions: pd.DataFrame, representatives: pd.DataFrame):
    selected = []
    for class_name in TARGET_CLASSES:
        class_id = WM811K_CLASS_NAMES.index(class_name)
        correct_group = predictions.loc[
            predictions["correct"] & predictions["label"].eq(class_name)
        ].copy()
        if correct_group.empty:
            raise ValueError(f"No correct predictions for {class_name}.")
        median_confidence = float(correct_group["confidence"].median())
        correct_group["distance_to_median"] = (
            correct_group["confidence"] - median_confidence
        ).abs()
        correct_row = correct_group.sort_values(
            ["distance_to_median", "source_index"], ascending=[True, True]
        ).iloc[0]

        error_group = representatives.loc[
            representatives["label"].eq(class_name)
            & representatives["unanimous_predicted_label"].eq("none")
            & representatives["selection_label"].eq("50th percentile")
        ]
        if len(error_group) != 1:
            raise ValueError(
                f"Expected one frozen 50th-percentile {class_name} -> none error."
            )
        error_row = error_group.iloc[0]
        seed42_match = predictions.loc[
            predictions["source_index"].eq(int(error_row["source_index"]))
        ]
        if len(seed42_match) != 1:
            raise ValueError("Frozen error sample is missing from seed 42 predictions.")
        error_prediction = seed42_match.iloc[0]
        if (
            bool(error_prediction["correct"])
            or error_prediction["predicted_label"] != "none"
            or int(error_prediction["label_id"]) != class_id
        ):
            raise ValueError("Frozen error sample no longer matches the protocol.")

        selected.append(
            {
                "class_name": class_name,
                "class_id": class_id,
                "correct_source_index": int(correct_row["source_index"]),
                "correct_confidence_seed42": float(correct_row["confidence"]),
                "correct_class_confidence_median": median_confidence,
                "error_source_index": int(error_prediction["source_index"]),
                "error_predicted_label": "none",
                "error_confidence_seed42": float(error_prediction["confidence"]),
                "error_mean_confidence_three_seeds": float(
                    error_row["mean_prediction_confidence"]
                ),
                "error_selection_label": "50th percentile",
            }
        )
    selection = pd.DataFrame(selected)
    if selection[["correct_source_index", "error_source_index"]].stack().duplicated().any():
        raise ValueError("Grad-CAM sample selection contains duplicate samples.")
    return selection


def image_for_source(source_index: int, metadata: pd.DataFrame, images):
    matches = metadata.index[metadata["source_index"].eq(source_index)].to_numpy()
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely map source_index {source_index} to an image.")
    image = np.asarray(images[int(matches[0])]).copy()
    tensor = torch.from_numpy(image).to(torch.float32).unsqueeze(0).unsqueeze(0) / 2.0
    return image, tensor


WAFER_CMAP = ListedColormap(("#ffffff", "#5b6573", "#d62728"))
WAFER_NORM = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), WAFER_CMAP.N)


def draw_map(axis, image: np.ndarray, title: str):
    axis.imshow(image, cmap=WAFER_CMAP, norm=WAFER_NORM, interpolation="nearest")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def draw_overlay(axis, image: np.ndarray, cam: np.ndarray, title: str):
    axis.imshow(image, cmap="gray", vmin=0, vmax=2, interpolation="nearest")
    axis.imshow(cam, cmap="inferno", vmin=0, vmax=1, alpha=0.62, interpolation="bilinear")
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def save_individual(path: Path, image: np.ndarray, cam: np.ndarray, title: str):
    figure, axis = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
    draw_overlay(axis, image, cam, title)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def generate_artifacts(model, selection, metadata, images):
    OUTPUT_DIR.mkdir(parents=True)
    ARRAY_DIR.mkdir()
    INDIVIDUAL_DIR.mkdir(parents=True)
    selection.to_csv(OUTPUT_DIR / "sample_selection.csv", index=False)

    figure, axes = plt.subplots(3, 5, figsize=(14, 8.2), constrained_layout=True)
    target_layer = model.network.conv5[0]
    cam_records = []

    for row_index, row in selection.iterrows():
        class_name = row["class_name"]
        class_id = int(row["class_id"])
        slug = class_name.lower().replace("-", "_")
        correct_image, correct_tensor = image_for_source(
            int(row["correct_source_index"]), metadata, images
        )
        error_image, error_tensor = image_for_source(
            int(row["error_source_index"]), metadata, images
        )

        correct_cam, correct_logits = compute_gradcam(
            model, target_layer, correct_tensor, class_id
        )
        error_none_cam, error_logits = compute_gradcam(
            model, target_layer, error_tensor, NONE_CLASS_ID
        )
        error_true_cam, _ = compute_gradcam(
            model, target_layer, error_tensor, class_id
        )
        correct_cam = correct_cam.numpy().astype(np.float32)
        error_none_cam = error_none_cam.numpy().astype(np.float32)
        error_true_cam = error_true_cam.numpy().astype(np.float32)

        correct_probabilities = torch.softmax(correct_logits, dim=1)[0].numpy()
        error_probabilities = torch.softmax(error_logits, dim=1)[0].numpy()
        if int(correct_probabilities.argmax()) != class_id:
            raise RuntimeError("Selected correct sample changed prediction.")
        if int(error_probabilities.argmax()) != NONE_CLASS_ID:
            raise RuntimeError("Selected frozen error sample changed prediction.")

        arrays = {
            "correct_predicted": correct_cam,
            "error_predicted_none": error_none_cam,
            "error_true_class": error_true_cam,
        }
        for cam_kind, array in arrays.items():
            array_path = ARRAY_DIR / f"{slug}_{cam_kind}.npy"
            np.save(array_path, array)
            source_index = (
                int(row["correct_source_index"])
                if cam_kind == "correct_predicted"
                else int(row["error_source_index"])
            )
            target_name = (
                class_name if cam_kind != "error_predicted_none" else "none"
            )
            cam_records.append(
                {
                    "class_name": class_name,
                    "sample_role": "correct" if cam_kind == "correct_predicted" else "error",
                    "source_index": source_index,
                    "cam_kind": cam_kind,
                    "target_class": target_name,
                    "array_path": str(array_path.relative_to(PROJECT_ROOT)),
                    "cam_min": float(array.min()),
                    "cam_max": float(array.max()),
                    "cam_mean": float(array.mean()),
                }
            )

        draw_map(axes[row_index, 0], correct_image, f"Correct map\nsource {int(row['correct_source_index'])}")
        draw_overlay(axes[row_index, 1], correct_image, correct_cam, f"CAM: predicted {class_name}")
        draw_map(axes[row_index, 2], error_image, f"Error map\nsource {int(row['error_source_index'])}")
        draw_overlay(axes[row_index, 3], error_image, error_none_cam, "CAM: predicted none")
        draw_overlay(axes[row_index, 4], error_image, error_true_cam, f"CAM: true {class_name}")
        axes[row_index, 0].set_ylabel(class_name, fontsize=11, fontweight="bold")

        save_individual(
            INDIVIDUAL_DIR / f"{slug}_correct_predicted_cam.png",
            correct_image,
            correct_cam,
            f"{class_name}: correct prediction CAM",
        )
        save_individual(
            INDIVIDUAL_DIR / f"{slug}_error_predicted_none_cam.png",
            error_image,
            error_none_cam,
            f"{class_name}: error CAM for predicted none",
        )
        save_individual(
            INDIVIDUAL_DIR / f"{slug}_error_true_class_cam.png",
            error_image,
            error_true_cam,
            f"{class_name}: error CAM for true class",
        )

    figure.suptitle(
        "HighRes ShuffleNetV2 Grad-CAM on frozen representative samples",
        fontsize=13,
    )
    figure.savefig(COMPOSITE_PNG, dpi=300)
    figure.savefig(COMPOSITE_PDF)
    plt.close(figure)

    cam_summary = pd.DataFrame(cam_records)
    cam_summary.to_csv(OUTPUT_DIR / "cam_summary.csv", index=False)
    artifact_paths = sorted(
        [path for path in OUTPUT_DIR.rglob("*") if path.is_file()]
        + [path for path in INDIVIDUAL_DIR.rglob("*") if path.is_file()]
        + [COMPOSITE_PNG, COMPOSITE_PDF]
    )
    manifest = {
        "analysis_id": ANALYSIS_ID,
        "completed_at": datetime.now().astimezone().isoformat(),
        "interpretation_boundary": (
            "Grad-CAM shows model-sensitive regions only; it is not defect "
            "localization accuracy or causal evidence."
        ),
        "model": "ShuffleNetV2HighRes",
        "seed": 42,
        "target_layer": "network.conv5[0]",
        "device": "cpu",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sample_classes": list(TARGET_CLASSES),
        "input_hashes": {
            str(path.relative_to(PROJECT_ROOT)): expected
            for path, expected in INPUT_HASHES.items()
        },
        "script_sha256": sha256_file(Path(__file__)),
        "artifacts": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in artifact_paths
        ],
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    highres, seed42 = verify_inputs()
    ensure_outputs_absent()
    model = load_model(highres, seed42)
    predictions, representatives, metadata, images = load_and_validate_data()
    selection = choose_samples(predictions, representatives)

    if args.preflight:
        with torch.no_grad():
            sample_image, sample_tensor = image_for_source(
                int(selection.iloc[0]["correct_source_index"]), metadata, images
            )
            logits = model(sample_tensor)
        if sample_image.shape != (64, 64) or logits.shape != (1, NUM_CLASSES):
            raise RuntimeError("Model dry-run failed.")
        print("Grad-CAM preflight passed.")
        print("Frozen model: ShuffleNetV2HighRes seed 42")
        print("Target layer: network.conv5[0]")
        print(selection.to_string(index=False))
        print("No formal Grad-CAM output was created.")
        return

    generate_artifacts(model, selection, metadata, images)
    print(f"Grad-CAM outputs: {OUTPUT_DIR}")
    print(f"Composite figure: {COMPOSITE_PNG}")


if __name__ == "__main__":
    main()
