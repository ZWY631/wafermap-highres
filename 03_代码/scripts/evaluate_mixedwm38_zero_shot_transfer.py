#!/usr/bin/env python3
"""Zero-shot transfer evaluation of frozen WM-811K checkpoints on MixedWM38.

Frozen protocol: 00_项目管理/20260822_MixedWM38零样本迁移冻结协议.md

Phases (single run, no overwrite of frozen outputs):
  1. Verify MixedWM38 data hash and checkpoint SHA-256s from the manifest.
  2. Preprocess MixedWM38 (3->2, zero-pad 52x52->64x64, /2.0).
  3. Pre-registered label-mapping verification (fingerprint correlation).
  4. Inference: one forward pass per (model, seed); logits saved as .npy.
  5. Read-only analysis: Track A (defect-vs-normal), B (per-type AUC),
     C (9-class zero-shot on single-type subset) + exact McNemar.

Refuses to run if any frozen output already exists (use --force to rerun
inference; analysis is recomputed from saved logits on every run).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "03_代码/src"))

from wafermap.constants import WM811K_CLASS_NAMES  # noqa: E402
from wafermap.models import ResNet18Baseline, ShuffleNetV2Baseline  # noqa: E402
from wafermap.models_improved import ShuffleNetV2HighRes  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen configuration (protocol 20260822)
# ---------------------------------------------------------------------------

DATA_FILE = PROJECT_ROOT / "02_数据/external/mixedwm38/MixedWM38.npz"
DATA_SHA256 = "a19e791c85ccd5051c080169a3d2b17902a42545cf749e9869f6c96312bcdc69"
MANIFEST_FILE = PROJECT_ROOT / "00_项目管理/20260801_五模型统一评估冻结清单.json"
OUT_DIR = PROJECT_ROOT / "04_实验/metrics/20260822_mixedwm38_zero_shot_transfer"
PRED_DIR = OUT_DIR / "predictions"

MODEL_IDS = {
    "highres": "highres_shufflenet_v2",
    "standard": "standard_shufflenet_v2",
    "resnet18": "resnet18",
}
MODEL_FACTORY = {
    "highres": ShuffleNetV2HighRes,
    "standard": ShuffleNetV2Baseline,
    "resnet18": ResNet18Baseline,
}
SEEDS = (42, 123, 2026)
BATCH_SIZE = 256
PAD = 6  # 52x52 -> 64x64

# Pre-registered mapping hypothesis (amended 2026-08-22): dims 6/7 swapped
# relative to the CA-AIIT paper-order hypothesis (see protocol section 2.2).
HYPOTHESIS = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7, 7: 6}
MIN_SINGLE_COUNT = 100
CORR_THRESHOLD = 0.5
CORR_MARGIN = 0.05
MIN_AGREE = 7
DESCRIPTOR_CAP = 3000  # WM-811K per-class cap for descriptor computation


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint_info():
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    info = {}
    for mid, model_id in MODEL_IDS.items():
        entry = next(m for m in manifest["models"] if m["model_id"] == model_id)
        ckpts = {}
        for c in entry["checkpoints"]:
            ckpts[c["seed"]] = {
                "path": PROJECT_ROOT / c["checkpoint_path"],
                "sha256": c["checkpoint_sha256"],
                "run_name": c["run_name"],
                "best_epoch": c["best_epoch"],
            }
        info[mid] = ckpts
    return info


def load_data():
    if sha256_file(DATA_FILE) != DATA_SHA256:
        raise ValueError("MixedWM38.npz hash mismatch — refusing to run.")
    d = np.load(DATA_FILE)
    arr0 = d["arr_0"].astype(np.int32)
    arr1 = d["arr_1"].astype(np.int32)
    if arr0.shape != (38015, 52, 52) or arr1.shape != (38015, 8):
        raise ValueError("Unexpected MixedWM38 shapes.")
    arr0[arr0 == 3] = 2
    if not np.isin(arr0, (0, 1, 2)).all():
        raise ValueError("Unexpected values after remap.")
    # zero-pad 52x52 -> 64x64
    padded = np.zeros((len(arr0), 64, 64), dtype=np.uint8)
    padded[:, PAD:PAD + 52, PAD:PAD + 52] = arr0
    return padded, arr1, arr0


def load_model(model_id: str, seed: int, ckpt_info: dict, device: torch.device):
    info = ckpt_info[model_id][seed]
    if sha256_file(info["path"]) != info["sha256"]:
        raise ValueError(f"Checkpoint hash mismatch: {info['path']}")
    ckpt = torch.load(info["path"], map_location="cpu", weights_only=False)
    if ckpt["class_names"] != list(WM811K_CLASS_NAMES):
        raise ValueError("Checkpoint class order differs from constants.")
    model = MODEL_FACTORY[model_id](num_classes=9)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, info


def wafer_descriptors(mask2d, wafer_mask):
    """Fail-cell geometry descriptors (8-connectivity), protocol section 2.2."""
    from scipy.ndimage import label

    fail = mask2d == 2
    n = fail.sum()
    if n == 0:
        return None
    tot = n / wafer_mask.sum()
    lab, k = label(fail, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())[1:]
    largest = sizes.max() / n
    sing = (sizes == 1).sum() / n
    ncomp = k
    ys, xs = np.nonzero(fail)
    wy, wx = np.nonzero(wafer_mask)
    wcy, wcx = wy.mean(), wx.mean()
    off = np.hypot(cy := ys.mean() - wcy, cx := xs.mean() - wcx) / max(
        np.hypot(wy - wcy, wx - wcx).max(), 1
    )
    r = np.hypot(ys - wcy, xs - wcx)
    rmax = np.hypot(wy - wcy, wx - wcx).max()
    rmean = (r / rmax).mean()
    edge = (r > 0.75 * rmax).sum() / n
    cov = np.cov(np.stack([ys, xs]))
    ev = np.linalg.eigvalsh(cov)
    aniso = 1 - ev[0] / max(ev[1], 1e-9)
    return np.array([tot, largest, sing, ncomp / n, off, rmean, edge, aniso])


def mapping_verification(arr0, arr1):
    """Amended pre-registered mapping verification (protocol section 2.2).

    Geometry descriptors + Hungarian assignment + domain-signature anchors.
    Writes/returns the report including the adopted mapping.
    """
    from scipy.optimize import linear_sum_assignment

    single = arr1.sum(axis=1) == 1
    mw_desc = {}
    mw_counts = {}
    for i in range(8):
        mask = single & (arr1[:, i] == 1)
        n = int(mask.sum())
        mw_counts[i] = n
        if n < MIN_SINGLE_COUNT:
            mw_desc[i] = None
            continue
        rows = np.flatnonzero(mask)
        vecs = []
        for j in rows:
            v = wafer_descriptors(arr0[j], arr0[j] != 0)
            if v is not None:
                vecs.append(v)
        mw_desc[i] = np.mean(vecs, axis=0)

    wimg = np.load(
        PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/images_uint8.npy",
        mmap_mode="r",
    )
    import pandas as pd

    meta = pd.read_csv(
        PROJECT_ROOT / "02_数据/processed/wm811k_labeled_64x64/metadata.csv"
    )
    labels = meta["label_id"].to_numpy(dtype=np.int64)
    wm_desc = {}
    for c in range(8):
        idx = np.flatnonzero(labels == c)[:DESCRIPTOR_CAP]
        vecs = []
        for j in idx:
            v = wafer_descriptors(wimg[j], wimg[j] != 0)
            if v is not None:
                vecs.append(v)
        wm_desc[c] = np.mean(vecs, axis=0)

    V = np.stack([mw_desc[i] for i in range(8)] + [wm_desc[c] for c in range(8)])
    mu, sd = V.mean(axis=0), V.std(axis=0) + 1e-9
    V = (V - mu) / sd
    D = np.zeros((8, 8))
    for i in range(8):
        for c in range(8):
            D[i, c] = float(np.linalg.norm(V[i] - V[8 + c]))
    ri, ci = linear_sum_assignment(D)
    assignment = {int(i): int(c) for i, c in zip(ri, ci)}

    margins = {}
    for k in range(8):
        i, c = ri[k], ci[k]
        d_sorted = np.sort(D[i])
        margins[int(i)] = float(d_sorted[1] - D[i, c])

    agree = sum(1 for i in range(8) if assignment[i] == HYPOTHESIS[i])
    report = {
        "method": "geometry descriptors + Hungarian + domain signatures "
                  "(amended 2026-08-22, protocol section 2.2)",
        "hypothesis": HYPOTHESIS,
        "assignment": assignment,
        "agreement_with_hypothesis": agree,
        "per_dim_margin": {str(k): round(v, 3) for k, v in margins.items()},
        "distance_matrix": {str(i): {str(c): round(D[i, c], 3) for c in range(8)} for i in range(8)},
        "single_bit_counts": mw_counts,
        "accepted": agree >= MIN_AGREE,
        "adopted_mapping": HYPOTHESIS if agree >= MIN_AGREE else None,
    }
    return report


def run_inference(model, padded, device):
    logits = []
    with torch.no_grad():
        for s in range(0, len(padded), BATCH_SIZE):
            chunk = padded[s:s + BATCH_SIZE].astype(np.float32) / 2.0
            batch = torch.from_numpy(chunk).unsqueeze(1).to(device)
            logits.append(model(batch).cpu().float())
    return torch.cat(logits, dim=0).numpy()


def analyze(logits_by_model, arr1, mapping):
    """Read-only analysis on saved logits (tracks A/B/C).

    logits_by_model: {model_id: {seed: (N, 9) logits array}} with exactly one
    model and one seed; returns the per-model metric dict.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    mid = next(iter(logits_by_model))
    seed = next(iter(logits_by_model[mid]))
    logits = logits_by_model[mid][seed]

    y_def = arr1.sum(axis=1) > 0
    single = arr1.sum(axis=1) == 1
    normal = arr1.sum(axis=1) == 0
    y9 = np.full(len(arr1), 8, dtype=np.int64)
    if mapping is not None:
        bit = arr1[single].argmax(axis=1)
        y9[single] = np.array([mapping[int(b)] for b in bit], dtype=np.int64)
    # Track C covers the 9 single-type groups: 8 single-defect types + normal->none
    c_mask = single | normal

    sm = np.exp(logits - logits.max(axis=1, keepdims=True))
    sm /= sm.sum(axis=1, keepdims=True)
    s_def = 1.0 - sm[:, 8]

    # Track A
    auc = roc_auc_score(y_def, s_def)
    pred = s_def > 0.5
    tp = int(((pred == 1) & y_def).sum())
    fp = int(((pred == 1) & ~y_def).sum())
    tn = int(((pred == 0) & ~y_def).sum())
    fn = int(((pred == 0) & y_def).sum())
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bacc = (recall + specificity) / 2
    f1 = f1_score(y_def, pred)
    track_a = {
        "auc": float(auc),
        "recall@0.5": float(recall),
        "specificity@0.5": float(specificity),
        "balanced_accuracy@0.5": float(bacc),
        "f1@0.5": float(f1),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }

    # Track B: per-type AUC — score must be the model output of the MAPPED
    # WM-811K class (mapping is not identity for dims 6/7)
    per_type = {}
    for i in range(8):
        if mapping is None:
            per_type[str(i)] = float(roc_auc_score(arr1[:, i], sm[:, i]))
        else:
            m = mapping[i]
            per_type[str(i)] = float(roc_auc_score(arr1[:, i], sm[:, m]))
    per_type["mean"] = float(np.mean(list(per_type.values())))
    track_b = per_type

    # Track C
    yhat = logits.argmax(axis=1)
    # c_mask defined above: single | normal (9 single-type groups, normal -> none)
    acc = accuracy_score(y9[c_mask], yhat[c_mask])
    bacc_c = balanced_accuracy_score(y9[c_mask], yhat[c_mask])
    mf1 = f1_score(y9[c_mask], yhat[c_mask], average="macro", zero_division=0)
    per_class = {}
    for c in range(9):
        per_class[WM811K_CLASS_NAMES[c]] = float(
            f1_score(y9[c_mask] == c, yhat[c_mask] == c, zero_division=0)
        )
    track_c = {
        "n": int(c_mask.sum()),
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc_c),
        "macro_f1": float(mf1),
        "per_class_f1": per_class,
    }

    return {"track_A": track_a, "track_B": track_b, "track_C": track_c}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="rerun inference")
    parser.add_argument("--no-infer", action="store_true", help="analysis only")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"[{datetime.now():%H:%M:%S}] device={device}")

    ckpt_info = load_checkpoint_info()
    tensors, arr1, arr0 = load_data()
    print(f"[{datetime.now():%H:%M:%S}] data loaded: {tensors.shape}")

    # --- mapping verification (amended protocol section 2.2) ---
    verif_path = OUT_DIR / "mapping_verification.json"
    if verif_path.exists():
        report = json.loads(verif_path.read_text(encoding="utf-8"))
        print("mapping verification: reuse frozen report")
    else:
        report = mapping_verification(arr0, arr1)
        verif_path.write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    mapping = report.get("adopted_mapping")
    if mapping is not None:
        mapping = {int(k): int(v) for k, v in mapping.items()}
    print(
        f"mapping verification: agreement {report['agreement_with_hypothesis']}/8, "
        f"accepted={report['accepted']}, adopted={mapping}"
    )
    if mapping is None:
        print("WARNING: mapping not verified — per-type tracks (B/C) disabled.")

    # --- inference ---
    logits_by_model = {}
    run_manifest = {
        "protocol": "20260822_MixedWM38零样本迁移冻结协议.md",
        "data_file": str(DATA_FILE),
        "data_sha256": DATA_SHA256,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "runs": {},
    }
    if not args.no_infer:
        for mid in MODEL_IDS:
            logits_by_model[mid] = {}
            for seed in SEEDS:
                out = PRED_DIR / f"logits_{mid}_seed{seed}.npy"
                if out.exists() and not args.force:
                    print(f"exists, skip: {out}")
                    logits_by_model[mid][seed] = np.load(out)
                    continue
                model, info = load_model(mid, seed, ckpt_info, device)
                print(
                    f"[{datetime.now():%H:%M:%S}] infer {mid} seed={seed} "
                    f"({info['run_name']}, best_epoch={info['best_epoch']})"
                )
                logits = run_inference(model, tensors, device)
                np.save(out, logits)
                run_manifest["runs"][f"{mid}_seed{seed}"] = {
                    "checkpoint": str(info["path"]),
                    "checkpoint_sha256": info["sha256"],
                    "logits_sha256": hashlib.sha256(logits.tobytes()).hexdigest(),
                    "shape": list(logits.shape),
                }
                logits_by_model[mid][seed] = logits
    else:
        for mid in MODEL_IDS:
            logits_by_model[mid] = {
                seed: np.load(PRED_DIR / f"logits_{mid}_seed{seed}.npy")
                for seed in SEEDS
            }

    # aggregate per model across seeds
    per_model = {mid: {} for mid in MODEL_IDS}
    for mid in MODEL_IDS:
        for seed in SEEDS:
            per_model[mid][seed] = analyze(
                {mid: {seed: logits_by_model[mid][seed]}}, arr1, mapping
            )

    # build full summary: per-seed + mean/std across seeds
    def agg(fn):
        return {
            "per_seed": {str(s): fn(s) for s in SEEDS},
            "mean": float(np.mean([fn(s) for s in SEEDS])),
            "std": float(np.std([fn(s) for s in SEEDS], ddof=1)),
        }

    summary = {}
    for track, keys in (
        ("track_A", ("auc", "recall@0.5", "specificity@0.5", "balanced_accuracy@0.5", "f1@0.5")),
        ("track_B", tuple(str(i) for i in range(8)) + ("mean",)),
        ("track_C", ("accuracy", "balanced_accuracy", "macro_f1")),
    ):
        summary[track] = {
            mid: {key: agg(lambda s, k=key, mid=mid: per_model[mid][s][track][k]) for key in keys}
            for mid in MODEL_IDS
        }
    summary["track_C"]["per_class_f1"] = {
        mid: {c: agg(lambda s, c=c, mid=mid: per_model[mid][s]["track_C"]["per_class_f1"][c]) for c in WM811K_CLASS_NAMES}
        for mid in MODEL_IDS
    }

    # deltas HighRes - Standard (per seed + mean/std), from per_model
    deltas = {"track_A": {}, "track_B": {}, "track_C": {}}
    for track, keys in (
        ("track_A", ("auc", "balanced_accuracy@0.5", "recall@0.5")),
        ("track_B", ("mean",)),
        ("track_C", ("accuracy", "macro_f1", "balanced_accuracy")),
    ):
        for key in keys:
            vals = [per_model["highres"][s][track][key] - per_model["standard"][s][track][key] for s in SEEDS]
            deltas[track][key] = {
                "per_seed": {str(s): float(v) for s, v in zip(SEEDS, vals)},
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
            }
    summary["deltas_highres_minus_standard"] = deltas

    summary["track_B"]["labels"] = {
        str(i): WM811K_CLASS_NAMES[mapping[int(i)]] if mapping else None
        for i in range(8)
    }

    # exact McNemar on Track C per seed
    single = arr1.sum(axis=1) == 1
    normal = arr1.sum(axis=1) == 0
    c_mask = single | normal
    y9 = np.full(len(arr1), 8, dtype=np.int64)
    if mapping is not None:
        bit = arr1[single].argmax(axis=1)
        y9[single] = np.array([mapping[int(b)] for b in bit], dtype=np.int64)
    from scipy.stats import binomtest

    mcnemar = {}
    for seed in SEEDS:
        a = logits_by_model["highres"][seed].argmax(axis=1)[c_mask]
        b = logits_by_model["standard"][seed].argmax(axis=1)[c_mask]
        y = y9[c_mask]
        b_only = int(((a != y) & (b == y)).sum())
        c_only = int(((a == y) & (b != y)).sum())
        p = binomtest(b_only, b_only + c_only).pvalue if (b_only + c_only) else 1.0
        mcnemar[str(seed)] = {
            "highres_only_wrong": b_only,
            "standard_only_wrong": c_only,
            "p_value": float(p),
        }
    summary["mcnemar_track_C"] = mcnemar

    out_json = OUT_DIR / "metrics_summary.json"
    out_json.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    # CSV
    import csv

    rows = []
    for track in ("track_A", "track_B", "track_C"):
        for mid in MODEL_IDS:
            for key, val in summary[track][mid].items():
                if isinstance(val, dict) and "mean" in val:
                    rows.append(
                        [track, mid, key, val["mean"], val["std"], val["per_seed"]]
                    )
    with open(OUT_DIR / "metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["track", "model", "metric", "mean", "std", "per_seed"])
        w.writerows(rows)

    if run_manifest["runs"]:
        (OUT_DIR / "run_manifest.json").write_text(
            json.dumps(run_manifest, indent=1, ensure_ascii=False), encoding="utf-8"
        )
    print(f"wrote {out_json}")

    # quick console summary
    for track in ("track_A", "track_B", "track_C"):
        for mid in ("highres", "standard", "resnet18"):
            line = []
            for key, val in summary[track][mid].items():
                if isinstance(val, dict) and "mean" in val:
                    line.append(f"{key}={val['mean']:.4f}±{val['std']:.4f}")
            print(f"{track} {mid}: " + "  ".join(line))


if __name__ == "__main__":
    main()
