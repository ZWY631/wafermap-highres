# WaferMap-HighRes: Preserving Early Spatial Resolution for Wafer Bin Map Failure-Pattern Classification

![Status](https://img.shields.io/badge/status-under_review-lightgrey)

Code, fixed data split, and frozen results for the manuscript:

> **Preserving Early Spatial Resolution in ShuffleNetV2 for Wafer Bin Map Failure-Pattern Classification: A Controlled Multi-Seed Ablation of Downsampling and Aliasing**

Under review at *Journal of Intelligent Manufacturing*.

## Abstract (short version)

Wafer bin maps are small (64 × 64) discrete spatial arrays in which narrow, localized, and edge-related failure patterns occupy few cells. Standard lightweight backbones downsample these maps aggressively **before** their main feature stages. We test the hypothesis that preserving early spatial resolution improves wafer-map failure-pattern classification *without increasing model parameters*: the ShuffleNetV2 x1.0 stem is changed from a stride-two convolution plus max pooling to a stride-one convolution without initial pooling, while all subsequent stages, the 1,262,397-parameter count, cross-entropy objective, data split, and training protocol are held constant. A zero-parameter, resolution-matched anti-aliased (binomial blur) stem isolates retained spatial resolution from aliasing; retained resolution dominates, and the cost side is reported explicitly because the stem choices span a factor of 15.75 in Conv/Linear FLOPs at identical parameter count.

Across three random seeds (42, 123, 2026) on the full labeled WM-811K subset (172,950 maps, lot-disjoint split):

| Metric | HighRes ShuffleNetV2 | Standard ShuffleNetV2 (matched) |
|---|---:|---:|
| Accuracy | **98.0342% ± 0.0039%** | 96.7750% ± 0.0425% |
| Macro-F1 | **90.2029% ± 0.1965%** | 80.4947% ± 0.1316% |
| Balanced accuracy | **89.1151% ± 0.5514%** | 78.5458% ± 0.3015% |

The mean Macro-F1 gain over the parameter-matched standard stem is **+9.7082 pp**, concentrated in Scratch (+56.68), Loc (+10.58), and Edge-Loc (+8.03). A same-protocol adapted reproduction of the wafer-specific Kang & Kang stacking ensemble (14.7 M parameters) scores 97.8427% / 88.5738%, with paired (McNemar + bootstrap + randomization) support in two of three seeds.

Evaluation additionally covers: 4-configuration stem ablation, class-imbalance controls, per-class analysis, error-morphology audit, 12-condition robustness stress tests, Grad-CAM audit, complexity/throughput benchmarking (Apple M1 Pro/MPS), and frozen Windows x86-64 ONNX FP32/INT8 deployment.

## Repository layout

The layout mirrors the local project structure so that `src/wafermap/paths.py` works out of the box.

```
├── 02_数据/splits/wm811k_labeled_lot_disjoint.csv   # Fixed lot-disjoint split (paper Table 1)
├── 03_代码/
│   ├── src/wafermap/                                # Core modules (14)
│   ├── scripts/                                     # Training, evaluation, analysis (52)
│   └── tests/                                       # Unit tests (17)
├── 04_实验/metrics/                                 # Frozen canonical outputs (hash-locked)
│   ├── 20260729_highres_ce_multiseed_final_test/    # Final model predictions (seeds 42/123/2026)
│   ├── 20260807_kang_kang_stacking_lot_reproduction/# Stacking reproduction predictions
│   ├── 20260817_kang_vs_highres_significance_canonical/  # Paired significance (paper Table 5)
│   └── ...                                          # Robustness, morphology, Grad-CAM, significance
├── 05_结果/
│   ├── tables/                                      # All paper tables as CSV
│   └── figures/                                     # All paper figures (PNG + vector PDF)
└── 03_代码/requirements-core.txt
```

**Excluded from the repository** (by design): raw WM-811K data (~1.2 GB, download link below), training checkpoints (~2.6 GB, available on request), training logs, internal management documents, and the manuscript itself (under review).

## Environment

Verified on Apple Silicon (macOS, `osx-arm64`), 2026-07-24:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r 03_代码/requirements-core.txt   # torch 2.13.0, torchvision 0.28.0, numpy 2.4.6, ...
```

## Reproduction

### 1. Data

1. Download the WM-811K dataset (QINGYI release, Version 1, CC0 1.0):  
   https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
2. Place the raw pickle at `02_数据/raw/wm811k_kaggle_qingyi_v1/LSWMD.pkl` (create the directory).

### 2. Preprocess and split (deterministic)

```bash
python 03_代码/scripts/preprocess_wm811k.py          # resizes labeled maps to 64 × 64 (nearest neighbor)
python 03_代码/scripts/make_wm811k_splits.py         # StratifiedGroupKFold (20 folds, seed 42) lot-disjoint split
```

The committed split file `02_数据/splits/wm811k_labeled_lot_disjoint.csv` is the exact one used in the paper (SHA-256 `c429a4a7...`); regenerating it is deterministic and must match byte-for-byte.

### 3. Training (three seeds)

```bash
python 03_代码/scripts/train_shufflenet_highres_ce.py --seed 42
python 03_代码/scripts/train_shufflenet_highres_ce.py --seed 123
python 03_代码/scripts/train_shufflenet_highres_ce.py --seed 2026
```

Optional reproductions:

```bash
python 03_代码/scripts/run_stem_ablation_training.py --execute      # 4 stem configurations (validation only)
python 03_代码/scripts/run_kang_stacking_multiseed.py --execute     # 3-seed stacking reproduction
```

Key training/evaluation entry scripts support `--preflight` to validate inputs and output paths without starting work.

### 4. Evaluation (frozen, one-time)

```bash
python 03_代码/scripts/evaluate_final_highres_ce_multiseed.py --run-final-test
python 03_代码/scripts/synthesize_final_evidence.py --run
python 03_代码/scripts/analyze_kang_vs_highres_significance_canonical.py --run
```

The final-test evaluator refuses to overwrite existing outputs and validates checkpoint SHA-256 hashes. The committed predictions under `04_实验/metrics/` are the canonical paper evidence.

### 5. Tests

```bash
python -m pytest 03_代码/tests -q
```

## Reproducibility notes

- **Lot-disjoint split**: maps are grouped by `lotName` before splitting, so no production lot appears in more than one subset.
- **Three random seeds** (42, 123, 2026) with matched protocol; results reported as mean ± sample SD.
- **Statistical analysis**: two-sided exact McNemar test, 10,000-replicate stratified paired bootstrap, 10,000-replicate prediction-swap randomization, Holm adjustment at α = 0.05, per-seed reporting.
- **Frozen evidence**: every table, figure, prediction, and manifest in `04_实验/metrics/` and `05_结果/` is hash-locked by the pre-submission freeze manifest; the canonical paired-significance manifest records SHA-256 of all input prediction files.
- **Determinism**: seeded Python/NumPy/PyTorch and data-loader generator initialization; bitwise determinism is not claimed (deterministic-algorithm enforcement is not enabled).

## Data and licensing

- Dataset: WM-811K wafer maps, QINGYI Kaggle release (Version 1), **CC0 1.0**. See https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
- Code: released under the **MIT License** (see `LICENSE`).
- The model is a **whole-map classifier** (nine classes, single label). It does not perform AOI optical defect detection, bounding-box localization, or pixel-level segmentation.

## Citation

If you use this repository, please cite the manuscript once it is published:

```bibtex
@article{zhang2026preserving,
  title   = {Preserving Early Spatial Resolution in {ShuffleNetV2} for Wafer Bin Map
             Failure-Pattern Classification: A Controlled Multi-Seed and
             Same-Protocol Domain Comparison},
  author  = {Zhang, Wangyang and TODO: co-author names},
  journal = {Journal of Intelligent Manufacturing},
  year    = {2026},
  note    = {under review}
}
```

## Contact

Wangyang Zhang (ORCID 0009-0001-1818-0930) — [TODO: e-mail address]

## Acknowledgements

This work uses the public WM-811K dataset (CC0 1.0). Checkpoint files (~2.6 GB) and training logs are available from the authors on request.
