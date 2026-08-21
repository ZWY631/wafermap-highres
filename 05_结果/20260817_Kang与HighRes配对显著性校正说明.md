# Kang 与 HighRes 配对显著性校正说明

## 目的

这是对 25,943 张固定测试集预测的事后配对分析。它不重新训练模型、不读取晶圆图像，也不读取模型 checkpoint。Kang 与 HighRes 按 `source_index` 对齐，在相同的九分类、lot-disjoint 测试协议下进行比较。

## 为什么新增校正版本

原始分析目录 `04_实验/metrics/20260810_kang_vs_highres_significance` 保留不变。核验时发现：

1. 原始表的 `macro_f1_randomization_discordant_predictions` 字段实际保存的是随机化检验的极端重复次数，不是 discordant 样本数；校正版本分别写为 `mcnemar_discordant_pairs` 和 `macro_f1_randomization_extreme_count`。
2. `05_结果/predictions/.../predictions_seed42.csv` 在原分析完成后被编辑过，当前哈希与原分析清单不同。其整数类别 ID 与原始 canonical 文件一致，但有一行 `predicted_label` 文本为 `nne`。校正版本使用未改动的 canonical 文件，不改写任何原始预测。

## 校正版本

- 脚本：`03_代码/scripts/analyze_kang_vs_highres_significance_corrected.py`
- canonical 驱动：`03_代码/scripts/analyze_kang_vs_highres_significance_canonical.py`
- 输出清单：`04_实验/metrics/20260817_kang_vs_highres_significance_canonical/analysis_manifest.json`
- 逐 seed 表：`04_实验/metrics/20260817_kang_vs_highres_significance_canonical/paired_per_seed.csv`
- 论文表：`05_结果/tables/table_kang_vs_highres_paired_significance_canonical.csv`

运行前只读核验：

```bash
PYTHONPATH=03_代码/src /Users/mima0000/miniforge3/envs/wafer-sci/bin/python \
  03_代码/scripts/analyze_kang_vs_highres_significance_canonical.py \
  --check-inputs
```

正式运行（输出目录和论文表均采用“存在即拒绝覆盖”策略）：

```bash
PYTHONPATH=03_代码/src /usr/bin/caffeinate -dimsu \
  /Users/mima0000/miniforge3/envs/wafer-sci/bin/python \
  03_代码/scripts/analyze_kang_vs_highres_significance_canonical.py \
  --run
```

## 结果摘要

HighRes 相对 Kang 的三 seed 平均提升为 Accuracy `+0.191` 个百分点、Macro-F1 `+1.629` 个百分点；由于 seed 2026 的配对差异接近零，不能把“每个 seed 均显著”作为结论。应报告逐 seed 结果和 Holm 校正后的 p 值：seed 42/123 显著，seed 2026 不显著。

