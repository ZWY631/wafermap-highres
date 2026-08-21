# Kang & Kang Stacking 同协议适配复现结果

## 1. 实验边界

- 方法：Kang & Kang (2021) 完整 stacking ensemble，DOI `10.1016/j.compind.2021.103450`。
- 定位：同协议适配复现，不是对原论文数值的逐位复现。
- 数据：WM-811K 全部 172,950 张有标签晶圆图，九分类。
- 划分：固定 lot-disjoint train/validation/test；测试集 25,943 张。
- 方法组成：59 维晶圆手工特征 FNN、单通道 VGG16、2 折 lot-grouped OOF 和 Ridge meta-classifier。
- 种子：42、123、2026；每个种子只用 validation Macro-F1 选模。

## 2. 总体结果

| 指标 | Kang Stacking（均值 ± 样本标准差） | HighRes ShuffleNetV2 | 差值（Kang - HighRes） |
|---|---:|---:|---:|
| Accuracy | 97.84% ± 0.14% | 98.03% ± 0.00% | -0.19 个百分点 |
| Macro-F1 | 88.57% ± 1.28% | 90.20% ± 0.20% | -1.63 个百分点 |
| Balanced Accuracy | 87.10% ± 1.87% | 89.12% ± 0.55% | -2.02 个百分点 |

Kang Stacking 在完全相同的九分类、lot-disjoint 协议下没有超过本文 HighRes 模型。三个主指标都略低，而且 Macro-F1 和 Balanced Accuracy 的种子间波动更大。这一结果应表述为“本文 HighRes 方法在同协议领域专用对照中保持了更好的综合表现”，不应声称对所有 stacking 实现或原论文协议普遍占优。

## 3. 逐类结果

| 类别 | 测试数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Center | 644 | 96.17% | 89.65% | 92.79% |
| Donut | 83 | 79.17% | 89.96% | 84.22% |
| Edge-Loc | 778 | 85.30% | 83.29% | 84.28% |
| Edge-Ring | 1,452 | 97.82% | 98.74% | 98.27% |
| Loc | 539 | 91.35% | 69.26% | 78.77% |
| Near-full | 21 | 95.30% | 96.83% | 96.01% |
| Random | 130 | 93.11% | 78.97% | 85.42% |
| Scratch | 179 | 79.00% | 77.65% | 78.29% |
| none | 22,117 | 98.69% | 99.54% | 99.11% |

Loc 的 Recall 最低（69.26%），Scratch 的 F1 也较低（78.29%），与本项目已有的稀疏、局部、形态多变缺陷错误机理分析一致。Near-full 只有 21 张测试样本，虽然 F1 高，但不确定性大，只能作参考。

## 4. 论文可用结论

1. 本文已补充一个含晶圆专用手工特征的领域方法，不再只与通用 CNN 基线对比。
2. 在相同数据、lot 划分、九分类和三种子协议下，HighRes 模型的 Macro-F1 高 1.63 个百分点。
3. Kang Stacking 在少数类上仍有明显波动，说明手工特征与深层分支融合不会自动消除长尾和形态多样性问题。
4. 原论文的样本筛选、划分和初始化与本项目不同，因此本结果只能称为同协议适配复现。

## 5. 证据位置

- 原始结果：`04_实验/metrics/20260807_kang_kang_stacking_lot_reproduction`
- 总体表：`05_结果/tables/table_kang_stacking_multiseed_test.csv`
- 逐类表：`05_结果/tables/table_kang_stacking_per_class_test.csv`
- 同协议对比表：`05_结果/tables/table_highres_vs_kang_stacking_same_protocol.csv`

## 6. 配对统计检验

HighRes 相对 Kang 的 Accuracy 平均提升 0.191 个百分点，Macro-F1 平均提升 1.629 个百分点。但三个 seed 的效应并不完全一致：seed 42 和 123 的 McNemar 和 Macro-F1 配对随机化检验经 Holm 校正后显著，seed 2026 不显著。因此论文应写“三种子均值显示 HighRes 更高，且两个 seed 的配对检验支持该差异”，不应写成“所有 seed 均具有统计显著性”。

## 7. 复杂度与实测延迟

Kang 包含约 14.74M 个可学参数，约为 HighRes 的 11.68 倍；静态 Conv/Linear FLOPs 约为 2.496G，为 HighRes 的 7.00 倍。在本机 MPS 合成输入前向测试中，Kang 的 FNN 和 CNN 分支串行相加延迟约为 batch=1 时 1.60 ms、batch=128 时 72.70 ms；该数值不包含原始图手工特征提取、数据搬运和后处理，不能代替完整工业端到端延迟。这表明参数量/FLOPs 与特定 MPS 硬件延迟不总是同方向，是应在讨论中明确披露的工程权衡。

## 8. 新增证据位置

- 配对统计：`04_实验/metrics/20260810_kang_vs_highres_significance`
- 复杂度基准：`04_实验/benchmarks/20260810_kang_stacking_efficiency`
- 配对检验表：`05_结果/tables/table_kang_vs_highres_paired_significance.csv`
- 效率表：`05_结果/tables/table_kang_vs_highres_efficiency.csv`
