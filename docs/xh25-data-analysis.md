# XH25 官方数据分析

本文记录官方 XH25 数据准备后生成的 JSON 报告关键结果，用于复现实验配置、
解释类别分布，并区分正式指标与诊断指标。

## 数据完整性

- 图像数：4481
- 标签数：4481
- 标注框总数：20933
- 缺失图像/标签配对：0
- 无效标签：0
- source groups：3625

所有 4481 张图像都有对应标签文件，生成报告中缺失配对和无效标签均为 0。

## 切分配置与结果

- 验证集比例：0.15
- 随机种子：42
- 训练集图像：3807
- 验证集图像：674
- 训练集 source groups：3071
- 验证集 source groups：554
- 训练/验证 source-group overlap：0

切分按 source group 隔离，训练集和验证集之间没有 source-group 重叠。

## 粗粒度类别分布

| 类别 | 总框数 | 训练集框数 | 验证集框数 |
| --- | ---: | ---: | ---: |
| ship | 2682 | 2280 | 402 |
| aircraft | 17849 | 15103 | 2746 |
| vehicle | 402 | 324 | 78 |
| 合计 | 20933 | 17707 | 3226 |

舰船类别为 class IDs 0..3，飞机类别为 class IDs 4..23，车辆类别为 class ID 24。

## 25 类源数据框数

下表为生成 JSON 报告中的 25 类 source counts。

| Class ID | Class name | Coarse category | Boxes |
| ---: | --- | --- | ---: |
| 0 | HM | ship | 17 |
| 1 | LQS | ship | 30 |
| 2 | QHS | ship | 641 |
| 3 | MS | ship | 1994 |
| 4 | A1_SU-35 | aircraft | 1317 |
| 5 | A2_C-130 | aircraft | 1297 |
| 6 | A3_C-17 | aircraft | 998 |
| 7 | A4_C-5 | aircraft | 500 |
| 8 | A5_F-16 | aircraft | 1017 |
| 9 | A6_TU-160 | aircraft | 361 |
| 10 | A7_E-3 | aircraft | 547 |
| 11 | A8_B-52 | aircraft | 750 |
| 12 | A9_P-3C | aircraft | 895 |
| 13 | A10_B-1B | aircraft | 762 |
| 14 | A11_E-8 | aircraft | 432 |
| 15 | A12_TU-22 | aircraft | 583 |
| 16 | A13_F-15 | aircraft | 1265 |
| 17 | A14_KC-135 | aircraft | 1424 |
| 18 | A15_F-22 | aircraft | 493 |
| 19 | A16_FA-18 | aircraft | 2147 |
| 20 | A17_TU-95 | aircraft | 1114 |
| 21 | A18_KC-10 | aircraft | 262 |
| 22 | A19_SU-34 | aircraft | 933 |
| 23 | A20_SU-24 | aircraft | 752 |
| 24 | FSC | vehicle | 402 |

HM 只有 17 个框，LQS 只有 30 个框，是明确的稀缺类。它们的细粒度指标容易受
单张图像、单次切分和少量预测波动影响，训练和阈值调整时应谨慎解读。

## 指标口径

- 官方整体指标是主要 competition-style 分数，应作为模型选择和最终汇报的主指标。
- 粗粒度诊断指标将 25 类聚合为 ship、aircraft、vehicle，用于定位大类短板，例如
  舰船与车辆样本更少时的召回或误检问题。
- 细粒度诊断指标保留 25 类逐类表现，用于发现单类问题。不要只针对 HM、LQS
  这类噪声较高的稀缺类优化，而忽略官方整体指标和粗粒度 ship/aircraft/vehicle
  表现。
