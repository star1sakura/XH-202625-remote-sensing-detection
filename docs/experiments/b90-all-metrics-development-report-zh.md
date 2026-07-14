# B90 同权重多尺度最优精度方案开发报告

## 1. 报告结论

当前固定验证集上的最优精度候选是 **B90 同权重多尺度方案**。它只使用一个
MKSNet-Lite 检查点，但分别以 `1024`、`1280`、`1536` 三个输入尺度推理，再按目标大类
选择主尺度并补充跨尺度独有检测框。

相对冻结的 Alpha050-Fine 单尺度候选，该方案同时提高：

- aircraft、ship、vehicle 各自的 Recall；
- aircraft、ship、vehicle 各自的 FDR（数值越低越好）；
- Ultralytics Precision、Recall、mAP50、mAP50-95。

因此它在本项目定义的 10 项精度门槛上达到 **10/10**。不过，三尺度推理的耗时尚未纳入
本轮晋级条件，融合参数也在同一固定验证集上选择，所以它应被称为“当前验证集精度最优
候选”，不能直接等同于隐藏测试集或最终比赛总分最优。

## 2. 任务和评估约束

数据使用 XH25 的 25 个细分类，粗粒度映射为：

- class `0..3`：ship；
- class `4..23`：aircraft；
- class `24`：vehicle。

比赛风格评估按粗粒度统计 TP、FP、FN、Recall 和 FDR。匹配 IoU 为 aircraft/ship
`0.50`、vehicle `0.35`。本轮还同时保留 Ultralytics 验证输出中的 Precision、Recall、
mAP50 和 mAP50-95，避免只靠阈值移动改善比赛风格统计，却损失模型的整体排序能力。

固定晋级门槛来自此前最好的 Alpha050-Fine 单检查点结果：

| 指标 | 冻结门槛 |
| --- | ---: |
| Aircraft Recall | 0.992353 |
| Aircraft FDR | 0.011965 |
| Ship Recall | 0.833333 |
| Ship FDR | 0.149746 |
| Vehicle Recall | 0.730769 |
| Vehicle FDR | 0.185714 |
| Ultralytics Precision | 0.939505 |
| Ultralytics Recall | 0.919452 |
| Ultralytics mAP50 | 0.948095 |
| Ultralytics mAP50-95 | 0.767672 |

候选必须在 Recall、Precision、AP 上更高，在 FDR 上更低，才算通过对应门槛。

## 3. 方案是怎样得到的

### 3.1 从 main 到单权重 MKSNet-Lite

最初直接训练的 MKSNet/SPH 结构没有稳定超过 main。随后改为从历史 main 检查点初始化
MKSNet-Lite：所有可兼容层继承 main 权重，新增的两个 MKS 残差块使用零门控初始化。
初始化后随机输入的最大输出差为 `0.0`，保证训练起点与 main 等价。

对 MKS 模块和后部检测头做低学习率微调后，将微调权重与原始 main-seeded 权重做参数空间
插值。Alpha050-Fine 最终得到单检查点结果：aircraft `2725 TP / 33 FP`、ship
`335 / 59`、vehicle `57 / 13`。这一步证明 MKSNet 方向可以产生比 main 更好的单权重
候选，但 vehicle Recall 和 ship Recall 仍有明显提升空间。

### 3.2 搜索兼顾 AP 的 B90 权重

只优化粗粒度 Recall/FDR 容易把置信度阈值变成对验证集的局部修补，因此下一阶段把
Ultralytics 的四项整体指标也设为硬约束。在 Alpha050 与 nowarmup 微调权重之间进行参数
插值后，B90 候选首先通过了四项整体指标：

| 指标 | B90 单权重 |
| --- | ---: |
| Precision | 0.943814 |
| Recall | 0.934280 |
| mAP50 | 0.954427 |
| mAP50-95 | 0.773016 |

`B90` 表示输出参数由 10% Alpha050 和 90% nowarmup best 组成，检查点内也保存了这组
插值元数据。生成命令为：

```bash
.venv/bin/xh-detect interpolate-checkpoints \
  --base-checkpoint outputs/xh25/single-student/main-seeded-alpha050.pt \
  --tuned-checkpoint runs/train/xh25-main-seeded-mks-head-nowarmup/weights/best.pt \
  --output-checkpoint outputs/xh25/all-metrics-search/b90-multiscale/final/best-all-metrics.pt \
  --alpha 0.9
```

但 B90 在单一 `1024` 尺度经过细分类阈值后，只通过 6 项粗粒度门槛中的 3 项。粗/细
阈值前沿搜索表明，不存在一个纯置信度阈值组合能够同时补齐全部 Recall 并压住全部 FDR。

### 3.3 被否决的训练和校准路线

在转向多尺度前，先验证了“继续训练是否能解决问题”：

1. 从 Alpha050 在原训练集上以低学习率继续训练 12 epoch；没有一个直接 epoch 同时通过
   四项整体指标和六项粗粒度指标。
2. 对继续训练的 epoch 做权重插值和 SWA；仍没有得到可晋级候选。
3. 使用 B90 在训练集上挖掘困难负样本。第一轮 `confidence=0.15`、`crop=512` 只得到
   19 个 ship 和 1 个 vehicle 样本，覆盖不足。
4. 第二轮降低到 `confidence=0.10`、`crop=256`，得到 99 个 ship 和 8 个 vehicle 困难
   样本，并将 vehicle 正样本扩充 3 倍；再叠加 QHS/MS 各 2 倍重采样，训练集达到
   5163 张图、972 个 vehicle 目标。
5. 以 `5e-6` 学习率校准 10 epoch 后，直接权重仍不能同时过门槛。B90 到 balanced
   epoch 4 的 `t=0.40` 插值虽然保住四项整体指标，但粗粒度结果退化为 aircraft
   `2721/38`、ship `325/59`、vehicle `63/30`，因此被否决。

这些实验说明短板不只是训练样本权重问题。不同尺度已经学到互补检测框，直接继续训练反而
容易在新增 Recall 与新增 FP 之间交换，无法同时改善 10 项指标。

### 3.4 同一权重的多尺度互补

本轮先以精度为目标，保持单个模型权重，并暂不将前向次数和耗时纳入晋级门槛。最终固定
同一个 B90 检查点，以低置信度分别生成三个尺度的原始预测。应用细分类阈值后的粗粒度
计数为：

| 尺度 | Aircraft TP/FP | Ship TP/FP | Vehicle TP/FP |
| --- | ---: | ---: | ---: |
| 1024 | 2724 / 31 | 327 / 57 | 62 / 18 |
| 1280 | 2730 / 47 | 315 / 53 | 62 / 14 |
| 1536 | 2716 / 54 | 323 / 47 | 61 / 13 |

三个尺度的优势不同：`1024` 的 aircraft 误检较少，`1280` 能补充 aircraft 漏检，
`1536` 对 ship/vehicle 的误检控制更好，而 `1024` 仍能补充部分 ship。于是搜索不再把
所有尺度无差别 NMS，而是按大类固定主尺度和补充尺度。

## 4. 最终实现

最终策略位于 `configs/xh25-b90-multiscale-fusion.yaml`：

- aircraft：以 `1024` 为主；从 `1280` 补充 `score >= 0.75` 且与已有 aircraft 的
  HBB IoU `< 0.30` 的检测框；
- ship：以 `1536` 为主；从 `1024` 补充 `score >= 0.56` 且与已有 ship 的 HBB IoU
  `< 0.70` 的检测框；
- vehicle：只使用 `1536`；对 class 24 同时满足 `score < 0.21` 和 HBB area `< 700`
  的小框做过滤；
- 三个主尺度都使用配置中的 25 个细分类阈值。

`src/xh_detect/same_weight_multiscale.py` 实现以下流程：

1. 严格校验概率、面积参数以及 25 个类别阈值是否完整；
2. 按 taxonomy 将检测框分成 aircraft、ship、vehicle；
3. 选择各类主尺度并应用细分类阈值；
4. 按置信度从高到低加入补充尺度的非重复框；
5. 对 vehicle 应用低分小面积过滤；
6. 按 image、class、score、polygon 做确定性排序并导出 COCO Detection JSON。

命令行入口为 `xh-detect fuse-same-weight-multiscale`。三个原始推理配置均指向同一个
检查点，只修改 `image_size`、`tile_size` 和批量大小；因此这不是三个模型的权重集成。

## 5. 最终结果

检查点：

```text
outputs/xh25/all-metrics-search/b90-multiscale/final/best-all-metrics.pt
```

SHA256：

```text
f0b8c2dcf439437035d1f7afc2211443d411593a02a7f4ce9f5d6e5ad1423b17
```

### 5.1 10 项精度对比

| 指标 | Alpha050-Fine | B90 多尺度 | 变化 | 通过 |
| --- | ---: | ---: | ---: | --- |
| Aircraft Recall | 0.992353 | 0.995994 | +0.003642 | 是 |
| Aircraft FDR | 0.011965 | 0.011565 | -0.000400 | 是 |
| Ship Recall | 0.833333 | 0.845771 | +0.012438 | 是 |
| Ship FDR | 0.149746 | 0.147870 | -0.001877 | 是 |
| Vehicle Recall | 0.730769 | 0.782051 | +0.051282 | 是 |
| Vehicle FDR | 0.185714 | 0.175676 | -0.010039 | 是 |
| Precision | 0.939505 | 0.943814 | +0.004309 | 是 |
| Recall | 0.919452 | 0.934280 | +0.014828 | 是 |
| mAP50 | 0.948095 | 0.954427 | +0.006333 | 是 |
| mAP50-95 | 0.767672 | 0.773016 | +0.005344 | 是 |

最终比赛风格计数：

| 大类 | TP | FP | FN | Recall | FDR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aircraft | 2735 | 32 | 11 | 0.995994 | 0.011565 |
| Ship | 340 | 59 | 62 | 0.845771 | 0.147870 |
| Vehicle | 61 | 13 | 17 | 0.782051 | 0.175676 |

机器可读结果保存在本地实验目录的
`outputs/xh25/all-metrics-search/b90-multiscale/final/ten-metric-report.json`。`outputs/`
和 `.pt` 权重不提交 Git；复现实验前应将 SHA256 相同的权重放到上述路径。

## 6. 复现步骤

安装环境：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple uv==0.11.23
bash scripts/bootstrap_gpu_server.sh
.venv/bin/xh-detect env
```

用同一个检查点生成三个尺度的低阈值预测：

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1024.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1024.json

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1280.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1280.json

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1536.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1536.json
```

融合并评估：

```bash
.venv/bin/xh-detect fuse-same-weight-multiscale \
  --predictions-1024 outputs/xh25/b90-multiscale/raw-1024.json \
  --predictions-1280 outputs/xh25/b90-multiscale/raw-1280.json \
  --predictions-1536 outputs/xh25/b90-multiscale/raw-1536.json \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --policy-yaml configs/xh25-b90-multiscale-fusion.yaml \
  --output-json outputs/xh25/b90-multiscale/predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/b90-multiscale/predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/b90-multiscale/report.json
```

开发机与服务器上的融合输出已经做过 canonical JSON 对比，均为 `3240` 条记录且完全
一致；三类 TP/FP/FN 也一致。

## 7. 代码和测试地图

| 路径 | 作用 |
| --- | --- |
| `src/xh_detect/same_weight_multiscale.py` | 同权重多尺度融合核心逻辑 |
| `src/xh_detect/cli.py` | `fuse-same-weight-multiscale` CLI |
| `configs/xh25-b90-multiscale-1024.yaml` | 1024 原始推理配置 |
| `configs/xh25-b90-multiscale-1280.yaml` | 1280 原始推理配置 |
| `configs/xh25-b90-multiscale-1536.yaml` | 1536 原始推理配置 |
| `configs/xh25-b90-multiscale-fusion.yaml` | 最终阈值和融合策略 |
| `tests/test_same_weight_multiscale.py` | 参数校验、补框、去重、面积过滤、确定性测试 |
| `tests/test_cli.py` | CLI 输入、输出和异常路径测试 |

提交前验证命令：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
```

## 8. 已知边界和下一步

1. **耗时未纳入 10 项门槛。** 三尺度需要三次前向推理，必须在比赛指定硬件和完整
   `10000 x 10000` 图像上重新测端到端时间；不满足时回退 Alpha050-Fine 单尺度方案。
2. **存在验证集调参偏差。** 细分类阈值、补充阈值、重复 IoU 和 vehicle 面积规则都在
   固定验证集上选择，应以隐藏测试集或新的冻结 holdout 决定是否最终提交。
3. **权重不在 Git 中。** 使用前必须核对 SHA256，避免配置静默加载错误版本。
4. **它是一个权重、三种尺度。** 若比赛规则把多尺度前向视为模型集成或限制推理次数，
   需要先向主办方确认；代码层面没有加载第二个检查点。
5. **最终选择原则。** 隐藏集精度保持且耗时过线时提交 B90 多尺度；否则提交已经验证
   通过 20 秒门槛的 Alpha050-Fine 单权重单尺度版本。
