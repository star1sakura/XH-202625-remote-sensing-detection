# XH-202625 遥感目标检测 Demo

面向飞机、舰船、车辆三类光学遥感目标的比赛基线。项目提供 DOTA-v1.5
数据转换、YOLO26s-OBB 训练、超大图滑窗推理、跨切片合并、COCO JSON
导出、比赛规则评估、Gradio Demo 和 10,000×10,000 性能基准。

当前代码闭环已完成；正式 XH25 数据使用 25 类 taxonomy 和 HBB 配置完成训练、
推理与 Demo。没有正式数据或只做冒烟时，仍可使用 DOTA-v1.5 或 Ultralytics
`dota8.yaml` 完成服务器、训练和推理检查。

## 1. 目标与指标

比赛初赛关注：

- 三类合并 Recall 不低于 85%；
- 三类合并 FDR 不高于 20%；
- 10,000×10,000 图像单幅处理时间不超过 20 秒；
- 飞机、舰船匹配 IoU 阈值为 0.50，车辆为 0.35。

本仓库第一阶段目标是形成可复现、可测试、可展示的基线，不承诺在未获得正式
数据和指定评测硬件前达到上述指标。

## 2. 推荐服务器环境

- Ubuntu 22.04；
- Python 3.11（支持 3.11–3.12）；
- NVIDIA RTX 4090 24GB；
- CUDA 驱动与 PyTorch 版本兼容；
- 至少 80GB 可用磁盘空间。

```bash
python -m pip install uv==0.11.23
git clone <your-repository-url> xh-detect
cd xh-detect
bash scripts/bootstrap_gpu_server.sh
.venv/bin/xh-detect version
.venv/bin/xh-detect env
```

`env` 输出应包含 Python、PyTorch、Ultralytics、CUDA 和 GPU 型号。4090
服务器上 `cuda_available` 应为 `true`。这个脚本会复用镜像预装的 CUDA
PyTorch，防止依赖解析器下载另一套 PyTorch/CUDA。普通 CPU 开发机仍可使用
`uv sync --extra dev`。

## 3. 官方 XH25 工作流

正式数据流程使用 `configs/xh25-hbb.yaml`、25 类 XH25 taxonomy，以及 HBB
检测任务；其中类别 0..3 为舰船，4..23 为飞机，24 为车辆。数据统计和切分结果
记录在 [docs/xh25-data-analysis.md](docs/xh25-data-analysis.md)。

准备官方 XH25 数据：

```bash
.venv/bin/xh-detect prepare-xh25 \
  --source-root data \
  --output-root datasets/xh25 \
  --val-ratio 0.15 \
  --seed 42
```

一轮官方 HBB baseline 训练：

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model yolo26s.pt \
  --epochs 1 \
  --image-size 1024 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --name xh25-baseline \
  --device 0
```

启动官方 Demo：

```bash
.venv/bin/xh-detect serve \
  --config-path configs/xh25-hbb.yaml \
  --host 127.0.0.1 \
  --port 7860
```

对验证集导出提交/评估用 COCO Detection JSON：

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-hbb.yaml \
  --output-json outputs/xh25/val-predictions.json
```

官方整体指标是主比赛风格分数；粗粒度诊断按舰船、飞机、车辆聚合，用于定位大类
短板；细粒度诊断保留 25 类逐类表现。HM、LQS 样本极少，调参时不要只追逐这些
稀缺类的噪声收益，应同时监控整体和粗粒度表现。

## 4. MKSNet-Lite 实验

`xh25-mksnet-lite` 是一个 MKSNet-inspired 中等改动实验。它保留现有
YOLO26s/HBB、滑窗推理和比赛评估流程，只在 YOLO neck 中加入轻量多核空间/通道
注意力模块，用来判断这类结构是否值得进一步完整复刻。

训练前先准备官方 XH25 数据：

```bash
.venv/bin/xh-detect prepare-xh25 \
  --source-root data \
  --output-root datasets/xh25 \
  --val-ratio 0.15 \
  --seed 42
```

训练 MKSNet-Lite：

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo26s-mksnet-lite.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --batch 8 \
  --workers 4 \
  --name xh25-mksnet-lite \
  --device 0
```

导出验证集预测：

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-mksnet-lite.yaml \
  --output-json outputs/xh25/mksnet-lite/val-predictions.json
```

用同一套 XH25 taxonomy 分别评估 baseline 与实验报告：

```bash
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/baseline/report.json \
  --taxonomy xh25

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/mksnet-lite/report.json \
  --taxonomy xh25
```

最后生成对比报告：

```bash
.venv/bin/xh-detect compare-experiments \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-report outputs/xh25/mksnet-lite/report.json \
  --output-dir outputs/xh25/mksnet-lite
```

保留 `comparison.json` 和 `comparison.md`，作为是否继续完整复刻 MKSNet 的依据。

### 阈值优化

MKSNet-Lite 的 80 epoch 结果显示全局阈值 `0.30` 比 `0.25` 更稳，但 ship 类仍是主要短板。
可以在不重新训练的情况下，用验证集预测搜索逐类别置信度阈值：

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-report outputs/xh25/baseline/report.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized
```

输出目录包含：

- `optimized-thresholds.yaml`：可复制到 `configs/xh25-mksnet-lite.yaml` 的 `class_thresholds`；
- `report.json`：优化阈值后的验证集评估；
- `comparison.json` 和 `comparison.md`：和 main 线 baseline 的对比；
- `search-summary.json` 和 `search-summary.md`：搜索网格、选择原因和 ship 类检查。

### 比赛评分优先实验

评分方案 `比赛评分方案-V1.5.pdf` 的初赛硬门槛是整体 Recall `>=0.85`、整体 FDR
`<=0.20`、单幅 `10000x10000` 图像推理时间 `<=20s`。通过硬门槛后，专家评分还会
参考 ship、aircraft、vehicle 各自 Recall/FDR 和总时效性 7 个排序信号。

当前 MKSNet-Lite 的阈值优化版配置为：

```bash
configs/xh25-mksnet-lite-thresholded.yaml
```

生成比赛评分代理报告：

```bash
.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-lite/threshold-optimized/report.json \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized \
  --experiment-name xh25-mksnet-lite-thresholded
```

构建 QHS/MS 轻度重采样训练集：

```bash
.venv/bin/xh-detect build-ship-balanced-xh25 \
  --source-root datasets/xh25 \
  --output-root datasets/xh25-ship-balanced \
  --qhs-factor 2 \
  --ms-factor 2
```

训练 ship-balanced MKSNet-Lite：

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-ship-balanced/dataset.yaml \
  --model configs/models/xh25-yolo26s-mksnet-lite.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-lite-ship-balanced \
  --no-resume
```

## 5. 无正式数据时的快速检查

不下载权重、不需要 GPU 的完整假检测器闭环：

```bash
.venv/bin/python -m pytest tests/test_e2e.py -v
```

可联网时运行 Ultralytics 的一轮 OBB 冒烟训练：

```bash
.venv/bin/yolo obb train \
  model=yolo26s-obb.pt \
  data=dota8.yaml \
  epochs=1 \
  imgsz=640 \
  device=0
```

该命令会下载公开样例数据和预训练权重。

## 6. DOTA-v1.5 数据准备

从 [DOTA 官方页面](https://captain-whu.github.io/DOTA/dataset.html) 获取
train、val 图像和 `labelTxt-v1.5`，整理为：

```text
datasets/DOTA-v1.5/
├── images/
│   ├── train/
│   └── val/
└── labelTxt/
    ├── train/
    └── val/
```

转换为飞机、舰船、车辆三类：

```bash
.venv/bin/xh-detect prepare-dota \
  --source-root datasets/DOTA-v1.5 \
  --output-root datasets/dota3
```

输出：

```text
datasets/dota3/
├── dataset.yaml
├── images/train
├── images/val
├── labels/train
└── labels/val
```

转换器保留无目标图像作为负样本，跳过损坏图像和 `difficult=1` 标注，并报告
无效标注数量。

## 7. 三类基线训练

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/dota3/dataset.yaml \
  --model yolo26s-obb.pt \
  --epochs 30 \
  --image-size 1024 \
  --device 0
```

默认结果目录：

```text
runs/train/baseline/
└── weights/
    ├── best.pt
    └── last.pt
```

训练参数固定 `seed=42` 和 `deterministic=true`。正式冲榜前应记录数据版本、
代码 commit、配置、最佳权重和训练曲线。

## 8. 单图与超大图推理

编辑 [configs/baseline.yaml](configs/baseline.yaml) 中的 `model_path`，然后：

```bash
.venv/bin/xh-detect infer \
  --image-path datasets/DOTA-v1.5/images/val/P0003.png \
  --config-path configs/baseline.yaml \
  --output-dir outputs/infer
```

输出包括：

- 标注后的 JPG；
- 经过严格校验的 COCO Detection JSON；
- 预处理、推理、后处理和总耗时 JSON；
- `outputs/infer/cache/` 下的可恢复切片缓存。

流水线会自动滑窗、补边、按类别阈值过滤、过滤内部切片边缘碎片、回映坐标并做
同类旋转框 NMS。CUDA OOM 时会自动降低 batch size。

## 9. 比赛规则评估与阈值扫描

```bash
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/infer/P0003.json \
  --ground-truth-json datasets/demo-ground-truth.json \
  --output-path outputs/evaluation/P0003-report.json

.venv/bin/xh-detect sweep-thresholds \
  --predictions-json outputs/infer/P0003.json \
  --ground-truth-json datasets/demo-ground-truth.json \
  --output-path outputs/evaluation/P0003-threshold-sweep.json
```

评估器按置信度降序执行一对一贪心匹配，重复预测计 FP，并输出整体、分类别和
分图 TP、FP、FN、Recall、FDR。

## 10. Gradio Demo

```bash
.venv/bin/xh-detect serve \
  --config-path configs/baseline.yaml \
  --host 0.0.0.0 \
  --port 7860
```

在租赁平台映射 7860 端口。页面支持：

- 上传普通图或超大图；
- OBB/HBB 显示切换；
- 三类目标计数和四阶段耗时；
- 下载 COCO Detection JSON；
- 可选上传当前图像的 COCO 真值并显示 Recall/FDR。

Demo 默认限制单并发，避免同一 GPU 同时运行多个大图任务。

## 11. 10,000×10,000 性能基准

PyTorch FP16：

```bash
.venv/bin/xh-detect benchmark \
  --config-path configs/baseline.yaml \
  --repeats 5 | tee outputs/benchmark/pytorch-fp16.json
```

首次执行会生成 `outputs/benchmark/synthetic-10000.png`。计时在图像已加载到
内存后开始，先预热一次，再输出总耗时及预处理、推理、后处理的 median/P95。

TensorRT FP16：

```bash
.venv/bin/xh-detect export-engine \
  --model-path runs/train/baseline/weights/best.pt \
  --image-size 1024 \
  --device 0

.venv/bin/xh-detect benchmark \
  --config-path configs/tensorrt.yaml \
  --repeats 5 | tee outputs/benchmark/tensorrt-fp16.json
```

不要使用切片缓存做正式速度验收；benchmark 命令已关闭缓存。

## 12. 质量检查

```bash
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest
.venv/bin/python -m pytest --cov=xh_detect --cov-report=term-missing
```

GPU、权重下载和 DOTA 数据不进入本地单元测试。提交前还应在干净服务器执行：

```bash
bash scripts/bootstrap_gpu_server.sh
.venv/bin/xh-detect env
.venv/bin/python -m pytest
.venv/bin/yolo obb predict \
  model=yolo26s-obb.pt \
  source=https://ultralytics.com/images/boats.jpg \
  imgsz=1024 \
  device=0
```

## 13. 一周推进建议

1. 第 1 天：4090 环境、DOTA8 冒烟、DOTA-v1.5 转换；
2. 第 2 天：30 epoch 三类基线并检查错误样本；
3. 第 3 天：普通图/大图 Demo 与阈值初扫；
4. 第 4 天：hard negatives、数据增强和分类别阈值；
5. 第 5 天：TensorRT 与 10k 性能优化；
6. 第 6 天：正式数据适配、消融和稳定性；
7. 第 7 天：最终权重、基准、Demo、提交 JSON 复核。

一天完成两到三天的计划量是可行的，但 GPU 训练、数据下载和错误分析存在串行
依赖。应以“可验证产物”推进，而不是只累计代码量。

## 14. 许可与数据使用

- DOTA 图像来自 Google Earth、卫星和航拍来源，使用前阅读
  [DOTA 数据说明](https://captain-whu.github.io/DOTA/dataset.html)及各图像来源条款；
- Ultralytics YOLO26 OBB 文档见
  [官方 OBB 指南](https://docs.ultralytics.com/tasks/obb/)；
- Ultralytics 代码和模型提供 AGPL-3.0 与 Enterprise 许可选项，商业或闭源使用前
  阅读[官方许可说明](https://docs.ultralytics.com/)；
- 比赛正式数据、模型权重、训练产物和输出结果不要提交到公开仓库，除非许可明确允许。
