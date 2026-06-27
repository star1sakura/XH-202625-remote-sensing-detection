# XH25 YOLO26s e80 复现记录设计

## 目的

为已经完成的 `xh25-yolo26s-e80` 正式训练生成一份可审计、可复现且适合提交到私有
GitHub 仓库的记录。记录应让没有服务器访问权限的协作者理解训练输入、环境、参数、
主要结果、阈值选择和已知短板，同时不泄露比赛数据、服务器凭据或原始影像。

## 交付文件

创建以下受 Git 跟踪的文件：

```text
docs/experiments/
├── xh25-yolo26s-e80.md
└── assets/xh25-yolo26s-e80/
    ├── training-args.yaml
    ├── results.csv
    ├── results.png
    ├── box-f1-curve.png
    ├── box-pr-curve.png
    └── confusion-matrix-normalized.png
```

其中 Markdown 是主记录，YAML 和 CSV 提供机器可读的参数与逐 epoch 指标，PNG 仅包含
训练指标、置信度曲线和混淆统计，不包含原始遥感图像。

## 数据来源

复现记录以以下已保存产物为事实来源：

- 服务器训练目录中的 `args.yaml`、`results.csv` 和 Ultralytics 指标图；
- `outputs/xh25/server/train-e80.log` 的环境、优化器、完成时间和最终验证摘要；
- `outputs/xh25/val-evaluation-e80.json` 的 Recall/FDR 诊断；
- `outputs/xh25/threshold-sweep-e80-low005.json` 的阈值扫描；
- 本地 `configs/xh25-yolo26s-e80.yaml` 和正式 taxonomy；
- 本地数据 manifest、底模以及服务器权重的 SHA256。

报告中的数值必须能追溯到上述来源。自定义 Recall/FDR 与 Ultralytics mAP 使用不同评估
语义，必须分表呈现，不能混为同一指标。

## 主报告结构

`docs/experiments/xh25-yolo26s-e80.md` 包含：

1. 实验摘要和适用范围；
2. 代码提交、环境、GPU 和依赖版本；
3. 数据集规模、切分规则与 manifest 哈希；
4. 底模、训练命令、关键参数与优化器；
5. 权重、CSV 和曲线文件哈希；
6. Ultralytics 最佳权重指标、逐 epoch 峰值和末轮趋势；
7. 自定义评估器的整体及飞机、舰船、车辆 Recall/FDR；
8. 代表性阈值点、F1 最优阈值及使用建议；
9. 类别级主要短板和下一轮实验建议；
10. Demo 验收摘要、复现步骤和未提交产物说明。

报告明确标注：

- `best.pt` 的最终验证指标与 `results.csv` 单 epoch 记录不是同一轮输出；
- 阈值扫描用于 Demo 和自定义诊断，不替代比赛 mAP；
- 稀缺类 HM 只有两个验证实例，指标不稳定；
- 当前瓶颈集中在 QHS、MS、FSC 和部分飞机细分类别。

## 参数脱敏

`training-args.yaml` 保留影响训练结果的参数，包括 task、model、data、epochs、batch、
imgsz、device、workers、seed、deterministic、optimizer、AMP、学习率、损失权重和数据
增强设置。

删除或改写以下机器相关内容：

- 服务器绝对 `project` 路径；
- 服务器绝对 `save_dir` 路径；
- 不影响复现且会制造噪声的推理、展示和跟踪器默认参数。

使用仓库相对值 `project: runs/train` 和 `name: xh25-yolo26s-e80` 表达输出位置。

## 不提交内容

以下内容继续留在被忽略目录或服务器：

- `best.pt`、`last.pt` 和其他模型文件；
- 原始 6 MB 训练日志；
- 全量预测、逐图评估和阈值扫描 JSON；
- train/val batch 图、预测图、标签图和任何原始比赛影像；
- 数据集、SSH 密钥、密码、token、主机名、端口和本机绝对路径。

## 验证

提交前执行：

1. 对 Markdown、YAML 和 CSV 扫描凭据、主机名、绝对路径和占位符；
2. 校验 CSV 为 80 行且关键指标峰值与报告一致；
3. 重新计算阈值 F1，确认最优阈值和表格数值；
4. 校验报告中的 manifest、底模、权重和结果文件 SHA256；
5. 打开全部 PNG，确认没有原始影像、路径或裁剪错误；
6. 运行 `git diff --check`、Ruff 和完整 pytest；
7. 只 stage `docs/experiments/`、本设计及后续实施计划，不包含 `demo_assets/` 或
   `outputs/`。

## 发布

实现提交在功能分支完成并验证后合并到 `main`，随后推送 `origin/main`。推送后比较
本地和远程 `main` SHA，并通过 GitHub 确认私有仓库中的报告与资产可见。
