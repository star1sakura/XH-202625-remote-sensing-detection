# SPH-YOLOv5 Small-Object Adaptation Design

## Goal

Build a SPH-YOLOv5-inspired YOLO experiment for the XH25 optical remote-sensing
competition, focused on improving the vehicle/FSC ranking signals while keeping the
current main YOLO pipeline usable as the safety baseline.

The implementation should adapt the paper's small-object ideas to this repository's
Ultralytics YOLO26-style detector instead of replacing the training, tiled inference,
evaluation, and competition proxy tools.

## Paper Basis

Primary paper:

- Hang Gong et al., "Swin-Transformer-Enabled YOLOv5 with Attention Mechanism for
  Small Object Detection on Satellite Images", Remote Sensing, 2022.
- Relevant ideas: an extra shallow prediction head for small objects, residual
  connections from shallow backbone features into the fusion path, NAM attention,
  and Swin Transformer encoder blocks in prediction heads.
- Relevance to the competition: one author affiliation is Shanghai Institute of
  Satellite Engineering, and the method targets satellite-image small object detection.

Supporting paper route:

- 2025 Shanghai Aerospace paper on constellation remote-sensing detection for
  embedded deployment.
- Relevant ideas for later experiments: coordinate attention, bidirectional feature
  pyramid fusion, and adaptive NMS for lower false alarms.

The first implementation focuses on the most directly useful SPH-YOLOv5 component:
the shallow P2 detection path. NAM and Swin blocks are implemented as follow-up
ablation stages only after the P2 path is loadable and trainable.

## Baseline Facts

Use `xh25-yolo26s-e80` as the baseline:

| Metric | Baseline |
| --- | ---: |
| Overall Recall | 0.961562 |
| Overall FDR | 0.037244 |
| Aircraft Recall | 0.989075 |
| Aircraft FDR | 0.015942 |
| Ship Recall | 0.823383 |
| Ship FDR | 0.157761 |
| Vehicle Recall | 0.705128 |
| Vehicle FDR | 0.202899 |

The data distribution explains the target:

| Group | Train boxes | Val boxes |
| --- | ---: | ---: |
| ship | 2280 | 402 |
| aircraft | 15103 | 2746 |
| vehicle/FSC | 324 | 78 |

FSC vehicle boxes are also much smaller than most other classes. In the prepared
validation split, the median FSC box is about 45 x 45 pixels at the 1024 crop scale,
with median area near 0.24 percent of the image. This makes shallow high-resolution
features more important than another full backbone replacement.

## Scope

Implement three experiment levels, but train them sequentially:

1. `sph-p2`
   - Add a P2 detection feature from the shallow backbone path.
   - Change Detect from three scales to four scales: P2, P3, P4, P5.
   - Keep the rest of the YOLO26-style backbone and neck as close to main as possible.

2. `sph-p2-nam`
   - Add NAM attention to the small-object and neck fusion features after `sph-p2`
     is verified.
   - Use it to test whether vehicle and ship false discovery can be reduced without
     suppressing recall.

3. `sph-full`
   - Add Swin-style prediction blocks after `sph-p2-nam` is verified.
   - Treat this as the closest SPH-YOLOv5 adaptation, not the first competition
     candidate.

Do not combine this with MKSNet full backbone replacement. MKSNet remains a separate
negative/ablation result because the completed full run lowered vehicle and ship
recall.

## Architecture

### P2 Path

Start from the current YOLO26s-style backbone used by the MKSNet-Lite YAML:

- Backbone stage after the second downsample provides a shallow feature suitable for
  P2-style detection.
- The existing head already fuses deeper features into P3, P4, and P5.

The new `sph-p2` head should:

- upsample the P3 neck output by 2x;
- concatenate it with the shallow backbone feature before the P3 downsample stage;
- process the fused feature with a lightweight C3k2 block;
- feed the resulting P2 feature into Detect alongside P3/P4/P5.

This keeps the strongest existing main-line behavior and adds only the shallow
small-object branch needed for FSC.

### NAM

Implement a channel-preserving `NAMBlock`:

- channel attention derived from BatchNorm scale weights;
- optional spatial attention derived from BatchNorm-normalized feature response;
- residual or multiplicative gating that preserves tensor shape.

Insert NAM only in the `sph-p2-nam` YAML, initially after the P2 and P3 fusion blocks.
Avoid putting it in every backbone stage until a smaller insertion is proven useful.

### Swin Prediction Block

Implement a channel-preserving `SwinPredictionBlock` for `sph-full`:

- windowed self-attention on feature maps;
- residual MLP path;
- shape-preserving output for Ultralytics YAML parsing.

Keep the first Swin variant conservative. If the full block is too heavy or unstable
on RTX3090, keep it as a paper-completeness ablation and do not make it the final
competition candidate.

## Files

Add or update:

- `src/xh_detect/models/sph_yolo.py`
- `src/xh_detect/models/__init__.py`
- `src/xh_detect/models/ultralytics.py`
- `configs/models/xh25-yolo26s-sph-p2.yaml`
- `configs/models/xh25-yolo26s-sph-p2-nam.yaml`
- `configs/models/xh25-yolo26s-sph-full.yaml`
- `configs/xh25-sph-p2.yaml`
- `configs/xh25-sph-p2-nam.yaml`
- `configs/xh25-sph-full.yaml`
- `tests/test_sph_yolo.py`
- `tests/test_sph_configs.py`
- `docs/experiments/sph-yolov5-small-object.md`

## Training And Evaluation

Train `sph-p2` first:

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo26s-sph-p2.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-sph-p2 \
  --no-resume
```

After training:

1. Run `infer-dataset` on the fixed validation split.
2. Run `evaluate`.
3. Run `competition-report`.
4. Run `optimize-thresholds` against the main baseline report.
5. Compare main, `sph-p2`, `sph-p2-nam`, and `sph-full` with the same fixed split.

## Keep Criteria

Prefer a SPH candidate only if it improves the seven competition ranking signals
without sacrificing the hard gate:

- Overall Recall remains at least 0.85.
- Overall FDR remains at most 0.20.
- Vehicle Recall improves over 0.705128.
- Vehicle FDR is at most 0.202899, or threshold optimization can bring it below that
  without losing the vehicle recall gain.
- Ship Recall does not drop by more than 0.02 versus main.
- Aircraft Recall does not drop by more than 0.005 versus main.
- 10000 x 10000 tiled inference remains within the RTX3090 timing budget.

If `sph-p2` increases vehicle recall but raises false discovery, continue with NAM or
threshold calibration. If `sph-p2` lowers vehicle recall, stop the SPH architecture
path and switch to vehicle-focused data augmentation around main.

## Risks

| Risk | Mitigation |
| --- | --- |
| Four-scale Detect makes pretrained transfer partial | Use `--pretrained yolo26s.pt`, inspect transferred parameter count, and smoke-load YAML before training. |
| P2 improves vehicle recall but increases FDR | Evaluate NAM, class thresholds, and merge IoU before rejecting the path. |
| Swin head is too heavy for RTX3090 timing | Keep Swin as an ablation; final candidate can be `sph-p2` or `sph-p2-nam`. |
| Small vehicle validation count makes metrics noisy | Compare TP/FP/FN counts, not only ratios; keep main as safety baseline. |
| Ship/aircraft regression offsets vehicle gain | Keep ship and aircraft drop limits in the keep criteria. |

## Acceptance

The design is complete when:

- SPH modules are unit-tested.
- All SPH YAML files parse and smoke-load with `DetectionModel`.
- Pipeline configs load.
- The first `sph-p2` training/evaluation report is produced.
- Results are documented against main using Overall, Ship, Aircraft, and Vehicle
  Recall/FDR.
