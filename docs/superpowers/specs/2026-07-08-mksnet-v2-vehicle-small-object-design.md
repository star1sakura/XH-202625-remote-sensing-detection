# MKSNet-v2 Vehicle Small-Object Design

## Goal

Build a closer MKSNet reproduction for the XH25 optical remote-sensing detector, focused on improving the vehicle/FSC small-object score versus the main `xh25-yolo26s-e80` baseline.

This is not a threshold-only experiment. Threshold search remains a final deployment calibration step, while the primary change must be a model-architecture change derived from the MKSNet paper: Multi-Kernel Selection plus spatial and channel dual attention.

## Paper Basis

Target paper:

- Jiahao Zhang, Xiao Zhao, Guangyu Gao, "MKSNet: Advanced Small Object Detection in Remote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms", MMM 2025, LNCS 15521, pp. 394-407.
- DOI: https://doi.org/10.1007/978-981-96-2061-6_29
- BIT publication page: https://pure.bit.edu.cn/en/publications/mksnet-advanced-small-object-detection-inremote-sensing-imagery-w/
- Springer page: https://link.springer.com/chapter/10.1007/978-981-96-2061-6_29

The paper targets remote-sensing small-object detection. Its relevant mechanisms are:

- Multi-Kernel Selection: large and different-size convolutional kernels capture broader contextual information and adaptively select useful spatial scales.
- Spatial Attention: feature maps are spatially reweighted to focus on target regions and suppress redundant background.
- Channel Attention: channels are reweighted to improve feature representation and detection accuracy.

## Current Evidence

Main line baseline, `xh25-yolo26s-e80`, remains the strongest stable comparison:

| Metric | Main baseline |
| --- | ---: |
| Overall Recall | 0.961562 |
| Overall FDR | 0.037244 |
| Ship Recall | 0.823383 |
| Ship FDR | 0.157761 |
| Vehicle Recall | 0.705128 |
| Vehicle FDR | 0.202899 |

The first MKSNet-Lite experiment did not improve vehicle recall:

| Candidate | Overall Recall | Overall FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: |
| MKSNet-Lite thresholded | 0.958772 | 0.029190 | 0.692308 | 0.129032 |
| MKSNet ship-priority | 0.960322 | 0.030663 | 0.602564 | 0.145455 |

Interpretation: the earlier lightweight neck-only insertion reduced false alarms but did not recover more vehicle targets. The next experiment must move closer to the paper's small-object design instead of reusing the ship-balanced setup.

## Design Principles

1. Keep the main line as the comparison anchor.
   - The new model should be trained on `datasets/xh25/dataset.yaml`, not the ship-balanced dataset.
   - Training settings should mirror `xh25-yolo26s-e80` unless a change is explicitly part of the architecture experiment.

2. Reproduce the MKSNet mechanisms more directly.
   - Implement a full MKS block with adaptive branch selection, not only summed parallel kernels.
   - Include both spatial and channel attention.
   - Use larger kernels than the first lite version, while controlling cost with depthwise or grouped convolution.

3. Focus the placement on small-object features.
   - Vehicle/FSC is the main weak class and is expected to benefit most from shallow or mid-level feature enhancement.
   - The first implementation should strengthen existing P3/P4 feature paths without adding a new detection head.
   - A P2 detection head is a second-stage YOLO adaptation, not part of the first closer MKSNet reproduction.

4. Keep ablations separable.
   - MKS module effect should be evaluated before adding dataset balancing, P2 heads, or vehicle-specific sampling.
   - Threshold calibration should be reported separately from raw model improvement.

## Proposed Architecture

### MKSSelectionBlock

Create a new `MKSSelectionBlock` as a channel-preserving PyTorch module:

- Input/output shape: `[B, C, H, W] -> [B, C, H, W]`.
- Branches:
  - depthwise or grouped convolution with kernel sizes such as 5, 7, 9, and 11;
  - each branch followed by pointwise projection or normalization as needed;
  - padding preserves spatial size.
- Selection gate:
  - global pooling over the input or fused branch features;
  - small MLP or 1x1 convolution gate;
  - softmax over kernel branches;
  - weighted sum of branch outputs.
- Residual output:
  - `output = input + projected_selected_features` when channel dimensions match.

This is closer to MKSNet than the earlier `MKSNetLiteBlock`, which used parallel depthwise kernels and attention but did not strongly model adaptive kernel selection.

### DualAttentionBlock

Attach dual attention after multi-kernel selection:

- Spatial attention:
  - derive a spatial weight map from average/max pooled channel summaries or a lightweight convolution stack;
  - output one spatial mask `[B, 1, H, W]`;
  - multiply feature map by the mask.
- Channel attention:
  - squeeze spatial dimensions by average and/or max pooling;
  - small MLP or 1x1 layers produce channel weights `[B, C, 1, 1]`;
  - multiply feature map by channel weights.
- Attention order:
  - use `MKS -> Spatial Attention -> Channel Attention -> residual projection` for the first implementation.

### YOLO Integration

Add an Ultralytics-compatible model YAML:

- Start from the main YOLO26s-style HBB detector, not the ship-balanced MKSNet config.
- Insert MKS blocks in small-object sensitive locations:
  - after the P3 feature fusion path;
  - after the P4-to-P3 fusion or immediately before the P3 detect input;
  - optionally one P4 block if memory and speed remain acceptable.
- Do not add a P2 detection head in v2-a.

Model names:

- `MKSSelectionBlock`: reusable module.
- `configs/models/xh25-yolo26s-mksnet-v2.yaml`: first closer reproduction.
- Experiment name: `xh25-mksnet-v2-vehicle`.

## Training And Evaluation

### Training

Use the existing main dataset split:

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo26s-mksnet-v2.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-v2-vehicle \
  --no-resume
```

The first run should not use ship-balanced or vehicle-balanced duplication. If vehicle recall does not improve, the next design can add vehicle-focused data sampling as a separate factor.

### Evaluation

Run the same validation workflow used by main and MKSNet-Lite:

1. `infer-dataset` on `datasets/xh25/images/val`.
2. `evaluate` against `datasets/xh25/reports/val-ground-truth.json`.
3. `competition-report` for hard gates and ranking proxy.
4. Optional threshold optimization reported as a separate calibrated candidate.

Primary comparison table:

| Candidate | Overall Recall | Overall FDR | Ship Recall | Ship FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main / xh25-yolo26s-e80 | baseline | baseline | baseline | baseline | baseline | baseline |
| MKSNet-Lite | previous | previous | previous | previous | previous | previous |
| MKSNet-v2 raw | new | new | new | new | new | new |
| MKSNet-v2 calibrated | new | new | new | new | new | new |

## Success Criteria

The experiment is worth keeping if all required conditions pass:

- Overall Recall is at least 0.85.
- Overall FDR is at most 0.20.
- Vehicle Recall improves over main baseline by at least 0.03, targeting 0.75 or higher.
- Vehicle FDR does not exceed 0.25 in raw evaluation, and the calibrated candidate should target 0.20 or lower if possible.
- Ship Recall does not drop by more than 0.02 versus main baseline.
- Aircraft Recall does not drop by more than 0.005 versus main baseline.

If vehicle recall does not improve but FDR improves substantially, keep the run as a negative/precision-oriented ablation but do not make it the main competition candidate.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Large kernels increase latency | Use depthwise/grouped kernels; keep insertion count small; benchmark after training. |
| Module improves FDR but hurts recall | Report raw and calibrated results separately; use vehicle recall as the primary design metric. |
| P3-only insertion is insufficient for very small vehicles | Treat P2 head as v2-b, after v2-a gives a clean result. |
| Pretrained weight transfer becomes weaker | Keep YOLO26s-compatible surrounding layers and verify transferred parameter count. |
| Results are confused by data balancing | Do not use duplication or class-balanced sampling in the first run. |

## Non-Goals

- Do not claim a bit-for-bit reproduction of the Springer implementation unless official code is found.
- Do not add P2 detection head in the first implementation.
- Do not change evaluator matching rules.
- Do not optimize only thresholds and call it a model improvement.
- Do not train on the ship-balanced dataset for this vehicle-focused experiment.

## Deliverables

- `src/xh_detect/models/mksnet_v2.py`
  - `MKSSelectionBlock`
  - `DualAttentionBlock` or integrated dual attention implementation
- Ultralytics registration for the new module.
- `configs/models/xh25-yolo26s-mksnet-v2.yaml`
- `configs/xh25-mksnet-v2.yaml`
- Unit tests for shape preservation, kernel selection weights, registration, and YAML loading.
- Server training run on the 3090 instance.
- Evaluation artifacts under `outputs/xh25/mksnet-v2-vehicle/`.
- A concise comparison against main, MKSNet-Lite, and calibrated variants.

## Decision

Proceed with MKSNet-v2-a:

- closer MKSNet reproduction;
- vehicle/FSC as the primary target;
- no P2 head in the first run;
- no class-balanced data in the first run;
- threshold search only after raw evaluation.
