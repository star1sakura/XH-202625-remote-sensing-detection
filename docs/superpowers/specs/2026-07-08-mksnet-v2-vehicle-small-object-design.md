# MKSNet-v2 Full Vehicle Small-Object Design

## Goal

Build a substantially closer MKSNet reproduction for the XH25 optical remote-sensing detector, focused on improving the vehicle/FSC small-object score versus the main `xh25-yolo26s-e80` baseline.

This replaces the previous v2-a idea. The new direction is not only to insert a lightweight module into the YOLO neck. It should implement a MKSNet-style backbone with repeated MKS blocks, then connect that backbone to the existing Ultralytics-compatible detection head, training CLI, tiled inference, and competition evaluator.

The intended claim after implementation is:

- "MKSNet-style full backbone reproduction adapted to the XH25 YOLO detection pipeline."

The intended claim is not:

- "bit-for-bit reproduction of the authors' official code."

Official code has not been found in the current project. Public paper text is enough to reproduce the architecture principles, but not enough to guarantee identical hidden implementation details.

## Paper Basis

Target paper:

- Jiahao Zhang, Xiao Zhao, Guangyu Gao, "MKSNet: Advanced Small Object Detection in Remote Sensing Imagery with Multi-Kernel and Dual Attention Mechanisms", MMM 2025, LNCS 15521, pp. 394-407.
- DOI: https://doi.org/10.1007/978-981-96-2061-6_29
- BIT publication page: https://pure.bit.edu.cn/en/publications/mksnet-advanced-small-object-detection-inremote-sensing-imagery-w/
- Springer page: https://link.springer.com/chapter/10.1007/978-981-96-2061-6_29
- arXiv HTML copy used for implementation detail reading: https://arxiv.org/html/2512.03640

The public paper text describes these core pieces:

- MKSNet uses a sequence of MKS blocks.
- The network starts by splitting the input image into patches via a convolutional layer.
- MKS dynamically selects multiple kernel sizes to capture and integrate multi-scale contextual information.
- The Spatial Attention module uses convolution kernels with varying sizes and dilation rates, transforms multi-scale features to a common channel dimension, computes average and maximum channel summaries, then produces spatial weights with a convolution and sigmoid.
- The Channel Attention module is SENet-like: global average pooling and global max pooling, channel reduction, channel expansion, fusion, sigmoid weighting, and element-wise channel reweighting.
- A feature fusion module consolidates enhanced features for detection.

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

Interpretation: the earlier lightweight neck insertion reduced false alarms but did not recover more vehicle targets. A closer paper reproduction should change the feature extractor itself, especially the small-object and complex-background representation learned before neck fusion.

## Design Principles

1. Reproduce the paper's architecture level more directly.
   - Implement explicit MKS blocks, not only a YOLO plug-in block.
   - Implement patch/stem convolution, repeated MKS stages, spatial attention, channel attention, and feature fusion.
   - Keep module names and tests close to paper terminology.

2. Adapt only the detection wrapper to the competition.
   - The competition pipeline already supports YOLO-style HBB detection, tiled inference, JSON exports, and Recall/FDR evaluation.
   - The new backbone should emit multi-scale feature maps that can feed an Ultralytics-compatible PAN/FPN head and `Detect` layer.
   - This is a practical adaptation because the paper reports DOTA/HRSC mAP, while the competition scores Recall/FDR on XH25 classes.

3. Keep the main line as the comparison anchor.
   - Train on `datasets/xh25/dataset.yaml`, not the ship-balanced dataset.
   - Use the same image size, 80-epoch first run, and evaluation commands as main unless the custom backbone requires a clearly documented adjustment.

4. Treat P2 as a YOLO adaptation, not a MKSNet requirement.
   - MKSNet targets small objects through large/multi-kernel context and dual attention.
   - A P2 detection head may help tiny vehicles, but it is not the first "full MKSNet" step.
   - If the full backbone improves representation but still misses vehicles, add P2 as a separate v2-b experiment.

5. Separate raw model quality from threshold calibration.
   - Raw validation should be reported before class threshold search.
   - Calibrated thresholds may be used for competition submission, but should not be described as the architecture improvement.

## Proposed Architecture

### MKSChannelAttention

Create a reusable channel attention module:

- Input/output shape: `[B, C, H, W] -> [B, C, H, W]`.
- Compute global average pooling and global max pooling.
- Feed both descriptors through reduction and expansion layers.
- Fuse the two channel descriptors by average or learned weighted sum.
- Apply sigmoid and multiply the input feature map channel-wise.
- Expose `reduction` as a YAML parameter, defaulting to 16.

This follows the paper's CA description and is closer than the previous lite block, which used only average pooling.

### MKSSpatialAttention

Create a spatial attention module with explicit multi-kernel spatial feature extraction:

- Input/output shape: `[B, C, H, W] -> [B, C, H, W]`.
- Build several branches with increasing odd kernel sizes and dilation rates.
  - Initial values: `(3, 5, 7, 9)` kernels and `(1, 1, 2, 2)` dilations.
  - Use depthwise/grouped convolution when needed to keep RTX 3090 memory under control.
- Each branch applies convolution, batch normalization, activation, and a 1x1 channel transform to a common dimension.
- Add adaptive kernel selection over branches:
  - summarize branch features with global pooling;
  - pass the summary through a small gate;
  - apply softmax across kernel branches;
  - weight each branch before final fusion.
- Concatenate transformed branch outputs along the channel dimension.
- Compute channel-average and channel-maximum maps from the concatenated features.
- Feed the two-map summary through a convolution and sigmoid to produce a spatial attention map.
- Reweight multi-scale features, fuse them, project back to `C`, and add a residual connection.

This models the paper's SA and MKS flow more directly than simply averaging parallel depthwise kernels.

### MKSBlock

Create the paper-facing block that combines CA and SA:

- Input/output shape: `[B, C, H, W] -> [B, C, H, W]`.
- Default order: `Channel Attention -> Spatial Attention -> residual`.
- Add a constructor flag for the order, with allowed values:
  - `ca_sa`: default, matching the paper figure caption that describes CA followed by SA.
  - `sa_ca`: optional ablation, matching the order in which the public text explains SA before CA.
- Always preserve shape and channels.
- Use residual connections around the block to stabilize training from partial or scratch initialization.

This avoids the earlier ambiguity where the design described `MKS -> Spatial -> Channel` as if MKS were separate from attention. In the full design, CA and SA are the two major modules inside each MKS block.

### MKSNetBackbone

Create a backbone that replaces the YOLO26s backbone:

- Stem/patch embedding:
  - convolutional patch/stem layer at the image input;
  - batch normalization and activation;
  - no transformer tokenization, because the paper describes convolutional patch embedding.
- Stages:
  - stage 1: downsample to shallow high-resolution features;
  - stage 2: repeated MKS blocks for small-object detail;
  - stage 3: repeated MKS blocks for mid-level context;
  - stage 4: repeated MKS blocks for deeper semantic context;
  - optional stage 5 if needed to align with YOLO P5.
- Outputs:
  - return P3, P4, and P5 feature maps to the existing head;
  - keep P2 internally available for a later P2-head ablation but do not use it in the first full run.
- Suggested first scale:
  - channels: `[64, 128, 256, 512, 768]`;
  - block depths: `[1, 2, 2, 2]`;
  - kernels per MKS block: `(3, 5, 7, 9)`.

These values are sized for a single RTX 3090 and can be reduced if model construction or dry-run memory checks fail.

### Detection Head Integration

Use an Ultralytics-compatible model YAML:

- Replace the original YOLO backbone with `MKSNetBackbone` or explicit YAML layers that implement the same stages.
- Reuse the existing YOLO-style neck/head where practical:
  - PAN/FPN fusion over P3/P4/P5;
  - final `Detect` layer for 25 HBB classes;
  - `end2end: true` and `reg_max: 1` remain aligned with the current HBB setup unless model loading proves incompatible.
- First config name:
  - `configs/models/xh25-yolo-mksnet-v2-full.yaml`
- First pipeline config:
  - `configs/xh25-mksnet-v2-full.yaml`
- First run name:
  - `xh25-mksnet-v2-full-vehicle`

This gives us a fuller MKSNet backbone while preserving the competition infrastructure.

## Training And Evaluation

### Training

Use the existing main dataset split:

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25/dataset.yaml \
  --model configs/models/xh25-yolo-mksnet-v2-full.yaml \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-mksnet-v2-full-vehicle \
  --no-resume
```

Pretrained transfer expectations:

- The custom backbone will not fully match `yolo26s.pt`.
- The run should log transferred parameter count.
- If too few weights transfer and early training is unstable, run a second controlled training with scratch initialization or longer epochs, but record it as a separate candidate.

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
| MKSNet-v2-full raw | new | new | new | new | new | new |
| MKSNet-v2-full calibrated | new | new | new | new | new | new |

## Success Criteria

The experiment is worth keeping if all required conditions pass:

- Overall Recall is at least 0.85.
- Overall FDR is at most 0.20.
- Vehicle Recall improves over main baseline by at least 0.03, targeting 0.75 or higher.
- Vehicle FDR does not exceed 0.25 in raw evaluation, and the calibrated candidate should target 0.20 or lower if possible.
- Ship Recall does not drop by more than 0.02 versus main baseline.
- Aircraft Recall does not drop by more than 0.005 versus main baseline.
- Model latency remains plausible for the competition 3090 limit; if it is too slow, prune block depth before changing the evaluation pipeline.

If vehicle recall does not improve but FDR improves substantially, keep the run as a negative or precision-oriented ablation but do not make it the main competition candidate.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Full custom backbone transfers fewer pretrained weights | Log transfer count, run a construction test, and compare partial-pretrain versus scratch only as separate candidates. |
| Large kernels increase memory or latency | Use grouped/depthwise kernels, reduce block depth, and benchmark before long training if dry-run memory is high. |
| Full backbone hurts mature YOLO baseline performance | Keep main as anchor and treat full MKSNet as a research candidate until validation beats main on competition proxy. |
| Vehicle recall still does not improve | Add P2 head or vehicle-focused sampling only after the full backbone raw result is known. |
| Paper detail ambiguity remains | Keep implementation comments and docs explicit about choices: `ca_sa` default, configurable `sa_ca` ablation, and no claim of official-code identity. |

## Non-Goals

- Do not claim official bit-for-bit reproduction without official code.
- Do not change evaluator matching rules.
- Do not optimize only thresholds and call it a model improvement.
- Do not train on the ship-balanced dataset for this vehicle-focused full-reproduction run.
- Do not add P2 detection head in the first full run; keep it as a separate YOLO adaptation.

## Deliverables

- `src/xh_detect/models/mksnet_v2.py`
  - `MKSChannelAttention`
  - `MKSSpatialAttention`
  - `MKSBlock`
  - `MKSStage`
  - `MKSNetBackbone`
- Ultralytics registration for the new modules.
- `configs/models/xh25-yolo-mksnet-v2-full.yaml`
- `configs/xh25-mksnet-v2-full.yaml`
- Unit tests for:
  - shape preservation;
  - average plus max channel attention path;
  - multi-kernel spatial branch construction;
  - adaptive branch selection weights summing to 1;
  - `ca_sa` and `sa_ca` order validation;
  - Ultralytics custom module registration;
  - model YAML loading or dry construction.
- Server dry-run model construction on the 3090 instance.
- Server training run on the 3090 instance.
- Evaluation artifacts under `outputs/xh25/mksnet-v2-full-vehicle/`.
- A concise comparison against main, MKSNet-Lite, raw MKSNet-v2-full, and calibrated MKSNet-v2-full.

## Decision

Proceed with MKSNet-v2-full:

- implement a MKSNet-style backbone, not only neck insertion;
- preserve the existing competition training, inference, and evaluation pipeline;
- use vehicle/FSC as the primary target;
- use P3/P4/P5 detection first for a clean paper-derived run;
- keep P2 head, class-balanced sampling, and threshold tuning as separate later factors;
- describe the result as a close practical reproduction adapted to XH25, not official-code identity.
