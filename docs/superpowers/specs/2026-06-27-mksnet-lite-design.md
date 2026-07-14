# MKSNet-Lite Experiment Design

## Goal

Build a first-stage MKSNet-inspired YOLO experiment for the XH25 optical remote-sensing detection competition. The experiment should test whether a lightweight multi-kernel and dual-attention feature module improves competition-style Recall and FDR on ships, aircraft, and vehicles without replacing the current Ultralytics-based training, tiled inference, evaluation, and benchmark pipeline.

This is not a full MKSNet reproduction. It is a controlled adaptation that keeps the current project usable while giving the team a measurable innovation path.

## Motivation

The XH25 dataset is strongly imbalanced:

- 20,933 total boxes.
- 17,849 aircraft boxes.
- 2,682 ship boxes.
- 402 vehicle boxes.
- HM has 17 boxes and LQS has 30 boxes.

The competition requires high combined Recall, low FDR, and a 10,000 x 10,000 image inference time under 20 seconds on RTX 3090-class hardware. The MKSNet paper is relevant because it targets small object detection in remote-sensing imagery with multi-kernel feature extraction and dual attention. Those ideas map naturally to vehicle detection, rare ship categories, and small aircraft targets, but a full network rewrite would delay usable metrics.

## Recommended Approach

Create an independent experiment named `xh25-mksnet-lite`.

The experiment will:

1. Keep the existing official XH25 HBB data preparation unchanged.
2. Keep `xh-detect train`, `infer-dataset`, `evaluate`, `sweep-thresholds`, and `benchmark` as the experiment harness.
3. Add a lightweight MKSNet-inspired module that combines multi-kernel depthwise convolution with spatial and channel attention.
4. Register the module for Ultralytics model YAML parsing.
5. Add an experiment model/config path that trains into `runs/train/xh25-mksnet-lite`.
6. Compare baseline and experiment metrics using the same validation split and evaluation code.

## Model Design

### MKSNet-Lite Block

The first version should be small and stable:

- Input and output channel count stay the same.
- Multi-kernel branch uses depthwise convolution with kernel sizes such as 3, 5, and 7.
- Branch outputs are combined by summation or concatenation followed by a 1 x 1 projection.
- Channel attention uses global average pooling plus a small bottleneck MLP or 1 x 1 convolution pair.
- Spatial attention uses channel pooling followed by a small convolution, similar to a lightweight CBAM-style spatial gate.
- A residual connection preserves the original feature when attention weights are poorly initialized.

The block must be scriptable inside normal PyTorch/Ultralytics training and should not require custom CUDA kernels.

### Placement

The first experiment should avoid changing detection heads. Insert MKSNet-Lite in a small number of feature stages where it can improve small-object representation without creating a large latency cost.

Preferred first placement:

- One block in the neck where multi-scale features have already been fused.
- Optionally one block in the shallower feature path if the model YAML supports it cleanly.

Avoid inserting the block everywhere in the first pass. If the first run is promising, a second ablation can test more placements.

## Configuration

Add an experiment model YAML and pipeline config:

- Model YAML: a local Ultralytics-compatible model definition for `xh25-mksnet-lite`.
- Inference config: `configs/xh25-mksnet-lite.yaml`.
- Training run name: `xh25-mksnet-lite`.
- Model path after training: `runs/train/xh25-mksnet-lite/weights/best.pt`.

Training should initially mirror the current official baseline settings:

- Dataset: `datasets/xh25/dataset.yaml`.
- Task: HBB detect.
- Image size: 1024.
- Batch: 8 unless RTX 3090 memory requires lowering it.
- Epochs: match the baseline comparison run, not a one-epoch smoke run.
- AMP: follow baseline settings for a fair comparison.
- Seed: keep 42 through the existing training wrapper.

## Evaluation

Use the same validation split and existing evaluation tools for both baseline and MKSNet-Lite.

Required metrics:

- Official-style overall TP, FP, FN, Recall, and FDR.
- Coarse group metrics for ship, aircraft, and vehicle.
- 25-class diagnostic metrics, especially HM, LQS, and FSC.
- 10,000 x 10,000 benchmark median and P95 time.

The first experiment is promising only if it improves Recall or fixes a specific weak group without causing unacceptable FDR or latency regression.

## Comparison Protocol

Run the baseline and experiment with the same prepared dataset:

1. Prepare XH25 once with `val_ratio=0.15` and `seed=42`.
2. Train the baseline run or identify an existing baseline weight produced from the same split.
3. Train `xh25-mksnet-lite`.
4. Run `infer-dataset` for both weights.
5. Run `evaluate` and `sweep-thresholds` for both outputs.
6. Run `benchmark` for both inference configs.
7. Write a comparison report under `outputs/xh25/mksnet-lite/`.

The comparison report should make the trade-offs visible rather than only reporting the best number.

## Non-Goals

- Do not replace the current pipeline with a standalone MKSNet training framework in this phase.
- Do not claim paper reproduction.
- Do not alter official XH25 label taxonomy or validation split logic.
- Do not optimize only HM or LQS at the expense of overall competition metrics.
- Do not introduce TensorRT-specific work until PyTorch results show the experiment is worth keeping.

## Risks

- Custom Ultralytics module registration may be brittle across Ultralytics versions.
- Attention can increase false positives if it amplifies background texture.
- Extra feature blocks may reduce 10,000 x 10,000 throughput.
- Gains from larger image size or longer training can be mistaken for gains from MKSNet-Lite. The baseline comparison must match training settings.
- Full paper reproduction may still be needed later if the competition report needs stronger novelty.

## Success Criteria

The design is successful when implementation produces:

- A loadable MKSNet-Lite module with unit coverage for tensor shape preservation.
- A trainable Ultralytics model YAML using that module.
- A dedicated `xh25-mksnet-lite` config.
- A repeatable baseline-vs-experiment comparison workflow.
- A report that includes overall Recall/FDR, ship/aircraft/vehicle metrics, selected 25-class diagnostics, and 10k benchmark timing.

The experiment itself is successful only if measured validation results justify keeping the module.

## Follow-Up Path

If MKSNet-Lite improves the measured competition metrics, the next phase can move toward a closer MKSNet reproduction:

- Test additional placements of the multi-kernel module.
- Separate spatial attention and channel attention ablations.
- Add a stronger small-object feature path or P2 head.
- Compare against a full MKSNet-style backbone if time and GPU budget allow.

If it does not improve metrics, stop at the documented negative result and focus on data balancing, thresholding, hard negatives, or P2-style small-object detection instead.
