# Competition Scoring And Ship Fix Design

## Goal

Build the next MKSNet-Lite iteration around the official competition scoring scheme, then run a focused ship-class training experiment that improves the seven ranking signals without risking the hard pass/fail gates.

## Source Scoring Rules

The scoring reference is `C:\Users\feng\project\fight\比赛评分方案-V1.5.pdf`.

Initial pass/fail gates:

- Overall detection Recall must be at least `0.85`.
- Overall FDR must be at most `0.20`.
- Inference time for one `10000x10000` image must be at most `20s` on one RTX3090 or equivalent accelerator.

Additional ranking reference after passing the gates:

- Ship Recall.
- Ship FDR.
- Aircraft Recall.
- Aircraft FDR.
- Vehicle Recall.
- Vehicle FDR.
- Overall timeliness.

The project evaluator already matches the key metric definition: class-agnostic overall matching, coarse ship/aircraft/vehicle metrics, IoU threshold `0.50` for ship and aircraft, and IoU threshold `0.35` for vehicle.

## Current Baseline For This Iteration

The current MKSNet-Lite validation result with optimized class thresholds is:

| Metric | Value |
| --- | ---: |
| Overall F1 | `0.964754` |
| Overall Recall | `0.958772` |
| Overall FDR | `0.029190` |
| TP / FP / FN | `3093 / 93 / 133` |

Coarse ship remains the weakest ranking signal:

| Group | Recall | FDR | TP | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| ship | `0.800995` | `0.154856` | `322` | `59` | `80` |

Ship fine-class diagnosis:

| Class | Train Boxes | Val Boxes | Recall | FDR | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| HM | `15` | `2` | `0.50` | `0.00` | Too sparse to optimize directly |
| LQS | `25` | `5` | `0.80` | `0.20` | Too sparse to optimize directly |
| QHS | `544` | `97` | `0.64` | `0.22` | Primary ship target |
| MS | `1696` | `298` | `0.82` | `0.18` | Primary ship target |

## Chosen Approach

Use a competition proxy report and a ship-balanced data experiment before any larger architecture rewrite.

Rejected alternatives:

- Full paper reimplementation first: higher risk and slower feedback; it may also harm the `20s` timing gate.
- Threshold-only ship tuning: useful for reducing FDR, but it already lowered ship Recall and cannot recover missed detections.
- HM/LQS-specific training: their validation support is too small for stable experiment selection.

## Design

### 1. Competition Proxy Report

Add a small reporting layer that converts an existing `EvaluationReport` plus an optional timing measurement into the exact fields needed for competition decisions.

The report should contain:

- Hard gate status for overall Recall, overall FDR, and optional large-image latency.
- Overall Recall/FDR/TP/FP/FN.
- Coarse Recall/FDR/TP/FP/FN for ship, aircraft, and vehicle.
- A ranking proxy table with the seven ranking signals listed in the scoring PDF.
- A recommendation string:
  - `pass_candidate` when hard gates pass and timing is absent or passing.
  - `accuracy_gate_failed` when Recall or FDR misses the hard gate.
  - `timing_gate_failed` when latency exceeds `20s`.

This should be a report artifact, not a replacement for the evaluator. The evaluator remains the source of truth for matching.

### 2. Thresholded MKSNet-Lite Config

Create a separate inference config instead of overwriting the original MKSNet-Lite config.

Expected file:

- `configs/xh25-mksnet-lite-thresholded.yaml`

It should copy the current MKSNet-Lite pipeline settings and use the optimized thresholds:

- Global `0.30` for most classes.
- `QHS` class ID `2`: `0.40`.
- `A1_SU-35` class ID `4`: `0.55`.
- `A2_C-130` class ID `5`: `0.50`.

This config becomes the stable baseline for the next ship experiment.

### 3. Ship-Balanced Training Dataset Variant

Create a deterministic training dataset variant that keeps validation unchanged and increases the sampling frequency of images containing QHS and MS in the training split.

Expected behavior:

- Input: existing prepared `datasets/xh25`.
- Output: `datasets/xh25-ship-balanced`.
- Validation images and labels are linked or copied exactly once from `datasets/xh25/images/val` and `datasets/xh25/labels/val`.
- Training images and labels are linked or copied once for normal samples, and additional duplicated training entries are created for samples containing class IDs `2` or `3`.
- Duplicate names must be deterministic and must not change label contents.
- Dataset YAML points to the balanced train split and the unchanged validation split.
- A JSON and Markdown report records per-class original counts, duplicated sample counts, final train counts, and the duplication policy.

Initial duplication policy:

- QHS-containing train images: `2x` total frequency.
- MS-containing train images: `2x` total frequency.
- Images containing both QHS and MS: cap at `2x`, not `4x`.
- HM and LQS are not separately oversampled in this first experiment.

The policy is intentionally mild. It should nudge ship learning without overwhelming aircraft, which currently carries most true positives.

### 4. Training And Evaluation Experiment

Train one new MKSNet-Lite run on the ship-balanced dataset using the existing architecture and the existing 80 epoch recipe:

- Model: `configs/models/xh25-yolo26s-mksnet-lite.yaml`.
- Pretrained: `yolo26s.pt`.
- Epochs: `80`.
- Image size: `1024`.
- Batch: `8`.
- Workers: `4`.
- AMP: off.
- Seed and deterministic behavior remain fixed through the existing training wrapper.
- Run name: `xh25-mksnet-lite-ship-balanced`.

After training:

- Create `configs/xh25-mksnet-lite-ship-balanced.yaml` pointing to the new `best.pt`.
- Run validation inference and evaluation.
- Run threshold optimization from the new validation predictions.
- Write a competition proxy report comparing:
  - main-line baseline,
  - current MKSNet-Lite thresholded baseline,
  - new ship-balanced MKSNet-Lite with optimized thresholds.

### 5. Selection Criteria

The new ship-balanced result is preferred only if:

- Overall Recall remains at least `0.85`.
- Overall FDR remains at most `0.20`.
- Ship Recall improves versus current thresholded MKSNet-Lite, or ship FDR improves without lowering ship Recall by more than `0.01`.
- Overall F1 does not fall by more than `0.002` versus current thresholded MKSNet-Lite.
- Large-image timing remains at or below `20s` if timing is measured in this iteration.

If the result fails these criteria, keep the current thresholded MKSNet-Lite as the competition candidate and use the ship-balanced report as evidence for the next experiment.

## Data Flow

```text
datasets/xh25
  -> build ship-balanced dataset
  -> train xh25-mksnet-lite-ship-balanced
  -> infer validation predictions
  -> evaluate with official-style matching
  -> optimize thresholds
  -> produce competition proxy report
  -> compare with current thresholded baseline
```

## Error Handling

- Refuse to create the balanced dataset if the source dataset is missing `dataset.yaml`, train/val image folders, train/val label folders, or class names.
- Refuse unsafe output paths that overlap the input dataset root.
- Refuse duplicate output filenames before writing.
- Refuse non-positive duplication factors.
- Do not alter `datasets/xh25`; all generated data goes under a new output root.
- If training or inference fails on the server, preserve logs and report the failing command plus the latest checkpoint state.

## Testing Strategy

Unit tests:

- Competition proxy report computes hard gate pass/fail and coarse metric fields from an `EvaluationReport`.
- Timing gate is optional and fails only when provided latency exceeds `20s`.
- Ship-balanced dataset builder duplicates only train samples containing class IDs `2` or `3`.
- Images containing both QHS and MS are capped at `2x`.
- Validation split remains unchanged.
- Dataset YAML and reports are deterministic.

Integration checks:

- CLI command creates `configs/xh25-mksnet-lite-thresholded.yaml` or equivalent committed config.
- CLI command builds a small synthetic ship-balanced dataset.
- Server run writes validation predictions, evaluation report, optimized thresholds, comparison, and competition proxy report.

## Non-Goals

- Do not implement the full MKSNet paper in this iteration.
- Do not change the evaluator matching logic unless the scoring PDF changes.
- Do not optimize only HM/LQS in this iteration.
- Do not commit model weights, raw datasets, or large generated prediction files.

## Expected Deliverables

- A committed thresholded MKSNet-Lite config.
- A committed ship-balanced dataset builder and CLI entry.
- Tests for the new report and data builder behavior.
- Server artifacts under ignored `outputs/` and `datasets/` paths.
- A concise result summary with official hard-gate status, seven ranking proxy signals, and recommendation for the next competition candidate.
