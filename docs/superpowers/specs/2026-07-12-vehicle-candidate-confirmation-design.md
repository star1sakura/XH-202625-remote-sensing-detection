# Vehicle Candidate Confirmation Design

## Objective

Improve the official vehicle Recall of the strongest historical main model
without worsening its vehicle FDR, aircraft metrics, ship metrics, or the
20-second inference gate. The historical main checkpoint is the fixed base;
the primary detector is not retrained in this experiment.

## Fixed Baseline

The baseline checkpoint is supplied outside Git as `best.pt` with SHA256:

```text
930CF7E1C698A8850523CE42D2565D1B2652E5AE01BF7F049A35D05778DD5424
```

Its training metadata is YOLO26s, XH25 25-class HBB, 80 epochs, image size
1024, batch 8, seed 42, deterministic training, and AMP disabled. Before any
candidate work, it must reproduce the historical official-style report:

| Metric | Recall | FDR |
| --- | ---: | ---: |
| Overall | 0.961562 | 0.037244 |
| Aircraft | 0.989075 | 0.015942 |
| Ship | 0.823383 | 0.157761 |
| Vehicle | 0.705128 | 0.202899 |

If the fixed split and evaluator do not reproduce those values, stop and
resolve the baseline mismatch before tuning a vehicle confirmer.

## Evidence

Against the reproduced main predictions, SPH-P2 recovered 15 vehicle truths
missed by main but produced 35 vehicle false positives. MKSNet-Lite recovered
11 missed vehicle truths with 9 vehicle false positives. This establishes
proposal complementarity but not a deployable fusion rule. Main-HN and
main-HN-density improved general mAP while failing the official vehicle
Recall/FDR objective, so another full detector retraining is excluded from
this phase.

## Architecture

The runtime candidate consists of three bounded stages:

1. Historical main runs normally and supplies the complete aircraft, ship,
   and vehicle result set.
2. SPH-P2 runs as a high-recall proposal teacher. Only class-24 FSC boxes are
   retained. A proposal overlapping an existing main vehicle detection at
   official vehicle IoU 0.35 or higher is discarded to prevent duplicate FP.
3. A lightweight vehicle confirmer classifies a context crop around each
   remaining proposal. Only accepted proposals are appended to main. Existing
   main detections are never deleted or rescored.

This boundary guarantees exact aircraft and ship invariance. MKSNet-Lite is
used only for offline complementarity analysis and training diagnostics; it
is not part of runtime inference.

## Phase 0: Feasibility Gates

Before training a confirmer:

1. Run the historical main checkpoint on the fixed validation split and
   reproduce the baseline report.
2. Generate train-only predictions for historical main, SPH-P2, and
   MKSNet-Lite using stable train image IDs.
3. Measure SPH-only, MKS-only, and SPH/MKS-consensus vehicle TP and FP sets
   under official 0.35 IoU matching.
4. Benchmark historical main and `main + SPH-P2` on one 10000 x 10000 image,
   with five measured repetitions after warm-up.

Continue only when SPH-P2 has at least three train-holdout recoverable vehicle
TPs and `main + SPH-P2` leaves at least one second inside the 20-second limit.
The one-second reserve covers crop extraction, batched confirmation, fusion,
and runtime variance. If the proposal pair exceeds 19 seconds, stop this
runtime design and prepare a separate single-model distillation design.

## Confirmer Dataset

All confirmer examples come from `datasets/xh25` train source groups.
Validation images, labels, predictions, and error analysis are forbidden as
training inputs.

For each SPH-P2 class-24 train proposal not duplicated by main:

- positive: the proposal matches one unused FSC truth at IoU 0.35 or higher;
- negative: the proposal does not match an unused FSC truth;
- ambiguous duplicate: another higher-score proposal already matched the same
  truth; retain it as a negative because the official evaluator counts it FP.

Assign these labels by processing SPH proposals in descending score and stable
original-index order, exactly matching the official greedy matching procedure.

Create a square context crop centered on the proposal with side length twice
the longer HBB side, clamped to at least 64 pixels and at most 256 pixels, then
resize it to 160 x 160. Preserve aspect ratio with zero padding. Store the SPH
score and normalized box width/height as scalar features alongside the image.

Split these examples 80/20 by source group with seed 42. No source group may
appear in both confirmer train and internal holdout. Class imbalance is handled
with a weighted sampler; do not duplicate the external validation set.

## Confirmer Model And Decision

Use a torchvision MobileNetV3-Small image encoder with a small MLP that joins
the image embedding, SPH confidence, and normalized box dimensions. Initialize
from ImageNet weights and train a binary vehicle-correctness target. The model
must support batched inference and TensorRT export.

Select the SPH proposal floor and confirmer acceptance threshold using only the
train-group holdout. The selected operating point must satisfy both:

- at least three additional matched vehicle truths;
- the added candidate FP/TP ratio is low enough that fused vehicle FDR does not
  exceed the baseline value.

For baseline vehicle counts `TP_b` and `FP_b`, accepted additions `TP_a` and
`FP_a` must satisfy:

```text
(FP_b + FP_a) / (TP_b + TP_a + FP_b + FP_a) <= 0.202899
```

Freeze both thresholds before the single fixed-validation evaluation.

## Fusion

Fusion is deterministic:

1. Copy all main detections unchanged.
2. Sort accepted SPH vehicle proposals by descending confirmer probability,
   descending SPH score, then original index.
3. Reject a proposal with HBB IoU 0.35 or higher against any main vehicle or
   previously accepted proposal.
4. Append remaining class-24 proposals with a calibrated fused score.

The fused score is used only for deterministic official matching order. It is
fit on the train-group holdout and must never reorder or alter existing main
detections.

## Evaluation And Promotion

Run the official-style evaluator on the fixed validation predictions. Promotion
requires all conditions:

1. Overall Recall >= 0.961562 and Overall FDR <= 0.20.
2. Aircraft Recall >= 0.989075 and Aircraft FDR <= 0.015942.
3. Ship Recall >= 0.823383 and Ship FDR <= 0.157761.
4. Vehicle Recall is strictly greater than 0.705128.
5. Vehicle FDR <= 0.202899.
6. At least three additional vehicle TP are recovered.
7. Five-run 10000 x 10000 mean latency and every measured run are <= 20 seconds.

General Precision, Recall, mAP50, and mAP50-95 are reported for diagnostics but
do not override these official seven-item constraints.

## Failure Handling

- Baseline mismatch: stop; do not tune against a different main.
- Insufficient SPH complementarity on train holdout: retain historical main.
- Proposal pair latency above 19 seconds: stop runtime fusion and design
  single-model distillation.
- Confirmer cannot meet the holdout TP/FDR constraint: retain historical main.
- Validation improves Recall but worsens vehicle FDR or another official item:
  reject the candidate without further validation tuning.

## Testing

Add focused tests for official vehicle matching, duplicate labeling, source
group isolation, crop bounds and padding, deterministic fusion, unchanged
aircraft/ship output, FDR constraint calculation, threshold freezing, model
batch shapes, and latency-report parsing. Integration tests use synthetic
predictions; generated datasets, checkpoints, predictions, and credentials
remain untracked.

## Artifacts

Generated artifacts use these ignored locations:

```text
outputs/xh25/historical-main/
outputs/xh25/vehicle-confirmation/
datasets/xh25-vehicle-confirmer/
runs/train/xh25-vehicle-confirmer/
```

The final report records the checkpoint hash, split hashes, proposal and
confirmation thresholds, TP/FP/FN for all three coarse classes, all seven
official ranking inputs, mAP diagnostics, and five-run latency measurements.
