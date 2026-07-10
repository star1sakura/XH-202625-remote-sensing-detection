# Main-Line Score Improvement Design

## Goal

Improve the XH25 competition score without replacing the proven
`xh25-yolo26s-e80` main model. The next candidate must preserve the main
model's combined Recall/FDR result while improving at least one of those two
metrics and avoiding regressions hidden by aircraft-heavy class counts.

The work is intentionally staged:

1. measure whether post-processing can remove ship duplicate detections;
2. improve the training data seen by main with vehicle and ship hard examples;
3. only then test a narrowly scoped, vehicle-only density-aware assignment
   adaptation.

This is a score-improvement program, not another full-backbone reproduction.

## Competition And Baseline Facts

The official score uses combined Recall and FDR as hard gates, then ranks the
three coarse categories and inference latency separately. A candidate must run
on one RTX3090 within the 10,000 x 10,000 image timing budget.

The main baseline is `xh25-yolo26s-e80`:

| Metric | Main |
| --- | ---: |
| Overall Recall | 0.961562 |
| Overall FDR | 0.037244 |
| Aircraft Recall / FDR | 0.989075 / 0.015942 |
| Ship Recall / FDR | 0.823383 / 0.157761 |
| Vehicle Recall / FDR | 0.705128 / 0.202899 |

The prepared grouped split contains 3,807 train images and 674 validation
images, with no source-group overlap. It has 20,933 labeled objects:

| Coarse group | Train boxes | Val boxes | All boxes | Share |
| --- | ---: | ---: | ---: | ---: |
| aircraft | 15,103 | 2,746 | 17,849 | 85.3% |
| ship | 2,280 | 402 | 2,682 | 12.8% |
| vehicle/FSC | 324 | 78 | 402 | 1.9% |

At the 1,024 input scale, the median equivalent vehicle side length is about
49 pixels, versus 120 pixels for aircraft and 160 pixels for ships. Therefore,
overall Recall can mask vehicle regression, while ship false positives can
materially affect FDR.

## Paper Basis

### Ship Suppression

The Shanghai Institute of Satellite Engineering coauthored the remote-sensing
ship YOLOv5 paper "Detection method of remote sensing image ship based on
YOLOv5". Its relevant ideas are CIoU box regression and DIoU-NMS for dense or
overlapping ships. The first experiment applies only the suppression idea,
because the current score counts duplicate detections as false positives.

### Vehicle Density Assignment

The Shanghai Institute of Satellite Engineering coauthored "Vehicle Detection
in High-Resolution Aerial Images with Parallel RPN and Density-Assigner". The
paper targets tiny and dense aerial vehicles by improving positive assignment.
Its complete two-stage architecture is out of scope. The later experiment will
adapt the principle to the existing YOLO training path through vehicle-only
density-aware positive weighting/assignment.

### Prior Results That Bound Scope

SPH-P2 increased vehicle Recall but created excessive vehicle false positives.
MKSNet-Lite reduced FDR but did not improve vehicle Recall. SPH-Full and the
full MKSNet adaptation both failed to beat main under competition-style
Recall/FDR. These results rule out another full feature-extractor replacement
as the next experiment.

## Candidate Sequence

### Phase A: Main Post-Processing Audit

Create an immutable validation-prediction artifact for main and evaluate these
post-processing variants independently:

1. existing class threshold search, retained as the calibration reference;
2. class-aware tiled-prediction merge IoU for ship classes;
3. DIoU-NMS or DIoU-based merge suppression for ships, with the current merge
   behavior retained for aircraft and vehicles unless metrics justify a change.

The audit must report ship duplicate/overlap false positives separately from
background false positives. It must also compare the original and filtered
predictions using the existing official-style evaluator. Post-processing never
changes the validation ground truth or trains on validation samples.

Keep a Phase A candidate only when its overall Recall is at least the main
baseline and its overall FDR is lower, with no ship Recall regression. A
post-processing-only gain is a valid short-term submission candidate, but not
the project's sole innovation claim.

### Phase B: Hard-Negative Main Training

Build `main-hn` from the existing main architecture and default training
configuration. The new dataset builder operates only on original training
source groups:

1. infer train images using the current main weights;
2. match predictions against train labels with the competition IoU thresholds;
3. collect high-confidence ship or vehicle false positives whose expanded crop
   contains no labeled object;
4. write those crops with empty labels as hard-negative images;
5. create a deterministic train manifest that modestly upweights images
   containing FSC vehicles without duplicating validation or source-group
   images.

The hard-negative crop margin, confidence floor, maximum crops per source
group, and vehicle sampling multiplier are explicit configuration values. A
crop overlapping a labeled target is rejected. This prevents accidental false
negative labels and avoids using validation labels for training decisions.

Train `main-hn` with the same image size, epoch count, device, and evaluation
pipeline as main. Keep its architecture, pretrained initialization, and base
augmentation unchanged so that the data intervention is measurable.

### Phase C: Vehicle Density-Aware Main Training

Run this phase only if `main-hn` preserves FDR but vehicle Recall remains at or
below main. Create `main-hn-density` by extending the current assignment/loss
path only for class 24 (FSC):

- compute a local density signal from nearby FSC ground-truth centers and box
  scales during training;
- use that signal to expand or increase the weight of valid small-vehicle
  positive assignments;
- leave aircraft and ship assignment behavior unchanged;
- retain the Phase B hard-negative data and the main detection architecture.

This is an inspired adaptation of the Density-Assigner paper, not a claim of a
full Parallel RPN reproduction. The scope keeps the change compatible with the
current Ultralytics YOLO detector and 3090 timing target.

## Interfaces And Artifacts

The implementation will add bounded components rather than put data mining,
suppression, and training policy into the CLI directly:

- a post-processing module for class-aware merge/suppression policy;
- a hard-negative builder that writes images, empty labels, and a manifest;
- a vehicle-density assignment helper called by the existing training path;
- dedicated model/runtime YAML files for `main-postprocess`, `main-hn`, and
  `main-hn-density`;
- experiment runbooks and comparison reports under `outputs/xh25/`.

Every component must be deterministic from a seed, reject invalid paths and
target-overlapping negative crops, and leave the current main configuration
unchanged.

## Evaluation And Promotion Rules

For every phase, run the same fixed validation inference, official-style
evaluation, threshold optimization, competition proxy, and 10,000 x 10,000
RTX3090 benchmark.

Promote a new candidate over main only if all conditions hold:

1. Overall Recall is at least 0.961562.
2. Overall FDR is at most 0.037244.
3. Ship Recall is at least 0.823383.
4. Vehicle Recall is at least 0.705128.
5. Aircraft Recall is at least 0.989075.
6. At least one overall metric is strictly better than main.
7. End-to-end 10,000 x 10,000 inference remains within 20 seconds on RTX3090.

In addition to ratios, the result report must include TP, FP, and FN counts for
overall, aircraft, ship, and vehicle. With only 78 vehicle boxes in validation,
count changes are necessary to interpret small ratio differences.

## Tests

Add focused tests before implementing each component:

- DIoU suppression keeps the highest-score overlapping detection and preserves
  non-overlapping detections;
- class-aware suppression uses the default policy when no class override is
  configured;
- hard-negative crops never overlap any annotated object after the configured
  margin;
- generated empty labels and manifests are deterministic for a fixed seed;
- source-group and validation paths are rejected by the hard-negative builder;
- vehicle density weighting affects only class 24 and preserves tensor/loss
  shapes for all classes;
- all new YAMLs load through the current model and inference configuration
  parsers.

## Risks And Stop Rules

| Risk | Mitigation |
| --- | --- |
| Ship FPs are background errors rather than duplicates | Report FP source before training; skip DIoU-NMS as a candidate if it cannot reduce FDR without Recall loss. |
| Hard-negative crops introduce unlabeled targets | Reject any crop with an expanded overlap against original labels. |
| Vehicle upweighting increases FDR like SPH-P2 | Use hard negatives first, keep the sampling multiplier conservative, and reject any run below the promotion rules. |
| Density-aware assignment destabilizes training | Restrict it to class 24, keep main as the default assignment path, and train it only after Phase B evidence. |
| Validation tuning overfits the holdout | Mine only training source groups; use validation only for fixed candidate comparison. |
| Improvements miss the timing limit | Benchmark each candidate before promotion and retain main as the latency-safe fallback. |

## Acceptance

The design is complete when:

- the main post-processing audit produces a reproducible comparison;
- `main-hn` is trained and evaluated against the fixed main baseline;
- `main-hn-density` is attempted only if its stated entry condition is met;
- each candidate has validation, calibrated, competition-proxy, and timing
  artifacts; and
- the recommended submission candidate is selected exclusively by the
  promotion rules above.
