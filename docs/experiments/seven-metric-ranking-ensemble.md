# Seven-Metric Ranking Ensemble

> **Submission status: inadmissible.** The competition submission accepts one
> model weight, while this experiment runs three independent detector weights.
> Keep it only as a teacher upper bound for single-student training.

## Objective

The V1.5 competition document ranks seven signals: aircraft, ship, and
vehicle Recall/FDR (six items), plus end-to-end time for one 10000 x 10000
image. AP and mAP are not ranking signals.

## Frozen Ensemble

The selected policy is implemented by the fuse-ranking-ensemble command:

| Role | Checkpoint | Threshold |
| --- | --- | ---: |
| Aircraft primary | MKSNet-Lite | 0.25 |
| Ship primary | SPH-P2-NAM | 0.31 |
| Vehicle primary | MKSNet-Lite | 0.25 |
| Vehicle supplement | SPH-P2 | 0.74 |

Vehicle supplements are accepted only when their HBB IoU with every selected
vehicle prediction in the same image is below 0.30. The 0.74 supplement point
was selected from the vehicle Pareto frontier: it gives 58 TP / 12 FP instead
of the less robust 0.64 point's 59 TP / 15 FP.

## Fixed Validation Result

The evaluator uses IoU 0.35 for vehicle and 0.50 for aircraft/ship, matching
the competition document.

| Ranking item | Historical main rerun | Ensemble | Result |
| --- | ---: | ---: | --- |
| Aircraft Recall | 0.988711 (2715 TP) | 0.990896 (2721 TP) | improved |
| Aircraft FDR | 0.015948 (44 FP) | 0.014130 (39 FP) | improved |
| Ship Recall | 0.823383 (331 TP) | 0.835821 (336 TP) | improved |
| Ship FDR | 0.157761 (62 FP) | 0.151515 (60 FP) | improved |
| Vehicle Recall | 0.705128 (55 TP) | 0.743590 (58 TP) | improved |
| Vehicle FDR | 0.202899 (14 FP) | 0.171429 (12 FP) | improved |
| Median latency | 1.382013 s | 4.148135 s | regressed |

Summary: **6 improved, 0 tied, 1 regressed, but not submission-eligible.**

The ensemble also improves the unranked combined hard-gate metrics:

| Metric | Historical main rerun | Ensemble |
| --- | ---: | ---: |
| Overall TP / FP / FN | 3101 / 120 / 125 | 3115 / 111 / 111 |
| Overall Recall | 0.961252 | 0.965592 |
| Overall FDR | 0.037256 | 0.034408 |

The earlier saved historical-main result was 3102 / 120 / 124, with aircraft
2716 / 44 / 30. The same checkpoint rerun differed by one boundary aircraft
match. The ensemble still strictly improves all six accuracy ranking items
against either baseline result.

## RTX3090 Timing

Five sequential wall-clock repeats include all three detector calls and the
fusion post-process:

- median: 4.148135 s;
- p95: 4.452750 s;
- maximum: 4.483680 s;
- 20-second gate: passed.

## Reproduction

    .venv/bin/xh-detect infer-dataset \
      --images-dir datasets/xh25/images/val \
      --image-map-json datasets/xh25/manifests/val-image-map.json \
      --config-path configs/xh25-mksnet-lite.yaml \
      --output-json outputs/xh25/mksnet-lite/val-predictions.json

    .venv/bin/xh-detect infer-dataset \
      --images-dir datasets/xh25/images/val \
      --image-map-json datasets/xh25/manifests/val-image-map.json \
      --config-path configs/xh25-sph-p2-nam.yaml \
      --output-json outputs/xh25/sph-p2-nam/val-predictions.json

    .venv/bin/xh-detect infer-dataset \
      --images-dir datasets/xh25/images/val \
      --image-map-json datasets/xh25/manifests/val-image-map.json \
      --config-path configs/xh25-sph-p2.yaml \
      --output-json outputs/xh25/sph-p2/val-predictions.json

    .venv/bin/xh-detect fuse-ranking-ensemble \
      --aircraft-predictions outputs/xh25/mksnet-lite/val-predictions.json \
      --ship-predictions outputs/xh25/sph-p2-nam/val-predictions.json \
      --vehicle-primary-predictions outputs/xh25/mksnet-lite/val-predictions.json \
      --vehicle-supplement-predictions outputs/xh25/sph-p2/val-predictions.json \
      --image-map-json datasets/xh25/manifests/val-image-map.json \
      --output-json outputs/xh25/ranking-ensemble/val-predictions.json

    .venv/bin/xh-detect benchmark-ranking-ensemble --repeats 5

## Artifact Hashes

| Artifact | SHA256 |
| --- | --- |
| historical main checkpoint | 930cf7e1c698a8850523ce42d2565d1b2652e5ae01bf7f049a35d05778dd5424 |
| MKSNet-Lite checkpoint | 164223f1dcbe53278110cfa4d83018f668f4bca8c64cf260d967f2274329e881 |
| SPH-P2-NAM checkpoint | 6dbf8d39d43495f80435e2fab7ab559b8818f3b6329690b4d89f598c58d947c9 |
| SPH-P2 checkpoint | 72631a1c2a2a9e018c62c6be630c01f7b9f98423ce42972914ddd1a4180390f7 |
| ensemble predictions | 7898f12444639c5983372b027bdb5c17a2aa1810df2f2e9546a2d7bd2433a3fa |
| ensemble report | 9150f871df9742a77029af931a9c735e39119f661d40b928ef2aa8b8b9f24f0a |
| ensemble benchmark | e747d004abc31293425070ceefa982de8af7ef4b0a25e713bfb444e38c5b721c |
| seven-metric comparison | 4a054b9cfa858ce5777210fb41a3c71a3291ce8abe7c361dedfc017f8a401115 |

## Decision

**RETAIN HISTORICAL MAIN FOR SUBMISSION.** The ensemble proves that the saved
specialists contain enough complementary predictions to improve all six class
accuracy ranking items, but it violates the single-weight constraint. Use its
class-specific behavior only to guide a single-student checkpoint. Thresholds
were selected on the fixed validation split, so the official hidden set remains
the final generalization check.
