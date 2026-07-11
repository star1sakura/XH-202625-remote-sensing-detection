# Main Score Improvement Experiments

This runbook keeps `xh25-yolo26s-e80` as the fixed baseline and evaluates five
competition-oriented steps: model complementarity, ship suppression, train-only
hard negatives, FSC density assignment, and final threshold/latency selection.

## Fixed Constraints

- Split, seed, image size, epochs, batch size, and baseline architecture stay fixed.
- Validation predictions are never used to create training samples.
- Promotion requires overall Recall >= 0.961562, overall FDR <= 0.037244,
  ship Recall >= 0.823383, aircraft Recall >= 0.989075, vehicle Recall >=
  0.705128, and 10000 x 10000 latency <= 20 seconds.

## 1. Finish And Evaluate Main

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-yolo26s-e80.yaml \
  --output-json outputs/xh25/baseline/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/baseline/report.json

.venv/bin/xh-detect analyze-complementarity \
  --prediction main=outputs/xh25/baseline/val-predictions.json \
  --prediction sph-p2=outputs/xh25/sph-p2/val-predictions.json \
  --prediction mksnet-lite=outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-name main \
  --taxonomy xh25 \
  --output-path outputs/xh25/complementarity/main-report.json
```

## 2. Ship Suppression Ablation

Run once with `xh25-main-ship-iou.yaml`, then repeat with
`xh25-main-ship-diou.yaml`.

```bash
.venv/bin/xh-detect audit-false-positives \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/main-postprocess/baseline-fp-audit.json

.venv/bin/xh-detect apply-suppression \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-main-ship-diou.yaml \
  --output-json outputs/xh25/main-postprocess/diou-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/main-postprocess/diou-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/main-postprocess/diou-report.json
```

## 3. Build Main-HN

Upgrade the existing prepared dataset in place, infer train only, then build
the derived dataset. The builder rejects every candidate whose padded crop
intersects any train annotation.

```bash
.venv/bin/xh-detect publish-xh25-train-artifacts \
  --dataset-root datasets/xh25

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/train \
  --image-map-json datasets/xh25/manifests/train-image-map.json \
  --config-path configs/xh25-yolo26s-e80.yaml \
  --output-json outputs/xh25/main-hn/train-predictions.json

.venv/bin/xh-detect build-main-hn-xh25 \
  --source-root datasets/xh25 \
  --predictions-json outputs/xh25/main-hn/train-predictions.json \
  --output-root datasets/xh25-main-hn \
  --confidence-floor 0.60 \
  --crop-size 512 \
  --object-margin 16 \
  --max-crops-per-group 2 \
  --vehicle-multiplier 2 \
  --seed 42
```

## 4. Train Data And Density Ablations

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-main-hn/dataset.yaml \
  --model yolo26s.pt --pretrained yolo26s.pt \
  --epochs 80 --image-size 1024 --device 0 --batch 8 --workers 4 \
  --no-amp --project runs/train --name xh25-main-hn --no-resume

.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-main-hn/dataset.yaml \
  --model yolo26s.pt --pretrained yolo26s.pt \
  --epochs 80 --image-size 1024 --device 0 --batch 8 --workers 4 \
  --no-amp --project runs/train --name xh25-main-hn-density --no-resume \
  --density-assignment --density-constant 12 --density-threshold 0.25
```

The density run follows the SISE coauthored density-assigner rule: dense FSC
ground truths keep only the highest-quality positive candidate; sparse FSC and
all non-vehicle classes retain the default task-aligned top-k assignment.

## 5. Final Evaluation

For both candidate configs, run `infer-dataset`, `evaluate`,
`audit-false-positives`, `optimize-thresholds`, `competition-report`,
`compare-experiments`, and `benchmark --repeats 5`. Record TP, FP, FN, Recall,
FDR, and latency. Promote only a candidate satisfying every fixed constraint;
otherwise retain main.
