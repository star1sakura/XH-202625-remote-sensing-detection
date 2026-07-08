# SPH-YOLOv5 Small-Object Experiment

This experiment adapts SPH-YOLOv5 ideas to the XH25 YOLO26-style detector. The
first trainable candidate is `sph-p2`, which adds a shallow P2 detection path for
FSC vehicle targets and uses four Detect scales: P2, P3, P4, and P5. Within this
runbook, `sph-full` means the approved SPH ablation that combines P2 + NAM + Swin
prediction blocks; it does not mean an MKSNet full-backbone replacement. NAM and
Swin variants are follow-up ablations.

## Baseline

| Candidate | Overall Recall | Overall FDR | Ship Recall | Ship FDR | Aircraft Recall | Aircraft FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| main / xh25-yolo26s-e80 | 0.961562 | 0.037244 | 0.823383 | 0.157761 | 0.989075 | 0.015942 | 0.705128 | 0.202899 |
| sph-p2 raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-p2 thresholded | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-p2-nam raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |
| sph-full raw | not_run | not_run | not_run | not_run | not_run | not_run | not_run | not_run |

## Local Smoke Tests

```bash
python -m pytest tests/test_sph_yolo.py tests/test_sph_configs.py -q
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo26s-sph-p2.yaml'); print(model.model.__class__.__name__)"
```

## Train P2 Candidate

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

## Evaluate P2 Candidate

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-sph-p2.yaml \
  --output-json outputs/xh25/sph-p2/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/sph-p2/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/sph-p2/report.json \
  --taxonomy xh25

.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/sph-p2/report.json \
  --output-dir outputs/xh25/sph-p2/competition-proxy \
  --experiment-name xh25-sph-p2
```

## Threshold Calibration

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/sph-p2/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-dir outputs/xh25/sph-p2/threshold-optimized \
  --taxonomy xh25 \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-name xh25-sph-p2-threshold-optimized
```

## Keep Criteria

Prefer the SPH candidate only if:

- Vehicle Recall is greater than 0.705128.
- Vehicle FDR is at most 0.202899, or threshold optimization reaches that value while keeping the recall gain.
- Ship Recall is at least 0.803383.
- Aircraft Recall is at least 0.984075.
- Overall Recall and Overall FDR pass the competition hard gates.
- `10000 x 10000` tiled inference on RTX3090 remains within the competition timing budget.
