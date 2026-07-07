# MKSNet-v2-full Vehicle Experiment

This experiment implements a MKSNet-style full backbone adapted to the XH25 YOLO HBB detection pipeline. It is compared against `xh25-yolo26s-e80` and the previous `xh25-mksnet-lite` run.

## Local Smoke

```bash
python -m pytest tests/test_mksnet_v2.py tests/test_mksnet_configs.py -q
python -c "from xh_detect.models.ultralytics import register_custom_modules; from ultralytics import YOLO; register_custom_modules(); model=YOLO('configs/models/xh25-yolo-mksnet-v2-full.yaml'); print(model.model.__class__.__name__)"
```

## Training

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

## Raw Evaluation

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-mksnet-v2-full.yaml \
  --output-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-path outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --taxonomy xh25

.venv/bin/xh-detect competition-report \
  --report-json outputs/xh25/mksnet-v2-full-vehicle/report.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/competition-proxy \
  --experiment-name xh25-mksnet-v2-full-vehicle
```

## Threshold Calibration

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-v2-full-vehicle/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --output-dir outputs/xh25/mksnet-v2-full-vehicle/threshold-optimized \
  --taxonomy xh25 \
  --baseline-report outputs/xh25/baseline/report.json \
  --experiment-name xh25-mksnet-v2-full-vehicle-threshold-optimized
```

## Result Table

| Candidate | Overall Recall | Overall FDR | Ship Recall | Ship FDR | Vehicle Recall | Vehicle FDR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| main / xh25-yolo26s-e80 | 0.961562 | 0.037244 | 0.823383 | 0.157761 | 0.705128 | 0.202899 |
| MKSNet-Lite thresholded | 0.958772 | 0.029190 | 0.800995 | 0.154856 | 0.692308 | 0.129032 |

After raw and calibrated evaluation complete, copy the six printed metrics from Task 5 Step 7 into a dated result note under `outputs/xh25/mksnet-v2-full-vehicle/`.

## Keep Criteria

- Overall Recall >= 0.85.
- Overall FDR <= 0.20.
- Vehicle Recall >= 0.735128, which is main vehicle recall plus 0.03.
- Raw Vehicle FDR <= 0.25.
- Ship Recall >= 0.803383, which is main ship recall minus 0.02.
- Aircraft Recall drop <= 0.005 versus main.
