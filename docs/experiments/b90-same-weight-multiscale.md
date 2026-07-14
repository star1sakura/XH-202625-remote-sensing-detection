# B90 Same-Weight Multiscale Result

This experiment uses one checkpoint at three input sizes. It does not fuse
predictions from different checkpoints. Latency is intentionally excluded from
the promotion gate.

## Checkpoint

- Path: `outputs/xh25/all-metrics-search/b90-multiscale/final/best-all-metrics.pt`
- SHA256: `f0b8c2dcf439437035d1f7afc2211443d411593a02a7f4ce9f5d6e5ad1423b17`
- Sizes: 1024, 1280, and 1536

## Formal Result

| Metric | Alpha050-Fine / Common Gate | B90 Multiscale | Passed |
| --- | ---: | ---: | --- |
| Aircraft Recall | 0.992353 | 0.995994 | Yes |
| Aircraft FDR | 0.011965 | 0.011565 | Yes |
| Ship Recall | 0.833333 | 0.845771 | Yes |
| Ship FDR | 0.149746 | 0.147870 | Yes |
| Vehicle Recall | 0.730769 | 0.782051 | Yes |
| Vehicle FDR | 0.185714 | 0.175676 | Yes |
| Ultralytics Precision | 0.939505 | 0.943814 | Yes |
| Ultralytics Recall | 0.919452 | 0.934280 | Yes |
| Ultralytics mAP50 | 0.948095 | 0.954427 | Yes |
| Ultralytics mAP50-95 | 0.767672 | 0.773016 | Yes |

The competition counts are aircraft `2735 TP / 32 FP / 11 FN`, ship
`340 / 59 / 62`, and vehicle `61 / 13 / 17`. The formal machine-readable
report is `outputs/xh25/all-metrics-search/b90-multiscale/final/ten-metric-report.json`.

## Fusion Policy

- Aircraft uses 1024 as primary and adds 1280 predictions at score 0.75 when
  HBB IoU with every selected aircraft is below 0.30.
- Ship uses 1536 as primary and adds 1024 predictions at score 0.56 when HBB
  IoU with every selected ship is below 0.70.
- Vehicle uses 1536 only. Class-24 predictions below score 0.21 are removed
  when HBB area is below 700.
- Primary predictions use the 25 thresholds in
  `configs/xh25-b90-multiscale-fusion.yaml`.

## Reproduction

Generate all three prediction files with the same checkpoint:

```bash
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1024.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1024.json

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1280.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1280.json

.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-b90-multiscale-1536.yaml \
  --output-json outputs/xh25/b90-multiscale/raw-1536.json
```

Fuse and evaluate:

```bash
.venv/bin/xh-detect fuse-same-weight-multiscale \
  --predictions-1024 outputs/xh25/b90-multiscale/raw-1024.json \
  --predictions-1280 outputs/xh25/b90-multiscale/raw-1280.json \
  --predictions-1536 outputs/xh25/b90-multiscale/raw-1536.json \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --policy-yaml configs/xh25-b90-multiscale-fusion.yaml \
  --output-json outputs/xh25/b90-multiscale/predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/b90-multiscale/predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/b90-multiscale/report.json
```
