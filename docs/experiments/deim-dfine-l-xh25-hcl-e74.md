# DEIM-D-FINE-L HCL Epoch 74

This branch archives the code and configuration used by the selected XH-202625
single-model candidate, HCL epoch 74. It intentionally excludes later prototype
mixing, FSC class weighting, and class-specific MAL target-exponent experiments.

## Model Contract

- Upstream DEIM commit: `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`.
- Initialization: official DEIM-D-FINE-L COCO checkpoint.
- Data: original official split only, 3,807 train images and 674 validation images.
- Training: seed 42, 1024 x 1024, batch size 8, AMP, EMA, 80 epochs.
- Architecture change: classification and localization decoder-query streams are
  decoupled after shared self-attention.
- Training-only objective: hierarchy-aware supervised contrastive loss on the
  classification queries, weighted by `loss_bhcl: 0.6`.
- Inference: the HCL projection head is removed; no ensemble or extra data is used.

The exact training configuration is
`configs/deim/deim_dfine_l_xh25_1024_80e_hcl.yml`.

## Selected Artifact

The selected checkpoint is `checkpoint0074.pth` (662,058,585 bytes):

```text
sha256 fae8282f73bc7a0e79010c1c0932233b853229ddbff6f05649786fce0bfe4e49
```

The checkpoint is deliberately not stored in ordinary Git. Verify any separately
transferred copy against this digest. Thresholds are committed in
`configs/deim/deim_dfine_l_xh25_hcl_e74_competition7_thresholds.yaml`, and the
machine-readable artifact record is in
`configs/deim/deim_dfine_l_xh25_hcl_e74_artifact.json`.

## Validation Result

Under the pinned Ultralytics 8.4.71 and competition evaluation protocol:

| Metric | HCL e74 |
| --- | ---: |
| Precision | 0.9204496704 |
| Recall | 0.9678521428 |
| mAP50 | 0.9453607916 |
| mAP50-95 | 0.7747561692 |
| Aircraft Recall / FDR | 1.0000000000 / 0.0115190785 |
| Ship Recall / FDR | 0.8557213930 / 0.1442786070 |
| Vehicle Recall / FDR | 0.8333333333 / 0.1666666667 |

The thresholded model strictly beats the frozen B90 baseline on mAP50-95 and all
six coarse-class Recall/FDR metrics. The raw pinned mAP50-95 before applying the
competition thresholds is 0.7862628600.

## Reproduce Training

```bash
bash scripts/bootstrap_deim.sh

DEIM_CONFIG=configs/deim/deim_dfine_l_xh25_1024_80e_hcl.yml \
DEIM_DATA_ROOT=/path/to/xh25 \
DEIM_OUTPUT_DIR=runs/train/deim-dfine-l-xh25-hcl-1024-b8-80e-s42 \
DEIM_BATCH_SIZE=8 \
DEIM_SEED=42 \
bash scripts/train_deim_dfine_l_xh25.sh
```

The data root must contain the official `reports/train-ground-truth.json` and
`reports/val-ground-truth.json` files. Do not substitute a regrouped split when
reproducing this checkpoint.
