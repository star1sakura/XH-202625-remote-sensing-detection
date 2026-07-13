# Single-Weight Main-Seeded MKSNet-Lite

## Constraint and decision

The competition submission may load only one detector checkpoint. The earlier
three-checkpoint ranking ensemble is therefore a teacher upper bound only. This
experiment produces one 26-layer MKSNet-Lite checkpoint and performs one model
forward pass per tile.

## Reproduction

Initialize MKSNet-Lite from every compatible historical-main layer. The two new
MKS residual blocks are zero-gated, so the initialized model is exactly equal to
main (`max_abs_diff=0.0` on a random tensor).

```bash
.venv/bin/xh-detect init-mksnet-lite-from-main \
  --main-checkpoint outputs/xh25/historical-main/best.pt
```

Fine-tune the MKS blocks and the later head without warmup amplification:

```bash
.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-hard-example/dataset.yaml \
  --model outputs/xh25/single-student/main-seeded-mksnet-lite.pt \
  --epochs 12 --image-size 1024 --device 0 --batch 8 --workers 4 \
  --no-amp --project runs/train --name xh25-main-seeded-mks-head-nowarmup \
  --no-resume --optimizer AdamW --learning-rate 0.00005 --freeze 17 \
  --save-period 1 --warmup-epochs 0 --warmup-bias-lr 0
```

Epoch 3 is the tuned endpoint. Interpolate it with the exact-main seed in weight
space. This creates one checkpoint; it is not prediction fusion.

```bash
.venv/bin/xh-detect interpolate-checkpoints \
  --base-checkpoint outputs/xh25/single-student/main-seeded-mksnet-lite.pt \
  --tuned-checkpoint runs/train/xh25-main-seeded-mks-head-nowarmup/weights/epoch2.pt \
  --output-checkpoint outputs/xh25/single-student/main-seeded-alpha050.pt \
  --alpha 0.5
```

Use `configs/xh25-main-seeded-mks-alpha050.yaml` for final inference.

## Fixed validation result

| Ranking item | Historical main | Single MKS weight | Result |
|---|---:|---:|---|
| Aircraft Recall | 0.988711 | 0.991260 | Improved |
| Aircraft FDR | 0.015948 | 0.012695 | Improved |
| Ship Recall | 0.823383 | 0.825871 | Improved |
| Ship FDR | 0.157761 | 0.153061 | Improved |
| Vehicle Recall | 0.705128 | 0.717949 | Improved |
| Vehicle FDR | 0.202899 | 0.188406 | Improved |

Counts are aircraft `2722 TP / 35 FP`, ship `332 / 60`, and vehicle `56 / 13`.
The checkpoint SHA256 is
`400a29ac9c505252bbdd2c411edeecda5468aeb25130618cc8f7138e452bafc0`.

## Latency

On the same RTX3090 and synthetic 10000 x 10000 image, a same-session 10-run
rerun measured main at `1.258274 s` median and this checkpoint at `1.399714 s`.
The candidate therefore improves six accuracy ranking items but not latency. It
passes the 20-second competition limit, but the honest result is 6/7 rather than
7/7. Thresholds were selected on the fixed validation split, so hidden-test
transfer remains to be confirmed by submission.
