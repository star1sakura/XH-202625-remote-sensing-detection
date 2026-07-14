# SPH-P2 Vehicle Expert

This experiment trains a one-class vehicle detector while keeping the historical
main detector immutable. All model selection and fusion checks use a source-group
isolated subset of the original train split. The formal validation split remains
sealed.

## Dataset

- Positive crops: `324`
- Negative crops: `324`
- Train crops: `555` (`288` positive)
- Internal validation crops: `93` (`36` positive)
- Crop size: `512`
- Training image size: `1024`
- Expert inference tile size: `512`
- Source-group overlap: `0`
- Source-group manifest SHA256:
  `10ed9815bdb1d96d2ce9cddeddadb3e9b455c8e844e9935f3c6220226b89c1a1`
- Internal source-image map SHA256:
  `1128e23854a7c044e2890333993242dc51384fabf7389fdae27789d0bdbe706b`

SPH background proposals are retained first. Deterministic crops from source
groups without vehicle truth fill the remaining negative quota to a 1:1 ratio.

## Training

The initial full-network run used Ultralytics automatic AdamW at `lr0=0.002`.
It collapsed from epoch-1 mAP50-95 `0.2550` to `0.0199` by epoch 12 and was
stopped.

The accepted schedule froze the 11-layer backbone and trained the one-class P2
head with AdamW at `lr0=1e-4` for 12 epochs. Its best result was epoch 4:

- Precision: `0.77284`
- Recall: `0.80473`
- mAP50: `0.83184`
- mAP50-95: `0.53699`
- Checkpoint SHA256:
  `d3c3a25665fd0b31408c6031f75d71fa1d60d1184a2fc17952499dff5acd50a9`

A second full-network stage initialized from that checkpoint with `lr0=2e-5`.
Its best mAP50-95 was only `0.38363` by epoch 10, so it was stopped and the
frozen-backbone checkpoint remained selected.

## Original-Image Holdout Gate

Historical main on the train-internal original-image holdout:

- TP: `33`
- FP: `1`
- FN: `3`
- Recall: `0.916667`
- FDR: `0.029412`

The expert recovered at most one of the three main misses at every tested
threshold. At threshold `0.65`, it added `1 TP / 1 FP`, producing recall
`0.944444` and FDR `0.055556`. This satisfies the FDR ceiling but fails the
required minimum `+3 TP` gain.

- Holdout predictions SHA256:
  `39d813f3ea648068544d274219c4491ec45d36d44a5720636e7b9ea5fe125acc`
- Holdout report SHA256:
  `276f5df03532f5fa093c9ad3135b6102a4f31a70eaad5f22822679d33f61f063`

**Decision: RETAIN MAIN.** Do not fuse this expert and do not inspect the formal
validation split for this branch. The crop-level model is competent, but its
performance does not transfer to enough main-missed vehicles on original images.
