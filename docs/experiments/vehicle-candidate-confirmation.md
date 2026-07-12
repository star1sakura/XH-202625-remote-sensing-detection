# Vehicle Candidate Confirmation

This experiment keeps the historical main detector frozen and tests whether
SPH-P2 and MKSNet-Lite can propose vehicle candidates for a small binary
confirmation model.

## Frozen baseline

- Config: `configs/xh25-historical-main.yaml`
- Checkpoint: `outputs/xh25/historical-main/best.pt`
- SHA256: `930CF7E1C698A8850523CE42D2565D1B2652E5AE01BF7F049A35D05778DD5424`
- Matching IoU: `0.35`
- Vehicle class id: `24`
- Historical vehicle result: TP `55`, FP `14`, FN `23`
- Historical vehicle Recall: `0.705128`
- Historical vehicle FDR: `0.202899`

## Acceptance gate

- Aircraft and ship metrics must not change.
- Vehicle true positives must increase by at least 3.
- Vehicle Recall must improve.
- Vehicle FDR must remain at or below `0.202899`.
- End-to-end latency must remain at or below 20 seconds per image.

The train split is used for proposal diagnostics and threshold selection. The
validation split stays sealed until the proposal rule and operating point are
frozen.

## RTX 3090 results

All diagnostics below use only the train split. Validation remained sealed.

### Paired latency gate

- Image: synthetic 10000 x 10000
- Main median: `1.287953 s`
- SPH-P2 median: `1.432796 s`
- Combined median: `2.699695 s`
- Combined P95: `2.793715 s`
- Combined maximum: `2.809280 s`
- Gate: PASS (`2.809280 <= 19.0 s`)

### Proposal diagnostics

- Historical main vehicle: TP `303`, FP `28`
- SPH-P2: recoverable TP `7`, proposal FP `394`, duplicate-main `301`
- MKSNet-Lite: recoverable TP `8`, proposal FP `331`, duplicate-main `293`
- SPH/MKS consensus: recoverable TP `5`, FP `19`
- Direct consensus FDR gate: FAIL

### Confirmer dataset

- Train: `4` positive, `71` negative
- Holdout: `3` positive, `22` negative
- Train manifest SHA256: `63217e630f7d1673b862bca38ceaeaa4d6888b7582d5f941dc47e1fa27b7d498`
- Holdout manifest SHA256: `97d1b0180e1790ad57d141b63cc7971f74b5d28bdeb061a8a03313551763aede`

### MobileNetV3-Small gate

- Best epoch: `30`
- Holdout AP: `0.131585`
- Holdout BCE: `0.549089`
- Checkpoint SHA256: `cacad457ca9e5819006ff752e93b01d5dd29dbe55f91d36e17b413a912d8dade`
- Repeated holdout score SHA256: `b5e98c5a76553702613cad6d1fdfd6bbaf76e2f6f21419ad173989d2a0f0a7bc`

The three holdout positives scored `0.387135`, `0.363165`, and `0.217557`.
The frozen confirmation grid starts at `0.50`, so every allowed point adds zero
true positives and fails the minimum `+3 TP` gate.

**Decision: RETAIN MAIN.** Do not run validation fusion or promote the vehicle
confirmation branch. The SPH/MKS models contain complementary vehicle truth,
but this train set does not contain enough recoverable positives to train a
reliable second-stage visual confirmer.
