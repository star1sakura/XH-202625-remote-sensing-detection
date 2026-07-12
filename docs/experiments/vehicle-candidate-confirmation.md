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
