# Task 4 Report: Add SPH Experiment Runbook

## Scope

Implemented only the requested Task 4 documentation file:

- `docs/experiments/sph-yolov5-small-object.md`

No config, model, test, or runtime files were changed.

## What Changed

Added the SPH experiment runbook exactly as specified in the brief, including:

- baseline comparison table with the provided XH25 metrics
- local smoke-test commands
- training command for the `sph-p2` candidate
- evaluation and competition-report commands
- threshold-optimization command
- keep criteria for deciding whether the SPH candidate should replace baseline

## Verification

Fresh verification steps:

1. Confirmed the runbook intro now states that `sph-p2` uses four Detect scales: P2, P3, P4, and P5.
2. Confirmed the runbook explicitly defines `sph-full` as the SPH P2 + NAM + Swin prediction-block variant and not an MKSNet full-backbone replacement.
3. Confirmed the existing `sph-full raw` baseline row remains in place as part of the approved ablation set.
4. Confirmed Keep Criteria now include the `10000 x 10000` tiled-inference timing-budget gate for RTX3090.
5. Ran `git diff --check` after the doc-only patch.

Verification commands run:

```bash
git diff -- docs/experiments/sph-yolov5-small-object.md .superpowers/sdd/sph-task-4-report.md
git diff --check
git commit -m "docs: clarify sph runbook gates"
```

## Commit

- `docs: clarify sph runbook gates`

## Concerns

- None.

## Fix Note

Addressed the runbook review findings by clarifying the SPH naming in the intro,
keeping the approved `sph-full raw` ablation row intact, and adding the RTX3090
tiled-inference timing-budget keep gate.
