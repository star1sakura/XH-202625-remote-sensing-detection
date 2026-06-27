# Threshold Optimization Design

## Purpose

Build a reproducible threshold optimization workflow for the completed
`xh25-mksnet-lite` experiment. The goal is to improve competition-style
validation performance without retraining by selecting per-class confidence
thresholds from the existing validation predictions.

The primary success criterion is better overall F1 than both:

- `xh25-yolo26s-e80` baseline at its reported threshold 0.25;
- `xh25-mksnet-lite` with a global threshold 0.30.

The optimizer should prefer lower FDR when F1 is effectively tied, while
preventing a threshold set that wins only by sacrificing too much Recall.

## Current Evidence

`xh25-mksnet-lite` has completed an 80 epoch run on the 3090 server. The current
best global threshold from the existing sweep is 0.30:

| Run | Recall | FDR | F1 |
| --- | ---: | ---: | ---: |
| `xh25-yolo26s-e80` baseline | 0.961562 | 0.037244 | 0.962159 |
| `xh25-mksnet-lite` global 0.25 | 0.962492 | 0.037806 | 0.962343 |
| `xh25-mksnet-lite` global 0.30 | 0.960012 | 0.032188 | 0.963897 |

The main regression is ship:

- aircraft improves slightly and has low FDR;
- vehicle keeps similar Recall and lowers FDR;
- ship drops in Recall and worsens FDR, especially QHS and MS.

This makes a threshold-only experiment the right next step: it is cheap,
reproducible, and can tell us whether MKSNet-Lite is worth further architecture
or training work.

## Proposed CLI

Add a command:

```bash
xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized
```

Optional parameters:

- `--baseline-report`: default `outputs/xh25/baseline/report.json` when present;
- `--experiment-name`: default `xh25-mksnet-lite-threshold-optimized`;
- `--thresholds`: comma-separated grid, default
  `0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70`;
- `--recall-floor-delta`: default `0.003`, meaning optimized Recall should not
  be more than 0.003 below the baseline when a baseline report is supplied;
- `--tie-epsilon`: default `0.0005`, used to prefer lower FDR when F1 is nearly
  tied.

## Outputs

The command writes:

- `optimized-thresholds.yaml`: a compact mapping of class ID to confidence
  threshold, ready to copy into `configs/xh25-mksnet-lite.yaml`;
- `report.json`: evaluation report with optimized thresholds;
- `comparison.json`: machine-readable comparison against the baseline report
  when available;
- `comparison.md`: human-readable summary covering overall, coarse classes, and
  watchlist classes;
- `search-summary.json`: selected objective values, search grid, chosen
  thresholds, and intermediate candidates;
- `search-summary.md`: concise explanation of why the threshold set was chosen.

## Architecture

Create a focused threshold optimization module:

- `src/xh_detect/thresholds.py`
  - loads COCO predictions and ground truth through existing evaluator helpers;
  - filters predictions per class according to a threshold map;
  - evaluates filtered predictions with the existing competition evaluator;
  - computes Precision, Recall, FDR, F1, and objective ordering;
  - performs deterministic per-class greedy threshold search;
  - writes YAML, JSON, and Markdown artifacts.

Modify `src/xh_detect/cli.py` only to expose the command and convert validation
errors into Typer-friendly messages.

No model, dataset conversion, inference, or evaluator matching logic should be
changed.

## Search Strategy

The search must be deterministic and cheap enough to run repeatedly.

1. Start from the best global threshold in the grid by overall F1.
2. For each class ID from 0 to 24, try every threshold in the grid while holding
   other class thresholds fixed.
3. Accept the candidate if it improves the objective order:
   - higher F1 is better;
   - when F1 is within `tie_epsilon`, lower FDR is better;
   - when F1 and FDR are effectively tied, higher Recall is better;
   - if still tied, prefer the simpler threshold closer to the current global
     threshold.
4. Repeat the class pass once more to catch interactions.
5. If a baseline report and recall floor are supplied, reject candidates whose
   overall Recall falls below `baseline_recall - recall_floor_delta`, unless no
   candidate satisfies the floor.

This is intentionally conservative. A full combinatorial search is unnecessary
for 25 classes and a 14-value grid.

## Reporting Requirements

The Markdown report should include:

- selected threshold map grouped by coarse class;
- overall baseline vs optimized metrics;
- coarse class deltas;
- low Recall classes after optimization;
- high FDR classes after optimization;
- a short recommendation: keep threshold-only configuration, retrain, or abandon
  the MKSNet-Lite branch.

The report should explicitly call out whether ship recovered enough to justify
the optimized configuration. It should not hide a ship regression behind a small
overall F1 win.

## Testing Strategy

Add focused unit tests with tiny synthetic reports and detections:

- per-class threshold filtering keeps predictions at or above the class-specific
  threshold and drops lower-scoring predictions;
- objective comparison prefers F1, then lower FDR, then higher Recall;
- optimizer can choose different thresholds for two classes and writes a YAML
  threshold map;
- CLI forwards parameters and writes the expected output files;
- invalid threshold grids and class IDs raise clear errors.

Existing evaluator tests remain the authority for matching semantics; these new
tests should not duplicate IoU matching internals.

## Operational Workflow

After implementation, run on the 3090 server:

```bash
.venv/bin/xh-detect optimize-thresholds \
  --predictions-json outputs/xh25/mksnet-lite/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --baseline-report outputs/xh25/baseline/report.json \
  --taxonomy xh25 \
  --output-dir outputs/xh25/mksnet-lite/threshold-optimized
```

If the optimized report improves F1 while keeping Recall within the configured
floor, update documentation to recommend this threshold set for MKSNet-Lite.
If it does not, keep global threshold 0.30 as the best known post-processing
setting and move to a ship-focused data or training experiment.

## Non-Goals

- Do not retrain the model.
- Do not change NMS, merge IoU, tiling, or inference code.
- Do not implement Bayesian, genetic, or exhaustive combinatorial search.
- Do not tune thresholds on test data or any data outside the fixed validation
  split.
