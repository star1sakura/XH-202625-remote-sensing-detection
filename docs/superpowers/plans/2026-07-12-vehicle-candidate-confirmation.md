# Vehicle Candidate Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add verified SPH-P2 vehicle proposals to the strongest historical main model while strictly improving vehicle Recall, preserving vehicle FDR and all aircraft/ship metrics, and remaining below the 20-second latency limit.

**Architecture:** Historical main remains immutable. SPH-P2 proposes only class-24 FSC boxes missed by main; a MobileNetV3-Small binary confirmer accepts or rejects context crops, and deterministic fusion appends accepted nonduplicate vehicles after all main predictions. Train source groups alone determine labels, model weights, proposal floor, and confirmation threshold; the fixed validation split is evaluated once after thresholds are frozen.

**Tech Stack:** Python 3.11/3.12, PyTorch, torchvision, Ultralytics 8.4.71, OpenCV, Typer, PyYAML, pytest, Ruff, RTX3090.

---

## Global Constraints

- Use the supplied historical checkpoint with SHA256 `930CF7E1C698A8850523CE42D2565D1B2652E5AE01BF7F049A35D05778DD5424`.
- Keep the grouped XH25 train/validation split, taxonomy, official IoU rules, and historical main detections immutable.
- Never use validation images, labels, predictions, or errors to train the confirmer or select either threshold.
- Vehicle matching uses IoU 0.35; aircraft and ship use IoU 0.50.
- Existing main detections are never deleted, rescored, or reordered relative to each other.
- Promotion requires vehicle Recall > 0.705128, vehicle FDR <= 0.202899, at least three additional vehicle TP, no aircraft/ship regression, and every measured 10000 x 10000 runtime <= 20 seconds.
- Checkpoints, copied weights, predictions, crop datasets, caches, engines, and credentials remain untracked.
- Use the configured mirror when a package or pretrained classifier weight must be downloaded.

## File Structure

- Create `src/xh_detect/vehicle_confirmation/__init__.py`: public types and functions.
- Create `src/xh_detect/vehicle_confirmation/proposals.py`: official-order vehicle proposal labeling, consensus analysis, and added-candidate FDR constraint.
- Create `src/xh_detect/vehicle_confirmation/data.py`: deterministic group split, context crop materialization, JSONL manifests, and dataset loading.
- Create `src/xh_detect/vehicle_confirmation/model.py`: MobileNetV3-Small confirmer, training, checkpoint loading, score inference, and ONNX export.
- Create `src/xh_detect/vehicle_confirmation/fusion.py`: frozen threshold config, deterministic offline fusion, and runtime pipeline.
- Create `src/xh_detect/vehicle_confirmation/benchmark.py`: sequential main/SPH/confirmer timing and 20-second gate report.
- Modify `src/xh_detect/cli.py`: thin commands for proposal analysis, dataset build, confirmer train/score, threshold selection, fusion, and benchmark.
- Create `configs/xh25-historical-main.yaml`: immutable historical-main runtime configuration.
- Create `configs/xh25-vehicle-confirmation.yaml`: runtime paths and frozen operating-point fields.
- Create `docs/experiments/vehicle-candidate-confirmation.md`: exact local/server workflow and promotion table.
- Create focused tests under `tests/test_vehicle_confirmation_*.py` and update `tests/test_cli.py` and `tests/test_config.py`.

---

### Task 1: Register And Reproduce Historical Main

**Files:**
- Create: `configs/xh25-historical-main.yaml`
- Modify: `tests/test_config.py`
- Create: `docs/experiments/vehicle-candidate-confirmation.md`

- [ ] **Step 1: Write a failing config test**

Add:

```python
def test_historical_main_config_uses_supplied_checkpoint() -> None:
    config = PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "xh25-historical-main.yaml"
    )
    assert config.model_path == "outputs/xh25/historical-main/best.pt"
    assert config.taxonomy == "xh25"
    assert config.image_size == 1024
    assert config.tile_size == 1024
    assert config.merge_iou == 0.3
    assert config.class_thresholds == {class_id: 0.25 for class_id in range(25)}
```

- [ ] **Step 2: Verify the test fails**

Run: `python -m pytest tests/test_config.py::test_historical_main_config_uses_supplied_checkpoint -q`

Expected: FAIL because `configs/xh25-historical-main.yaml` is absent.

- [ ] **Step 3: Create the config**

Copy `configs/xh25-yolo26s-e80.yaml` and change only:

```yaml
model_path: outputs/xh25/historical-main/best.pt
```

- [ ] **Step 4: Verify config compatibility**

Run: `python -m pytest tests/test_config.py -q; python -m ruff check tests/test_config.py`

Expected: PASS.

- [ ] **Step 5: Copy and verify the external artifact**

On Windows, copy the user-provided file to the ignored artifact path, then run:

```powershell
Get-FileHash outputs/xh25/historical-main/best.pt -Algorithm SHA256
```

Expected hash: `930CF7E1C698A8850523CE42D2565D1B2652E5AE01BF7F049A35D05778DD5424`.

On the RTX3090 server, copy the same artifact and run:

```bash
sha256sum outputs/xh25/historical-main/best.pt
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/val \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-historical-main.yaml \
  --output-json outputs/xh25/historical-main/val-predictions.json
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/historical-main/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/historical-main/report.json
```

Expected official-style counts: Overall `TP=3102, FP=120, FN=124`; Aircraft `TP=2716, FP=44, FN=30`; Ship `TP=331, FP=62, FN=71`; Vehicle `TP=55, FP=14, FN=23`. Stop if any count differs.

- [ ] **Step 6: Commit**

```bash
git add configs/xh25-historical-main.yaml tests/test_config.py docs/experiments/vehicle-candidate-confirmation.md
git commit -m "config: register historical main checkpoint"
```

---

### Task 2: Label Recoverable Vehicle Proposals In Official Order

**Files:**
- Create: `src/xh_detect/vehicle_confirmation/__init__.py`
- Create: `src/xh_detect/vehicle_confirmation/proposals.py`
- Create: `tests/test_vehicle_confirmation_proposals.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class LabeledVehicleProposal:
    proposal_index: int
    detection: Detection
    label: int
    reason: str
    matched_truth_index: int | None
    duplicate_main: bool


@dataclass(frozen=True)
class VehicleProposalReport:
    main_vehicle_tp: int
    main_vehicle_fp: int
    recoverable_tp: int
    proposal_fp: int
    duplicate_main: int
    duplicate_proposal: int


def label_vehicle_proposals(
    main_predictions: Iterable[Detection],
    proposal_predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    vehicle_class_id: int = 24,
    iou_threshold: float = 0.35,
) -> tuple[tuple[LabeledVehicleProposal, ...], VehicleProposalReport]: ...


def satisfies_vehicle_fdr(
    baseline_tp: int,
    baseline_fp: int,
    added_tp: int,
    added_fp: int,
    *,
    ceiling: float = 0.202899,
) -> bool: ...
```

- [ ] **Step 1: Write failing matching tests**

Create tests covering this exact sequence:

```python
def test_labels_only_recoverable_vehicle_as_positive() -> None:
    truth = [vehicle_truth("img", 0, 0, 10, 10), vehicle_truth("img", 30, 0, 40, 10)]
    main = [vehicle_detection("img", 0.95, 0, 0, 10, 10)]
    proposals = [
        vehicle_detection("img", 0.90, 0, 0, 10, 10),
        vehicle_detection("img", 0.80, 30, 0, 40, 10),
        vehicle_detection("img", 0.70, 30, 0, 40, 10),
        vehicle_detection("img", 0.60, 70, 0, 80, 10),
    ]
    labels, report = label_vehicle_proposals(main, proposals, truth)
    assert [(item.label, item.reason) for item in labels] == [
        (0, "duplicate_main"),
        (1, "recoverable_truth"),
        (0, "duplicate_proposal"),
        (0, "background"),
    ]
    assert report.recoverable_tp == 1
    assert report.proposal_fp == 3
```

Add tests for nonvehicle predictions being ignored, exact IoU 0.35 matching, descending-score/stable-index order, difficult truth exclusion, cross-image isolation, invalid class IDs, and immutable returned tuples.

Add FDR boundary tests:

```python
def test_vehicle_fdr_constraint_uses_fused_counts() -> None:
    assert satisfies_vehicle_fdr(55, 14, 4, 1)
    assert not satisfies_vehicle_fdr(55, 14, 3, 1)
```

- [ ] **Step 2: Verify the tests fail**

Run: `python -m pytest tests/test_vehicle_confirmation_proposals.py -q`

Expected: collection FAIL because the package is absent.

- [ ] **Step 3: Implement official-order labeling**

Use `xh_detect.geometry.hbb_iou`, `obb_to_hbb`, and the same score ordering as `xh_detect.evaluator`. First greedily match main vehicle predictions to vehicle truth. For every SPH proposal in `(-score, original_index)` order:

```python
if overlaps_any_main_vehicle_at_035:
    label = 0
    reason = "duplicate_main"
elif matches_unclaimed_vehicle_truth_at_035:
    label = 1
    reason = "recoverable_truth"
    claim_truth()
elif overlaps_already_claimed_truth_at_035:
    label = 0
    reason = "duplicate_proposal"
else:
    label = 0
    reason = "background"
```

Reject boolean integer arguments, nonfinite thresholds, thresholds outside `[0, 1]`, and labels outside the selected taxonomy. Compute FDR using `(baseline_fp + added_fp) / (baseline_tp + added_tp + baseline_fp + added_fp)`.

- [ ] **Step 4: Verify implementation**

Run: `python -m pytest tests/test_vehicle_confirmation_proposals.py tests/test_evaluator.py -q; python -m ruff check src/xh_detect/vehicle_confirmation tests/test_vehicle_confirmation_proposals.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xh_detect/vehicle_confirmation tests/test_vehicle_confirmation_proposals.py
git commit -m "feat: label recoverable vehicle proposals"
```

---

### Task 3: Add Proposal Analysis Command And Consensus Report

**Files:**
- Modify: `src/xh_detect/vehicle_confirmation/proposals.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_vehicle_confirmation_proposals.py`
- Modify: `tests/test_cli.py`

**Command:** `xh-detect analyze-vehicle-proposals`

- [ ] **Step 1: Write failing consensus and CLI tests**

Add a three-model fixture where SPH and MKS share one recoverable vehicle and one background FP. Assert:

```python
report = analyze_vehicle_consensus(main, sph, mks, truth)
assert report.sph.recoverable_tp == 2
assert report.mks.recoverable_tp == 1
assert report.consensus_recoverable_tp == 1
assert report.consensus_fp == 1
```

Add a CLI test invoking:

```text
analyze-vehicle-proposals
--main-predictions main.json
--sph-predictions sph.json
--mks-predictions mks.json
--ground-truth-json truth.json
--output-path report.json
```

Assert the command writes JSON and prints the output path.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_proposals.py tests/test_cli.py -q -k vehicle_proposal`

Expected: FAIL because consensus analysis and the command are absent.

- [ ] **Step 3: Implement consensus matching and serialization**

Use HBB IoU 0.35 between SPH and MKS proposals on the same image. Greedily pair in descending SPH score, then MKS score, then stable indexes. A consensus pair is recoverable only when its SPH proposal is labeled positive; otherwise it is FP. Serialize individual proposal reports plus consensus TP/FP, accepted indexes, and the FDR constraint result for historical baseline counts `55/14`.

Implement the Typer wrapper by loading XH25 predictions/truth with existing evaluator loaders and `_write_json`.

- [ ] **Step 4: Verify command and regression coverage**

Run: `python -m pytest tests/test_vehicle_confirmation_proposals.py tests/test_cli.py -q; python -m ruff check src/xh_detect/vehicle_confirmation/proposals.py src/xh_detect/cli.py tests/test_vehicle_confirmation_proposals.py tests/test_cli.py`

Expected: PASS.

- [ ] **Step 5: Run train-only diagnostics**

Generate historical-main train predictions and run the command only on train. Do not generate or inspect SPH/MKS consensus results on validation before Task 9. Continue only when the train internal holdout constructed in Task 5 contains at least three recoverable SPH proposals.

- [ ] **Step 6: Commit**

```bash
git add src/xh_detect/vehicle_confirmation/proposals.py src/xh_detect/cli.py tests/test_vehicle_confirmation_proposals.py tests/test_cli.py
git commit -m "feat: analyze vehicle proposal consensus"
```

---

### Task 4: Implement The Sequential Latency Gate

**Files:**
- Create: `src/xh_detect/vehicle_confirmation/benchmark.py`
- Create: `tests/test_vehicle_confirmation_benchmark.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VehicleLatencyReport:
    main_seconds: tuple[float, ...]
    sph_seconds: tuple[float, ...]
    combined_seconds: tuple[float, ...]
    reserve_seconds: float
    proposal_gate_passed: bool


def benchmark_vehicle_proposal_pair(
    main: InferencePipeline,
    sph: InferencePipeline,
    image: ImageArray,
    image_id: str,
    repeats: int = 5,
    reserve_seconds: float = 1.0,
    limit_seconds: float = 20.0,
) -> VehicleLatencyReport: ...
```

- [ ] **Step 1: Write failing timing tests**

Use fake pipelines with deterministic `InferenceResult.timings.total_s` values and assert one warm-up per pipeline, five measured runs, `combined_seconds[i] = main[i] + sph[i]`, and `proposal_gate_passed` only when every combined run is at most `limit_seconds - reserve_seconds`. Add validation tests for nonpositive repeats, negative/nonfinite reserve, and nonpositive limit.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_benchmark.py -q`

Expected: collection FAIL because the module is absent.

- [ ] **Step 3: Implement timing and JSON serialization**

Run main then SPH sequentially for each repetition with unique image IDs to avoid cache reuse. Record every sample, median, p95, maximum, reserve, limit, and pass/fail. Do not use the sum of independently benchmarked medians as the gate; gate actual paired repetitions.

Add `benchmark-vehicle-proposals` to CLI with historical-main config, SPH-P2 config, image path, repeats, reserve, and output path.

- [ ] **Step 4: Verify locally**

Run: `python -m pytest tests/test_vehicle_confirmation_benchmark.py tests/test_cli.py -q; python -m ruff check src/xh_detect/vehicle_confirmation/benchmark.py tests/test_vehicle_confirmation_benchmark.py`

Expected: PASS.

- [ ] **Step 5: Execute the RTX3090 gate**

Run five paired repetitions on a real representative 10000 x 10000 image when available; otherwise use the repository synthetic benchmark image and explicitly mark it synthetic in the report.

Expected: every paired repetition <= 19 seconds. If any run exceeds 19 seconds, record `STOP: proposal pair exceeds latency reserve`, retain historical main, and do not execute Tasks 5-9.

- [ ] **Step 6: Commit**

```bash
git add src/xh_detect/vehicle_confirmation/benchmark.py src/xh_detect/cli.py tests/test_vehicle_confirmation_benchmark.py tests/test_cli.py
git commit -m "feat: gate vehicle proposals by paired latency"
```

---

### Task 5: Build A Train-Only Vehicle Confirmer Dataset

**Files:**
- Create: `src/xh_detect/vehicle_confirmation/data.py`
- Create: `tests/test_vehicle_confirmation_data.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class VehicleCropPolicy:
    context_scale: float = 2.0
    min_side: int = 64
    max_side: int = 256
    output_size: int = 160
    holdout_ratio: float = 0.20
    seed: int = 42


@dataclass(frozen=True)
class VehicleConfirmerDatasetResult:
    output_root: Path
    train_examples: int
    holdout_examples: int
    train_positive: int
    train_negative: int
    holdout_positive: int
    holdout_negative: int
    train_groups: frozenset[str]
    holdout_groups: frozenset[str]


def build_vehicle_confirmer_dataset(
    source_root: Path,
    main_predictions_json: Path,
    sph_predictions_json: Path,
    output_root: Path,
    policy: VehicleCropPolicy,
) -> VehicleConfirmerDatasetResult: ...
```

- [ ] **Step 1: Write failing crop and isolation tests**

Create a two-group fixture and assert:

- positives, background FP, and duplicate-proposal FP receive exact labels from Task 2;
- no duplicate-main proposal is materialized because it is not a runtime candidate;
- crop side is `clamp(2 * max(width, height), 64, 256)`;
- edge crops are zero-padded and resized to exactly 160 x 160;
- train and holdout source groups are disjoint;
- the same seed produces byte-identical JSONL manifests;
- a validation source-group record, unknown image ID, overlapping source/output root, nonempty output, or missing label/image fails before writing;
- class counts and group IDs in `reports/vehicle-confirmer-dataset.json` match manifests.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_data.py -q`

Expected: collection FAIL because the module is absent.

- [ ] **Step 3: Implement deterministic materialization**

Load train image IDs, train truth, and source groups from the prepared dataset. Label proposals with Task 2, discard `duplicate_main`, group records by source group, and assign groups by SHA256 rank `sha256(f"42:{group}")` while preserving at least one positive and one negative in both partitions. Fail with a clear error if this is impossible.

Write crops as lossless PNG and JSONL records with exactly:

```json
{"crop":"crops/train/000001.png","image_id":"17","proposal_index":4,"label":1,"reason":"recoverable_truth","sph_score":0.73,"width_norm":0.012,"height_norm":0.009,"source_group":"group-a"}
```

Write `manifests/train.jsonl`, `manifests/holdout.jsonl`, `reports/vehicle-confirmer-dataset.json`, and a concise Markdown report. Use temporary-stage plus atomic directory publication so failed builds leave no partial dataset.

Add `build-vehicle-confirmer-dataset` to CLI with all policy options.

- [ ] **Step 4: Verify dataset and adjacent regressions**

Run: `python -m pytest tests/test_vehicle_confirmation_data.py tests/test_hard_negative.py tests/test_xh25.py tests/test_cli.py -q; python -m ruff check src/xh_detect/vehicle_confirmation/data.py tests/test_vehicle_confirmation_data.py`

Expected: PASS.

- [ ] **Step 5: Build the server dataset**

Use only train predictions and confirm the report contains at least three holdout positives. Record train/holdout group hashes and counts before training.

- [ ] **Step 6: Commit**

```bash
git add src/xh_detect/vehicle_confirmation/data.py src/xh_detect/cli.py tests/test_vehicle_confirmation_data.py tests/test_cli.py
git commit -m "feat: build vehicle confirmer dataset"
```

---

### Task 6: Train And Export MobileNetV3-Small Confirmer

**Files:**
- Modify: `pyproject.toml`
- Create: `src/xh_detect/vehicle_confirmation/model.py`
- Create: `tests/test_vehicle_confirmation_model.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**

```python
class VehicleConfirmer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None: ...
    def forward(self, images: Tensor, scalar_features: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class VehicleConfirmerTrainingConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    workers: int = 4
    seed: int = 42


def train_vehicle_confirmer(
    dataset_root: Path,
    output_dir: Path,
    config: VehicleConfirmerTrainingConfig,
    device: str,
) -> Path: ...
```

- [ ] **Step 1: Write failing model and training tests**

Add tests asserting:

```python
model = VehicleConfirmer(pretrained=False)
logits = model(torch.zeros(4, 3, 160, 160), torch.zeros(4, 3))
assert logits.shape == (4,)
```

Test invalid image/scalar shapes, deterministic holdout scoring, weighted-sampler counts, BCE loss finiteness, CPU one-batch training, checkpoint fields, checkpoint SHA recording, and ONNX export input names `images` and `scalar_features` with dynamic batch axes. Test TensorRT command construction and nonzero `trtexec` failure with a mocked subprocess. Monkeypatch torchvision weights in tests so no network is used.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_model.py -q`

Expected: collection FAIL because the model module is absent.

- [ ] **Step 3: Declare the direct dependency**

Add `torchvision>=0.22,<0.27` to project dependencies. Do not pin torch independently; the server image owns its CUDA-compatible torch build. Install editable dependencies from the configured mirror only when the existing environment lacks torchvision.

- [ ] **Step 4: Implement model, dataset loader, and training**

Build `torchvision.models.mobilenet_v3_small` with default ImageNet weights when `pretrained=True`. Use its feature extractor and average pool, encode the three scalar features through `Linear(3, 16) -> ReLU`, concatenate, and classify with `Dropout(0.2) -> Linear(feature_dim + 16, 1)`.

Normalize images with ImageNet mean/std. Use `WeightedRandomSampler`, `BCEWithLogitsLoss`, AdamW, fixed seeds, and deterministic algorithms. Compute binary average precision by sorting holdout probabilities descending, calculating cumulative positive precision at every rank, and averaging precision at positive ranks; fail dataset construction if holdout has no positives. Select `best.pt` by highest holdout average precision and break ties by lower holdout BCE. Save model state, config, epoch, AP, BCE, dataset report hash, and code commit. Training must never search an acceptance threshold.

Implement batched score inference and `torch.onnx.export` with opset 17. Add TensorRT export by invoking:

```text
trtexec --onnx=<model.onnx> --saveEngine=<model.engine> --fp16 \
  --minShapes=images:1x3x160x160,scalar_features:1x3 \
  --optShapes=images:128x3x160x160,scalar_features:128x3 \
  --maxShapes=images:512x3x160x160,scalar_features:512x3
```

Capture stdout/stderr, require exit code zero and a nonempty engine file, and record the command plus engine SHA. If ONNX or `trtexec` is unavailable, the export command fails with an actionable dependency error while PyTorch training/inference remains usable.

Add `train-vehicle-confirmer`, `score-vehicle-confirmer`, `export-vehicle-confirmer-onnx`, and `export-vehicle-confirmer-engine` commands.

- [ ] **Step 5: Verify implementation**

Run: `python -m pytest tests/test_vehicle_confirmation_model.py tests/test_cli.py -q; python -m ruff format --check .; python -m ruff check .`

Expected: PASS.

- [ ] **Step 6: Train on RTX3090**

Run with ImageNet weights from the configured mirror, save all epoch metrics, and verify the selected checkpoint can score the holdout manifest twice with byte-identical output.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/xh_detect/vehicle_confirmation/model.py src/xh_detect/cli.py tests/test_vehicle_confirmation_model.py tests/test_cli.py
git commit -m "feat: train vehicle proposal confirmer"
```

---

### Task 7: Select And Freeze The Operating Point On Train Holdout

**Files:**
- Create: `src/xh_detect/vehicle_confirmation/fusion.py`
- Create: `tests/test_vehicle_confirmation_fusion.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ScoredVehicleProposal:
    proposal_index: int
    detection: Detection
    sph_score: float
    confirmation_probability: float
    label: int | None = None


@dataclass(frozen=True)
class VehicleConfirmationThresholds:
    proposal_floor: float
    confirmation_threshold: float
    duplicate_iou: float = 0.35


@dataclass(frozen=True)
class OperatingPoint:
    thresholds: VehicleConfirmationThresholds
    added_tp: int
    added_fp: int
    fused_vehicle_recall: float
    fused_vehicle_fdr: float


def select_operating_point(
    scored_proposals: Iterable[ScoredVehicleProposal],
    *,
    baseline_tp: int,
    baseline_fp: int,
    baseline_fn: int,
    proposal_floors: tuple[float, ...],
    confirmation_thresholds: tuple[float, ...],
    fdr_ceiling: float,
    minimum_added_tp: int,
) -> OperatingPoint: ...
```

`select_operating_point` rejects records whose `label` is `None`; runtime fusion
accepts unlabeled records and never reads validation labels.

- [ ] **Step 1: Write failing selection tests**

Test a grid where the unconstrained highest-Recall point violates FDR and assert selection chooses the highest added TP among feasible points, then lowest added FP, highest confirmation threshold, and highest proposal floor. Add tests for no feasible point, duplicate grids, nonfinite scores, thresholds outside `[0,1]`, and baseline count validation.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_fusion.py -q`

Expected: collection FAIL because fusion is absent.

- [ ] **Step 3: Implement grid selection and frozen config**

Use fixed grids:

```python
proposal_floors = (0.05, 0.10, 0.15, 0.20, 0.25)
confirmation_thresholds = tuple(round(0.50 + index * 0.05, 2) for index in range(10))
```

Count labels after both thresholds and enforce Task 2 FDR math. Serialize the selected point, all evaluated points, model SHA, dataset-report SHA, split-group hashes, and timestamp to `outputs/xh25/vehicle-confirmation/frozen-operating-point.json`. The validation command accepts only this frozen file and never raw threshold options.

Add `select-vehicle-confirmation-thresholds` to CLI.

- [ ] **Step 4: Verify selection**

Run: `python -m pytest tests/test_vehicle_confirmation_fusion.py tests/test_cli.py -q; python -m ruff check src/xh_detect/vehicle_confirmation/fusion.py tests/test_vehicle_confirmation_fusion.py`

Expected: PASS.

- [ ] **Step 5: Execute the holdout gate**

Select and freeze one point. Continue only if holdout added TP >= 3 and fused holdout vehicle FDR does not exceed its main-only baseline. Otherwise write `RETAIN MAIN: no feasible confirmer operating point` and stop.

- [ ] **Step 6: Commit**

```bash
git add src/xh_detect/vehicle_confirmation/fusion.py src/xh_detect/cli.py tests/test_vehicle_confirmation_fusion.py tests/test_cli.py
git commit -m "feat: freeze vehicle confirmation operating point"
```

---

### Task 8: Implement Deterministic Offline And Runtime Fusion

**Files:**
- Modify: `src/xh_detect/vehicle_confirmation/fusion.py`
- Modify: `src/xh_detect/vehicle_confirmation/benchmark.py`
- Modify: `tests/test_vehicle_confirmation_fusion.py`
- Modify: `tests/test_vehicle_confirmation_benchmark.py`
- Modify: `src/xh_detect/cli.py`
- Modify: `tests/test_cli.py`
- Create: `configs/xh25-vehicle-confirmation.yaml`

- [ ] **Step 1: Write failing fusion invariance tests**

Construct main aircraft, ship, and vehicle detections plus accepted/rejected SPH proposals. Assert:

- every main detection is byte-for-byte equal and in the same relative order;
- aircraft and ship outputs are exactly unchanged;
- proposals below either frozen threshold are absent;
- IoU 0.35 with main or a higher-ranked accepted proposal is rejected;
- accepted proposals sort by confirmation probability, SPH score, then original index;
- candidate fused scores are below every existing main vehicle score, preserving official main-first matching;
- no main vehicle case remains deterministic;
- repeated calls produce equal tuples.

- [ ] **Step 2: Verify failure**

Run: `python -m pytest tests/test_vehicle_confirmation_fusion.py -q`

Expected: FAIL because fusion behavior is absent.

- [ ] **Step 3: Implement offline fusion**

Add:

```python
def fuse_vehicle_confirmations(
    main_detections: Iterable[Detection],
    scored_proposals: Iterable[ScoredVehicleProposal],
    thresholds: VehicleConfirmationThresholds,
) -> tuple[Detection, ...]: ...
```

Copy main first. Scale accepted candidate scores into `[0, min_main_vehicle_score)` while preserving candidate order. If no main vehicle exists, scale below `0.25`, the historical main class threshold. Use HBB IoU for duplicate rejection.

- [ ] **Step 4: Implement runtime pipeline and final benchmark**

Add `VehicleConfirmationPipeline` that runs main and SPH pipelines, filters class 24, extracts/resizes crops with Task 5 policy, scores candidates in one batch, fuses, and reports stage timings. Cache keys include both detector configs, both checkpoint metadata, confirmer SHA, crop policy, and frozen thresholds.

Extend the benchmark report to include crop, confirmer, fusion, and total timings. The final gate uses five paired end-to-end runs and requires every total <= 20 seconds.

Add `infer-vehicle-confirmation-dataset`, `fuse-vehicle-confirmation-predictions`, and `benchmark-vehicle-confirmation` to CLI. The offline fusion command consumes validation proposal scores produced by the frozen model and cannot accept threshold overrides.

- [ ] **Step 5: Create runtime config**

Write YAML containing exact paths:

```yaml
main_config: configs/xh25-historical-main.yaml
proposal_config: configs/xh25-sph-p2.yaml
confirmer_model: runs/train/xh25-vehicle-confirmer/best.pt
operating_point: outputs/xh25/vehicle-confirmation/frozen-operating-point.json
device: "0"
confirmer_batch_size: 128
```

Validate all keys and reject unknown keys, missing files at runtime, invalid batch size, and non-XH25 detector configs.

- [ ] **Step 6: Verify runtime and CLI**

Run: `python -m pytest tests/test_vehicle_confirmation_fusion.py tests/test_vehicle_confirmation_benchmark.py tests/test_cli.py tests/test_config.py -q; python -m ruff format --check .; python -m ruff check .`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/xh_detect/vehicle_confirmation/fusion.py src/xh_detect/vehicle_confirmation/benchmark.py src/xh_detect/cli.py configs/xh25-vehicle-confirmation.yaml tests/test_vehicle_confirmation_fusion.py tests/test_vehicle_confirmation_benchmark.py tests/test_cli.py tests/test_config.py
git commit -m "feat: fuse confirmed vehicle proposals"
```

---

### Task 9: Execute The Single Fixed-Validation Evaluation

**Files:**
- Modify: `docs/experiments/vehicle-candidate-confirmation.md`
- Generated, ignored: `outputs/xh25/vehicle-confirmation/`
- Generated, ignored: `runs/train/xh25-vehicle-confirmer/`
- Generated, ignored: `datasets/xh25-vehicle-confirmer/`

- [ ] **Step 1: Verify repository and artifact identity**

Run on RTX3090:

```bash
git status --short
.venv/bin/python -m pytest tests/test_vehicle_confirmation_proposals.py tests/test_vehicle_confirmation_data.py tests/test_vehicle_confirmation_model.py tests/test_vehicle_confirmation_fusion.py tests/test_vehicle_confirmation_benchmark.py tests/test_cli.py -q
sha256sum outputs/xh25/historical-main/best.pt
sha256sum datasets/xh25/manifests/train.txt datasets/xh25/manifests/val.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Expected: clean tracked worktree, all tests pass, historical hash matches, split hashes are recorded, and one 24 GB RTX3090 is present.

- [ ] **Step 2: Score frozen validation proposals exactly once**

Run historical main and SPH-P2 on validation if immutable predictions are absent, score only runtime-eligible SPH vehicle proposals with the frozen confirmer, then fuse with the frozen operating point. Do not rerun Task 7 or alter configs after reading validation output.

- [ ] **Step 3: Produce official and diagnostic reports**

Run:

```bash
.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/vehicle-confirmation/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/vehicle-confirmation/report.json
.venv/bin/xh-detect audit-false-positives \
  --predictions-json outputs/xh25/vehicle-confirmation/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/vehicle-confirmation/fp-audit.json
.venv/bin/xh-detect benchmark-vehicle-confirmation \
  --config-path configs/xh25-vehicle-confirmation.yaml \
  --image-path outputs/benchmark/representative-10000.png \
  --repeats 5 \
  --output-path outputs/xh25/vehicle-confirmation/benchmark.json
```

Record immutable component-checkpoint Precision, Recall, mAP50, and mAP50-95
from their checkpoint metadata or `results.csv`, clearly labeled as nonofficial.
Do not derive fused mAP from the official TP/FP/FN report or present those
point metrics as AP.

- [ ] **Step 4: Apply promotion constraints mechanically**

Write `promotion.json` with one boolean and observed value per constraint. Assert aircraft and ship TP/FP/FN are exactly equal to historical main. Print exactly one final decision:

```text
PROMOTE VEHICLE CONFIRMATION: vehicle Recall improved by <N> TP, vehicle FDR did not worsen, all other official items and latency passed.
```

or:

```text
RETAIN HISTORICAL MAIN: vehicle confirmation failed <named constraints>.
```

- [ ] **Step 5: Complete runbook and commit**

Record artifact hashes, commands, train/holdout counts, frozen thresholds, all seven official values, mAP diagnostics, latency samples, and the decision. Never stage generated artifacts.

```bash
git add docs/experiments/vehicle-candidate-confirmation.md
git commit -m "docs: report vehicle confirmation experiment"
git status --short
```

Expected: only ignored runtime artifacts remain outside Git.

---

## Plan Self-Review

- **Spec coverage:** Tasks 1-4 implement immutable baseline reproduction, official proposal labels, consensus evidence, and the pretraining latency gate. Tasks 5-7 implement train-only crops, group isolation, MobileNetV3 training, and frozen threshold selection. Tasks 8-9 implement invariant fusion, end-to-end timing, single validation evaluation, and mechanical promotion.
- **Placeholder scan:** No unresolved task marker, unspecified handler, or deferred implementation remains. Conditional stopping is explicit and produces a retained-main decision rather than incomplete work.
- **Type consistency:** `LabeledVehicleProposal` flows from proposal labeling into dataset records; `ScoredVehicleProposal` flows from model scoring into operating-point selection and fusion; `VehicleConfirmationThresholds` is frozen once and consumed by both offline and runtime fusion; timing reports use measured per-run tuples throughout.
- **Leakage check:** Only train source groups feed labels, crop data, model selection, proposal floor, or confirmation threshold. Validation is consumed once in Task 9 after the operating point is frozen.
