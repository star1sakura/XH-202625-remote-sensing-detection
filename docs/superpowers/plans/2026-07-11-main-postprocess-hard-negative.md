# Main Post-Process and Hard-Negative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Improve the XH25 competition candidate without replacing xh25-yolo26s-e80, first by removing provable ship duplicate detections and then by training main-hn on deterministic train-only hard negatives and modest FSC upsampling.

**Architecture:** Phase A adds a class-scoped secondary NMS policy after the existing tiled merge. It works in InferencePipeline and against immutable baseline COCO predictions, so ship IoU and DIoU variants are comparable without retraining. Phase B extends the prepared dataset with train IDs and train COCO truth, mines only high-confidence ship/vehicle train false positives whose padded crop is label-free, materializes empty-label crops, and duplicates only FSC-containing train images at a fixed multiplier.

**Tech Stack:** Python 3.11/3.12, PyTorch/Ultralytics, OpenCV, Typer, PyYAML, pytest, Ruff, existing xh_detect evaluator/exporter/pipeline.

## Global Constraints

- Keep configs/xh25-yolo26s-e80.yaml, its weights, the grouped 85/15 split, base architecture, image size 1024, epochs 80, batch 8, workers 4, seed 42, and base augmentation unchanged.
- Use official-style HBB matching: IoU 0.50 for ship/aircraft and 0.35 for FSC vehicle. Never alter evaluator matching semantics.
- Mine only from datasets/xh25 train artifacts: images/train, labels/train, manifests/train-image-map.json, reports/train-ground-truth.json, and source-groups.json. Never inspect validation predictions as training data.
- Reject a hard-negative crop when its rectangle enlarged by object_margin intersects any non-difficult original train GT HBB.
- Promote only when Overall Recall >= 0.961562, Overall FDR <= 0.037244, Ship Recall >= 0.823383, Aircraft Recall >= 0.989075, Vehicle Recall >= 0.705128, one overall value is strictly better, and one 10000 x 10000 RTX3090 inference takes <= 20 seconds.
- Every report records TP, FP, FN, Recall, and FDR for Overall, Aircraft, Ship, and Vehicle. Data, weights, predictions, and caches remain untracked.
- Phase C, main-hn-density, is implemented behind an explicit training flag and
  evaluated as a separate ablation. It is never silently enabled for main or
  main-hn.

## File Structure

- Create src/xh_detect/postprocess.py: validated IoU/DIoU rules and deterministic class-specific suppression.
- Create src/xh_detect/complementarity.py: evaluator-aligned true-positive set and pairwise oracle analysis.
- Modify src/xh_detect/config.py and src/xh_detect/pipeline.py: parse class_suppression and run it after current tile merge.
- Modify src/xh_detect/evaluator.py: overlap-versus-background false-positive audit.
- Modify src/xh_detect/cli.py: apply-suppression, audit-false-positives, and build-main-hn-xh25 commands.
- Modify src/xh_detect/data/xh25.py: train image ID map and train COCO truth.
- Create src/xh_detect/data/hard_negative.py: train-only candidate selection and main-hn materialization.
- Create configs/xh25-main-ship-iou.yaml, configs/xh25-main-ship-diou.yaml, and docs/experiments/main-postprocess-hard-negative.md.
- Create or modify focused tests: tests/test_postprocess.py, tests/test_hard_negative.py, tests/test_config.py, tests/test_pipeline.py, tests/test_evaluator.py, tests/test_xh25.py, tests/test_cli.py.

---

### Task 0: Analyze Existing Model Complementarity Before Retraining

**Files:**
- Create: src/xh_detect/complementarity.py
- Create: tests/test_complementarity.py
- Modify: src/xh_detect/cli.py
- Modify: tests/test_cli.py

**Interfaces:**
- Consumes: Mapping[str, Iterable[Detection]], Iterable[ObjectAnnotation], Taxonomy, hbb_iou, obb_to_hbb.
- Produces: analyze_complementarity(predictions_by_model, ground_truth, taxonomy, baseline_name) -> ComplementarityReport and xh-detect analyze-complementarity.

- [ ] **Step 1: Write failing matching and pairwise tests**

Create two vehicle truths, let main match only the first, let sph-p2 match both
plus one background false positive, and assert the pairwise report contains
shared_tp=1, baseline_only_tp=0, candidate_only_tp=1, oracle_tp=2,
oracle_recall=1.0, and candidate_fp=1. Add a ship case to verify its 0.50 IoU
threshold and deterministic score-order matching.

- [ ] **Step 2: Verify the tests fail for the missing module**

Run: python -m pytest tests/test_complementarity.py -q

Expected: collection fails because xh_detect.complementarity does not exist.

- [ ] **Step 3: Implement evaluator-aligned assignment and serialization**

Represent each non-difficult truth by its original integer index. For every
model and coarse class, greedily match predictions by descending score and
original index using the competition threshold. Store matched truth indices,
TP, FP, FN, Recall, and FDR. For each non-baseline model compute set
intersection/differences with baseline and the union oracle. Reject a missing
baseline, duplicate model names, and fewer than two models.

Register analyze-complementarity with repeatable --prediction NAME=PATH,
--ground-truth-json, --baseline-name, --taxonomy, and --output-path options.
Load all files with the existing COCO loaders, write JSON with _write_json, and
print the output path.

- [ ] **Step 4: Verify the module and command**

Run: python -m pytest tests/test_complementarity.py tests/test_cli.py -q; python -m ruff check src/xh_detect/complementarity.py tests/test_complementarity.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/complementarity.py src/xh_detect/cli.py tests/test_complementarity.py tests/test_cli.py docs/superpowers/specs/2026-07-11-main-score-improvement-design.md docs/superpowers/plans/2026-07-11-main-postprocess-hard-negative.md
git commit -m "feat: analyze detection model complementarity"
~~~

### Task 1: Add Deterministic Class-Scoped IoU and DIoU Suppression

**Files:**
- Create: src/xh_detect/postprocess.py
- Create: tests/test_postprocess.py

**Interfaces:**
- Consumes: Detection, hbb_iou, and obb_to_hbb.
- Produces: SuppressionRule(method: str, threshold: float) and suppress_class_detections(detections: Iterable[Detection], rules: Mapping[int, SuppressionRule]) -> list[Detection].

- [ ] **Step 1: Write failing suppression tests**

Create tests/test_postprocess.py:

~~~
from xh_detect.postprocess import SuppressionRule, suppress_class_detections
from xh_detect.types import Detection


def _box(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_diou_suppression_keeps_highest_score_and_non_overlapping_detection() -> None:
    detections = [
        Detection("img", 3, 0.95, _box(0, 0, 10, 10)),
        Detection("img", 3, 0.90, _box(1, 0, 11, 10)),
        Detection("img", 3, 0.80, _box(30, 0, 40, 10)),
    ]
    kept = suppress_class_detections(
        detections, {3: SuppressionRule(method="diou", threshold=0.30)}
    )
    assert [(item.score, item.polygon) for item in kept] == [
        (0.95, _box(0, 0, 10, 10)),
        (0.80, _box(30, 0, 40, 10)),
    ]


def test_unconfigured_class_preserves_existing_detection_order() -> None:
    detections = [
        Detection("img", 24, 0.70, _box(0, 0, 10, 10)),
        Detection("img", 24, 0.60, _box(0, 0, 10, 10)),
    ]
    assert suppress_class_detections(
        detections, {3: SuppressionRule(method="iou", threshold=0.30)}
    ) == detections
~~~

Add parametrized validation tests for invalid method, invalid boolean class ID, IoU threshold outside [0, 1], and DIoU threshold outside [-1, 1].

- [ ] **Step 2: Run the test to verify it fails**

Run: python -m pytest tests/test_postprocess.py -q

Expected: FAIL during collection because xh_detect.postprocess does not exist.

- [ ] **Step 3: Write minimal implementation**

Create postprocess.py with:

~~~
@dataclass(frozen=True)
class SuppressionRule:
    method: str
    threshold: float

    def __post_init__(self) -> None:
        if self.method not in {"iou", "diou"}:
            raise ValueError("suppression method must be iou or diou")
        if not math.isfinite(self.threshold):
            raise ValueError("suppression threshold must be finite")
        lower, upper = (-1.0, 1.0) if self.method == "diou" else (0.0, 1.0)
        if not lower <= self.threshold <= upper:
            raise ValueError("suppression threshold is outside the method range")


def diou(box_a: HBB, box_b: HBB) -> float:
    overlap = hbb_iou(box_a, box_b)
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    distance_sq = ((ax1 + ax2 - bx1 - bx2) / 2) ** 2 + ((ay1 + ay2 - by1 - by2) / 2) ** 2
    cx1, cy1 = min(ax1, bx1), min(ay1, by1)
    cx2, cy2 = max(ax2, bx2), max(ay2, by2)
    diagonal_sq = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    return overlap if diagonal_sq == 0.0 else overlap - distance_sq / diagonal_sq
~~~

Group detections by (image_id, class_id). Classes without a rule are copied unchanged. For configured classes, sort (original_index, detection) by (-score, original_index); retain the first item and remove later items when rule overlap is >= threshold. Compute all overlap from HBBs, which matches competition scoring. Return all retained records sorted again by (-score, original_index).

- [ ] **Step 4: Run focused verification**

Run: python -m pytest tests/test_postprocess.py -q; python -m ruff check src/xh_detect/postprocess.py tests/test_postprocess.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/postprocess.py tests/test_postprocess.py
git commit -m "feat: add class scoped iou and diou suppression"
~~~

### Task 2: Wire Suppression Into Runtime Configuration and Pipeline

**Files:**
- Modify: src/xh_detect/config.py
- Modify: src/xh_detect/pipeline.py
- Modify: tests/test_config.py
- Modify: tests/test_pipeline.py

**Interfaces:**
- Consumes: Task 1 SuppressionRule and suppress_class_detections.
- Produces: PipelineConfig.class_suppression: Mapping[int, SuppressionRule], empty by default.

- [ ] **Step 1: Write failing parser and pipeline tests**

Add this config test:

~~~
def test_pipeline_config_loads_ship_only_suppression(tmp_path: Path) -> None:
    path = tmp_path / "ship.yaml"
    path.write_text(
        "task: detect\ntaxonomy: xh25\nmodel_path: model.pt\n"
        "class_suppression:\n  3: {method: diou, threshold: 0.15}\n"
        "class_thresholds:\n" + "".join(f"  {i}: 0.25\n" for i in range(25)),
        encoding="utf-8",
    )
    config = PipelineConfig.from_yaml(path)
    assert config.class_suppression[3].method == "diou"
    assert config.class_suppression[3].threshold == 0.15
~~~

Add a pipeline fixture with overlapping class 3 predictions and non-overlapping class 24 predictions. Assert the class 3 duplicate is removed by class_suppression while class 24 detections remain.

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_config.py tests/test_pipeline.py -q

Expected: FAIL because PipelineConfig does not accept class_suppression.

- [ ] **Step 3: Implement configuration and post-merge behavior**

Add this field after class_thresholds:

~~~
class_suppression: Mapping[int, SuppressionRule] = field(default_factory=dict)
~~~

In __post_init__, copy it, require each non-boolean int key to belong to the selected taxonomy, require each value to be SuppressionRule, and store MappingProxyType(copy). In to_dict, serialize:

~~~
"class_suppression": {
    class_id: {"method": rule.method, "threshold": rule.threshold}
    for class_id, rule in self.class_suppression.items()
},
~~~

In from_yaml, pop class_suppression with {}, reject non-mappings, convert YAML keys to int, construct SuppressionRule(**dict(rule_payload)), and add the new key to valid_keys. No existing YAML should need editing.

In pipeline.py, retain the existing merge_detections call then add:

~~~
merged = merge_detections(detections, iou_threshold=self.config.merge_iou)
final_detections = suppress_class_detections(merged, self.config.class_suppression)
~~~

Return final_detections. Because cache namespace uses config.to_dict, policies are isolated automatically.

- [ ] **Step 4: Verify behavior and compatibility**

Run: python -m pytest tests/test_config.py tests/test_pipeline.py tests/test_postprocess.py -q

Expected: PASS, including unchanged existing XH25 YAML loading.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/config.py src/xh_detect/pipeline.py tests/test_config.py tests/test_pipeline.py
git commit -m "feat: configure ship specific suppression in pipeline"
~~~

### Task 3: Audit Ship False Positives as Overlap or Background

**Files:**
- Modify: src/xh_detect/evaluator.py
- Modify: tests/test_evaluator.py

**Interfaces:**
- Produces: FalsePositiveSources(overlap: int, background: int), FalsePositiveAudit(by_coarse_class: dict[str, FalsePositiveSources]), audit_false_positives(predictions, ground_truth, taxonomy), and false_positive_audit_to_dict(audit).

- [ ] **Step 1: Write failing audit tests**

Append:

~~~
def test_false_positive_audit_separates_ship_overlap_from_background() -> None:
    truth = [ObjectAnnotation("img", 3, GT)]
    predictions = [
        Detection("img", 3, 0.95, GT),
        Detection("img", 3, 0.90, GT),
        Detection("img", 3, 0.80, ((30, 0), (40, 0), (40, 10), (30, 10))),
    ]
    audit = audit_false_positives(predictions, truth, taxonomy=get_taxonomy("xh25"))
    assert audit.by_coarse_class["ship"] == FalsePositiveSources(overlap=1, background=1)
    assert audit.by_coarse_class["ship"].total == 2
~~~

Add a vehicle case whose 0.35-IoU prediction matches and produces neither FP source.

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_evaluator.py -q

Expected: FAIL because audit_false_positives is absent.

- [ ] **Step 3: Implement audit with the existing coarse matching rules**

Reuse exact score ordering, difficult-object exclusion, coarse-class keying, and _iou_threshold from _match. For an unmatched prediction, calculate max HBB IoU against all same-image/same-coarse non-difficult truth. Add it to overlap when max IoU > 0, otherwise background. Initialize aircraft, ship, and vehicle with zero counts. Validate source counts are non-negative and serialize each category as overlap, background, total.

- [ ] **Step 4: Verify evaluator coverage**

Run: python -m pytest tests/test_evaluator.py -q; python -m ruff check src/xh_detect/evaluator.py tests/test_evaluator.py

Expected: PASS, and all existing evaluate() assertions remain unchanged.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/evaluator.py tests/test_evaluator.py
git commit -m "feat: audit overlapping and background false positives"
~~~

### Task 4: Expose Reproducible Suppression and Audit Commands

**Files:**
- Modify: src/xh_detect/cli.py
- Modify: tests/test_cli.py

**Interfaces:**
- Produces: xh-detect apply-suppression and xh-detect audit-false-positives.

- [ ] **Step 1: Write failing CLI tests**

Use Typer CliRunner and mocks to verify:

~~~
result = runner.invoke(app, [
    "apply-suppression", "--predictions-json", str(predictions),
    "--image-map-json", str(image_map), "--config-path", str(config),
    "--output-json", str(output),
])
assert result.exit_code == 0
assert result.stdout.strip() == str(output)

result = runner.invoke(app, [
    "audit-false-positives", "--predictions-json", str(predictions),
    "--ground-truth-json", str(truth), "--output-path", str(audit), "--taxonomy", "xh25",
])
assert result.exit_code == 0
assert json.loads(audit.read_text(encoding="utf-8"))["by_coarse_class"]["ship"]["total"] == 2
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_cli.py -q

Expected: FAIL because neither command is registered.

- [ ] **Step 3: Implement thin command wrappers**

apply-suppression must load a PipelineConfig, load predictions with its taxonomy, apply only config.class_suppression, and write through export_coco_results using _load_image_id_map:

~~~
@app.command("apply-suppression")
def apply_suppression_command(
    predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_json: Annotated[Path, typer.Option()] = Path("outputs/xh25/postprocess/predictions.json"),
) -> None:
    config = PipelineConfig.from_yaml(config_path)
    taxonomy = get_taxonomy(config.taxonomy)
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy)
    kept = suppress_class_detections(predictions, config.class_suppression)
    export_coco_results(kept, _load_image_id_map(image_map_json), output_json, taxonomy.valid_ids)
    typer.echo(str(output_json))
~~~

audit-false-positives loads the chosen taxonomy, invokes the evaluator audit, writes its dict with _write_json, and prints the path. Do not add training behavior to either command.

- [ ] **Step 4: Verify CLI and Phase A units**

Run: python -m pytest tests/test_cli.py tests/test_config.py tests/test_pipeline.py tests/test_postprocess.py tests/test_evaluator.py -q

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/cli.py tests/test_cli.py
git commit -m "feat: add suppression and false positive audit commands"
~~~

### Task 5: Publish Ship-IoU/DIoU Configs and Phase A Runbook

**Files:**
- Create: configs/xh25-main-ship-iou.yaml
- Create: configs/xh25-main-ship-diou.yaml
- Create: docs/experiments/main-postprocess-hard-negative.md
- Modify: tests/test_config.py

**Interfaces:**
- Consumes: immutable outputs/xh25/baseline/val-predictions.json and fixed validation truth.
- Produces: two runnable main configs and a promotion table.

- [ ] **Step 1: Write failing config tests**

~~~
@pytest.mark.parametrize(
    ("name", "method", "threshold"),
    [("xh25-main-ship-iou.yaml", "iou", 0.30), ("xh25-main-ship-diou.yaml", "diou", 0.15)],
)
def test_main_ship_postprocess_configs_are_ship_only(name: str, method: str, threshold: float) -> None:
    config = PipelineConfig.from_yaml(Path("configs") / name)
    assert config.model_path == "runs/train/xh25-yolo26s-e80/weights/best.pt"
    assert set(config.class_suppression) == {0, 1, 2, 3}
    assert {rule.method for rule in config.class_suppression.values()} == {method}
    assert {rule.threshold for rule in config.class_suppression.values()} == {threshold}
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_config.py -q

Expected: FAIL because both YAML files are absent.

- [ ] **Step 3: Create configurations and document Phase A**

Copy every baseline field and all 25 thresholds from configs/xh25-yolo26s-e80.yaml. Append exactly this to the IoU config:

~~~
class_suppression:
  0: {method: iou, threshold: 0.30}
  1: {method: iou, threshold: 0.30}
  2: {method: iou, threshold: 0.30}
  3: {method: iou, threshold: 0.30}
~~~

Append exactly this to the DIoU config:

~~~
class_suppression:
  0: {method: diou, threshold: 0.15}
  1: {method: diou, threshold: 0.15}
  2: {method: diou, threshold: 0.15}
  3: {method: diou, threshold: 0.15}
~~~

Document this complete DIoU flow; repeat the suppression/evaluate/audit triplet for IoU:

~~~
.venv/bin/xh-detect audit-false-positives \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/main-postprocess/baseline-fp-audit.json

.venv/bin/xh-detect apply-suppression \
  --predictions-json outputs/xh25/baseline/val-predictions.json \
  --image-map-json datasets/xh25/manifests/val-image-map.json \
  --config-path configs/xh25-main-ship-diou.yaml \
  --output-json outputs/xh25/main-postprocess/ship-diou-predictions.json

.venv/bin/xh-detect evaluate \
  --predictions-json outputs/xh25/main-postprocess/ship-diou-predictions.json \
  --ground-truth-json datasets/xh25/reports/val-ground-truth.json \
  --taxonomy xh25 \
  --output-path outputs/xh25/main-postprocess/ship-diou-report.json
~~~

Then specify optimize-thresholds, competition-report, compare-experiments, audit-false-positives, and benchmark --repeats 5 for both candidates. The benchmark must use the same YAML applied during inference, not just offline JSON suppression.

- [ ] **Step 4: Verify config and Phase A checks**

Run: python -m pytest tests/test_config.py tests/test_postprocess.py tests/test_pipeline.py -q; python -m ruff check src tests

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add configs/xh25-main-ship-iou.yaml configs/xh25-main-ship-diou.yaml docs/experiments/main-postprocess-hard-negative.md tests/test_config.py
git commit -m "docs: add main ship suppression experiment"
~~~

### Task 6: Publish Train IDs and COCO Truth for Safe Mining

**Files:**
- Modify: src/xh_detect/data/xh25.py
- Modify: tests/test_xh25.py

**Interfaces:**
- Consumes: ImageRecord, _coco_ground_truth, and transaction-safe data writers.
- Produces: manifests/train-image-map.json and reports/train-ground-truth.json with IDs 1..N over sorted train stems.

- [ ] **Step 1: Write failing preparation tests**

In the existing prepared-dataset fixture assert:

~~~
train_map = json.loads((output / "manifests" / "train-image-map.json").read_text(encoding="utf-8"))
assert train_map == {"train-a": 1, "train-b": 2}
train_truth = json.loads((output / "reports" / "train-ground-truth.json").read_text(encoding="utf-8"))
assert {image["file_name"] for image in train_truth["images"]} == {
    "images/train/train-a.jpg", "images/train/train-b.jpg"
}
~~~

Also mutate each artifact and assert _validate_materialized_dataset rejects the changed train map and train COCO ID mapping.

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_xh25.py -q

Expected: FAIL because train artifacts do not exist.

- [ ] **Step 3: Materialize and validate both split artifacts**

Rename _coco_ground_truth(val_records, image_map) parameters to records, image_map without changing payload semantics. In _materialize_locked_stage create:

~~~
train_image_map = {
    record.stem: image_id
    for image_id, record in enumerate(sorted(train_records, key=lambda item: item.stem), start=1)
}
val_image_map = {
    record.stem: image_id
    for image_id, record in enumerate(sorted(val_records, key=lambda item: item.stem), start=1)
}
train_coco = _coco_ground_truth(train_records, train_image_map)
val_coco = _coco_ground_truth(val_records, val_image_map)
~~~

Atomically write the two train files next to current validation files. Extend _validate_materialized_dataset to compute and compare both maps and both COCO file_name/ID mappings. Do not alter split selection, PreparedDataset, or dataset.yaml.

- [ ] **Step 4: Verify**

Run: python -m pytest tests/test_xh25.py -q; python -m ruff check src/xh_detect/data/xh25.py tests/test_xh25.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/data/xh25.py tests/test_xh25.py
git commit -m "feat: publish train ids and ground truth for hard negative mining"
~~~

### Task 7: Select and Materialize Train-Only Hard Negatives

**Files:**
- Create: src/xh_detect/data/hard_negative.py
- Create: tests/test_hard_negative.py

**Interfaces:**
- Produces:
~~~
@dataclass(frozen=True)
class HardNegativePolicy:
    confidence_floor: float = 0.60
    crop_size: int = 512
    object_margin: int = 16
    max_crops_per_group: int = 2
    vehicle_multiplier: int = 2
    seed: int = 42


@dataclass(frozen=True)
class HardNegativeResult:
    output_root: Path
    original_train_images: int
    vehicle_upsampled_images: int
    selected_hard_negatives: int
    rejected_target_overlap: int
    selected_by_coarse_class: dict[str, int]


def build_main_hn_dataset(
    source_root: Path, predictions_json: Path, output_root: Path, policy: HardNegativePolicy
) -> HardNegativeResult
~~~

- [ ] **Step 1: Write failing safety and determinism tests**

Use a two-train-image/one-validation-image fixture and cover these cases:

~~~
def test_builder_rejects_crop_that_overlaps_any_train_target(tmp_path: Path) -> None:
    source, predictions = _prepared_source_with_prediction_near_target(tmp_path)
    with pytest.raises(ValueError, match="no label-safe hard negatives"):
        build_main_hn_dataset(source, predictions, tmp_path / "out",
                              HardNegativePolicy(crop_size=64, object_margin=8))


def test_builder_is_seed_deterministic_and_writes_empty_labels(tmp_path: Path) -> None:
    source, predictions = _prepared_source_with_background_ship_fp(tmp_path)
    policy = HardNegativePolicy(0.60, 64, 8, 1, 2, 42)
    first = build_main_hn_dataset(source, predictions, tmp_path / "out-a", policy)
    second = build_main_hn_dataset(source, predictions, tmp_path / "out-b", policy)
    assert first.selected_hard_negatives == second.selected_hard_negatives == 1
    assert (tmp_path / "out-a" / "labels" / "train" / "train-a__hn01.txt").read_text() == ""
    assert (tmp_path / "out-a" / "manifests" / "train.txt").read_text() == (
        tmp_path / "out-b" / "manifests" / "train.txt"
    ).read_text()
~~~

Also assert per-group caps, source/output overlap rejection, nonempty output rejection, a validation image ID rejection, one-time validation materialization, and vehicle_multiplier aliases.

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_hard_negative.py -q

Expected: FAIL during collection because xh_detect.data.hard_negative does not exist.

- [ ] **Step 3: Implement bounded selection and data materialization**

Validate policy values: confidence_floor in [0, 1], crop_size > 0, object_margin >= 0, max_crops_per_group > 0, vehicle_multiplier >= 1, seed >= 0, with booleans rejected.

Load train predictions and truth with xh25 taxonomy. Invert train-image-map.json; reject every prediction not mapped to a train stem. For predictions whose coarse class is ship or vehicle, score >= confidence_floor, and no same-coarse train truth meets the official threshold, form a clipped square crop centered on prediction HBB. Expand only the crop safety rectangle by object_margin. Reject it if it has positive intersection with any non-difficult train truth HBB.

Sort eligible candidates by (-score, sha256(f"{seed}:{stem}:{prediction_index}")); cap at max_crops_per_group using source-groups.json. Materialize:

~~~
images/train/<original>.jpg
labels/train/<original>.txt
images/train/<vehicle>__vehup01.jpg
labels/train/<vehicle>__vehup01.txt
images/train/<source>__hn01.jpg
labels/train/<source>__hn01.txt
images/val/<original>.jpg
labels/val/<original>.txt
manifests/train.txt
manifests/val.txt
manifests/source-groups.json
reports/hard-negative.json
reports/hard-negative.md
dataset.yaml
~~~

Use hardlink_to with shutil.copy2 fallback for original files, as ship_balance.py does. Hard-negative images are cv2 crops and labels are zero-byte text files. Copy validation once. Train aliases are sorted and distinct; a vehicle source appears vehicle_multiplier times total. dataset.yaml points to images/train and images/val and retains the source names map. Report all six policy values, selected/rejected counts, and ship/vehicle selections.

- [ ] **Step 4: Verify hard-negative and adjacent regressions**

Run: python -m pytest tests/test_hard_negative.py tests/test_xh25.py tests/test_ship_balance.py -q; python -m ruff check src/xh_detect/data/hard_negative.py tests/test_hard_negative.py

Expected: PASS.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/data/hard_negative.py tests/test_hard_negative.py
git commit -m "feat: build train only hard negative xh25 dataset"
~~~

### Task 8: Add Builder Command and Main-HN Evaluation Contract

**Files:**
- Modify: src/xh_detect/cli.py
- Modify: tests/test_cli.py
- Modify: docs/experiments/main-postprocess-hard-negative.md

**Interfaces:**
- Consumes: build_main_hn_dataset and HardNegativePolicy.
- Produces: xh-detect build-main-hn-xh25 and exact training/evaluation commands.

- [ ] **Step 1: Write failing command test**

Patch xh_detect.cli.build_main_hn_dataset, invoke:

~~~
result = runner.invoke(app, [
    "build-main-hn-xh25", "--source-root", str(source),
    "--predictions-json", str(predictions), "--output-root", str(output),
    "--confidence-floor", "0.60", "--crop-size", "512", "--object-margin", "16",
    "--max-crops-per-group", "2", "--vehicle-multiplier", "2", "--seed", "42",
])
assert result.exit_code == 0
build_main_hn_dataset.assert_called_once()
assert json.loads(result.stdout)["selected_hard_negatives"] == 3
~~~

- [ ] **Step 2: Verify failure**

Run: python -m pytest tests/test_cli.py -q

Expected: FAIL because build-main-hn-xh25 is not registered.

- [ ] **Step 3: Implement the command and complete runbook**

Register build-main-hn-xh25 with the six policy options, construct exactly one HardNegativePolicy, map TypeError/ValueError to typer.BadParameter, and print all HardNegativeResult fields as JSON.

Append these exact server commands to the runbook:

~~~
.venv/bin/xh-detect infer-dataset \
  --images-dir datasets/xh25/images/train \
  --image-map-json datasets/xh25/manifests/train-image-map.json \
  --config-path configs/xh25-yolo26s-e80.yaml \
  --output-json outputs/xh25/main-hn/train-predictions.json

.venv/bin/xh-detect build-main-hn-xh25 \
  --source-root datasets/xh25 \
  --predictions-json outputs/xh25/main-hn/train-predictions.json \
  --output-root datasets/xh25-main-hn \
  --confidence-floor 0.60 \
  --crop-size 512 \
  --object-margin 16 \
  --max-crops-per-group 2 \
  --vehicle-multiplier 2 \
  --seed 42

.venv/bin/xh-detect train \
  --dataset-yaml datasets/xh25-main-hn/dataset.yaml \
  --model yolo26s.pt \
  --pretrained yolo26s.pt \
  --epochs 80 \
  --image-size 1024 \
  --device 0 \
  --batch 8 \
  --workers 4 \
  --no-amp \
  --project runs/train \
  --name xh25-main-hn \
  --no-resume
~~~

Document infer-dataset, evaluate, audit-false-positives, optimize-thresholds, competition-report, compare-experiments, and benchmark --repeats 5 using outputs/xh25/main-hn. Create configs/xh25-main-hn.yaml by copying main config and changing only model_path to runs/train/xh25-main-hn/weights/best.pt.

- [ ] **Step 4: Run complete local verification**

Run: python -m pytest tests/test_postprocess.py tests/test_config.py tests/test_pipeline.py tests/test_evaluator.py tests/test_xh25.py tests/test_hard_negative.py tests/test_cli.py -q; python -m ruff format --check .; python -m ruff check .

Expected: all commands pass.

- [ ] **Step 5: Commit**

~~~
git add src/xh_detect/cli.py tests/test_cli.py docs/experiments/main-postprocess-hard-negative.md
git commit -m "docs: add main hard negative experiment workflow"
~~~

### Task 9: Execute Phase A, Main-HN, and Main-HN-Density on RTX3090

**Files:**
- Create, ignored: outputs/xh25/main-postprocess/
- Create, ignored: outputs/xh25/main-hn/
- Create, ignored: datasets/xh25-main-hn/
- Create, ignored: runs/train/xh25-main-hn/
- Create, ignored: runs/train/xh25-main-hn-density/

**Interfaces:**
- Consumes: Tasks 1-8, fixed datasets/xh25, xh25-yolo26s-e80 weights, and an RTX3090.
- Produces: raw/calibrated reports, audits, benchmarks, density ablation evidence,
  and one promotion decision.

- [ ] **Step 1: Verify server identity**

Run:

~~~
git status --short
.venv/bin/python -m pytest tests/test_postprocess.py tests/test_config.py tests/test_pipeline.py tests/test_evaluator.py tests/test_xh25.py tests/test_hard_negative.py tests/test_cli.py -q
sha256sum datasets/xh25/manifests/train.txt datasets/xh25/manifests/val.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
~~~

Expected: clean tracked worktree, passing tests, recorded split hashes, and one 24 GB RTX3090.

- [ ] **Step 2: Run Phase A before any retraining**

Run the IoU and DIoU suppression/evaluate/audit flows from Task 5. For both candidate prediction files, run optimize-thresholds against outputs/xh25/baseline/report.json, competition-report, compare-experiments, and benchmark --config-path with the matching YAML --repeats 5.

Expected: raw and calibrated reports, FP audits, proxy artifacts, and timing artifacts. If both candidates raise Overall FDR or lower Ship Recall below 0.823383, retain the audit evidence and move to Phase B without promoting either.

- [ ] **Step 3: Build and train main-hn once**

Run the Task 8 build/training commands. Afterwards use configs/xh25-main-hn.yaml for fixed-validation inference, raw report, FP audit, threshold optimization, proxy, comparison, and benchmark.

Expected: hard-negative report; raw and calibrated reports; proxy; comparison; benchmark; and training weights. No validation file should appear in the hard-negative train manifest.

- [ ] **Step 4: Apply the promotion rule from artifacts**

Load main/raw/calibrated reports and print every promotion constraint with TP/FP/FN. Record exactly one line in the runbook:

~~~
PROMOTE: <candidate> satisfies every global constraint and improves <recall|fdr>.
RETAIN MAIN: no candidate satisfies every global constraint; xh25-yolo26s-e80 remains the submission model.
~~~

Expected: one artifact-backed recommendation, not a choice based only on mAP.

- [ ] **Step 5: Leave generated data untracked**

Run: git status --short; git log --oneline -5

Expected: source/docs commits from Tasks 1-8 are present; datasets, weights, predictions, cache files, and credentials are never staged.

## Plan Self-Review

- **Spec coverage:** Tasks 1-5 implement the immutable prediction audit, ship-only IoU/DIoU suppression, YAML runtime configurations, calibrated evaluation, proxy, and timing. Tasks 6-8 implement train-only IDs/truth, crop safety, per-group caps, deterministic FSC upsampling, manifests, CLI, and unchanged-main training. Task 9 applies every promotion gate.
- **Placeholder scan:** No unresolved task marker or unspecified code step remains.
  Main-HN-density is isolated behind an explicit flag and evaluated separately.
- **Type consistency:** SuppressionRule flows from postprocess.py to PipelineConfig, InferencePipeline, and CLI. HardNegativePolicy and HardNegativeResult flow from data/hard_negative.py to CLI and runbook. YAML parsing remains centralized in PipelineConfig.from_yaml.
