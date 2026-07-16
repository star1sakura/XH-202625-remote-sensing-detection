from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import TypeVar

import yaml

from xh_detect.evaluator import EvaluationReport, evaluate, report_to_dict
from xh_detect.taxonomy import Taxonomy
from xh_detect.thresholds import ObjectiveScore, objective_from_report, parse_threshold_grid
from xh_detect.types import Detection, ObjectAnnotation

ItemT = TypeVar("ItemT", Detection, ObjectAnnotation)

DEFAULT_CALIBRATION_GRID: tuple[float, ...] = tuple(
    round(0.25 + index * 0.025, 3) for index in range(15)
)
DEFAULT_CALIBRATION_GRID_TEXT = ",".join(
    f"{threshold:.3f}" for threshold in DEFAULT_CALIBRATION_GRID
)


@dataclass(frozen=True)
class ImageGroupMapping:
    image_to_group: dict[str, str]
    image_to_stem: dict[str, str]


@dataclass(frozen=True)
class ThresholdCandidate:
    threshold: float
    report: EvaluationReport
    objective: ObjectiveScore


@dataclass(frozen=True)
class FoldCalibration:
    fold: int
    selected_threshold: float | None
    eligible_thresholds: tuple[float, ...]
    recall_floor: float
    fdr_cap: float
    baseline_calibration: EvaluationReport
    candidate_calibration: EvaluationReport | None
    baseline_holdout: EvaluationReport
    candidate_holdout: EvaluationReport | None


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    failure_reason: str | None
    seed: int
    folds: int
    raw_threshold: float
    grid: tuple[float, ...]
    group_to_fold: dict[str, int]
    image_to_fold: dict[str, int]
    image_to_group: dict[str, str]
    fold_results: tuple[FoldCalibration, ...]
    baseline_oof_report: EvaluationReport
    candidate_oof_report: EvaluationReport | None
    final_threshold: float | None
    threshold_range: float | None
    acceptance: dict[str, dict[str, object]]
    oof_predictions: tuple[Detection, ...]

    @property
    def passed(self) -> bool:
        return self.status == "passed"


def _load_json(path: Path | str, label: str) -> object:
    source = Path(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {exc.msg}") from exc


def _normalized_image_id(value: object, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise TypeError(f"{context} image ID must be a string or integer")
    image_id = str(value)
    if not image_id:
        raise ValueError(f"{context} image ID must not be empty")
    return image_id


def load_image_group_mapping(
    ground_truth_json: Path | str,
    source_groups_json: Path | str,
) -> ImageGroupMapping:
    ground_truth = _load_json(ground_truth_json, "ground truth")
    source_groups = _load_json(source_groups_json, "source groups")
    if not isinstance(ground_truth, Mapping):
        raise TypeError("COCO ground truth must be an object")
    images = ground_truth.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("COCO ground truth images must be a non-empty list")
    if not isinstance(source_groups, Mapping):
        raise TypeError("source groups must be an object")

    image_to_group: dict[str, str] = {}
    image_to_stem: dict[str, str] = {}
    seen_stems: set[str] = set()
    for index, image in enumerate(images):
        if not isinstance(image, Mapping) or "id" not in image or "file_name" not in image:
            raise ValueError(f"ground truth image {index} must contain id and file_name")
        image_id = _normalized_image_id(image["id"], f"ground truth image {index}")
        file_name = image["file_name"]
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError(f"ground truth image {index} file_name must be non-empty")
        stem = Path(file_name).stem
        if image_id in image_to_group:
            raise ValueError(f"duplicate ground truth image ID: {image_id}")
        if stem in seen_stems:
            raise ValueError(f"duplicate ground truth image stem: {stem}")
        seen_stems.add(stem)

        source_record = source_groups.get(stem)
        if not isinstance(source_record, Mapping):
            raise ValueError(f"source group is missing for ground truth image stem: {stem}")
        group = source_record.get("group")
        split = source_record.get("split")
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"source group for {stem} must contain a non-empty group")
        if split != "val":
            raise ValueError(f"source group for validation image {stem} has split {split!r}")
        image_to_group[image_id] = group
        image_to_stem[image_id] = stem
    return ImageGroupMapping(image_to_group=image_to_group, image_to_stem=image_to_stem)


def _stable_hash(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _validate_non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite number in [0, 1]")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return normalized


def _partition_score(
    fold_vectors: Sequence[Counter[str]],
    totals: Counter[str],
    folds: int,
) -> float:
    target = 1.0 / folds
    weights = {"images": 1.0, "groups": 0.25}
    score = 0.0
    for key, total in totals.items():
        if total <= 0:
            continue
        weight = weights.get(key, 3.0 if key.startswith("fine:") else 2.0)
        score += weight * sum((vector[key] / total - target) ** 2 for vector in fold_vectors)
    return score


def build_group_folds(
    image_to_group: Mapping[str, str],
    ground_truth: Sequence[ObjectAnnotation],
    taxonomy: Taxonomy,
    *,
    folds: int = 5,
    seed: int = 42,
) -> dict[str, int]:
    seed = _validate_non_negative_int(seed, "seed")
    if isinstance(folds, bool) or not isinstance(folds, int):
        raise TypeError("folds must be an integer")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if not image_to_group:
        raise ValueError("image groups must not be empty")
    groups = sorted(set(image_to_group.values()))
    if len(groups) < folds:
        raise ValueError("number of source groups must be at least the number of folds")
    image_ids = set(image_to_group)
    unknown_truth = sorted({item.image_id for item in ground_truth} - image_ids)
    if unknown_truth:
        raise ValueError(f"ground truth contains unmapped image IDs: {unknown_truth[:3]}")

    vector_by_group: dict[str, Counter[str]] = {group: Counter(groups=1) for group in groups}
    for _image_id, group in image_to_group.items():
        vector_by_group[group]["images"] += 1
    for item in ground_truth:
        if item.difficult:
            continue
        vector = vector_by_group[image_to_group[item.image_id]]
        vector[f"fine:{item.class_id}"] += 1
        vector[f"coarse:{taxonomy.coarse_name(item.class_id)}"] += 1
    totals = sum(vector_by_group.values(), Counter())

    def rarity(group: str) -> float:
        vector = vector_by_group[group]
        return sum(
            count / totals[key]
            for key, count in vector.items()
            if key.startswith("fine:") and totals[key]
        )

    ordered_groups = sorted(
        groups,
        key=lambda group: (
            -rarity(group),
            -sum(count for key, count in vector_by_group[group].items() if key.startswith("fine:")),
            -vector_by_group[group]["images"],
            _stable_hash(seed, group),
        ),
    )
    fold_vectors = [Counter() for _ in range(folds)]
    group_to_fold: dict[str, int] = {}
    for group in ordered_groups:
        choices: list[tuple[float, int, int, int]] = []
        for fold in range(folds):
            candidate_vectors = [vector.copy() for vector in fold_vectors]
            candidate_vectors[fold].update(vector_by_group[group])
            choices.append(
                (
                    _partition_score(candidate_vectors, totals, folds),
                    candidate_vectors[fold]["images"],
                    _stable_hash(seed, f"{group}:{fold}"),
                    fold,
                )
            )
        selected = min(choices)[-1]
        group_to_fold[group] = selected
        fold_vectors[selected].update(vector_by_group[group])
    return group_to_fold


def select_threshold_candidate(
    candidates: Sequence[ThresholdCandidate],
    *,
    recall_floor: float,
    fdr_cap: float,
    tie_epsilon: float = 0.0001,
) -> tuple[ThresholdCandidate | None, tuple[float, ...]]:
    recall_floor = _validate_probability(recall_floor, "recall_floor")
    fdr_cap = _validate_probability(fdr_cap, "fdr_cap")
    if isinstance(tie_epsilon, bool) or not isinstance(tie_epsilon, Real):
        raise TypeError("tie_epsilon must be finite and non-negative")
    tie_epsilon = float(tie_epsilon)
    if not math.isfinite(tie_epsilon) or tie_epsilon < 0:
        raise ValueError("tie_epsilon must be finite and non-negative")
    eligible = [
        candidate
        for candidate in candidates
        if candidate.objective.recall >= recall_floor and candidate.objective.fdr <= fdr_cap
    ]
    if not eligible:
        return None, ()
    best_f1 = max(candidate.objective.f1 for candidate in eligible)
    tied = [candidate for candidate in eligible if candidate.objective.f1 >= best_f1 - tie_epsilon]
    selected = min(
        tied,
        key=lambda candidate: (
            candidate.objective.fdr,
            -candidate.objective.recall,
            candidate.threshold,
        ),
    )
    return selected, tuple(candidate.threshold for candidate in eligible)


def _items_for_images(items: Sequence[ItemT], image_ids: set[str]) -> list[ItemT]:
    return [item for item in items if item.image_id in image_ids]


def _filter_at_threshold(predictions: Sequence[Detection], threshold: float) -> list[Detection]:
    return [prediction for prediction in predictions if prediction.score >= threshold]


def _acceptance_check(value: float, operator: str, limit: float) -> dict[str, object]:
    passed = value >= limit if operator == ">=" else value <= limit
    return {"value": value, "operator": operator, "limit": limit, "passed": passed}


def calibrate_thresholds(
    baseline_predictions: Sequence[Detection],
    candidate_predictions: Sequence[Detection],
    ground_truth: Sequence[ObjectAnnotation],
    image_to_group: Mapping[str, str],
    taxonomy: Taxonomy,
    *,
    folds: int = 5,
    seed: int = 42,
    thresholds: str | Sequence[float] = DEFAULT_CALIBRATION_GRID,
    raw_threshold: float = 0.25,
    recall_floor_delta: float = 0.005,
    fdr_cap_delta: float = 0.005,
    tie_epsilon: float = 0.0001,
    acceptance_recall: float = 0.953772,
    acceptance_fdr: float = 0.045037,
    acceptance_ship_recall: float = 0.80,
    acceptance_ship_fdr: float = 0.18,
    acceptance_threshold_range: float = 0.05,
) -> CalibrationResult:
    grid = tuple(parse_threshold_grid(thresholds))
    seed = _validate_non_negative_int(seed, "seed")
    raw_threshold = _validate_probability(raw_threshold, "raw_threshold")
    recall_floor_delta = _validate_probability(recall_floor_delta, "recall_floor_delta")
    fdr_cap_delta = _validate_probability(fdr_cap_delta, "fdr_cap_delta")
    acceptance_recall = _validate_probability(acceptance_recall, "acceptance_recall")
    acceptance_fdr = _validate_probability(acceptance_fdr, "acceptance_fdr")
    acceptance_ship_recall = _validate_probability(acceptance_ship_recall, "acceptance_ship_recall")
    acceptance_ship_fdr = _validate_probability(acceptance_ship_fdr, "acceptance_ship_fdr")
    acceptance_threshold_range = _validate_probability(
        acceptance_threshold_range, "acceptance_threshold_range"
    )
    image_ids = set(image_to_group)
    if not image_ids:
        raise ValueError("image groups must not be empty")
    for label, items in (
        ("baseline predictions", baseline_predictions),
        ("candidate predictions", candidate_predictions),
        ("ground truth", ground_truth),
    ):
        unknown = sorted({item.image_id for item in items} - image_ids)
        if unknown:
            raise ValueError(f"{label} contain unmapped image IDs: {unknown[:3]}")

    group_to_fold = build_group_folds(
        image_to_group, ground_truth, taxonomy, folds=folds, seed=seed
    )
    image_to_fold = {image_id: group_to_fold[group] for image_id, group in image_to_group.items()}
    raw_baseline = _filter_at_threshold(baseline_predictions, raw_threshold)
    fold_results: list[FoldCalibration] = []
    oof_baseline: list[Detection] = []
    oof_candidate: list[Detection] = []
    selected_thresholds: list[float] = []
    failure_reason: str | None = None

    for fold in range(folds):
        holdout_ids = {image_id for image_id, value in image_to_fold.items() if value == fold}
        calibration_ids = image_ids - holdout_ids
        baseline_calibration = evaluate(
            _items_for_images(raw_baseline, calibration_ids),
            _items_for_images(ground_truth, calibration_ids),
            taxonomy=taxonomy,
        )
        baseline_objective = objective_from_report(baseline_calibration)
        recall_floor = max(0.0, baseline_objective.recall - recall_floor_delta)
        fdr_cap = min(1.0, baseline_objective.fdr + fdr_cap_delta)
        calibration_candidate_predictions = _items_for_images(
            candidate_predictions, calibration_ids
        )
        calibration_truth = _items_for_images(ground_truth, calibration_ids)
        candidates: list[ThresholdCandidate] = []
        for threshold in grid:
            report = evaluate(
                _filter_at_threshold(calibration_candidate_predictions, threshold),
                calibration_truth,
                taxonomy=taxonomy,
            )
            candidates.append(
                ThresholdCandidate(
                    threshold=threshold,
                    report=report,
                    objective=objective_from_report(report),
                )
            )
        selected, eligible_thresholds = select_threshold_candidate(
            candidates,
            recall_floor=recall_floor,
            fdr_cap=fdr_cap,
            tie_epsilon=tie_epsilon,
        )
        baseline_holdout_predictions = _items_for_images(raw_baseline, holdout_ids)
        baseline_holdout = evaluate(
            baseline_holdout_predictions,
            _items_for_images(ground_truth, holdout_ids),
            taxonomy=taxonomy,
        )
        oof_baseline.extend(baseline_holdout_predictions)
        candidate_holdout = None
        candidate_calibration = None
        if selected is None:
            failure_reason = failure_reason or f"fold {fold} has no threshold satisfying its gates"
        else:
            selected_thresholds.append(selected.threshold)
            candidate_calibration = selected.report
            holdout_predictions = _filter_at_threshold(
                _items_for_images(candidate_predictions, holdout_ids), selected.threshold
            )
            candidate_holdout = evaluate(
                holdout_predictions,
                _items_for_images(ground_truth, holdout_ids),
                taxonomy=taxonomy,
            )
            oof_candidate.extend(holdout_predictions)
        fold_results.append(
            FoldCalibration(
                fold=fold,
                selected_threshold=None if selected is None else selected.threshold,
                eligible_thresholds=eligible_thresholds,
                recall_floor=recall_floor,
                fdr_cap=fdr_cap,
                baseline_calibration=baseline_calibration,
                candidate_calibration=candidate_calibration,
                baseline_holdout=baseline_holdout,
                candidate_holdout=candidate_holdout,
            )
        )

    baseline_oof_report = evaluate(oof_baseline, ground_truth, taxonomy=taxonomy)
    candidate_oof_report = None
    final_threshold = None
    threshold_range = None
    acceptance: dict[str, dict[str, object]] = {}
    status = "failed"
    if len(selected_thresholds) == folds:
        sorted_thresholds = sorted(selected_thresholds)
        final_threshold = sorted_thresholds[folds // 2]
        threshold_range = max(selected_thresholds) - min(selected_thresholds)
        candidate_oof_report = evaluate(oof_candidate, ground_truth, taxonomy=taxonomy)
        overall = candidate_oof_report.overall_class_agnostic
        ship = candidate_oof_report.by_coarse_class["ship"]
        acceptance = {
            "oof_recall": _acceptance_check(overall.recall, ">=", acceptance_recall),
            "oof_fdr": _acceptance_check(overall.fdr, "<=", acceptance_fdr),
            "ship_recall": _acceptance_check(ship.recall, ">=", acceptance_ship_recall),
            "ship_fdr": _acceptance_check(ship.fdr, "<=", acceptance_ship_fdr),
            "threshold_range": _acceptance_check(threshold_range, "<=", acceptance_threshold_range),
        }
        if all(bool(check["passed"]) for check in acceptance.values()):
            status = "passed"
            failure_reason = None
        else:
            failed_checks = ", ".join(
                name for name, check in acceptance.items() if not check["passed"]
            )
            failure_reason = f"acceptance checks failed: {failed_checks}"

    return CalibrationResult(
        status=status,
        failure_reason=failure_reason,
        seed=seed,
        folds=folds,
        raw_threshold=raw_threshold,
        grid=grid,
        group_to_fold=group_to_fold,
        image_to_fold=image_to_fold,
        image_to_group=dict(image_to_group),
        fold_results=tuple(fold_results),
        baseline_oof_report=baseline_oof_report,
        candidate_oof_report=candidate_oof_report,
        final_threshold=final_threshold,
        threshold_range=threshold_range,
        acceptance=acceptance,
        oof_predictions=tuple(oof_candidate),
    )


def _fold_to_dict(fold: FoldCalibration) -> dict[str, object]:
    return {
        "fold": fold.fold,
        "selected_threshold": fold.selected_threshold,
        "eligible_thresholds": list(fold.eligible_thresholds),
        "recall_floor": fold.recall_floor,
        "fdr_cap": fold.fdr_cap,
        "baseline_calibration": report_to_dict(fold.baseline_calibration),
        "candidate_calibration": (
            None
            if fold.candidate_calibration is None
            else report_to_dict(fold.candidate_calibration)
        ),
        "baseline_holdout": report_to_dict(fold.baseline_holdout),
        "candidate_holdout": (
            None if fold.candidate_holdout is None else report_to_dict(fold.candidate_holdout)
        ),
    }


def calibration_result_to_dict(result: CalibrationResult) -> dict[str, object]:
    return {
        "status": result.status,
        "failure_reason": result.failure_reason,
        "seed": result.seed,
        "folds": result.folds,
        "raw_threshold": result.raw_threshold,
        "threshold_grid": list(result.grid),
        "selected_thresholds": [fold.selected_threshold for fold in result.fold_results],
        "final_threshold": result.final_threshold,
        "threshold_range": result.threshold_range,
        "acceptance": result.acceptance,
        "baseline_oof_report": report_to_dict(result.baseline_oof_report),
        "candidate_oof_report": (
            None
            if result.candidate_oof_report is None
            else report_to_dict(result.candidate_oof_report)
        ),
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    temporary.replace(path)


def _coco_detection(detection: Detection) -> dict[str, object]:
    xs = [point[0] for point in detection.polygon]
    ys = [point[1] for point in detection.polygon]
    image_id: str | int = detection.image_id
    if detection.image_id.isdecimal():
        image_id = int(detection.image_id)
    return {
        "image_id": image_id,
        "category_id": detection.class_id,
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "score": detection.score,
    }


def write_calibration_artifacts(
    result: CalibrationResult,
    output_dir: Path | str,
    taxonomy: Taxonomy,
    *,
    base_config: Path | str | None = None,
    calibrated_config: Path | str | None = None,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": destination / "calibration-summary.json",
        "fold_assignments": destination / "fold-assignments.json",
        "fold_results": destination / "fold-results.json",
        "oof_report": destination / "oof-report.json",
        "oof_predictions": destination / "oof-predictions.json",
    }
    _write_json(paths["summary"], calibration_result_to_dict(result))
    _write_json(
        paths["fold_assignments"],
        {
            "seed": result.seed,
            "folds": result.folds,
            "group_to_fold": result.group_to_fold,
            "image_to_fold": result.image_to_fold,
            "image_to_group": result.image_to_group,
        },
    )
    _write_json(paths["fold_results"], [_fold_to_dict(fold) for fold in result.fold_results])
    _write_json(
        paths["oof_report"],
        {
            "baseline": report_to_dict(result.baseline_oof_report),
            "candidate": (
                None
                if result.candidate_oof_report is None
                else report_to_dict(result.candidate_oof_report)
            ),
        },
    )
    _write_json(
        paths["oof_predictions"],
        [_coco_detection(detection) for detection in result.oof_predictions],
    )

    if result.final_threshold is not None:
        threshold_path = destination / "calibrated-thresholds.yaml"
        threshold_payload = {
            "global_threshold": result.final_threshold,
            "class_thresholds": {
                class_id: result.final_threshold for class_id in sorted(taxonomy.valid_ids)
            },
        }
        _write_yaml(threshold_path, threshold_payload)
        paths["thresholds"] = threshold_path

        if (base_config is None) != (calibrated_config is None):
            raise ValueError("base_config and calibrated_config must be provided together")
        if base_config is not None and calibrated_config is not None:
            base_path = Path(base_config)
            payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("base config YAML root must be an object")
            payload["class_thresholds"] = threshold_payload["class_thresholds"]
            config_path = Path(calibrated_config)
            _write_yaml(config_path, payload)
            paths["config"] = config_path
    elif base_config is not None or calibrated_config is not None:
        raise ValueError("cannot write a calibrated config without a selected threshold")
    return paths
