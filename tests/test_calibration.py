from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from xh_detect.calibration import (
    ThresholdCandidate,
    build_group_folds,
    calibrate_thresholds,
    load_image_group_mapping,
    select_threshold_candidate,
    write_calibration_artifacts,
)
from xh_detect.evaluator import evaluate
from xh_detect.taxonomy import get_taxonomy
from xh_detect.thresholds import ObjectiveScore
from xh_detect.types import Detection, ObjectAnnotation

TAXONOMY = get_taxonomy("xh25")


def _polygon(x: float = 0.0, y: float = 0.0) -> tuple[tuple[float, float], ...]:
    return ((x, y), (x + 10.0, y), (x + 10.0, y + 10.0), (x, y + 10.0))


def _truth(image_id: str, class_id: int = 0) -> ObjectAnnotation:
    return ObjectAnnotation(image_id=image_id, class_id=class_id, polygon=_polygon())


def _prediction(
    image_id: str,
    score: float,
    *,
    class_id: int = 0,
    false_positive: bool = False,
) -> Detection:
    return Detection(
        image_id=image_id,
        class_id=class_id,
        score=score,
        polygon=_polygon(100.0 if false_positive else 0.0),
    )


def _synthetic_inputs() -> tuple[
    list[Detection], list[Detection], list[ObjectAnnotation], dict[str, str]
]:
    image_ids = [str(index) for index in range(1, 7)]
    truth = [_truth(image_id) for image_id in image_ids]
    baseline = [_prediction(image_id, 0.9) for image_id in image_ids]
    candidate = [
        detection
        for image_id in image_ids
        for detection in (
            _prediction(image_id, 0.9),
            _prediction(image_id, 0.4, false_positive=True),
        )
    ]
    image_to_group = {image_id: f"source-{image_id}" for image_id in image_ids}
    return baseline, candidate, truth, image_to_group


def test_build_group_folds_is_deterministic_and_has_no_group_leakage() -> None:
    _, _, truth, image_to_group = _synthetic_inputs()
    image_to_group["2"] = image_to_group["1"]

    first = build_group_folds(image_to_group, truth, TAXONOMY, folds=3, seed=42)
    second = build_group_folds(image_to_group, truth, TAXONOMY, folds=3, seed=42)

    assert first == second
    assert set(first) == set(image_to_group.values())
    image_to_fold = {image_id: first[group] for image_id, group in image_to_group.items()}
    assert image_to_fold["1"] == image_to_fold["2"]
    assert set(first.values()) == {0, 1, 2}


def test_build_group_folds_allows_rare_class_to_be_absent_from_a_fold() -> None:
    _, _, truth, image_to_group = _synthetic_inputs()
    truth[0] = _truth("1", class_id=1)

    assignment = build_group_folds(image_to_group, truth, TAXONOMY, folds=5, seed=42)

    assert len(assignment) == 6
    assert set(assignment.values()) == {0, 1, 2, 3, 4}


def test_select_threshold_candidate_applies_gate_and_tie_break() -> None:
    report = evaluate([], [], taxonomy=TAXONOMY)
    candidates = [
        ThresholdCandidate(
            threshold=0.4,
            report=report,
            objective=ObjectiveScore(0.9000, 0.95, 0.86, 0.05, 1, 0, 0),
        ),
        ThresholdCandidate(
            threshold=0.425,
            report=report,
            objective=ObjectiveScore(0.90005, 0.94, 0.87, 0.06, 1, 0, 0),
        ),
        ThresholdCandidate(
            threshold=0.45,
            report=report,
            objective=ObjectiveScore(0.99, 0.99, 0.70, 0.01, 1, 0, 0),
        ),
    ]

    selected, eligible = select_threshold_candidate(
        candidates, recall_floor=0.8, fdr_cap=0.1, tie_epsilon=0.0001
    )

    assert eligible == (0.4, 0.425)
    assert selected is candidates[0]


def test_calibration_uses_fold_specific_thresholds_and_counts_oof_once() -> None:
    baseline, candidate, truth, image_to_group = _synthetic_inputs()

    result = calibrate_thresholds(
        baseline,
        candidate,
        truth,
        image_to_group,
        TAXONOMY,
        folds=3,
        seed=42,
        thresholds=[0.25, 0.45, 0.50],
    )

    assert result.passed
    assert [fold.selected_threshold for fold in result.fold_results] == [0.45, 0.45, 0.45]
    assert result.final_threshold == 0.45
    assert result.threshold_range == 0.0
    assert result.candidate_oof_report is not None
    assert result.candidate_oof_report.overall.tp == len(truth)
    assert result.candidate_oof_report.overall.fp == 0
    assert set(result.candidate_oof_report.by_image) == set(image_to_group)
    assert len(result.oof_predictions) == len(truth)


def test_calibration_reports_failure_when_a_fold_has_no_eligible_threshold() -> None:
    baseline, _, truth, image_to_group = _synthetic_inputs()
    candidate = [
        detection
        for image_id in image_to_group
        for detection in (
            _prediction(image_id, 0.3),
            _prediction(image_id, 0.9, false_positive=True),
        )
    ]

    result = calibrate_thresholds(
        baseline,
        candidate,
        truth,
        image_to_group,
        TAXONOMY,
        folds=3,
        thresholds=[0.25, 0.45],
    )

    assert not result.passed
    assert result.final_threshold is None
    assert result.candidate_oof_report is None
    assert "no threshold" in (result.failure_reason or "")


def test_load_image_group_mapping_uses_coco_file_stems(tmp_path: Path) -> None:
    ground_truth_path = tmp_path / "truth.json"
    source_groups_path = tmp_path / "source-groups.json"
    ground_truth_path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "images/val/source-a_crop1.jpg"},
                    {"id": 2, "file_name": "images/val/source-a_crop2.png"},
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    source_groups_path.write_text(
        json.dumps(
            {
                "source-a_crop1": {"group": "source-a", "split": "val"},
                "source-a_crop2": {"group": "source-a", "split": "val"},
            }
        ),
        encoding="utf-8",
    )

    mapping = load_image_group_mapping(ground_truth_path, source_groups_path)

    assert mapping.image_to_group == {"1": "source-a", "2": "source-a"}
    assert mapping.image_to_stem == {"1": "source-a_crop1", "2": "source-a_crop2"}


def test_load_image_group_mapping_rejects_missing_or_non_val_group(tmp_path: Path) -> None:
    truth_path = tmp_path / "truth.json"
    groups_path = tmp_path / "groups.json"
    truth_path.write_text(
        json.dumps(
            {"images": [{"id": 1, "file_name": "images/val/missing.jpg"}], "annotations": []}
        ),
        encoding="utf-8",
    )
    groups_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_image_group_mapping(truth_path, groups_path)

    groups_path.write_text(
        json.dumps({"missing": {"group": "source", "split": "train"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="split"):
        load_image_group_mapping(truth_path, groups_path)


def test_write_artifacts_outputs_uniform_thresholds_and_only_changes_config_thresholds(
    tmp_path: Path,
) -> None:
    baseline, candidate, truth, image_to_group = _synthetic_inputs()
    result = calibrate_thresholds(
        baseline,
        candidate,
        truth,
        image_to_group,
        TAXONOMY,
        folds=3,
        thresholds=[0.25, 0.45],
    )
    base_config = tmp_path / "base.yaml"
    calibrated_config = tmp_path / "calibrated.yaml"
    base_payload = {
        "model": "best.pt",
        "tile_size": 1024,
        "class_thresholds": {class_id: 0.25 for class_id in range(25)},
        "merge": {"method": "nms", "iou": 0.5},
    }
    base_config.write_text(yaml.safe_dump(base_payload), encoding="utf-8")

    paths = write_calibration_artifacts(
        result,
        tmp_path / "artifacts",
        TAXONOMY,
        base_config=base_config,
        calibrated_config=calibrated_config,
    )

    thresholds = yaml.safe_load(paths["thresholds"].read_text(encoding="utf-8"))
    assert thresholds["class_thresholds"] == {class_id: 0.45 for class_id in range(25)}
    calibrated = yaml.safe_load(calibrated_config.read_text(encoding="utf-8"))
    assert calibrated["class_thresholds"] == thresholds["class_thresholds"]
    assert {key: value for key, value in calibrated.items() if key != "class_thresholds"} == {
        key: value for key, value in base_payload.items() if key != "class_thresholds"
    }
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    oof = json.loads(paths["oof_predictions"].read_text(encoding="utf-8"))
    assert len(oof) == len(truth)
