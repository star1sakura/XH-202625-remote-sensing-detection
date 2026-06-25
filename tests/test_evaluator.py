from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from xh_detect.evaluator import (
    EvaluationReport,
    Metrics,
    evaluate,
    load_coco_ground_truth,
    load_coco_predictions,
    report_to_dict,
    threshold_sweep,
)
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation

GT = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def box_with_iou(iou: float) -> tuple[tuple[float, float], ...]:
    width = 10.0 * iou
    return ((0.0, 0.0), (width, 0.0), (width, 10.0), (0.0, 10.0))


def test_xh25_overall_is_fine_class_agnostic_but_coarse_metrics_are_not() -> None:
    truth = [ObjectAnnotation("img", 0, GT)]
    predictions = [Detection("img", 4, 0.9, GT)]
    report = evaluate(predictions, truth, taxonomy=get_taxonomy("xh25"))
    assert report.overall_class_agnostic == Metrics(tp=1, fp=0, fn=0)
    assert report.by_coarse_class["ship"] == Metrics(tp=0, fp=0, fn=1)
    assert report.by_coarse_class["aircraft"] == Metrics(tp=0, fp=1, fn=0)
    assert report.by_fine_class[0] == Metrics(tp=0, fp=0, fn=1)
    assert report.by_fine_class[4] == Metrics(tp=0, fp=1, fn=0)


def test_xh25_same_coarse_different_fine_matches_coarse_only() -> None:
    truth = [ObjectAnnotation("img", 4, GT)]
    predictions = [Detection("img", 8, 0.9, GT)]
    report = evaluate(predictions, truth, taxonomy=get_taxonomy("xh25"))
    assert report.overall_class_agnostic.tp == 1
    assert report.by_coarse_class["aircraft"].tp == 1
    assert report.by_fine_class[4].fn == 1
    assert report.by_fine_class[8].fp == 1


def test_vehicle_truth_uses_point_35_threshold() -> None:
    truth = [ObjectAnnotation("img", 24, GT)]
    prediction = Detection("img", 24, 0.9, box_with_iou(0.35))
    report = evaluate([prediction], truth, taxonomy=get_taxonomy("xh25"))
    assert report.overall_class_agnostic.tp == 1
    assert report.by_coarse_class["vehicle"].tp == 1


def test_duplicate_predictions_produce_one_tp_and_one_fp() -> None:
    report = evaluate(
        [Detection("img", 0, 0.8, GT), Detection("img", 0, 0.9, GT)],
        [ObjectAnnotation("img", 0, GT)],
    )

    assert report.overall_class_agnostic == Metrics(tp=1, fp=1, fn=0)
    assert report.overall_class_agnostic.recall == 1.0
    assert report.overall_class_agnostic.fdr == 0.5


@pytest.mark.parametrize(
    ("class_id", "prediction_width", "expected_tp"),
    [(0, 5.0, 1), (1, 5.0, 1), (2, 3.5, 1), (2, 3.49, 0)],
)
def test_competition_class_iou_thresholds(
    class_id: int,
    prediction_width: float,
    expected_tp: int,
) -> None:
    prediction = (
        (0.0, 0.0),
        (prediction_width, 0.0),
        (prediction_width, 10.0),
        (0.0, 10.0),
    )

    report = evaluate(
        [Detection("img", class_id, 0.9, prediction)],
        [ObjectAnnotation("img", class_id, GT)],
    )

    assert report.overall_class_agnostic.tp == expected_tp
    assert report.overall_class_agnostic.fp == 1 - expected_tp
    assert report.overall_class_agnostic.fn == 1 - expected_tp


def test_greedy_matching_uses_best_unmatched_truth_and_fine_class_is_strict() -> None:
    second = ((20.0, 0.0), (30.0, 0.0), (30.0, 10.0), (20.0, 10.0))
    report = evaluate(
        [
            Detection("img", 0, 0.9, second),
            Detection("img", 1, 0.8, GT),
        ],
        [ObjectAnnotation("img", 0, GT), ObjectAnnotation("img", 0, second)],
    )

    assert report.overall_class_agnostic == Metrics(tp=2, fp=0, fn=0)
    assert report.by_fine_class[0] == Metrics(tp=1, fp=0, fn=1)
    assert report.by_fine_class[1] == Metrics(tp=0, fp=1, fn=0)


def test_difficult_truth_is_excluded_from_denominators() -> None:
    report = evaluate([], [ObjectAnnotation("img", 0, GT, difficult=True)])

    assert report.overall_class_agnostic == Metrics(tp=0, fp=0, fn=0)
    assert report.by_image == {}


def test_empty_inputs_and_truth_only_or_prediction_only_images() -> None:
    empty = evaluate([], [])
    assert empty.overall_class_agnostic == Metrics(0, 0, 0)
    assert set(empty.by_fine_class) == {0, 1, 2}

    truth_only = evaluate([], [ObjectAnnotation("truth", 0, GT)])
    assert truth_only.by_image["truth"] == Metrics(0, 0, 1)

    prediction_only = evaluate([Detection("pred", 1, 0.9, GT)], [])
    assert prediction_only.by_image["pred"] == Metrics(0, 1, 0)


def test_threshold_sweep_preserves_threshold_order_and_filters_inclusively() -> None:
    predictions = [
        Detection("img", 0, 0.9, GT),
        Detection("img", 0, 0.5, GT),
    ]
    truth = [ObjectAnnotation("img", 0, GT)]

    sweep = threshold_sweep(predictions, truth, [0.9, 0.5, 1.0])

    assert [threshold for threshold, _ in sweep] == [0.9, 0.5, 1.0]
    assert [report.overall_class_agnostic for _, report in sweep] == [
        Metrics(1, 0, 0),
        Metrics(1, 1, 0),
        Metrics(0, 0, 1),
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, math.nan, math.inf, True, "0.5"])
def test_threshold_sweep_rejects_invalid_thresholds(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        threshold_sweep([], [], [threshold])  # type: ignore[list-item]


def test_coco_loaders_validate_and_use_numeric_ids_as_strings(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    truth_path = tmp_path / "truth.json"
    predictions_path.write_text(
        json.dumps([{"image_id": 7, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]),
        encoding="utf-8",
    )
    truth_path.write_text(
        json.dumps(
            {
                "images": [{"id": 7, "file_name": "P0001.png"}],
                "annotations": [
                    {
                        "image_id": 7,
                        "category_id": 0,
                        "bbox": [0, 0, 10, 10],
                        "iscrowd": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    predictions = load_coco_predictions(predictions_path)
    truth = load_coco_ground_truth(truth_path)

    assert predictions == [Detection("7", 0, 0.9, GT)]
    assert truth == [ObjectAnnotation("7", 0, GT)]


def test_coco_loaders_accept_xh25_categories_when_taxonomy_is_xh25(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    truth_path = tmp_path / "truth.json"
    predictions_path.write_text(
        json.dumps([{"image_id": "img", "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9}]),
        encoding="utf-8",
    )
    truth_path.write_text(
        json.dumps(
            {"annotations": [{"image_id": "img", "category_id": 24, "bbox": [0, 0, 10, 10]}]}
        ),
        encoding="utf-8",
    )

    assert load_coco_predictions(predictions_path, taxonomy=get_taxonomy("xh25")) == [
        Detection("img", 24, 0.9, GT)
    ]
    assert load_coco_ground_truth(truth_path, taxonomy=get_taxonomy("xh25")) == [
        ObjectAnnotation("img", 24, GT)
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9}],
        [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 0, 10], "score": 0.9}],
        [{"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": math.nan}],
    ],
)
def test_prediction_loader_rejects_invalid_payload(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_coco_predictions(path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"annotations": "bad"},
        {"annotations": [{"image_id": 1, "category_id": 0, "bbox": [0, 0, -1, 10]}]},
        {"annotations": [{"image_id": 1, "category_id": 9, "bbox": [0, 0, 10, 10]}]},
    ],
)
def test_ground_truth_loader_rejects_invalid_payload(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "truth.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError)):
        load_coco_ground_truth(path)


def test_report_to_dict_contains_overall_class_and_image_metrics() -> None:
    report = EvaluationReport(
        overall_class_agnostic=Metrics(1, 2, 3),
        by_coarse_class={"aircraft": Metrics(1, 0, 1), "ship": Metrics(0, 2, 1)},
        by_fine_class={0: Metrics(1, 0, 1), 1: Metrics(0, 2, 1), 2: Metrics(0, 0, 1)},
        by_image={"img": Metrics(1, 2, 3)},
    )

    payload = report_to_dict(report)

    assert payload["overall_class_agnostic"] == {
        "tp": 1,
        "fp": 2,
        "fn": 3,
        "recall": 0.25,
        "fdr": pytest.approx(2 / 3),
    }
    assert payload["by_coarse_class"]["aircraft"]["recall"] == 0.5
    assert payload["by_fine_class"]["0"]["recall"] == 0.5
    assert payload["by_image"]["img"]["fn"] == 3
