from __future__ import annotations

import math

import pytest

from xh_detect.evaluator import EvaluationReport, Metrics
from xh_detect.taxonomy import get_taxonomy
from xh_detect.thresholds import (
    DEFAULT_THRESHOLD_GRID,
    DEFAULT_THRESHOLD_GRID_TEXT,
    ObjectiveScore,
    f1_score,
    filter_predictions_by_class_threshold,
    is_better_objective,
    objective_from_report,
    parse_threshold_grid,
    validate_threshold_map,
)
from xh_detect.types import Detection

BOX = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def test_default_threshold_grid_matches_specification() -> None:
    assert DEFAULT_THRESHOLD_GRID == (
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    )


def _report(tp: int, fp: int, fn: int) -> EvaluationReport:
    metrics = Metrics(tp, fp, fn)
    return EvaluationReport(
        overall_class_agnostic=metrics,
        by_coarse_class={"ship": metrics},
        by_fine_class={0: metrics},
        by_image={},
    )


def test_default_threshold_grid_text_matches_default_grid() -> None:
    assert parse_threshold_grid(DEFAULT_THRESHOLD_GRID_TEXT) == list(DEFAULT_THRESHOLD_GRID)


def test_parse_threshold_grid_returns_sorted_unique_values() -> None:
    assert parse_threshold_grid("0.30, 0.10,0.30,0.20") == [0.1, 0.2, 0.3]


@pytest.mark.parametrize(
    "grid",
    ["", "0.2,bad", "-0.1,0.2", "0.2,1.1", "nan", "inf", "0.1,,0.2"],
)
def test_parse_threshold_grid_rejects_invalid_values(grid: str) -> None:
    with pytest.raises(ValueError):
        parse_threshold_grid(grid)


@pytest.mark.parametrize(
    ("threshold", "error_type"),
    [
        (True, TypeError),
        (False, TypeError),
        ("0.2", TypeError),
        (object(), TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
        (-0.1, ValueError),
        (1.1, ValueError),
    ],
)
def test_parse_threshold_grid_rejects_invalid_sequence_values(
    threshold: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        parse_threshold_grid([threshold])


def test_parse_threshold_grid_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        parse_threshold_grid([])


def test_validate_threshold_map_accepts_integer_and_decimal_string_class_ids() -> None:
    taxonomy = get_taxonomy("legacy3")

    assert validate_threshold_map({0: 0.25, "1": 0.3}, taxonomy) == {0: 0.25, 1: 0.3}


def test_validate_threshold_map_rejects_unknown_class_ids() -> None:
    taxonomy = get_taxonomy("legacy3")

    with pytest.raises(ValueError, match="class ID"):
        validate_threshold_map({9: 0.25}, taxonomy)


@pytest.mark.parametrize("class_id", [True, False, 0.0, 1.5, "1.0", "ship"])
def test_validate_threshold_map_rejects_non_integer_class_ids(class_id: object) -> None:
    with pytest.raises(TypeError, match="class ID"):
        validate_threshold_map({class_id: 0.25}, get_taxonomy("legacy3"))


@pytest.mark.parametrize(
    ("threshold", "error_type"),
    [
        (True, TypeError),
        ("0.25", TypeError),
        (None, TypeError),
        (math.nan, ValueError),
        (math.inf, ValueError),
        (-math.inf, ValueError),
        (-0.1, ValueError),
        (1.1, ValueError),
    ],
)
def test_validate_threshold_map_rejects_invalid_threshold_values(
    threshold: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type, match="threshold"):
        validate_threshold_map({0: threshold}, get_taxonomy("legacy3"))


def test_validate_threshold_map_rejects_duplicate_normalized_class_ids() -> None:
    with pytest.raises(ValueError, match="duplicate class ID"):
        validate_threshold_map({0: 0.25, "0": 0.70}, get_taxonomy("legacy3"))


def test_filter_predictions_uses_class_specific_thresholds_inclusively() -> None:
    predictions = [
        Detection("img", 0, 0.50, BOX),
        Detection("img", 0, 0.49, BOX),
        Detection("img", 1, 0.20, BOX),
        Detection("img", 1, 0.19, BOX),
    ]

    filtered = filter_predictions_by_class_threshold(
        predictions,
        {0: 0.50, 1: 0.20},
        taxonomy=get_taxonomy("legacy3"),
    )

    assert filtered == [predictions[0], predictions[2]]


def test_filter_predictions_uses_zero_threshold_for_missing_classes() -> None:
    predictions = [
        Detection("img", 0, 0.49, BOX),
        Detection("img", 1, 0.00, BOX),
    ]

    filtered = filter_predictions_by_class_threshold(
        predictions,
        {0: 0.50},
        taxonomy=get_taxonomy("legacy3"),
    )

    assert filtered == [predictions[1]]


def test_objective_from_report_computes_precision_recall_fdr_and_f1() -> None:
    objective = objective_from_report(_report(tp=8, fp=2, fn=2))

    assert objective == ObjectiveScore(
        f1=pytest.approx(0.8),
        precision=0.8,
        recall=0.8,
        fdr=0.2,
        tp=8,
        fp=2,
        fn=2,
    )
    assert f1_score(recall=0.0, fdr=1.0) == 0.0


def test_is_better_objective_prefers_f1_then_lower_fdr_then_higher_recall() -> None:
    incumbent = ObjectiveScore(f1=0.90, precision=0.90, recall=0.90, fdr=0.10, tp=9, fp=1, fn=1)

    assert is_better_objective(
        ObjectiveScore(f1=0.91, precision=0.91, recall=0.90, fdr=0.09, tp=9, fp=1, fn=1),
        incumbent,
    )
    assert is_better_objective(
        ObjectiveScore(f1=0.9001, precision=0.92, recall=0.88, fdr=0.08, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
    assert is_better_objective(
        ObjectiveScore(f1=0.9001, precision=0.90, recall=0.91, fdr=0.1001, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
    assert not is_better_objective(
        ObjectiveScore(f1=0.8990, precision=0.99, recall=0.99, fdr=0.01, tp=9, fp=1, fn=1),
        incumbent,
        tie_epsilon=0.0005,
    )
