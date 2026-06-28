from __future__ import annotations

import json
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
    optimize_thresholds,
    parse_threshold_grid,
    validate_threshold_map,
)
from xh_detect.types import Detection, ObjectAnnotation

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


def test_parse_threshold_grid_accepts_sequence_values() -> None:
    assert parse_threshold_grid([0.30, 0.10, 0.30, 0.20]) == [0.1, 0.2, 0.3]


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


def test_optimize_thresholds_selects_different_thresholds_per_class() -> None:
    taxonomy = get_taxonomy("legacy3")
    truth = [
        ObjectAnnotation("ship-ok", 0, BOX),
        ObjectAnnotation("aircraft-low", 1, BOX),
    ]
    predictions = [
        Detection("ship-ok", 0, 0.90, BOX),
        Detection("ship-fp", 0, 0.20, BOX),
        Detection("aircraft-low", 1, 0.20, BOX),
    ]

    result = optimize_thresholds(
        predictions,
        truth,
        taxonomy=taxonomy,
        thresholds=(0.20, 0.50),
        passes=2,
    )

    assert result.global_threshold == 0.20
    assert result.thresholds[0] == 0.50
    assert result.thresholds[1] == 0.20
    assert result.objective.f1 == pytest.approx(1.0)
    assert result.report.overall_class_agnostic == Metrics(tp=2, fp=0, fn=0)
    assert result.candidates[0]["stage"] == "global"


def test_optimize_threshold_candidate_records_are_json_serializable() -> None:
    result = optimize_thresholds(
        [],
        [],
        taxonomy=get_taxonomy("legacy3"),
        thresholds=(0.20, 0.50),
    )

    json.dumps(result.candidates, allow_nan=False)
    assert isinstance(result.candidates[0]["objective"], dict)
    assert set(result.candidates[0]["objective"]) == {
        "f1",
        "precision",
        "recall",
        "fdr",
        "tp",
        "fp",
        "fn",
    }


def test_optimize_thresholds_keeps_first_sorted_threshold_on_exact_ties() -> None:
    result = optimize_thresholds(
        [],
        [],
        taxonomy=get_taxonomy("legacy3"),
        thresholds=(0.50, 0.20),
    )

    assert result.grid == [0.20, 0.50]
    assert result.global_threshold == 0.20


def test_optimize_thresholds_recall_floor_rejects_high_f1_low_recall_candidate() -> None:
    taxonomy = get_taxonomy("legacy3")
    truth = [
        ObjectAnnotation("keep-high", 0, BOX),
        ObjectAnnotation("keep-low", 0, BOX),
    ]
    predictions = [
        Detection("keep-high", 0, 0.90, BOX),
        Detection("keep-low", 0, 0.20, BOX),
        Detection("fp-1", 0, 0.20, BOX),
        Detection("fp-2", 0, 0.20, BOX),
        Detection("fp-3", 0, 0.20, BOX),
        Detection("fp-4", 0, 0.20, BOX),
    ]
    baseline = ObjectiveScore(
        f1=0.50,
        precision=0.33,
        recall=1.0,
        fdr=0.67,
        tp=2,
        fp=4,
        fn=0,
    )

    result = optimize_thresholds(
        predictions,
        truth,
        taxonomy=taxonomy,
        thresholds=(0.20, 0.50),
        baseline_objective=baseline,
        recall_floor_delta=0.0,
        passes=2,
    )

    assert result.recall_floor == 1.0
    assert result.thresholds[0] == 0.20
    assert result.report.overall_class_agnostic.recall == 1.0


@pytest.mark.parametrize("recall_floor_delta", [True, "0.1", object()])
def test_optimize_thresholds_rejects_non_real_recall_floor_delta(
    recall_floor_delta: object,
) -> None:
    with pytest.raises(TypeError, match="recall_floor_delta"):
        optimize_thresholds(
            [],
            [],
            taxonomy=get_taxonomy("legacy3"),
            recall_floor_delta=recall_floor_delta,  # type: ignore[arg-type]
        )


def test_optimize_thresholds_rejects_negative_recall_floor_delta() -> None:
    with pytest.raises(ValueError, match="recall_floor_delta"):
        optimize_thresholds(
            [],
            [],
            taxonomy=get_taxonomy("legacy3"),
            recall_floor_delta=-0.1,
        )


@pytest.mark.parametrize("tie_epsilon", [True, "0.1", object()])
def test_optimize_thresholds_rejects_non_real_tie_epsilon(tie_epsilon: object) -> None:
    with pytest.raises(TypeError, match="tie_epsilon"):
        optimize_thresholds(
            [],
            [],
            taxonomy=get_taxonomy("legacy3"),
            tie_epsilon=tie_epsilon,  # type: ignore[arg-type]
        )


def test_optimize_thresholds_rejects_negative_tie_epsilon() -> None:
    with pytest.raises(ValueError, match="tie_epsilon"):
        optimize_thresholds(
            [],
            [],
            taxonomy=get_taxonomy("legacy3"),
            tie_epsilon=-0.1,
        )


@pytest.mark.parametrize("passes", [True, 1.5, "2", object()])
def test_optimize_thresholds_rejects_non_integer_passes(passes: object) -> None:
    with pytest.raises(TypeError, match="passes"):
        optimize_thresholds(
            [],
            [],
            taxonomy=get_taxonomy("legacy3"),
            passes=passes,  # type: ignore[arg-type]
        )


def test_optimize_thresholds_rejects_non_positive_passes() -> None:
    with pytest.raises(ValueError, match="passes"):
        optimize_thresholds([], [], taxonomy=get_taxonomy("legacy3"), passes=0)
