from __future__ import annotations

from xh_detect.evaluator import EvaluationReport, Metrics
from xh_detect.ranking_thresholds import optimize_ranking_thresholds
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_optimizer_selects_six_strict_accuracy_improvements() -> None:
    taxonomy = get_taxonomy("xh25")
    classes = {"aircraft": 4, "ship": 3, "vehicle": 24}
    truth = []
    predictions = []
    for offset, (_group, class_id) in enumerate(classes.items()):
        image_id = str(offset)
        truth.extend(
            [
                ObjectAnnotation(image_id, class_id, _box(0, 0, 10, 10)),
                ObjectAnnotation(image_id, class_id, _box(20, 0, 30, 10)),
            ]
        )
        predictions.extend(
            [
                Detection(image_id, class_id, 0.90, _box(0, 0, 10, 10)),
                Detection(image_id, class_id, 0.80, _box(20, 0, 30, 10)),
                Detection(image_id, class_id, 0.40, _box(40, 0, 50, 10)),
            ]
        )
    baseline = EvaluationReport(
        overall_class_agnostic=Metrics(3, 3, 3),
        by_coarse_class={group: Metrics(1, 1, 1) for group in classes},
        by_fine_class={},
        by_image={},
    )

    result = optimize_ranking_thresholds(
        predictions,
        truth,
        baseline=baseline,
        taxonomy=taxonomy,
        thresholds=[0.25, 0.50, 0.75],
        passes=1,
    )

    assert result.improved_accuracy_items == 6
    assert result.thresholds[4] == 0.50
    assert result.thresholds[3] == 0.50
    assert result.thresholds[24] == 0.50
    assert all(item["recall_improved"] for item in result.group_scores.values())
    assert all(item["fdr_improved"] for item in result.group_scores.values())
