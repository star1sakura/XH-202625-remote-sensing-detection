from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ultralytics.engine.validator import BaseValidator

from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, ObjectAnnotation
from xh_detect.ultralytics_evaluation import (
    _match_predictions,
    evaluate_ultralytics,
    ultralytics_evaluation_to_dict,
)


def _box(x1: float, y1: float, x2: float, y2: float):
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def test_matching_is_identical_to_pinned_ultralytics() -> None:
    pred_classes = torch.tensor([4, 4, 24])
    true_classes = torch.tensor([4, 24])
    iou = torch.tensor([[0.9, 0.0, 0.0], [0.8, 0.0, 0.0]])
    thresholds = torch.linspace(0.5, 0.95, 10)

    class Matcher:
        iouv = thresholds

    expected = BaseValidator.match_predictions(  # type: ignore[arg-type]
        Matcher(), pred_classes, true_classes, iou
    ).numpy()

    assert np.array_equal(
        _match_predictions(pred_classes, true_classes, iou, thresholds), expected
    )


def test_perfect_predictions_produce_perfect_ultralytics_metrics() -> None:
    taxonomy = get_taxonomy("xh25")
    truth = [
        ObjectAnnotation("a", 4, _box(0, 0, 10, 10)),
        ObjectAnnotation("s", 0, _box(20, 20, 30, 30)),
        ObjectAnnotation("v", 24, _box(40, 40, 50, 50)),
    ]
    predictions = [
        Detection(item.image_id, item.class_id, 0.9, item.polygon) for item in truth
    ]

    result = evaluate_ultralytics(predictions, truth, taxonomy=taxonomy)

    assert result.images == 3
    assert result.predictions == 3
    assert result.targets == 3
    assert result.source_targets == 3
    assert result.duplicate_targets_removed == 0
    assert result.metrics == pytest.approx(
        {
            "metrics/precision(B)": 1.0,
            "metrics/recall(B)": 1.0,
            "metrics/mAP50(B)": 0.995,
            "metrics/mAP50-95(B)": 0.995,
        }
    )
    payload = ultralytics_evaluation_to_dict(result)
    assert payload["ultralytics_version"] == "8.4.71"
    json.dumps(payload, allow_nan=False)


def test_wrong_class_and_max_detections_affect_metrics() -> None:
    taxonomy = get_taxonomy("xh25")
    truth = [ObjectAnnotation("image", 4, _box(0, 0, 10, 10))]
    predictions = [
        Detection("image", 5, 0.95, _box(0, 0, 10, 10)),
        Detection("image", 4, 0.90, _box(0, 0, 10, 10)),
    ]

    truncated = evaluate_ultralytics(
        predictions, truth, taxonomy=taxonomy, max_detections=1
    )
    complete = evaluate_ultralytics(predictions, truth, taxonomy=taxonomy)

    assert truncated.metrics["metrics/recall(B)"] == 0.0
    assert complete.metrics["metrics/recall(B)"] == pytest.approx(1.0)


def test_exact_duplicate_truth_is_removed_like_ultralytics_dataset_cache() -> None:
    taxonomy = get_taxonomy("xh25")
    annotation = ObjectAnnotation("image", 24, _box(0, 0, 10, 10))
    prediction = Detection("image", 24, 0.9, annotation.polygon)

    deduplicated = evaluate_ultralytics(
        [prediction], [annotation, annotation], taxonomy=taxonomy
    )
    retained = evaluate_ultralytics(
        [prediction],
        [annotation, annotation],
        taxonomy=taxonomy,
        deduplicate_ground_truth=False,
    )

    assert deduplicated.targets == 1
    assert deduplicated.source_targets == 2
    assert deduplicated.duplicate_targets_removed == 1
    assert deduplicated.metrics["metrics/recall(B)"] == pytest.approx(1.0)
    assert retained.targets == 2
    assert retained.duplicate_targets_removed == 0
    assert retained.metrics["metrics/recall(B)"] < 1.0


@pytest.mark.parametrize("value", [0, -1])
def test_max_detections_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        evaluate_ultralytics([], [], taxonomy=get_taxonomy("xh25"), max_detections=value)
