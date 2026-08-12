from __future__ import annotations

import numpy as np
import pytest

from xh_detect.mmdet_predictions import instances_to_coco_predictions, positive_xyxy_mask


def test_positive_xyxy_mask_rejects_degenerate_boxes() -> None:
    mask = positive_xyxy_mask(
        np.array(
            [
                [0.0, 0.0, 2.0, 3.0],
                [1.0, 1.0, 1.0, 2.0],
                [2.0, 2.0, 1.0, 3.0],
                [4.0, 5.0, 6.0, 5.0],
            ]
        )
    )

    assert mask.tolist() == [True, False, False, False]


def test_instances_to_coco_predictions_converts_xyxy_and_filters_scores() -> None:
    rows = instances_to_coco_predictions(
        image_id=17,
        bboxes=np.array([[10.0, 20.0, 30.0, 50.0], [1.0, 2.0, 4.0, 6.0]]),
        scores=np.array([0.9, 0.2]),
        labels=np.array([24, 3]),
        confidence=0.5,
        valid_class_ids=range(25),
    )

    assert rows == [
        {
            "image_id": 17,
            "category_id": 24,
            "bbox": [10.0, 20.0, 20.0, 30.0],
            "score": pytest.approx(0.9),
        }
    ]


def test_instances_to_coco_predictions_rejects_invalid_shape() -> None:
    with pytest.raises(ValueError, match="bboxes must have shape"):
        instances_to_coco_predictions(
            image_id=1,
            bboxes=np.zeros((1, 5)),
            scores=np.zeros(1),
            labels=np.zeros(1),
            confidence=0.0,
            valid_class_ids={0},
        )


def test_instances_to_coco_predictions_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="valid_class_ids"):
        instances_to_coco_predictions(
            image_id="image",
            bboxes=np.array([[0.0, 0.0, 1.0, 1.0]]),
            scores=np.array([0.5]),
            labels=np.array([25]),
            confidence=0.0,
            valid_class_ids=range(25),
        )


def test_instances_to_coco_predictions_validates_filtered_rows_too() -> None:
    with pytest.raises(ValueError, match="positive width"):
        instances_to_coco_predictions(
            image_id=1,
            bboxes=np.array([[2.0, 0.0, 1.0, 1.0]]),
            scores=np.array([0.1]),
            labels=np.array([0]),
            confidence=0.5,
            valid_class_ids={0},
        )
