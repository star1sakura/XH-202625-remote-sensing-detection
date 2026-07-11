from __future__ import annotations

import torch
from torch import nn


def test_dense_vehicle_mask_uses_outline_and_center_distance() -> None:
    from xh_detect.models.density_assigner import compute_dense_vehicle_mask

    labels = torch.tensor([[[24.0], [24.0], [24.0]]])
    boxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [11.0, 0.0, 21.0, 10.0], [500.0, 0.0, 510.0, 10.0]]]
    )
    valid = torch.ones((1, 3, 1), dtype=torch.bool)

    dense = compute_dense_vehicle_mask(labels, boxes, valid, constant=12.0, threshold=0.25)

    assert dense.tolist() == [[True, True, False]]


def test_dense_mask_never_changes_non_vehicle_classes() -> None:
    from xh_detect.models.density_assigner import compute_dense_vehicle_mask

    labels = torch.tensor([[[3.0], [3.0], [24.0]]])
    boxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [1.0, 0.0, 11.0, 10.0], [100.0, 0.0, 110.0, 10.0]]]
    )
    valid = torch.ones((1, 3, 1), dtype=torch.bool)

    dense = compute_dense_vehicle_mask(labels, boxes, valid, constant=12.0, threshold=0.25)

    assert dense.tolist() == [[False, False, False]]


def test_density_candidate_selection_uses_top_one_only_for_dense_gt() -> None:
    from xh_detect.models.density_assigner import select_density_candidates

    metrics = torch.tensor(
        [
            [
                [0.9, 0.8, 0.7, 0.1, 0.0],
                [0.6, 0.5, 0.4, 0.3, 0.2],
            ]
        ]
    )
    dense = torch.tensor([[True, False]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    selected = select_density_candidates(metrics, dense, valid, sparse_topk=3)

    assert selected[0, 0].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert selected[0, 1].tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_density_assigner_preserves_ultralytics_output_shapes() -> None:
    from xh_detect.models.density_assigner import DensityAwareTaskAlignedAssigner

    assigner = DensityAwareTaskAlignedAssigner(
        topk=3,
        num_classes=25,
        alpha=0.5,
        beta=6.0,
        stride=[8, 16, 32],
        density_constant=12.0,
        density_threshold=0.25,
    )
    scores = torch.full((1, 4, 25), 0.5)
    predicted_boxes = torch.tensor(
        [
            [
                [0.0, 0.0, 10.0, 10.0],
                [11.0, 0.0, 21.0, 10.0],
                [0.0, 0.0, 21.0, 10.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ]
    )
    anchors = torch.tensor([[5.0, 5.0], [16.0, 5.0], [10.0, 5.0], [45.0, 45.0]])
    labels = torch.tensor([[[24.0], [24.0]]])
    truth_boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0], [11.0, 0.0, 21.0, 10.0]]])
    valid = torch.ones((1, 2, 1), dtype=torch.bool)

    target_labels, target_boxes, target_scores, foreground, target_indices = assigner(
        scores,
        predicted_boxes,
        anchors,
        labels,
        truth_boxes,
        valid,
    )

    assert target_labels.shape == (1, 4)
    assert target_boxes.shape == (1, 4, 4)
    assert target_scores.shape == (1, 4, 25)
    assert foreground.shape == (1, 4)
    assert target_indices.shape == (1, 4)


def test_density_detection_loss_installs_density_assigner() -> None:
    from types import SimpleNamespace

    from xh_detect.models.density_assigner import (
        DensityAssignerConfig,
        DensityAwareDetectionLoss,
        DensityAwareTaskAlignedAssigner,
    )

    class DummyHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stride = torch.tensor([8.0, 16.0, 32.0])
            self.nc = 25
            self.reg_max = 16

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.parameter = nn.Parameter(torch.zeros(1))
            self.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
            self.model = nn.ModuleList([DummyHead()])

    loss = DensityAwareDetectionLoss(
        DummyModel(),
        config=DensityAssignerConfig(constant=12.0, threshold=0.25),
    )

    assert isinstance(loss.assigner, DensityAwareTaskAlignedAssigner)
    assert loss.assigner.vehicle_class_id == 24
