from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from ultralytics.utils.loss import BboxLoss, E2ELoss


def _official_gcd_reference(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    center1 = (boxes1[..., :2] + boxes1[..., 2:]) / 2
    center2 = (boxes2[..., :2] + boxes2[..., 2:]) / 2
    delta = center1 - center2
    width1 = boxes1[..., 2] - boxes1[..., 0]
    height1 = boxes1[..., 3] - boxes1[..., 1]
    width2 = boxes2[..., 2] - boxes2[..., 0]
    height2 = boxes2[..., 3] - boxes2[..., 1]

    center1_distance = (delta[..., 0] / (width1 + eps)).square() + (
        delta[..., 1] / (height1 + eps)
    ).square()
    size1_distance = (
        ((width1 - width2) / (width2 + eps)).square()
        + ((height1 - height2) / (height2 + eps)).square()
    ) / 4
    center2_distance = (delta[..., 0] / (width2 + eps)).square() + (
        delta[..., 1] / (height2 + eps)
    ).square()
    size2_distance = (
        ((width1 - width2) / (width1 + eps)).square()
        + ((height1 - height2) / (height1 + eps)).square()
    ) / 4
    distance_squared = (center1_distance + size1_distance + center2_distance + size2_distance) / 2
    return torch.exp(-torch.sqrt(distance_squared))


def test_gcd_similarity_matches_official_reference() -> None:
    from xh_detect.models.gcd import gcd_similarity

    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 20.0], [4.0, 2.0, 16.0, 11.0]])
    boxes2 = torch.tensor([[1.0, 3.0, 12.0, 19.0], [5.0, 1.0, 19.0, 12.0]])

    actual = gcd_similarity(boxes1, boxes2)
    expected = _official_gcd_reference(boxes1, boxes2)

    assert torch.allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_gcd_is_symmetric_and_scale_invariant() -> None:
    from xh_detect.models.gcd import gcd_similarity

    first = torch.tensor([[0.0, 0.0, 8.0, 12.0]])
    second = torch.tensor([[1.0, 2.0, 10.0, 15.0]])

    forward = gcd_similarity(first, second)
    reverse = gcd_similarity(second, first)
    scaled = gcd_similarity(first * 10, second * 10)

    assert torch.allclose(forward, reverse)
    assert torch.allclose(forward, scaled, atol=1e-6, rtol=1e-6)


def test_gcd_identity_and_distance_ordering() -> None:
    from xh_detect.models.gcd import gcd_similarity

    reference = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    near = torch.tensor([[2.0, 0.0, 12.0, 10.0]])
    far = torch.tensor([[100.0, 0.0, 110.0, 10.0]])

    identical = gcd_similarity(reference, reference)
    near_similarity = gcd_similarity(reference, near)
    far_similarity = gcd_similarity(reference, far)

    assert identical.item() == 1.0
    assert 0.0 < far_similarity.item() < near_similarity.item() < 1.0


def test_gcd_has_finite_gradients_for_tiny_and_degenerate_boxes() -> None:
    from xh_detect.models.gcd import gcd_similarity

    predicted = torch.tensor(
        [[0.0, 0.0, 1e-4, 1e-4], [0.0, 0.0, 0.0, 0.0]],
        requires_grad=True,
    )
    target = predicted.detach().clone()

    loss = (1.0 - gcd_similarity(predicted, target)).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_gcd_assigner_supports_ultralytics_broadcast_shapes() -> None:
    from xh_detect.models.gcd import GCDTaskAlignedAssigner

    assigner = GCDTaskAlignedAssigner(
        topk=3,
        num_classes=25,
        alpha=0.5,
        beta=6.0,
        stride=[8, 16, 32],
    )
    truth = torch.tensor([[[[0.0, 0.0, 10.0, 10.0]], [[20.0, 20.0, 30.0, 30.0]]]])
    predicted = torch.tensor(
        [[[[0.0, 0.0, 10.0, 10.0], [2.0, 0.0, 12.0, 10.0], [40.0, 40.0, 50.0, 50.0]]]]
    )

    overlaps = assigner.iou_calculation(truth, predicted)

    assert overlaps.shape == (1, 2, 3)
    assert overlaps[0, 0, 0].item() == 1.0
    assert torch.all((overlaps >= 0.0) & (overlaps <= 1.0))


def test_gcd_bbox_loss_preserves_dfl() -> None:
    from xh_detect.models.gcd import GCDBboxLoss

    torch.manual_seed(42)
    pred_dist = torch.randn(1, 3, 64)
    pred_bboxes = torch.tensor(
        [[[0.0, 0.0, 10.0, 10.0], [9.0, 9.0, 20.0, 20.0], [18.0, 18.0, 30.0, 30.0]]]
    )
    anchor_points = torch.tensor([[5.0, 5.0], [15.0, 15.0], [25.0, 25.0]])
    target_bboxes = torch.tensor(
        [[[1.0, 0.0, 11.0, 10.0], [10.0, 10.0, 20.0, 20.0], [20.0, 20.0, 31.0, 31.0]]]
    )
    target_scores = torch.zeros(1, 3, 25)
    target_scores[0, :, 0] = torch.tensor([1.0, 0.8, 0.6])
    target_scores_sum = target_scores.sum()
    foreground = torch.ones(1, 3, dtype=torch.bool)
    imgsz = torch.tensor([1024.0, 1024.0])
    stride = torch.tensor([8.0, 8.0, 8.0])
    arguments = (
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        foreground,
        imgsz,
        stride,
    )

    baseline_box, baseline_dfl = BboxLoss(16)(*arguments)
    gcd_box, gcd_dfl = GCDBboxLoss(16)(*arguments)

    assert torch.isfinite(gcd_box)
    assert not torch.allclose(gcd_box, baseline_box)
    assert torch.equal(gcd_dfl, baseline_dfl)


def test_gcd_detection_loss_installs_selected_components() -> None:
    from xh_detect.models.gcd import (
        GCDBboxLoss,
        GCDDetectionLoss,
        GCDTaskAlignedAssigner,
        GCDTrainingConfig,
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

    loss = GCDDetectionLoss(
        DummyModel(),
        config=GCDTrainingConfig(use_loss=True, use_assignment=True),
    )

    assert isinstance(loss.bbox_loss, GCDBboxLoss)
    assert isinstance(loss.assigner, GCDTaskAlignedAssigner)


def test_gcd_detection_model_preserves_yolo26_end_to_end_loss() -> None:
    from xh_detect.models.gcd import (
        GCDBboxLoss,
        GCDDetectionModel,
        GCDTaskAlignedAssigner,
        GCDTrainingConfig,
    )

    model = GCDDetectionModel(
        "yolo26n.yaml",
        nc=25,
        ch=3,
        verbose=False,
        gcd_config=GCDTrainingConfig(use_loss=True, use_assignment=True),
    )
    model.args = SimpleNamespace()

    loss = model.init_criterion()

    assert isinstance(loss, E2ELoss)
    assert isinstance(loss.one2many.bbox_loss, GCDBboxLoss)
    assert isinstance(loss.one2one.bbox_loss, GCDBboxLoss)
    assert isinstance(loss.one2many.assigner, GCDTaskAlignedAssigner)
    assert isinstance(loss.one2one.assigner, GCDTaskAlignedAssigner)
    assert loss.one2one.assigner.topk2 == 1
