from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import BboxLoss, E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@dataclass(frozen=True)
class GCDTrainingConfig:
    use_loss: bool = False
    use_assignment: bool = False
    assignment_weight: float = 1.0
    eps: float = 1e-7
    root_eps: float = 1e-12

    def __post_init__(self) -> None:
        if not isinstance(self.use_loss, bool):
            raise TypeError("use_loss must be a boolean")
        if not isinstance(self.use_assignment, bool):
            raise TypeError("use_assignment must be a boolean")
        if (
            isinstance(self.assignment_weight, bool)
            or not isinstance(self.assignment_weight, (int, float))
            or not math.isfinite(self.assignment_weight)
            or not 0.0 <= self.assignment_weight <= 1.0
        ):
            raise ValueError("assignment_weight must be finite and in [0, 1]")
        if not math.isfinite(self.eps) or self.eps <= 0.0:
            raise ValueError("eps must be positive and finite")
        if not math.isfinite(self.root_eps) or self.root_eps <= 0.0:
            raise ValueError("root_eps must be positive and finite")


def gcd_similarity(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    *,
    eps: float = 1e-7,
    root_eps: float = 1e-12,
) -> torch.Tensor:
    """Return Gaussian Combined Distance similarity for broadcastable xyxy boxes."""
    if boxes1.shape[-1] != 4 or boxes2.shape[-1] != 4:
        raise ValueError("GCD expects xyxy boxes with a final dimension of 4")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")
    if not math.isfinite(root_eps) or root_eps <= 0.0:
        raise ValueError("root_eps must be positive and finite")

    first = boxes1.float()
    second = boxes2.float()
    center1 = (first[..., :2] + first[..., 2:]) / 2
    center2 = (second[..., :2] + second[..., 2:]) / 2
    center_delta = center1 - center2

    width1 = (first[..., 2] - first[..., 0]).clamp_min(0.0)
    height1 = (first[..., 3] - first[..., 1]).clamp_min(0.0)
    width2 = (second[..., 2] - second[..., 0]).clamp_min(0.0)
    height2 = (second[..., 3] - second[..., 1]).clamp_min(0.0)

    center_distance1 = (center_delta[..., 0] / (width1 + eps)).square() + (
        center_delta[..., 1] / (height1 + eps)
    ).square()
    size_distance1 = (
        ((width1 - width2) / (width2 + eps)).square()
        + ((height1 - height2) / (height2 + eps)).square()
    ) / 4

    center_distance2 = (center_delta[..., 0] / (width2 + eps)).square() + (
        center_delta[..., 1] / (height2 + eps)
    ).square()
    size_distance2 = (
        ((width1 - width2) / (width1 + eps)).square()
        + ((height1 - height2) / (height1 + eps)).square()
    ) / 4

    squared_distance = (center_distance1 + size_distance1 + center_distance2 + size_distance2) / 2
    distance = torch.sqrt(squared_distance.clamp_min(0.0) + root_eps) - math.sqrt(root_eps)
    return torch.exp(-distance).clamp(0.0, 1.0)


class GCDTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(
        self,
        *args: object,
        assignment_weight: float = 1.0,
        gcd_eps: float = 1e-7,
        gcd_root_eps: float = 1e-12,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if (
            isinstance(assignment_weight, bool)
            or not isinstance(assignment_weight, (int, float))
            or not math.isfinite(assignment_weight)
            or not 0.0 <= assignment_weight <= 1.0
        ):
            raise ValueError("assignment_weight must be finite and in [0, 1]")
        self.assignment_weight = float(assignment_weight)
        self.gcd_eps = gcd_eps
        self.gcd_root_eps = gcd_root_eps

    def iou_calculation(
        self,
        gt_bboxes: torch.Tensor,
        pd_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        gcd = gcd_similarity(
            gt_bboxes,
            pd_bboxes,
            eps=self.gcd_eps,
            root_eps=self.gcd_root_eps,
        )
        if self.assignment_weight == 1.0:
            return gcd
        ciou = super().iou_calculation(gt_bboxes, pd_bboxes)
        if self.assignment_weight == 0.0:
            return ciou
        return (1.0 - self.assignment_weight) * ciou + self.assignment_weight * gcd


class GCDBboxLoss(BboxLoss):
    def __init__(
        self,
        reg_max: int = 16,
        *,
        gcd_eps: float = 1e-7,
        gcd_root_eps: float = 1e-12,
    ) -> None:
        super().__init__(reg_max)
        self.gcd_eps = gcd_eps
        self.gcd_root_eps = gcd_root_eps

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _, loss_dfl = super().forward(
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride,
        )
        weight = target_scores.sum(-1)[fg_mask]
        similarity = gcd_similarity(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            eps=self.gcd_eps,
            root_eps=self.gcd_root_eps,
        )
        loss_gcd = ((1.0 - similarity) * weight).sum() / target_scores_sum
        return loss_gcd, loss_dfl


class GCDDetectionLoss(v8DetectionLoss):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: GCDTrainingConfig,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        if config.use_assignment:
            self.assigner = GCDTaskAlignedAssigner(
                topk=tal_topk,
                num_classes=self.nc,
                alpha=0.5,
                beta=6.0,
                stride=self.stride.tolist(),
                topk2=tal_topk2,
                assignment_weight=config.assignment_weight,
                gcd_eps=config.eps,
                gcd_root_eps=config.root_eps,
            )
        if config.use_loss:
            self.bbox_loss = GCDBboxLoss(
                self.reg_max,
                gcd_eps=config.eps,
                gcd_root_eps=config.root_eps,
            ).to(self.device)


class GCDDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        gcd_config: GCDTrainingConfig,
    ) -> None:
        self.gcd_config = gcd_config
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self) -> E2ELoss | GCDDetectionLoss:
        if getattr(self, "end2end", False):
            loss_fn = partial(GCDDetectionLoss, config=self.gcd_config)
            return E2ELoss(self, loss_fn=loss_fn)
        return GCDDetectionLoss(self, config=self.gcd_config)


class GCDDetectionTrainer(DetectionTrainer):
    gcd_config = GCDTrainingConfig()

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: object | None = None,
        verbose: bool = True,
    ) -> GCDDetectionModel:
        model = GCDDetectionModel(
            cfg or "yolo26n.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            gcd_config=self.gcd_config,
        )
        if weights:
            model.load(weights)
        return model
