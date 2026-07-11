from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial

import torch
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import E2ELoss, v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@dataclass(frozen=True)
class DensityAssignerConfig:
    constant: float = 12.0
    threshold: float = 0.25
    vehicle_class_id: int = 24

    def __post_init__(self) -> None:
        if not math.isfinite(self.constant) or self.constant <= 0.0:
            raise ValueError("density constant must be positive and finite")
        if not math.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("density threshold must be in [0, 1]")
        if isinstance(self.vehicle_class_id, bool) or not isinstance(self.vehicle_class_id, int):
            raise TypeError("vehicle_class_id must be an integer")


def compute_dense_vehicle_mask(
    gt_labels: torch.Tensor,
    gt_bboxes: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    constant: float,
    threshold: float,
    vehicle_class_id: int = 24,
) -> torch.Tensor:
    if not math.isfinite(constant) or constant <= 0.0:
        raise ValueError("density constant must be positive and finite")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("density threshold must be in [0, 1]")

    labels = gt_labels.squeeze(-1).long()
    valid = mask_gt.squeeze(-1).bool()
    widths = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp_min(0.0)
    heights = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp_min(0.0)
    center_x = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
    center_y = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2

    outline_distance_sq = (widths.unsqueeze(2) - widths.unsqueeze(1)).square() + (
        heights.unsqueeze(2) - heights.unsqueeze(1)
    ).square()
    center_distance_sq = (center_x.unsqueeze(2) - center_x.unsqueeze(1)).square() + (
        center_y.unsqueeze(2) - center_y.unsqueeze(1)
    ).square()
    mean_diagonal_sq = ((widths.unsqueeze(2) + widths.unsqueeze(1)) / 2).square() + (
        (heights.unsqueeze(2) + heights.unsqueeze(1)) / 2
    ).square()
    normalized_center_distance_sq = center_distance_sq / mean_diagonal_sq.clamp_min(1e-9)
    density_index = torch.exp(
        -torch.sqrt(outline_distance_sq + normalized_center_distance_sq) / constant
    )

    vehicle = labels.eq(vehicle_class_id) & valid
    pair_mask = vehicle.unsqueeze(2) & vehicle.unsqueeze(1)
    diagonal = torch.eye(labels.shape[1], dtype=torch.bool, device=labels.device).unsqueeze(0)
    pair_mask &= ~diagonal
    return ((density_index > threshold) & pair_mask).any(dim=-1)


def select_density_candidates(
    metrics: torch.Tensor,
    dense_gt: torch.Tensor,
    mask_gt: torch.Tensor,
    *,
    sparse_topk: int,
) -> torch.Tensor:
    if sparse_topk <= 0:
        raise ValueError("sparse_topk must be positive")
    topk = min(sparse_topk, metrics.shape[-1])
    _, topk_indices = torch.topk(metrics, topk, dim=-1, largest=True)
    valid_topk = mask_gt.expand(-1, -1, topk).bool()
    topk_indices = topk_indices.masked_fill(~valid_topk, 0)
    sparse = torch.zeros_like(metrics)
    sparse.scatter_add_(-1, topk_indices, valid_topk.to(metrics.dtype))
    sparse.clamp_(max=1.0)

    best_indices = topk_indices[..., :1]
    best_valid = valid_topk[..., :1]
    dense = torch.zeros_like(metrics)
    dense.scatter_add_(-1, best_indices, best_valid.to(metrics.dtype))
    return torch.where(dense_gt.unsqueeze(-1), dense, sparse)


class DensityAwareTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(
        self,
        *args: object,
        density_constant: float = 12.0,
        density_threshold: float = 0.25,
        vehicle_class_id: int = 24,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not math.isfinite(density_constant) or density_constant <= 0.0:
            raise ValueError("density constant must be positive and finite")
        if not math.isfinite(density_threshold) or not 0.0 <= density_threshold <= 1.0:
            raise ValueError("density threshold must be in [0, 1]")
        self.density_constant = density_constant
        self.density_threshold = density_threshold
        self.vehicle_class_id = vehicle_class_id

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores,
            pd_bboxes,
            gt_labels,
            gt_bboxes,
            mask_in_gts * mask_gt,
        )
        dense_gt = compute_dense_vehicle_mask(
            gt_labels,
            gt_bboxes,
            mask_gt,
            constant=self.density_constant,
            threshold=self.density_threshold,
            vehicle_class_id=self.vehicle_class_id,
        )
        candidate_mask = select_density_candidates(
            align_metric,
            dense_gt,
            mask_gt,
            sparse_topk=self.topk,
        )
        mask_pos = candidate_mask * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps


class DensityAwareDetectionLoss(v8DetectionLoss):
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        config: DensityAssignerConfig,
        tal_topk: int = 10,
        tal_topk2: int | None = None,
    ) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        self.assigner = DensityAwareTaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
            density_constant=config.constant,
            density_threshold=config.threshold,
            vehicle_class_id=config.vehicle_class_id,
        )


class DensityAwareDetectionModel(DetectionModel):
    def __init__(
        self,
        cfg: str | dict = "yolo26n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        density_config: DensityAssignerConfig,
    ) -> None:
        self.density_config = density_config
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

    def init_criterion(self) -> E2ELoss | DensityAwareDetectionLoss:
        if getattr(self, "end2end", False):
            loss_fn = partial(DensityAwareDetectionLoss, config=self.density_config)
            return E2ELoss(self, loss_fn=loss_fn)
        return DensityAwareDetectionLoss(self, config=self.density_config)


class DensityAwareDetectionTrainer(DetectionTrainer):
    density_config = DensityAssignerConfig()

    def get_model(
        self,
        cfg: str | dict | None = None,
        weights: object | None = None,
        verbose: bool = True,
    ) -> DensityAwareDetectionModel:
        model = DensityAwareDetectionModel(
            cfg or "yolo26n.yaml",
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            density_config=self.density_config,
        )
        if weights:
            model.load(weights)
        return model
