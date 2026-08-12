from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .taxonomy import get_taxonomy


@dataclass(frozen=True)
class HierarchySpec:
    """Leaf-to-node mappings for every non-root hierarchy level."""

    leaf_to_level: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.leaf_to_level:
            raise ValueError("hierarchy must contain at least one non-root level")

        leaf_count = len(self.leaf_to_level[0])
        if leaf_count == 0:
            raise ValueError("hierarchy must contain at least one leaf class")

        for level, mapping in enumerate(self.leaf_to_level):
            if len(mapping) != leaf_count:
                raise ValueError("all hierarchy levels must map every leaf class")
            if any(isinstance(class_id, bool) or class_id < 0 for class_id in mapping):
                raise ValueError(f"hierarchy level {level} contains an invalid class ID")
            class_ids = set(mapping)
            if class_ids != set(range(len(class_ids))):
                raise ValueError(
                    f"hierarchy level {level} class IDs must be contiguous from zero"
                )

    @property
    def num_leaf_classes(self) -> int:
        return len(self.leaf_to_level[0])

    @property
    def num_levels(self) -> int:
        return len(self.leaf_to_level)

    @property
    def level_class_counts(self) -> tuple[int, ...]:
        return tuple(len(set(mapping)) for mapping in self.leaf_to_level)

    @property
    def level_weights(self) -> tuple[float, ...]:
        unnormalized = [
            math.exp(1.0 / (self.num_levels + 1 - level))
            for level in range(1, self.num_levels + 1)
        ]
        denominator = sum(unnormalized)
        return tuple(weight / denominator for weight in unnormalized)

    def labels_at_level(self, leaf_labels: Tensor, level: int) -> Tensor:
        if leaf_labels.dtype != torch.long:
            leaf_labels = leaf_labels.long()
        if leaf_labels.numel() and (
            leaf_labels.min().item() < 0
            or leaf_labels.max().item() >= self.num_leaf_classes
        ):
            raise ValueError("leaf labels fall outside the hierarchy")
        mapping = torch.as_tensor(
            self.leaf_to_level[level], dtype=torch.long, device=leaf_labels.device
        )
        return mapping[leaf_labels]


def xh25_hierarchy() -> HierarchySpec:
    taxonomy = get_taxonomy("xh25")
    coarse_names = ("ship", "aircraft", "vehicle")
    coarse_ids = {name: class_id for class_id, name in enumerate(coarse_names)}
    coarse_mapping = tuple(
        coarse_ids[taxonomy.coarse_by_id[class_id]]
        for class_id in range(len(taxonomy.names))
    )
    return HierarchySpec(
        leaf_to_level=(coarse_mapping, tuple(range(len(taxonomy.names))))
    )


def _zero_loss(features: Tensor) -> Tensor:
    return features.sum() * 0.0


def _hcl_level_loss(features: Tensor, labels: Tensor, temperature: float) -> Tensor:
    instance_count = features.shape[0]
    if instance_count < 2:
        return _zero_loss(features)

    logits = features @ features.transpose(0, 1) / temperature
    self_mask = torch.eye(instance_count, dtype=torch.bool, device=features.device)
    denominator = torch.logsumexp(logits.masked_fill(self_mask, -torch.inf), dim=1)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return _zero_loss(features)

    positive_sum = logits.masked_fill(~positive_mask, 0.0).sum(dim=1)
    mean_positive = positive_sum / positive_count.clamp_min(1)
    per_anchor = torch.where(valid, denominator - mean_positive, 0.0)
    return per_anchor.sum() / instance_count


def _bhcl_level_loss(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    temperature: float,
) -> Tensor:
    instance_count = features.shape[0]
    class_count = prototypes.shape[0]
    if instance_count == 0:
        return _zero_loss(features)
    if class_count == 0:
        raise ValueError("BHCL requires at least one class prototype")

    sample_logits = features @ features.transpose(0, 1) / temperature
    prototype_logits = features @ prototypes.transpose(0, 1) / temperature
    self_mask = torch.eye(instance_count, dtype=torch.bool, device=features.device)
    sample_logits_without_self = sample_logits.masked_fill(self_mask, -torch.inf)

    class_log_means = []
    for class_id in range(class_count):
        class_mask = labels.eq(class_id)
        class_sample_sum = torch.logsumexp(
            sample_logits_without_self.masked_fill(~class_mask[None, :], -torch.inf),
            dim=1,
        )
        class_total = torch.logaddexp(
            class_sample_sum,
            prototype_logits[:, class_id],
        )
        # Equation (8) divides by |I'_c| even when the anchor itself is excluded.
        class_log_means.append(class_total - math.log(int(class_mask.sum()) + 1))

    log_denominator = torch.logsumexp(torch.stack(class_log_means, dim=1), dim=1)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    positive_count = positive_mask.sum(dim=1) + 1
    positive_sum = sample_logits.masked_fill(~positive_mask, 0.0).sum(dim=1)
    positive_sum = positive_sum + prototype_logits.gather(1, labels[:, None]).squeeze(1)
    mean_positive = positive_sum / positive_count
    return (log_denominator - mean_positive).mean()


def hierarchical_contrastive_loss(
    features: Tensor,
    leaf_labels: Tensor,
    hierarchy: HierarchySpec,
    *,
    mode: str,
    temperature: float = 0.1,
    prototypes: Sequence[Tensor] | None = None,
) -> Tensor:
    """Compute the paper's HCL or class-balanced BHCL objective."""

    if mode not in {"hcl", "bhcl"}:
        raise ValueError("mode must be 'hcl' or 'bhcl'")
    if features.ndim != 2:
        raise ValueError("features must have shape [instances, embedding_dim]")
    if leaf_labels.ndim != 1 or leaf_labels.shape[0] != features.shape[0]:
        raise ValueError("leaf_labels must have one entry per feature")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if mode == "bhcl" and (
        prototypes is None or len(prototypes) != hierarchy.num_levels
    ):
        raise ValueError("BHCL requires one prototype tensor per hierarchy level")

    if features.shape[0] == 0:
        return _zero_loss(features)

    normalized_features = F.normalize(features.float(), dim=-1)
    level_losses = []
    for level, level_weight in enumerate(hierarchy.level_weights):
        labels = hierarchy.labels_at_level(leaf_labels, level)
        if mode == "hcl":
            level_loss = _hcl_level_loss(normalized_features, labels, temperature)
        else:
            assert prototypes is not None
            level_prototypes = prototypes[level]
            expected_shape = (hierarchy.level_class_counts[level], features.shape[1])
            if tuple(level_prototypes.shape) != expected_shape:
                raise ValueError(
                    f"prototype level {level} has shape {tuple(level_prototypes.shape)}, "
                    f"expected {expected_shape}"
                )
            level_loss = _bhcl_level_loss(
                normalized_features,
                labels,
                F.normalize(level_prototypes.float(), dim=-1),
                temperature,
            )
        level_losses.append(level_loss * level_weight)

    loss = torch.stack(level_losses).sum()
    if not torch.isfinite(loss):
        raise FloatingPointError("hierarchical contrastive loss became non-finite")
    return loss


class HierarchicalPrototypeBank(nn.Module):
    """Checkpointable EMA prototype bank from BHCL equation (10)."""

    def __init__(
        self,
        hierarchy: HierarchySpec,
        embedding_dim: int,
        *,
        epsilon: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not 0 < epsilon <= 1 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be in (0, 1]")

        self.hierarchy = hierarchy
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.level_offsets = tuple(
            sum(hierarchy.level_class_counts[:level])
            for level in range(hierarchy.num_levels + 1)
        )
        self.register_buffer(
            "prototypes",
            torch.zeros((self.level_offsets[-1], embedding_dim), dtype=torch.float32),
        )

    def by_level(self, *, dtype: torch.dtype | None = None) -> tuple[Tensor, ...]:
        values = self.prototypes if dtype is None else self.prototypes.to(dtype=dtype)
        return tuple(
            values[self.level_offsets[level] : self.level_offsets[level + 1]]
            for level in range(self.hierarchy.num_levels)
        )

    @torch.no_grad()
    def update(self, features: Tensor, leaf_labels: Tensor) -> None:
        if features.ndim != 2 or features.shape[1] != self.embedding_dim:
            raise ValueError("prototype features have the wrong shape")
        if leaf_labels.ndim != 1 or leaf_labels.shape[0] != features.shape[0]:
            raise ValueError("prototype labels must have one entry per feature")
        if features.shape[0] == 0:
            return

        normalized = F.normalize(features.detach().float(), dim=-1)
        for level, class_count in enumerate(self.hierarchy.level_class_counts):
            labels = self.hierarchy.labels_at_level(leaf_labels, level)
            sums = torch.zeros(
                (class_count, self.embedding_dim),
                dtype=normalized.dtype,
                device=normalized.device,
            )
            counts = torch.zeros(class_count, dtype=normalized.dtype, device=normalized.device)
            sums.index_add_(0, labels, normalized)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=normalized.dtype))

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(sums)
                torch.distributed.all_reduce(counts)

            present = counts > 0
            if not present.any():
                continue
            means = sums[present] / counts[present, None]
            start, end = self.level_offsets[level : level + 2]
            current = self.prototypes[start:end]
            update_fraction = self.epsilon ** (self.hierarchy.num_levels - level - 1)
            current[present] = F.normalize(
                (1.0 - update_fraction) * current[present]
                + update_fraction * means.to(current.device),
                dim=-1,
            )
