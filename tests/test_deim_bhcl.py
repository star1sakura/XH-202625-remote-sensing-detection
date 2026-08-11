from __future__ import annotations

import math

import pytest
import torch

from xh_detect.deim_bhcl import (
    HierarchicalPrototypeBank,
    HierarchySpec,
    hierarchical_contrastive_loss,
    xh25_hierarchy,
)


def test_xh25_hierarchy_maps_coarse_and_leaf_classes() -> None:
    hierarchy = xh25_hierarchy()

    assert hierarchy.level_class_counts == (3, 25)
    assert hierarchy.leaf_to_level[0] == (
        0,
        0,
        0,
        0,
        *([1] * 20),
        2,
    )
    assert hierarchy.leaf_to_level[1] == tuple(range(25))
    assert sum(hierarchy.level_weights) == pytest.approx(1.0)
    assert hierarchy.level_weights[1] > hierarchy.level_weights[0]


def test_hierarchy_rejects_noncontiguous_level_ids() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        HierarchySpec(((0, 2),))


def test_hcl_is_zero_for_a_single_perfect_positive_pair() -> None:
    hierarchy = HierarchySpec(((0, 0),))
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 1])

    loss = hierarchical_contrastive_loss(
        features,
        labels,
        hierarchy,
        mode="hcl",
        temperature=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_bhcl_matches_equation_eight_class_averaged_denominator() -> None:
    hierarchy = HierarchySpec(((0, 0, 1),))
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    labels = torch.tensor([0, 0])
    prototypes = (torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),)

    loss = hierarchical_contrastive_loss(
        features,
        labels,
        hierarchy,
        mode="bhcl",
        temperature=1.0,
        prototypes=prototypes,
    )

    expected = math.log(2.0 * math.e / 3.0 + math.exp(-1.0)) - 1.0
    torch.testing.assert_close(loss, torch.tensor(expected), rtol=1e-6, atol=1e-6)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_bhcl_stays_finite_when_each_leaf_occurs_once() -> None:
    hierarchy = HierarchySpec(((0, 0, 1), (0, 1, 2)))
    features = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2])
    prototypes = (torch.zeros(2, 8), torch.zeros(3, 8))

    loss = hierarchical_contrastive_loss(
        features,
        labels,
        hierarchy,
        mode="bhcl",
        prototypes=prototypes,
    )

    assert torch.isfinite(loss)
    loss.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_prototype_bank_updates_coarse_nodes_from_descendants() -> None:
    hierarchy = HierarchySpec(((0, 0, 1), (0, 1, 2)))
    bank = HierarchicalPrototypeBank(hierarchy, embedding_dim=2, epsilon=0.1)
    features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([0, 1])

    bank.update(features, labels)

    coarse, leaves = bank.by_level()
    diagonal = torch.tensor([2**-0.5, 2**-0.5])
    torch.testing.assert_close(coarse[0], diagonal)
    torch.testing.assert_close(coarse[1], torch.zeros(2))
    torch.testing.assert_close(leaves[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(leaves[1], torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(leaves[2], torch.zeros(2))


def test_prototype_bank_state_dict_round_trip() -> None:
    hierarchy = HierarchySpec(((0, 1),))
    source = HierarchicalPrototypeBank(hierarchy, embedding_dim=2)
    source.update(torch.eye(2), torch.tensor([0, 1]))
    restored = HierarchicalPrototypeBank(hierarchy, embedding_dim=2)

    restored.load_state_dict(source.state_dict())

    torch.testing.assert_close(restored.prototypes, source.prototypes)
