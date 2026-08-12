from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEIM_ROOT = PROJECT_ROOT / ".third_party" / "DEIM"
pytest.importorskip("tensorboard")
pytest.importorskip("faster_coco_eval")
sys.path.insert(0, str(DEIM_ROOT))

from engine.core import YAMLConfig  # noqa: E402
from engine.deim.matcher import HungarianMatcher  # noqa: E402
from engine.solver.det_solver import DetSolver  # noqa: E402

from xh_detect.deim_bhcl_adapter import (  # noqa: E402
    BHCLDEIMCriterion,
    BHCLDFINETransformer,
    DecoupledTransformerDecoderLayer,
    _copy_attention_into_joint,
)


def test_joint_attention_starts_as_two_pretrained_blocks() -> None:
    source = nn.MultiheadAttention(8, 2, batch_first=True)
    destination = nn.MultiheadAttention(16, 4, batch_first=True)

    _copy_attention_into_joint(source, destination)

    for projection in range(3):
        source_slice = source.in_proj_weight[projection * 8 : (projection + 1) * 8]
        destination_slice = destination.in_proj_weight[
            projection * 16 : (projection + 1) * 16
        ]
        torch.testing.assert_close(destination_slice[:8, :8], source_slice)
        torch.testing.assert_close(destination_slice[8:, 8:], source_slice)
        torch.testing.assert_close(destination_slice[:8, 8:], torch.zeros(8, 8))
        torch.testing.assert_close(destination_slice[8:, :8], torch.zeros(8, 8))


def test_bhcl_configs_are_strict_official_split_ablations() -> None:
    paths = {
        "decoupled": "deim_dfine_l_xh25_1024_80e_decoupled.yml",
        "hcl": "deim_dfine_l_xh25_1024_80e_hcl.yml",
        "bhcl": "deim_dfine_l_xh25_1024_80e_bhcl.yml",
    }

    for mode, filename in paths.items():
        config = yaml.safe_load((PROJECT_ROOT / "configs/deim" / filename).read_text())
        expected_mode = "none" if mode == "decoupled" else mode.split("_")[0]
        decoder = config.get("BHCLDFINETransformer", {})
        assert decoder.get("bhcl_mode", expected_mode) == expected_mode
        assert "vehicle-source-balanced" not in str(config)


def _criterion(*, mode: str) -> BHCLDEIMCriterion:
    return BHCLDEIMCriterion(
        matcher=HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
        ),
        weight_dict={"loss_bhcl": 0.6},
        losses=["bhcl"],
        num_classes=25,
        bhcl_mode=mode,
        bhcl_embedding_dim=8,
    )


def test_hcl_mode_does_not_create_a_prototype_bank() -> None:
    criterion = _criterion(mode="hcl")

    assert criterion.bhcl_mode == "hcl"
    assert criterion.prototype_bank is None


def test_bhcl_mode_creates_a_prototype_bank() -> None:
    criterion = _criterion(mode="bhcl")

    assert criterion.bhcl_mode == "bhcl"
    assert criterion.prototype_bank is not None


def test_tiny_bhcl_model_has_finite_end_to_end_gradients() -> None:
    model = BHCLDFINETransformer(
        num_classes=25,
        hidden_dim=32,
        num_queries=6,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=2,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
        eval_spatial_size=[64, 64],
        bhcl_mode="bhcl",
        bhcl_embedding_dim=8,
    )
    model.train()
    model.initialize_after_tuning()
    assert isinstance(model.decoder.layers[0], DecoupledTransformerDecoderLayer)

    features = [
        torch.randn(2, 32, 8, 8),
        torch.randn(2, 32, 4, 4),
        torch.randn(2, 32, 2, 2),
    ]
    targets = [
        {
            "labels": torch.tensor([0, 4]),
            "boxes": torch.tensor([[0.2, 0.2, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1]]),
        },
        {
            "labels": torch.tensor([24]),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
    ]
    outputs = model(features, targets)
    criterion = BHCLDEIMCriterion(
        matcher=HungarianMatcher(
            weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2}
        ),
        weight_dict={
            "loss_mal": 1,
            "loss_bbox": 5,
            "loss_giou": 2,
            "loss_fgl": 0.15,
            "loss_ddf": 1.5,
            "loss_bhcl": 0.6,
        },
        losses=["mal", "boxes", "local", "bhcl"],
        num_classes=25,
        reg_max=32,
        bhcl_mode="bhcl",
        bhcl_embedding_dim=8,
    )
    criterion.train()

    losses = criterion(outputs, targets)
    total_loss = sum(losses.values())
    total_loss.backward()

    assert "loss_bhcl" in losses
    assert "loss_bhcl_aux_0" in losses
    assert torch.isfinite(total_loss)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert criterion.prototype_bank is not None
    assert torch.count_nonzero(criterion.prototype_bank.prototypes).item() > 0


def test_deploy_removes_training_only_projection_head() -> None:
    model = BHCLDFINETransformer(
        num_classes=25,
        hidden_dim=32,
        num_queries=6,
        feat_channels=[32, 32, 32],
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=2,
        nhead=4,
        num_layers=2,
        dim_feedforward=64,
        num_denoising=0,
        eval_spatial_size=[64, 64],
        bhcl_mode="bhcl",
        bhcl_embedding_dim=8,
    )
    model.convert_to_deploy()

    assert model.decoder.decoupled_ready
    assert isinstance(model.decoder.layers[0], DecoupledTransformerDecoderLayer)
    assert isinstance(model.bhcl_projection_head, nn.Identity)


def test_solver_rebuilds_ema_after_nested_decoder_initialization(tmp_path: Path) -> None:
    config = YAMLConfig(
        str(PROJECT_ROOT / "configs/deim/deim_dfine_l_xh25_1024_80e_bhcl.yml")
    )
    config.device = "cpu"
    config.tuning = None
    config.output_dir = str(tmp_path)

    solver = DetSolver(config)
    solver._setup()

    assert solver.model.decoder.decoder.decoupled_ready
    model_state = solver.model.state_dict()
    ema_state = solver.ema.module.state_dict()
    assert model_state.keys() == ema_state.keys()
    assert all(model_state[key].shape == ema_state[key].shape for key in model_state)
