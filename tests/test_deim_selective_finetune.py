from __future__ import annotations

import pytest
import torch
from torch import nn

from xh_detect.deim_selective_finetune import (
    register_class_row_gradient_masks,
    set_batch_norm_eval,
)


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc_score_head = nn.Linear(4, 3)
        self.dec_score_head = nn.ModuleList([nn.Linear(4, 3), nn.Linear(4, 3)])
        self.box_head = nn.Linear(4, 4)


class _Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder()


def test_set_batch_norm_eval_preserves_parent_training_mode() -> None:
    model = nn.Sequential(nn.BatchNorm2d(3), nn.Dropout(), nn.Conv2d(3, 4, 1))
    model.train()

    assert set_batch_norm_eval(model) == ("0",)
    assert model.training is True
    assert model[0].training is False
    assert model[1].training is True


def test_register_class_row_gradient_masks_keeps_only_requested_row() -> None:
    model = _Detector()
    names = register_class_row_gradient_masks(
        model,
        class_ids=[2],
        parameter_patterns=[
            r"^decoder\.enc_score_head\.(?:weight|bias)$",
            r"^decoder\.dec_score_head\.\d+\.(?:weight|bias)$",
        ],
        num_classes=3,
    )

    assert names == (
        "decoder.enc_score_head.weight",
        "decoder.enc_score_head.bias",
        "decoder.dec_score_head.0.weight",
        "decoder.dec_score_head.0.bias",
        "decoder.dec_score_head.1.weight",
        "decoder.dec_score_head.1.bias",
    )

    inputs = torch.ones((2, 4))
    loss = model.decoder.enc_score_head(inputs).sum()
    loss += sum(head(inputs).sum() for head in model.decoder.dec_score_head)
    loss.backward()

    for name, parameter in model.named_parameters():
        if "score_head" not in name:
            continue
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad[:2], torch.zeros_like(parameter.grad[:2]))
        assert torch.count_nonzero(parameter.grad[2]).item() > 0


def test_register_class_row_gradient_masks_rejects_wrong_classifier_shape() -> None:
    model = _Detector()

    with pytest.raises(ValueError, match="expected first dimension 25"):
        register_class_row_gradient_masks(
            model,
            class_ids=[24],
            parameter_patterns=[r"^decoder\.enc_score_head\.weight$"],
            num_classes=25,
        )
