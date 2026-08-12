from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml

from xh_detect.deim_integration import (
    initialize_deim_extensions,
    select_deim_checkpoint_state,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_deim_xh25_config_keeps_competition_dataset_contract() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs/deim/deim_dfine_l_xh25_1024.yml").read_text(
            encoding="utf-8"
        )
    )

    assert config["num_classes"] == 25
    assert config["remap_mscoco_category"] is False
    assert config["eval_spatial_size"] == [1024, 1024]
    assert config["epoches"] == 24
    assert config["train_dataloader"]["total_batch_size"] == 8
    assert config["train_dataloader"]["dataset"]["ann_file"].endswith(
        "reports/train-ground-truth.json"
    )
    assert config["val_dataloader"]["dataset"]["ann_file"].endswith(
        "reports/val-ground-truth.json"
    )
    assert config["train_dataloader"]["dataset"]["transforms"]["policy"]["epoch"] == [
        2,
        12,
        20,
    ]


def test_deim_upstream_is_pinned() -> None:
    upstream = json.loads(
        (PROJECT_ROOT / "configs/deim/upstream.json").read_text(encoding="utf-8")
    )

    assert upstream["commit"] == "09d35d53d39ee3145a1e61e3a989b28b9468d1dd"
    assert upstream["pretrained_checkpoint"]["google_drive_file_id"] == (
        "1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr"
    )


def test_select_deim_checkpoint_state_prefers_ema_and_strips_module_prefix() -> None:
    source, state = select_deim_checkpoint_state(
        {
            "ema": {"module": {"module.encoder.weight": "ema"}},
            "model": {"encoder.weight": "model"},
        }
    )

    assert source == "ema.module"
    assert state == {"encoder.weight": "ema"}


def test_select_deim_checkpoint_state_falls_back_to_model() -> None:
    source, state = select_deim_checkpoint_state({"model": {"decoder.bias": 1}})

    assert source == "model"
    assert state == {"decoder.bias": 1}


def test_select_deim_checkpoint_state_can_explicitly_select_model() -> None:
    source, state = select_deim_checkpoint_state(
        {
            "ema": {"module": {"encoder.weight": "ema"}},
            "model": {"module.encoder.weight": "model"},
        },
        preferred="model",
    )

    assert source == "model"
    assert state == {"encoder.weight": "model"}


def test_select_deim_checkpoint_state_rejects_invalid_checkpoint() -> None:
    with pytest.raises(ValueError, match="no ema.module or model"):
        select_deim_checkpoint_state({})


def test_initialize_deim_extensions_finds_nested_decoder_hook() -> None:
    class Extension(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.initialized = False

        def initialize_after_tuning(self) -> None:
            self.initialized = True

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), Extension())

    initialized = initialize_deim_extensions(model)

    assert initialized == 1
    assert model[1].initialized
