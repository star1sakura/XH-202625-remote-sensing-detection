from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest
import torch
from torch import nn

from xh_detect.vehicle_confirmation.model import (
    VehicleConfirmer,
    VehicleConfirmerTrainingConfig,
    binary_average_precision,
    export_vehicle_confirmer_engine,
    score_vehicle_confirmer,
    train_vehicle_confirmer,
    weighted_sampler_weights,
)


def test_vehicle_confirmer_returns_one_logit_per_crop() -> None:
    model = VehicleConfirmer(pretrained=False)

    logits = model(torch.zeros(4, 3, 160, 160), torch.zeros(4, 3))

    assert logits.shape == (4,)


@pytest.mark.parametrize(
    ("images", "features"),
    [
        (torch.zeros(3, 160, 160), torch.zeros(1, 3)),
        (torch.zeros(1, 1, 160, 160), torch.zeros(1, 3)),
        (torch.zeros(1, 3, 160, 160), torch.zeros(3)),
        (torch.zeros(2, 3, 160, 160), torch.zeros(1, 3)),
    ],
)
def test_vehicle_confirmer_rejects_invalid_input_shapes(
    images: torch.Tensor,
    features: torch.Tensor,
) -> None:
    model = VehicleConfirmer(pretrained=False)

    with pytest.raises(ValueError):
        model(images, features)


def test_binary_average_precision_and_sampler_balance() -> None:
    assert binary_average_precision([0.9, 0.8, 0.1], [1, 0, 1]) == pytest.approx(5 / 6)
    weights = weighted_sampler_weights([1, 0, 0, 0])
    assert weights == pytest.approx([1.0, 1 / 3, 1 / 3, 1 / 3])


class _TinyConfirmer(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, images: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        pooled = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return self.linear(torch.cat((pooled, scalar_features), dim=1)).squeeze(1)


def _write_dataset(root: Path) -> None:
    records = {
        "train": [(1, 0.9), (0, 0.2), (0, 0.1), (1, 0.8)],
        "holdout": [(1, 0.7), (0, 0.3)],
    }
    for split, items in records.items():
        lines = []
        for index, (label, score) in enumerate(items, start=1):
            crop = Path("crops") / split / f"{index:06d}.png"
            destination = root / crop
            destination.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(
                str(destination),
                np.full((32, 32, 3), 255 if label else 0, dtype=np.uint8),
            )
            lines.append(
                json.dumps(
                    {
                        "crop": crop.as_posix(),
                        "image_id": str(index),
                        "proposal_index": index,
                        "label": label,
                        "reason": "recoverable_truth" if label else "background",
                        "sph_score": score,
                        "width_norm": 0.1,
                        "height_norm": 0.1,
                        "source_group": f"group-{split}-{index}",
                    }
                )
            )
        manifest = root / "manifests" / f"{split}.jsonl"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = root / "reports" / "vehicle-confirmer-dataset.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("{}", encoding="utf-8")


def test_cpu_training_writes_checkpoint_and_scores_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    _write_dataset(dataset)
    config = VehicleConfirmerTrainingConfig(
        epochs=1,
        batch_size=2,
        workers=0,
        seed=42,
        pretrained=False,
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    with patch("xh_detect.vehicle_confirmation.model.VehicleConfirmer", _TinyConfirmer):
        checkpoint_path = train_vehicle_confirmer(dataset, output, config, "cpu")
        first = score_vehicle_confirmer(
            dataset,
            dataset / "manifests" / "holdout.jsonl",
            checkpoint_path,
            tmp_path / "first.jsonl",
            "cpu",
        )
        second = score_vehicle_confirmer(
            dataset,
            dataset / "manifests" / "holdout.jsonl",
            checkpoint_path,
            tmp_path / "second.jsonl",
            "cpu",
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert {"model_state", "config", "epoch", "holdout_ap", "holdout_bce"} <= checkpoint.keys()
    assert (output / "best.pt.sha256").read_text(encoding="utf-8").strip()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert first == second
    assert (tmp_path / "first.jsonl").read_bytes() == (tmp_path / "second.jsonl").read_bytes()


def test_tensorrt_export_builds_expected_command_and_rejects_failure(tmp_path: Path) -> None:
    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    completed = Mock(returncode=1, stdout="out", stderr="failed")

    with (
        patch(
            "xh_detect.vehicle_confirmation.model.subprocess.run",
            return_value=completed,
        ) as run,
        pytest.raises(RuntimeError, match="trtexec failed"),
    ):
        export_vehicle_confirmer_engine(onnx_path, tmp_path / "model.engine")

    command = run.call_args.args[0]
    assert command[0] == "trtexec"
    assert "--fp16" in command
    assert any(item.startswith("--minShapes=images:1x3x160x160") for item in command)
