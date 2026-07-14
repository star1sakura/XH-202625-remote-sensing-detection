from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from ultralytics import YOLO

from xh_detect.models.mksnet_lite import MKSNetLiteBlock
from xh_detect.models.ultralytics import register_custom_modules

MAIN_TO_MKS_LAYER_MAP = {
    **{index: index for index in range(17)},
    18: 17,
    19: 18,
    20: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
}
MKS_IDENTITY_LAYER_INDICES = (17, 21)
LayerContainer = torch.nn.ModuleList | torch.nn.Sequential


@dataclass(frozen=True)
class MKSNetSeedResult:
    source_checkpoint: Path
    target_model_yaml: Path
    output_checkpoint: Path
    source_sha256: str
    output_sha256: str
    source_layers: int
    target_layers: int
    transferred_layers: int
    transferred_state_values: int
    identity_blocks: tuple[int, ...]


@dataclass(frozen=True)
class CheckpointInterpolationResult:
    base_checkpoint: Path
    tuned_checkpoint: Path
    output_checkpoint: Path
    alpha: float
    base_sha256: str
    tuned_sha256: str
    output_sha256: str
    state_tensors: int
    interpolated_values: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_architectures(source_layers: object, target_layers: object) -> None:
    if (
        not isinstance(source_layers, (torch.nn.ModuleList, torch.nn.Sequential))
        or len(source_layers) != 24
    ):
        raise ValueError("main checkpoint must contain the expected 24-layer YOLO26s model")
    if (
        not isinstance(target_layers, (torch.nn.ModuleList, torch.nn.Sequential))
        or len(target_layers) != 26
    ):
        raise ValueError("target YAML must contain the expected 26-layer MKSNet-Lite model")
    for index in MKS_IDENTITY_LAYER_INDICES:
        if not isinstance(target_layers[index], MKSNetLiteBlock):
            raise ValueError(f"target layer {index} must be MKSNetLiteBlock")


def _copy_mapped_layers(
    source_layers: LayerContainer,
    target_layers: LayerContainer,
) -> int:
    transferred_values = 0
    for target_index, source_index in MAIN_TO_MKS_LAYER_MAP.items():
        source_state = source_layers[source_index].state_dict()
        target_state = target_layers[target_index].state_dict()
        source_shapes = {name: tuple(value.shape) for name, value in source_state.items()}
        target_shapes = {name: tuple(value.shape) for name, value in target_state.items()}
        if source_shapes != target_shapes:
            raise ValueError(
                f"layer state mismatch for main {source_index} -> MKSNet-Lite {target_index}"
            )
        target_layers[target_index].load_state_dict(source_state, strict=True)
        transferred_values += sum(value.numel() for value in source_state.values())
    return transferred_values


def _initialize_identity_blocks(target_layers: LayerContainer) -> None:
    with torch.no_grad():
        for index in MKS_IDENTITY_LAYER_INDICES:
            block = target_layers[index]
            block.norm.weight.zero_()
            block.norm.bias.zero_()


def initialize_mksnet_lite_from_main(
    main_checkpoint: Path,
    model_yaml: Path,
    output_checkpoint: Path,
    *,
    overwrite: bool = False,
) -> MKSNetSeedResult:
    main_checkpoint = Path(main_checkpoint).resolve()
    model_yaml = Path(model_yaml).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    if not main_checkpoint.is_file():
        raise ValueError(f"main checkpoint does not exist: {main_checkpoint}")
    if not model_yaml.is_file():
        raise ValueError(f"MKSNet-Lite model YAML does not exist: {model_yaml}")
    if output_checkpoint.exists() and not overwrite:
        raise ValueError(f"output checkpoint already exists: {output_checkpoint}")

    register_custom_modules()
    source = YOLO(str(main_checkpoint))
    target = YOLO(str(model_yaml))
    source_layers = source.model.model
    target_layers = target.model.model
    _validate_architectures(source_layers, target_layers)
    transferred_values = _copy_mapped_layers(source_layers, target_layers)
    _initialize_identity_blocks(target_layers)

    if hasattr(source.model, "names"):
        target.model.names = dict(source.model.names)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    target.save(str(output_checkpoint))
    if not output_checkpoint.is_file():
        raise RuntimeError(f"Ultralytics did not save checkpoint: {output_checkpoint}")

    return MKSNetSeedResult(
        source_checkpoint=main_checkpoint,
        target_model_yaml=model_yaml,
        output_checkpoint=output_checkpoint,
        source_sha256=_sha256(main_checkpoint),
        output_sha256=_sha256(output_checkpoint),
        source_layers=len(source_layers),
        target_layers=len(target_layers),
        transferred_layers=len(MAIN_TO_MKS_LAYER_MAP),
        transferred_state_values=transferred_values,
        identity_blocks=MKS_IDENTITY_LAYER_INDICES,
    )


def _checkpoint_model(payload: object, description: str) -> torch.nn.Module:
    if not isinstance(payload, dict):
        raise ValueError(f"{description} checkpoint must contain a mapping")
    model = payload.get("ema") or payload.get("model")
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"{description} checkpoint does not contain model weights")
    return model


def interpolate_checkpoints(
    base_checkpoint: Path,
    tuned_checkpoint: Path,
    output_checkpoint: Path,
    alpha: float,
    *,
    overwrite: bool = False,
) -> CheckpointInterpolationResult:
    base_checkpoint = Path(base_checkpoint).resolve()
    tuned_checkpoint = Path(tuned_checkpoint).resolve()
    output_checkpoint = Path(output_checkpoint).resolve()
    if not base_checkpoint.is_file():
        raise ValueError(f"base checkpoint does not exist: {base_checkpoint}")
    if not tuned_checkpoint.is_file():
        raise ValueError(f"tuned checkpoint does not exist: {tuned_checkpoint}")
    if output_checkpoint.exists() and not overwrite:
        raise ValueError(f"output checkpoint already exists: {output_checkpoint}")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
    ):
        raise ValueError("alpha must be finite and in [0, 1]")
    alpha = float(alpha)

    register_custom_modules()
    base_payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    tuned_payload = torch.load(tuned_checkpoint, map_location="cpu", weights_only=False)
    base_model = deepcopy(_checkpoint_model(base_payload, "base")).float()
    tuned_model = _checkpoint_model(tuned_payload, "tuned").float()
    base_state = base_model.state_dict()
    tuned_state = tuned_model.state_dict()
    if base_state.keys() != tuned_state.keys():
        raise ValueError("checkpoint state keys do not match")

    merged_state: dict[str, torch.Tensor] = {}
    interpolated_values = 0
    for name, base_value in base_state.items():
        tuned_value = tuned_state[name]
        if base_value.shape != tuned_value.shape:
            raise ValueError(f"checkpoint tensor shape mismatch: {name}")
        if base_value.is_floating_point():
            merged_state[name] = base_value.mul(1.0 - alpha).add(tuned_value, alpha=alpha)
            interpolated_values += base_value.numel()
        else:
            merged_state[name] = tuned_value.clone()
    base_model.load_state_dict(merged_state, strict=True)

    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_payload = dict(base_payload)
    output_payload.update(
        {
            "model": base_model.half(),
            "ema": None,
            "optimizer": None,
            "interpolation": {
                "base_checkpoint": str(base_checkpoint),
                "tuned_checkpoint": str(tuned_checkpoint),
                "alpha": alpha,
            },
        }
    )
    torch.save(output_payload, output_checkpoint)
    return CheckpointInterpolationResult(
        base_checkpoint=base_checkpoint,
        tuned_checkpoint=tuned_checkpoint,
        output_checkpoint=output_checkpoint,
        alpha=alpha,
        base_sha256=_sha256(base_checkpoint),
        tuned_sha256=_sha256(tuned_checkpoint),
        output_sha256=_sha256(output_checkpoint),
        state_tensors=len(merged_state),
        interpolated_values=interpolated_values,
    )
