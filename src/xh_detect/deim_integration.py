from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

CheckpointStatePreference = Literal["auto", "ema", "model"]


def initialize_deim_extensions(model: object) -> int:
    """Finalize project decoder extensions before loading their state dicts."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("DEIM model must provide modules()")

    initialized = 0
    for module in list(modules()):
        initializer = getattr(module, "initialize_after_tuning", None)
        if callable(initializer):
            initializer()
            initialized += 1
    return initialized


def select_deim_checkpoint_state(
    checkpoint: object,
    *,
    preferred: CheckpointStatePreference = "auto",
) -> tuple[str, dict[str, object]]:
    """Select the inference model state from an official DEIM checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise TypeError("DEIM checkpoint must be a mapping")

    ema = checkpoint.get("ema")
    model = checkpoint.get("model")
    if preferred == "ema":
        if not isinstance(ema, Mapping) or not isinstance(ema.get("module"), Mapping):
            raise ValueError("DEIM checkpoint has no ema.module state")
        source = "ema.module"
        state = ema["module"]
    elif preferred == "model":
        if not isinstance(model, Mapping):
            raise ValueError("DEIM checkpoint has no model state")
        source = "model"
        state = model
    elif isinstance(ema, Mapping) and isinstance(ema.get("module"), Mapping):
        source = "ema.module"
        state = ema["module"]
    elif isinstance(model, Mapping):
        source = "model"
        state = model
    elif checkpoint and all(isinstance(key, str) for key in checkpoint):
        source = "raw"
        state = checkpoint
    else:
        raise ValueError("DEIM checkpoint has no ema.module or model state")

    normalized: dict[str, object] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            raise TypeError("DEIM state-dict keys must be strings")
        normalized[key.removeprefix("module.")] = value
    if not normalized:
        raise ValueError("DEIM model state is empty")
    return source, normalized
