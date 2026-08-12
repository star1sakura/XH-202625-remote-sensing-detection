from __future__ import annotations

import re
from collections.abc import Sequence

from torch import nn


def set_batch_norm_eval(model: nn.Module) -> tuple[str, ...]:
    """Freeze running statistics while preserving the model's training outputs."""
    frozen: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            frozen.append(name)
    return tuple(frozen)


def register_class_row_gradient_masks(
    model: nn.Module,
    *,
    class_ids: Sequence[int],
    parameter_patterns: Sequence[str],
    num_classes: int,
) -> tuple[str, ...]:
    """Restrict selected classifier tensors to updates on specific class rows."""
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    raw_rows = tuple(class_ids)
    if not raw_rows:
        raise ValueError("class_ids must contain at least one class row")
    if any(isinstance(row, bool) or not isinstance(row, int) for row in raw_rows):
        raise TypeError("class_ids must contain integers")
    rows = tuple(sorted(set(raw_rows)))
    if rows[0] < 0 or rows[-1] >= num_classes:
        raise ValueError(f"class_ids must be in [0, {num_classes})")
    if not parameter_patterns:
        raise ValueError("parameter_patterns must not be empty")

    patterns = tuple(re.compile(pattern) for pattern in parameter_patterns)
    matched: list[str] = []
    for name, parameter in model.named_parameters():
        if not any(pattern.search(name) for pattern in patterns):
            continue
        if parameter.ndim < 1 or parameter.shape[0] != num_classes:
            raise ValueError(
                f"selected parameter {name!r} has shape {tuple(parameter.shape)}, "
                f"expected first dimension {num_classes}"
            )

        mask_shape = (num_classes,) + (1,) * (parameter.ndim - 1)
        mask = parameter.new_zeros(mask_shape)
        mask[list(rows)] = 1
        parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
        matched.append(name)

    if not matched:
        raise ValueError("parameter_patterns did not match any model parameters")
    return tuple(matched)
