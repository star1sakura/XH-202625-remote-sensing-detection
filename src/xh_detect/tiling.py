from __future__ import annotations

import math
from collections.abc import Iterator
from numbers import Integral, Real
from typing import cast

import numpy as np

from xh_detect.types import ImageArray, Tile, TileMeta

SUPPORTED_IMAGE_DTYPES = (
    np.uint8,
    np.uint16,
    np.int16,
    np.float32,
    np.float64,
)


def _validate_int(name: str, value: object, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")

    result = int(value)
    if minimum is not None and result < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be non-negative")
        raise ValueError(f"{name} must be a positive integer")
    return result


def axis_positions(length: int, tile_size: int, stride: int) -> list[int]:
    length = _validate_int("length", length, minimum=0)
    tile_size = _validate_int("tile_size", tile_size, minimum=1)
    stride = _validate_int("stride", stride, minimum=1)

    if length <= tile_size:
        return [0]

    final_position = length - tile_size
    positions = list(range(0, final_position + 1, stride))
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def _validate_overlap(overlap: object) -> float:
    if isinstance(overlap, bool) or not isinstance(overlap, Real):
        raise TypeError("overlap must be a finite real number")

    overlap_value = float(overlap)
    if not math.isfinite(overlap_value):
        raise ValueError("overlap must be finite")
    if overlap_value < 0 or overlap_value >= 1:
        raise ValueError("overlap must be in [0, 1)")
    return overlap_value


def _validate_image(image: object) -> ImageArray:
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale array or 3D HxWxC array")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("image height and width must be positive")
    if array.ndim == 3 and array.shape[2] <= 0:
        raise ValueError("image channel dimension must be positive")
    if array.dtype.type not in SUPPORTED_IMAGE_DTYPES:
        raise TypeError(
            "image dtype must be one of uint8, uint16, int16, float32, or float64"
        )
    return cast(ImageArray, array)


def _is_finite_real(value: Real) -> bool:
    if isinstance(value, Integral):
        return True
    if isinstance(value, np.floating):
        return bool(np.isfinite(value))
    return math.isfinite(value)


def _validate_pad_value(pad_value: object, dtype: np.dtype) -> np.generic:
    if not np.isscalar(pad_value):
        raise TypeError("pad_value must be a scalar")
    if isinstance(pad_value, (bool, np.bool_)) or not isinstance(pad_value, Real):
        raise TypeError("pad_value must be a non-boolean real numeric scalar")

    if np.issubdtype(dtype, np.integer):
        if isinstance(pad_value, Integral):
            integer_value = int(pad_value)
        else:
            float_value = float(pad_value)
            if not math.isfinite(float_value) or not float_value.is_integer():
                raise ValueError(
                    "pad_value must be a finite integer for integer image dtype"
                )
            integer_value = int(float_value)

        limits = np.iinfo(dtype)
        if integer_value < limits.min or integer_value > limits.max:
            raise ValueError(f"pad_value is outside the range of image dtype {dtype}")
        return dtype.type(integer_value)

    is_finite = _is_finite_real(pad_value)
    if is_finite and abs(pad_value) > np.finfo(dtype).max:
        raise ValueError(
            f"pad_value overflows finite range of image dtype {dtype}"
        )
    try:
        converted = dtype.type(pad_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"pad_value cannot be converted to image dtype {dtype}") from exc
    if is_finite and not np.isfinite(converted):
        raise ValueError(f"pad_value overflows finite range of image dtype {dtype}")
    return converted


def _empty_tile_shape(image: ImageArray, tile_size: int) -> tuple[int, ...]:
    if image.ndim == 2:
        return (tile_size, tile_size)
    return (tile_size, tile_size, image.shape[2])


def _make_tile(
    image: ImageArray,
    image_id: str,
    tile_size: int,
    x: int,
    y: int,
    pad_value: object,
) -> Tile:
    valid_width = min(tile_size, image.shape[1] - x)
    valid_height = min(tile_size, image.shape[0] - y)

    tile = np.full(_empty_tile_shape(image, tile_size), pad_value, dtype=image.dtype)
    tile_slices = (slice(0, valid_height), slice(0, valid_width))
    if image.ndim == 3:
        tile_slices += (slice(None),)
    tile[tile_slices] = image[y : y + valid_height, x : x + valid_width]

    meta = TileMeta(
        image_id=image_id,
        tile_id=f"{image_id}__x{x}_y{y}_s{tile_size}",
        x=x,
        y=y,
        width=tile_size,
        height=tile_size,
        valid_width=valid_width,
        valid_height=valid_height,
    )
    return Tile(image=cast(ImageArray, tile), meta=meta)


def iter_tiles(
    image: ImageArray,
    image_id: str,
    tile_size: int,
    overlap: float,
    pad_value: object = 114,
) -> Iterator[Tile]:
    array = _validate_image(image)
    tile_size = _validate_int("tile_size", tile_size, minimum=1)
    overlap_value = _validate_overlap(overlap)
    normalized_pad_value = _validate_pad_value(pad_value, array.dtype)

    stride = max(1, round(tile_size * (1.0 - overlap_value)))
    y_positions = axis_positions(array.shape[0], tile_size, stride)
    x_positions = axis_positions(array.shape[1], tile_size, stride)

    for y in y_positions:
        for x in x_positions:
            yield _make_tile(array, image_id, tile_size, x, y, normalized_pad_value)
