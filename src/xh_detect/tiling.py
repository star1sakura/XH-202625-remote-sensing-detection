from __future__ import annotations

import math
import warnings
from collections.abc import Iterator
from numbers import Integral, Number, Real
from typing import cast

import numpy as np

from xh_detect.types import ImageArray, Tile, TileMeta


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
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image dtype must be numeric")
    return cast(ImageArray, array)


def _real_values_match(original: object, converted: object) -> bool:
    original_value = original.item() if isinstance(original, np.generic) else original
    converted_value = (
        converted.item() if isinstance(converted, np.generic) else converted
    )
    if isinstance(original_value, (float, np.floating)):
        if np.isnan(original_value):
            return bool(np.isnan(converted_value))
        if np.isinf(original_value):
            return bool(
                np.isinf(converted_value)
                and np.signbit(original_value) == np.signbit(converted_value)
            )
    return bool(converted_value == original_value)


def _pad_value_matches_dtype(pad_value: object, converted: np.generic) -> bool:
    if np.iscomplexobj(converted):
        if np.iscomplexobj(pad_value):
            original_real = pad_value.real
            original_imag = pad_value.imag
        else:
            original_real = pad_value
            original_imag = 0
        return _real_values_match(
            original_real, converted.real
        ) and _real_values_match(original_imag, converted.imag)
    return _real_values_match(pad_value, converted)


def _validate_pad_value(pad_value: object, dtype: np.dtype) -> np.generic:
    if not np.isscalar(pad_value):
        raise TypeError("pad_value must be a scalar")
    if not isinstance(pad_value, (Number, np.number, np.bool_)):
        raise TypeError("pad_value must be a numeric scalar")
    if np.iscomplexobj(pad_value) and not np.issubdtype(dtype, np.complexfloating):
        raise ValueError(f"pad_value must be losslessly representable by image dtype {dtype}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            converted = dtype.type(pad_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            f"pad_value must be losslessly representable by image dtype {dtype}"
        ) from exc

    if not _pad_value_matches_dtype(pad_value, converted):
        raise ValueError(
            f"pad_value must be losslessly representable by image dtype {dtype}"
        )
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
