from __future__ import annotations

import math

import numpy as np
import pytest

from xh_detect.tiling import axis_positions, iter_tiles


def test_axis_positions_returns_origin_when_length_fits_in_one_tile() -> None:
    assert axis_positions(1024, 1024, 819) == [0]
    assert axis_positions(0, 1024, 819) == [0]


def test_axis_positions_includes_final_position_once() -> None:
    assert axis_positions(1500, 1024, 819) == [0, 476]


def test_axis_positions_large_image_uses_expected_grid_count() -> None:
    positions = axis_positions(10000, 1024, 819)

    assert len(positions) == 12
    assert len(positions) * len(positions) == 144
    assert positions[-1] == 8976
    assert len(positions) == len(set(positions))


@pytest.mark.parametrize(
    ("length", "tile_size", "stride", "exc_type"),
    [
        (-1, 1024, 819, ValueError),
        (10, 0, 1, ValueError),
        (10, 1024, 0, ValueError),
        (10.5, 1024, 819, TypeError),
        (10, 1024.0, 819, TypeError),
        (10, 1024, 819.0, TypeError),
    ],
)
def test_axis_positions_validates_inputs(length, tile_size, stride, exc_type) -> None:
    with pytest.raises(exc_type):
        axis_positions(length, tile_size, stride)


def test_iter_tiles_pads_small_grayscale_image_and_preserves_dtype() -> None:
    image = np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]], dtype=np.float32)

    tiles = list(iter_tiles(image, "img-1", tile_size=4, overlap=0.0, pad_value=-5.5))

    assert len(tiles) == 1
    tile = tiles[0]
    assert tile.image.shape == (4, 4)
    assert tile.image.dtype == image.dtype
    np.testing.assert_array_equal(tile.image[:2, :3], image)
    assert np.all(tile.image[2:, :] == -5.5)
    assert np.all(tile.image[:, 3:] == -5.5)
    assert tile.meta.image_id == "img-1"
    assert tile.meta.tile_id == "img-1__x0_y0_s4"
    assert tile.meta.x == 0
    assert tile.meta.y == 0
    assert tile.meta.width == 4
    assert tile.meta.height == 4
    assert tile.meta.valid_width == 3
    assert tile.meta.valid_height == 2


def test_iter_tiles_uses_default_pad_value_for_uint8() -> None:
    image = np.array([[7]], dtype=np.uint8)

    tile = next(iter_tiles(image, "default-pad", tile_size=2, overlap=0.0))

    assert tile.image.dtype == np.uint8
    np.testing.assert_array_equal(
        tile.image,
        np.array([[7, 114], [114, 114]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.int16, np.float32, np.float64],
)
def test_iter_tiles_accepts_supported_image_dtypes(dtype) -> None:
    image = np.array([[7]], dtype=dtype)

    tile = next(iter_tiles(image, "supported", tile_size=2, overlap=0.0))

    assert tile.image.dtype.type is dtype
    assert tile.image[1, 1] == dtype(114)


@pytest.mark.parametrize("pad_value", [-1, 300, 1.5, math.nan, math.inf, -math.inf])
def test_iter_tiles_rejects_invalid_uint8_pad_value(pad_value) -> None:
    image = np.ones((1, 1), dtype=np.uint8)

    with pytest.raises(ValueError, match="pad_value"):
        list(iter_tiles(image, "bad-pad", tile_size=2, overlap=0.0, pad_value=pad_value))


@pytest.mark.parametrize("pad_value", [[1], np.array(1, dtype=np.int64)])
def test_iter_tiles_rejects_non_scalar_pad_value(pad_value) -> None:
    image = np.ones((1, 1), dtype=np.uint8)

    with pytest.raises(TypeError, match="pad_value must be a scalar"):
        list(iter_tiles(image, "bad-pad", tile_size=2, overlap=0.0, pad_value=pad_value))


@pytest.mark.parametrize("pad_value", ["1", np.datetime64("2026-06-20"), True])
def test_iter_tiles_rejects_non_numeric_or_boolean_scalar_pad_value(pad_value) -> None:
    image = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(TypeError, match="pad_value"):
        list(iter_tiles(image, "bad-pad", tile_size=2, overlap=0.0, pad_value=pad_value))


def test_iter_tiles_accepts_ordinary_float_conversion_to_float32() -> None:
    image = np.ones((1, 1), dtype=np.float32)

    tile = next(
        iter_tiles(image, "float-pad", tile_size=2, overlap=0.0, pad_value=0.1)
    )

    assert tile.image[1, 1] == np.float32(0.1)


def test_iter_tiles_rejects_finite_float_overflow() -> None:
    image = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="pad_value"):
        list(
            iter_tiles(
                image,
                "overflow-pad",
                tile_size=2,
                overlap=0.0,
                pad_value=np.finfo(np.float64).max,
            )
        )


def test_iter_tiles_accepts_negative_int16_pad_value() -> None:
    image = np.ones((1, 1), dtype=np.int16)

    tile = next(iter_tiles(image, "int-pad", tile_size=2, overlap=0.0, pad_value=-5))

    assert tile.image[1, 1] == -5


@pytest.mark.parametrize("pad_value", [math.nan, math.inf, -math.inf])
def test_iter_tiles_allows_non_finite_pad_value_for_float_dtype(pad_value) -> None:
    image = np.ones((1, 1), dtype=np.float32)

    tile = next(
        iter_tiles(image, "float-pad", tile_size=2, overlap=0.0, pad_value=pad_value)
    )

    padded = tile.image[1, 1]
    if math.isnan(pad_value):
        assert np.isnan(padded)
    else:
        assert padded == pad_value


def test_iter_tiles_rejects_complex_pad_value() -> None:
    image = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(TypeError, match="pad_value"):
        list(
            iter_tiles(
                image,
                "complex-pad",
                tile_size=2,
                overlap=0.0,
                pad_value=1 + 0j,
            )
        )


def test_iter_tiles_y_then_x_order_and_multichannel_metadata() -> None:
    image = np.arange(5 * 6 * 2, dtype=np.int16).reshape(5, 6, 2)

    tiles = list(iter_tiles(image, "scene", tile_size=3, overlap=0.5, pad_value=99))

    assert [tile.meta.tile_id for tile in tiles] == [
        "scene__x0_y0_s3",
        "scene__x2_y0_s3",
        "scene__x3_y0_s3",
        "scene__x0_y2_s3",
        "scene__x2_y2_s3",
        "scene__x3_y2_s3",
    ]
    assert [(tile.meta.x, tile.meta.y) for tile in tiles] == [
        (0, 0),
        (2, 0),
        (3, 0),
        (0, 2),
        (2, 2),
        (3, 2),
    ]
    assert [(tile.meta.valid_width, tile.meta.valid_height) for tile in tiles] == [
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
        (3, 3),
    ]
    assert all(tile.image.shape == (3, 3, 2) for tile in tiles)
    assert all(tile.image.dtype == image.dtype for tile in tiles)
    np.testing.assert_array_equal(tiles[0].image, image[:3, :3])


@pytest.mark.parametrize(
    "image",
    [
        np.ones((1, 1), dtype=np.bool_),
        np.ones((1, 1), dtype=np.int8),
        np.ones((1, 1), dtype=np.uint32),
        np.ones((1, 1), dtype=np.int32),
        np.ones((1, 1), dtype=np.uint64),
        np.ones((1, 1), dtype=np.int64),
        np.ones((1, 1), dtype=np.float16),
        np.ones((1, 1), dtype=np.longdouble),
        np.ones((1, 1), dtype=np.complex64),
        np.ones((1, 1), dtype=np.complex128),
        np.array([[1]], dtype=object),
        np.array([["1"]], dtype=np.str_),
        np.array([["2026-06-20"]], dtype="datetime64[D]"),
    ],
)
def test_iter_tiles_rejects_unsupported_image_dtype(image) -> None:
    with pytest.raises(TypeError, match="image dtype"):
        list(iter_tiles(image, "bad-dtype", tile_size=2, overlap=0.0))


@pytest.mark.parametrize(
    ("image", "tile_size", "overlap"),
    [
        (np.zeros((0, 5), dtype=np.uint8), 4, 0.0),
        (np.zeros((5, 0), dtype=np.uint8), 4, 0.0),
        (np.zeros((5,), dtype=np.uint8), 4, 0.0),
        (np.zeros((5, 5, 5, 5), dtype=np.uint8), 4, 0.0),
        (np.zeros((5, 5), dtype=np.uint8), 0, 0.0),
        (np.zeros((5, 5), dtype=np.uint8), -1, 0.0),
        (np.zeros((5, 5), dtype=np.uint8), 4, -0.1),
        (np.zeros((5, 5), dtype=np.uint8), 4, 1.0),
        (np.zeros((5, 5), dtype=np.uint8), 4, math.inf),
        (np.zeros((5, 5), dtype=np.uint8), 4, math.nan),
    ],
)
def test_iter_tiles_validates_image_tile_size_and_overlap(image, tile_size, overlap) -> None:
    with pytest.raises((TypeError, ValueError)):
        list(iter_tiles(image, "bad", tile_size=tile_size, overlap=overlap))
