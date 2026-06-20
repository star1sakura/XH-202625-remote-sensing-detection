from __future__ import annotations

import math
from typing import get_args, get_type_hints

import numpy as np
import pytest

from xh_detect.tiling import axis_positions, iter_tiles
from xh_detect.types import ImageArray, Tile


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


def test_image_array_contract_accepts_non_uint8_dtypes() -> None:
    assert get_args(ImageArray)[1] == np.dtype[np.generic]
    assert get_type_hints(Tile)["image"] == ImageArray


def test_iter_tiles_public_annotation_uses_image_array_not_object() -> None:
    assert get_type_hints(iter_tiles)["image"] == ImageArray


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
