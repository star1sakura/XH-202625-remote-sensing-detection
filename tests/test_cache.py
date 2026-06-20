from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from xh_detect.types import BoxPrediction

PathLikeType = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def _prediction(
    class_id: int = 1,
    score: float = 0.375,
    polygon: tuple[tuple[float, float], ...] = (
        (1.25, 2.5),
        (3.75, 2.5),
        (3.75, 4.5),
        (1.25, 4.5),
    ),
) -> BoxPrediction:
    return BoxPrediction(class_id=class_id, score=score, polygon=polygon)  # type: ignore[arg-type]


def test_tile_prediction_cache_roundtrips_predictions_and_preserves_order(
    tmp_path: Path,
) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    predictions = [
        _prediction(class_id=2, score=0.95),
        _prediction(
            class_id=0,
            score=0.125,
            polygon=((9.0, 1.0), (10.0, 1.0), (10.0, 2.0), (9.0, 2.0)),
        ),
    ]

    cache.save("scene/切片:?.png", predictions)

    assert cache.load("scene/切片:?.png") == predictions


def test_tile_prediction_cache_roundtrips_empty_predictions(tmp_path: Path) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    cache.save("empty-tile", [])

    assert cache.load("empty-tile") == []


def test_tile_prediction_cache_uses_hashed_filenames_inside_root(tmp_path: Path) -> None:
    from xh_detect.cache import TilePredictionCache

    cache_root = tmp_path / "cache"
    cache = TilePredictionCache(cache_root)
    tile_id = "../odd/子目录\\:*?tile"

    cache.save(tile_id, [_prediction()])

    entries = list(cache_root.iterdir())
    assert len(entries) == 1
    entry = entries[0]
    assert entry.parent == cache_root
    assert entry.name.endswith(".json")
    assert tile_id not in entry.name
    assert ".." not in entry.name
    assert len(entry.stem) == 64
    int(entry.stem, 16)


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        "",
        "[]",
        json.dumps({"version": 2, "tile_id": "tile", "predictions": []}),
        json.dumps({"version": 1, "predictions": []}),
        json.dumps({"version": 1, "tile_id": "other", "predictions": []}),
        json.dumps({"version": 1, "tile_id": "tile", "predictions": {}}),
        json.dumps(
            {
                "version": 1,
                "tile_id": "tile",
                "predictions": [
                    {"class_id": True, "score": 0.5, "polygon": [[0, 0]] * 4}
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "tile_id": "tile",
                "predictions": [
                    {"class_id": 1, "score": 1.5, "polygon": [[0, 0]] * 4}
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "tile_id": "tile",
                "predictions": [
                    {
                        "class_id": 1,
                        "score": 0.5,
                        "polygon": [[0, 0], [1, 1], [2, 2]],
                    }
                ],
            }
        ),
        json.dumps(
            {
                "version": 1,
                "tile_id": "tile",
                "predictions": [
                    {
                        "class_id": 1,
                        "score": 0.5,
                        "polygon": [[0, 0], [1, 1], [2, 2], [float("nan"), 3]],
                    }
                ],
            }
        ),
    ],
)
def test_tile_prediction_cache_treats_corrupt_or_invalid_payload_as_miss(
    tmp_path: Path,
    payload: str,
) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    path = cache._path_for("tile")
    path.write_text(payload, encoding="utf-8")

    assert cache.load("tile") is None


@pytest.mark.parametrize(
    "payload",
    [
        {"version": 1, "tile_id": "tile"},
        {"version": 1, "tile_id": "tile", "predictions": [], "extra": True},
        {
            "version": 1,
            "tile_id": "tile",
            "predictions": [{"class_id": 1, "score": 0.5}],
        },
        {
            "version": 1,
            "tile_id": "tile",
            "predictions": [
                {
                    "class_id": 1,
                    "score": 0.5,
                    "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
                    "extra": True,
                }
            ],
        },
    ],
)
def test_tile_prediction_cache_requires_exact_payload_and_prediction_fields(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    cache._path_for("tile").write_text(json.dumps(payload), encoding="utf-8")

    assert cache.load("tile") is None


def test_tile_prediction_cache_save_rejects_invalid_predictions(tmp_path: Path) -> None:
    from xh_detect.cache import TilePredictionCache

    cache_root = tmp_path / "cache"
    cache = TilePredictionCache(cache_root)
    invalid = [
        BoxPrediction(  # type: ignore[arg-type]
            class_id=-1,
            score=0.5,
            polygon=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        )
    ]

    with pytest.raises(ValueError, match="class_id"):
        cache.save("bad-tile", invalid)

    assert list(cache_root.iterdir()) == []


def test_tile_prediction_cache_uses_atomic_replace_in_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    tile_id = "atomic-tile"
    cache.save(tile_id, [_prediction(score=0.25)])
    target = cache._path_for(tile_id)
    original_text = target.read_text(encoding="utf-8")
    original_replace = os.replace
    replace_calls: list[tuple[Path, Path]] = []

    def spy_replace(src: PathLikeType, dst: PathLikeType) -> None:
        src_path = Path(src)
        dst_path = Path(dst)
        replace_calls.append((src_path, dst_path))
        assert src_path.parent == dst_path.parent == cache.root
        assert src_path != dst_path
        assert src_path.exists()
        assert '"score":0.75' in src_path.read_text(encoding="utf-8")
        assert dst_path.read_text(encoding="utf-8") == original_text
        original_replace(src_path, dst_path)

    monkeypatch.setattr(os, "replace", spy_replace)

    cache.save(tile_id, [_prediction(score=0.75)])

    assert len(replace_calls) == 1
    assert target.read_text(encoding="utf-8") != original_text
    assert list(cache.root.glob("*.tmp")) == []


def test_tile_prediction_cache_cleans_up_temp_file_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    temp_candidates: list[Path] = []

    def failing_replace(src: PathLikeType, dst: PathLikeType) -> None:
        temp_candidates.append(Path(src))
        raise PermissionError("replace denied")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(PermissionError, match="replace denied"):
        cache.save("tile", [_prediction()])

    assert len(temp_candidates) == 1
    assert temp_candidates[0].exists() is False
    assert list(cache.root.iterdir()) == []


def test_tile_prediction_cache_preserves_int64_max_class_id(tmp_path: Path) -> None:
    from xh_detect.cache import TilePredictionCache

    cache = TilePredictionCache(tmp_path / "cache")
    max_class_id = int(np.iinfo(np.int64).max)
    predictions = [_prediction(class_id=max_class_id)]

    cache.save("tile", predictions)

    assert cache.load("tile") == predictions
