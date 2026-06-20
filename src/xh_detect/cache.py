from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np

from xh_detect.types import BoxPrediction, Polygon4

_CACHE_VERSION = 1
_INT64_MAX = int(np.iinfo(np.int64).max)


def _validate_tile_id(tile_id: object) -> str:
    if not isinstance(tile_id, str) or not tile_id:
        raise ValueError("tile_id must be a non-empty string")
    return tile_id


def _validate_class_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("class_id must be a non-boolean int")
    if value < 0 or value > _INT64_MAX:
        raise ValueError("class_id must be in [0, int64 max]")
    return value


def _validate_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("score must be a finite real number in [0, 1]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be a finite real number in [0, 1]")
    return score


def _validate_coordinate(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("polygon coordinates must be finite real numbers")
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError("polygon coordinates must be finite real numbers")
    return numeric_value


def _validate_polygon(value: object) -> Polygon4:
    if not isinstance(value, Sequence) or len(value) != 4:
        raise ValueError("polygon must contain exactly four points")

    points: list[tuple[float, float]] = []
    for point in value:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("polygon points must be length-2 sequences")
        x = _validate_coordinate(point[0])
        y = _validate_coordinate(point[1])
        points.append((x, y))
    return cast(Polygon4, tuple(points))


def _validate_prediction(prediction: object) -> BoxPrediction:
    if isinstance(prediction, BoxPrediction):
        class_id = _validate_class_id(prediction.class_id)
        score = _validate_score(prediction.score)
        polygon = _validate_polygon(prediction.polygon)
        return BoxPrediction(class_id=class_id, score=score, polygon=polygon)

    if not isinstance(prediction, Mapping):
        raise ValueError("prediction must be a mapping")

    class_id = _validate_class_id(prediction.get("class_id"))
    score = _validate_score(prediction.get("score"))
    polygon = _validate_polygon(prediction.get("polygon"))
    return BoxPrediction(class_id=class_id, score=score, polygon=polygon)


def _payload_from_predictions(
    tile_id: str,
    predictions: Sequence[BoxPrediction],
) -> dict[str, object]:
    serialized_predictions = [
        {
            "class_id": prediction.class_id,
            "score": prediction.score,
            "polygon": [[x, y] for x, y in prediction.polygon],
        }
        for prediction in (_validate_prediction(prediction) for prediction in predictions)
    ]
    return {
        "version": _CACHE_VERSION,
        "tile_id": tile_id,
        "predictions": serialized_predictions,
    }


def _deserialize_payload(payload: object, *, expected_tile_id: str) -> list[BoxPrediction]:
    if not isinstance(payload, Mapping):
        raise ValueError("cache payload must be a mapping")

    version = payload.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("cache version must be an int")
    if version != _CACHE_VERSION:
        raise ValueError("unsupported cache version")

    tile_id = payload.get("tile_id")
    if not isinstance(tile_id, str):
        raise ValueError("cache tile_id must be a string")
    if tile_id != expected_tile_id:
        raise ValueError("cache tile_id mismatch")

    raw_predictions = payload.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("cache predictions must be a list")

    return [_validate_prediction(item) for item in raw_predictions]


class TilePredictionCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path_for(self, tile_id: str) -> Path:
        normalized_tile_id = _validate_tile_id(tile_id)
        digest = hashlib.sha256(normalized_tile_id.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def save(self, tile_id: str, predictions: Sequence[BoxPrediction]) -> None:
        normalized_tile_id = _validate_tile_id(tile_id)
        payload = _payload_from_predictions(normalized_tile_id, predictions)
        payload_text = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

        target_path = self._path_for(normalized_tile_id)
        fd, temp_name = tempfile.mkstemp(
            dir=self.root,
            prefix=f"{target_path.stem}.",
            suffix=".tmp",
            text=True,
        )
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
                handle.write(payload_text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        else:
            if temp_path.exists():
                temp_path.unlink()

    def load(self, tile_id: str) -> list[BoxPrediction] | None:
        normalized_tile_id = _validate_tile_id(tile_id)
        path = self._path_for(normalized_tile_id)
        try:
            payload_text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        try:
            payload = json.loads(payload_text)
            return _deserialize_payload(payload, expected_tile_id=normalized_tile_id)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
