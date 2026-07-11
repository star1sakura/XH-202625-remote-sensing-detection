from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import cast

import numpy as np

from xh_detect.cache import TilePredictionCache
from xh_detect.config import PipelineConfig
from xh_detect.detector import Detector, predict_with_oom_backoff
from xh_detect.merge import keep_tile_prediction, merge_detections, project_prediction
from xh_detect.postprocess import suppress_class_detections
from xh_detect.tiling import iter_tiles
from xh_detect.types import (
    BoxPrediction,
    Detection,
    ImageArray,
    InferenceResult,
    StageTimings,
    Tile,
)


def _validate_non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _canonical_json_digest(payload: object, *, length: int = 16) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _image_fingerprint(image: ImageArray) -> str:
    array = np.asarray(image)
    contiguous = array if array.flags.c_contiguous else np.ascontiguousarray(array)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(repr(tuple(contiguous.shape)).encode("utf-8"))
    hasher.update(memoryview(contiguous).cast("B"))
    return hasher.hexdigest()


def _cache_key(tile_id: str, image_fingerprint: str) -> str:
    return f"{tile_id}::img={image_fingerprint}"


def _model_metadata(model_path: str) -> dict[str, object]:
    model_file = Path(model_path)
    metadata: dict[str, object] = {
        "configured_path": model_path,
    }
    if model_file.exists():
        resolved = model_file.resolve()
        stat = resolved.stat()
        metadata.update(
            {
                "resolved_path": str(resolved),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return metadata


def _cache_namespace(config: PipelineConfig) -> str:
    payload = {
        "config": config.to_dict(),
        "model": _model_metadata(config.model_path),
    }
    return _canonical_json_digest(payload)


def _filter_predictions(
    predictions: list[BoxPrediction],
    *,
    class_thresholds: dict[int, float],
    valid_class_ids: frozenset[int],
) -> list[BoxPrediction]:
    filtered: list[BoxPrediction] = []
    for prediction in predictions:
        if prediction.class_id not in valid_class_ids:
            raise ValueError(
                f"detector returned class_id {prediction.class_id}; "
                f"expected one of {sorted(valid_class_ids)}"
            )
        threshold = class_thresholds.get(prediction.class_id)
        if threshold is None or prediction.score < threshold:
            continue
        filtered.append(prediction)
    return filtered


class InferencePipeline:
    def __init__(
        self,
        detector: Detector,
        config: PipelineConfig,
        cache_root: Path | None = None,
    ) -> None:
        self.detector = detector
        self.config = config
        self._class_thresholds = dict(config.class_thresholds)
        self._valid_class_ids = config.valid_class_ids
        self._confidence = min(self._class_thresholds.values())
        self.cache = (
            TilePredictionCache(Path(cache_root) / _cache_namespace(config))
            if cache_root is not None
            else None
        )

    def run(self, image: ImageArray, image_id: str) -> InferenceResult:
        normalized_image_id = _validate_non_empty_string(image_id, "image_id")

        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        tiles = list(
            iter_tiles(
                image,
                normalized_image_id,
                tile_size=self.config.tile_size,
                overlap=self.config.overlap,
            )
        )
        image_fingerprint = _image_fingerprint(cast(ImageArray, np.asarray(image)))

        tile_predictions: list[list[BoxPrediction] | None] = [None] * len(tiles)
        missing_tiles: list[Tile] = []
        missing_indexes: list[int] = []
        missing_cache_keys: list[str] = []

        for index, tile in enumerate(tiles):
            cache_key = _cache_key(tile.meta.tile_id, image_fingerprint)
            cached_predictions = self.cache.load(cache_key) if self.cache is not None else None
            if cached_predictions is None:
                missing_tiles.append(tile)
                missing_indexes.append(index)
                missing_cache_keys.append(cache_key)
                continue
            tile_predictions[index] = cached_predictions
        preprocess_s = time.perf_counter() - preprocess_start

        inference_start = time.perf_counter()
        missing_predictions: list[list[BoxPrediction]] = []
        if missing_tiles:
            raw_predictions = predict_with_oom_backoff(
                detector=self.detector,
                images=[tile.image for tile in missing_tiles],
                confidence=self._confidence,
                initial_batch_size=self.config.batch_size,
            )
            missing_predictions = [
                _filter_predictions(
                    predictions,
                    class_thresholds=self._class_thresholds,
                    valid_class_ids=self._valid_class_ids,
                )
                for predictions in raw_predictions
            ]
        inference_s = time.perf_counter() - inference_start

        postprocess_start = time.perf_counter()
        if missing_predictions:
            for cache_key, index, predictions in zip(
                missing_cache_keys,
                missing_indexes,
                missing_predictions,
                strict=True,
            ):
                tile_predictions[index] = predictions
                if self.cache is not None:
                    try:
                        self.cache.save(cache_key, predictions)
                    except OSError as exc:
                        warnings.warn(
                            f"failed to save cache key {cache_key!r}: {type(exc).__name__}: {exc}",
                            RuntimeWarning,
                            stacklevel=2,
                        )

        detections: list[Detection] = []
        image_height, image_width = np.asarray(image).shape[:2]
        for tile, predictions in zip(tiles, tile_predictions, strict=True):
            local_predictions = [] if predictions is None else predictions
            for prediction in local_predictions:
                if not keep_tile_prediction(
                    prediction,
                    tile.meta,
                    image_width=image_width,
                    image_height=image_height,
                    margin=self.config.edge_margin,
                ):
                    continue
                detections.append(
                    project_prediction(
                        prediction,
                        tile.meta,
                        image_width=image_width,
                        image_height=image_height,
                    )
                )

        merged = merge_detections(detections, iou_threshold=self.config.merge_iou)
        final_detections = tuple(suppress_class_detections(merged, self.config.class_suppression))
        postprocess_s = time.perf_counter() - postprocess_start

        total_elapsed = time.perf_counter() - total_start
        measured_sum = preprocess_s + inference_s + postprocess_s
        timings = StageTimings(
            preprocess_s=preprocess_s,
            inference_s=inference_s,
            postprocess_s=postprocess_s,
            total_s=max(total_elapsed, measured_sum),
        )
        return InferenceResult(detections=final_detections, timings=timings)
