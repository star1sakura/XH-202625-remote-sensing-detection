from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xh_detect.config import PipelineConfig
from xh_detect.types import BoxPrediction, InferenceResult


def _box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _prediction(
    class_id: int = 0,
    score: float = 0.9,
    polygon: tuple[tuple[float, float], ...] = (
        (1.0, 1.0),
        (3.0, 1.0),
        (3.0, 3.0),
        (1.0, 3.0),
    ),
) -> BoxPrediction:
    return BoxPrediction(class_id=class_id, score=score, polygon=polygon)  # type: ignore[arg-type]


class RecordingDetector:
    def __init__(self, outputs: list[list[BoxPrediction]]) -> None:
        self.outputs = outputs
        self.calls: list[dict[str, object]] = []

    def predict(self, images: list[np.ndarray], confidence: float) -> list[list[BoxPrediction]]:
        self.calls.append(
            {
                "count": len(images),
                "confidence": confidence,
                "shapes": [image.shape for image in images],
            }
        )
        return self.outputs[: len(images)]


class PixelDrivenDetector:
    def __init__(self) -> None:
        self.calls: list[tuple[int, float]] = []

    def predict(self, images: list[np.ndarray], confidence: float) -> list[list[BoxPrediction]]:
        self.calls.append((len(images), confidence))
        results: list[list[BoxPrediction]] = []
        for image in images:
            class_id = int(image[0, 0, 0] % 3)
            results.append([_prediction(class_id=class_id, score=0.95)])
        return results


class FailingDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, images: list[np.ndarray], confidence: float) -> list[list[BoxPrediction]]:
        self.calls += 1
        raise RuntimeError("detector failed")


def test_inference_pipeline_runs_single_tile_and_reports_timings() -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0)
    detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    pipeline = InferencePipeline(detector=detector, config=config)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    result = pipeline.run(image, "scene-1")

    assert isinstance(result, InferenceResult)
    assert len(result.detections) == 1
    assert result.detections[0].image_id == "scene-1"
    assert result.detections[0].class_id == 2
    assert result.detections[0].score == 0.9
    assert result.detections[0].polygon == (
        (1.0, 1.0),
        (3.0, 1.0),
        (3.0, 3.0),
        (1.0, 3.0),
    )
    assert detector.calls == [{"count": 1, "confidence": 0.25, "shapes": [(4, 4, 3)]}]
    assert result.timings.preprocess_s >= 0.0
    assert result.timings.inference_s >= 0.0
    assert result.timings.postprocess_s >= 0.0
    assert result.timings.total_s + 1e-9 >= (
        result.timings.preprocess_s + result.timings.inference_s + result.timings.postprocess_s
    )


def test_inference_pipeline_merges_cross_tile_duplicates_in_stable_order() -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.5, batch_size=8, edge_margin=0)
    detector = RecordingDetector(
        [
            [
                _prediction(
                    class_id=1,
                    score=0.9,
                    polygon=((2.0, 1.0), (4.0, 1.0), (4.0, 3.0), (2.0, 3.0)),
                )
            ],
            [
                _prediction(
                    class_id=1,
                    score=0.8,
                    polygon=((0.0, 1.0), (2.0, 1.0), (2.0, 3.0), (0.0, 3.0)),
                )
            ],
        ]
    )
    pipeline = InferencePipeline(detector=detector, config=config)
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    result = pipeline.run(image, "scene-merge")

    assert [(item.class_id, item.score, item.polygon) for item in result.detections] == [
        (1, 0.9, ((2.0, 1.0), (4.0, 1.0), (4.0, 3.0), (2.0, 3.0)))
    ]


def test_inference_pipeline_filters_by_per_class_thresholds() -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(
        tile_size=4,
        overlap=0.0,
        batch_size=4,
        edge_margin=0,
        class_thresholds={0: 0.4, 1: 0.6, 2: 0.8},
    )
    detector = RecordingDetector(
        [
            [
                _prediction(class_id=0, score=0.39),
                _prediction(class_id=1, score=0.6),
                _prediction(class_id=2, score=0.79),
            ]
        ]
    )
    pipeline = InferencePipeline(detector=detector, config=config)

    result = pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), "threshold-scene")

    assert [(item.class_id, item.score) for item in result.detections] == [(1, 0.6)]
    assert detector.calls[0]["confidence"] == 0.4


def test_pipeline_applies_secondary_suppression_only_to_configured_ship() -> None:
    from xh_detect.pipeline import InferencePipeline
    from xh_detect.postprocess import SuppressionRule

    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        tile_size=20,
        overlap=0.0,
        batch_size=4,
        edge_margin=0,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
        class_suppression={3: SuppressionRule("iou", 0.20)},
    )
    detector = RecordingDetector(
        [
            [
                _prediction(class_id=3, score=0.95, polygon=((0, 0), (10, 0), (10, 10), (0, 10))),
                _prediction(class_id=3, score=0.90, polygon=((6, 0), (16, 0), (16, 10), (6, 10))),
                _prediction(
                    class_id=24,
                    score=0.85,
                    polygon=((0, 10), (10, 10), (10, 20), (0, 20)),
                ),
                _prediction(
                    class_id=24,
                    score=0.80,
                    polygon=((6, 10), (16, 10), (16, 20), (6, 20)),
                ),
            ]
        ]
    )

    result = InferencePipeline(detector, config).run(
        np.zeros((20, 20, 3), dtype=np.uint8),
        "ship-secondary",
    )

    assert [(item.class_id, item.score) for item in result.detections] == [
        (3, 0.95),
        (24, 0.85),
        (24, 0.80),
    ]


def test_pipeline_applies_low_score_area_filter_after_merge() -> None:
    from xh_detect.pipeline import InferencePipeline
    from xh_detect.postprocess import LowScoreAreaRule

    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        tile_size=40,
        overlap=0.0,
        batch_size=4,
        edge_margin=0,
        merge_iou=1.0,
        class_thresholds={class_id: 0.19 for class_id in range(25)},
        class_low_score_area_filters={24: LowScoreAreaRule(0.21, 700)},
    )
    detector = RecordingDetector(
        [
            [
                _prediction(class_id=24, score=0.20, polygon=_box(0, 0, 20, 20)),
                _prediction(class_id=24, score=0.20, polygon=_box(0, 0, 35, 20)),
                _prediction(class_id=24, score=0.22, polygon=_box(20, 0, 30, 10)),
                _prediction(class_id=3, score=0.20, polygon=_box(20, 20, 30, 30)),
            ]
        ]
    )

    result = InferencePipeline(detector, config).run(
        np.zeros((40, 40, 3), dtype=np.uint8),
        "area-filter",
    )

    assert [(item.class_id, item.score, item.polygon) for item in result.detections] == [
        (24, 0.22, _box(20, 0, 30, 10)),
        (24, 0.20, _box(0, 0, 35, 20)),
        (3, 0.20, _box(20, 20, 30, 30)),
    ]


def test_inference_pipeline_rejects_detector_class_outside_taxonomy() -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(
        tile_size=4,
        overlap=0.0,
        batch_size=4,
        edge_margin=0,
        class_thresholds={0: 0.4, 1: 0.6, 2: 0.8},
    )
    detector = RecordingDetector([[_prediction(class_id=24, score=0.99)]])
    pipeline = InferencePipeline(detector=detector, config=config)

    with pytest.raises(ValueError, match="class_id"):
        pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), "threshold-scene")


def test_pipeline_accepts_class_24_when_configured_for_xh25() -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        device="cpu",
        half=False,
        tile_size=64,
        overlap=0.0,
        edge_margin=0,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    pipeline = InferencePipeline(
        RecordingDetector([[_prediction(class_id=24, score=0.9)]]),
        config,
        cache_root=None,
    )

    result = pipeline.run(np.zeros((64, 64, 3), dtype=np.uint8), "vehicle")

    assert result.detections[0].class_id == 24


def test_inference_pipeline_second_run_hits_cache_without_detector_call(
    tmp_path: Path,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    first_detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    first_pipeline = InferencePipeline(
        detector=first_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    expected_result = first_pipeline.run(image, "scene-cache")

    second_detector = FailingDetector()
    second_pipeline = InferencePipeline(
        detector=second_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    result = second_pipeline.run(image, "scene-cache")

    assert result.detections == expected_result.detections
    assert second_detector.calls == 0


def test_inference_pipeline_only_predicts_missing_tiles_when_cache_is_partial(
    tmp_path: Path,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.5, batch_size=8, edge_margin=0)
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[:, 2:, :] = 2

    first_detector = PixelDrivenDetector()
    first_pipeline = InferencePipeline(
        detector=first_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    expected_result = first_pipeline.run(image, "scene-partial")

    namespace_root = next((tmp_path / "cache").iterdir())
    cached_files = sorted(namespace_root.glob("*.json"))
    assert len(cached_files) == 2
    cached_files[0].unlink()

    second_detector = PixelDrivenDetector()
    second_pipeline = InferencePipeline(
        detector=second_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    result = second_pipeline.run(image, "scene-partial")

    assert result.detections == expected_result.detections
    assert second_detector.calls == [(1, 0.25)]


def test_inference_pipeline_cache_key_includes_image_fingerprint(tmp_path: Path) -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0)
    detector = PixelDrivenDetector()
    pipeline = InferencePipeline(detector=detector, config=config, cache_root=tmp_path / "cache")

    image_a = np.zeros((4, 4, 3), dtype=np.uint8)
    image_b = np.full((4, 4, 3), fill_value=1, dtype=np.uint8)

    result_a = pipeline.run(image_a, "same-id")
    result_b = pipeline.run(image_b, "same-id")

    assert [item.class_id for item in result_a.detections] == [0]
    assert [item.class_id for item in result_b.detections] == [1]
    assert detector.calls == [(1, 0.25), (1, 0.25)]


def test_cache_namespace_is_stable_and_changes_with_config_or_model_metadata(
    tmp_path: Path,
) -> None:
    from xh_detect.pipeline import _cache_namespace

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"abc")
    config = PipelineConfig(model_path=str(model_path))
    same_config = PipelineConfig(model_path=str(model_path))

    namespace_a = _cache_namespace(config)
    namespace_b = _cache_namespace(same_config)

    assert namespace_a == namespace_b

    different_thresholds = PipelineConfig(
        model_path=str(model_path),
        class_thresholds={0: 0.2, 1: 0.25, 2: 0.25},
    )
    assert _cache_namespace(different_thresholds) != namespace_a

    model_path.write_bytes(b"abcdef")
    assert _cache_namespace(config) != namespace_a


def test_inference_pipeline_recomputes_when_cache_payload_is_corrupt(
    tmp_path: Path,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    config = PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    first_detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    first_pipeline = InferencePipeline(
        detector=first_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    expected_result = first_pipeline.run(image, "scene-corrupt")

    namespace_root = next((tmp_path / "cache").iterdir())
    cached_file = next(namespace_root.glob("*.json"))
    cached_file.write_text("{", encoding="utf-8")

    second_detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    second_pipeline = InferencePipeline(
        detector=second_detector,
        config=config,
        cache_root=tmp_path / "cache",
    )
    result = second_pipeline.run(image, "scene-corrupt")

    assert result.detections == expected_result.detections
    assert second_detector.calls == [{"count": 1, "confidence": 0.25, "shapes": [(4, 4, 3)]}]


def test_inference_pipeline_warns_and_returns_result_when_cache_save_has_oserror(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    pipeline = InferencePipeline(
        detector=detector,
        config=PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0),
        cache_root=tmp_path / "cache",
    )
    assert pipeline.cache is not None
    attempted_keys: list[str] = []

    def failing_save(cache_key: str, predictions: list[BoxPrediction]) -> None:
        attempted_keys.append(cache_key)
        raise PermissionError("cache write denied")

    monkeypatch.setattr(pipeline.cache, "save", failing_save)

    with pytest.warns(RuntimeWarning) as warning_records:
        result = pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), "save-failure")

    assert isinstance(result, InferenceResult)
    assert [(item.class_id, item.score) for item in result.detections] == [(2, 0.9)]
    assert len(attempted_keys) == 1
    warning_message = str(warning_records[0].message)
    assert attempted_keys[0] in warning_message
    assert "PermissionError" in warning_message
    assert "cache write denied" in warning_message
    assert detector.calls == [{"count": 1, "confidence": 0.25, "shapes": [(4, 4, 3)]}]


def test_inference_pipeline_propagates_non_oserror_from_cache_save(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    detector = RecordingDetector([[_prediction(class_id=2, score=0.9)]])
    pipeline = InferencePipeline(
        detector=detector,
        config=PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0),
        cache_root=tmp_path / "cache",
    )
    assert pipeline.cache is not None

    def failing_save(cache_key: str, predictions: list[BoxPrediction]) -> None:
        raise ValueError("programming error")

    monkeypatch.setattr(pipeline.cache, "save", failing_save)

    with pytest.raises(ValueError, match="programming error"):
        pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), "save-value-error")

    assert detector.calls == [{"count": 1, "confidence": 0.25, "shapes": [(4, 4, 3)]}]


def test_inference_pipeline_run_returns_inference_result_dataclass() -> None:
    from xh_detect.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        detector=RecordingDetector([[_prediction(class_id=1, score=0.9)]]),
        config=PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0),
    )

    result = pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), "shape-scene")

    assert isinstance(result, InferenceResult)
    assert hasattr(result, "detections")
    assert hasattr(result, "timings")


@pytest.mark.parametrize("image_id", ["", "   "])
def test_inference_pipeline_rejects_empty_image_id(image_id: str) -> None:
    from xh_detect.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        detector=RecordingDetector([[]]),
        config=PipelineConfig(),
    )

    with pytest.raises(ValueError, match="image_id must be a non-empty string"):
        pipeline.run(np.zeros((4, 4, 3), dtype=np.uint8), image_id)


def test_inference_pipeline_propagates_tiling_validation_errors() -> None:
    from xh_detect.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        detector=RecordingDetector([[]]),
        config=PipelineConfig(),
    )

    with pytest.raises(ValueError, match="image height and width must be positive"):
        pipeline.run(np.zeros((0, 4, 3), dtype=np.uint8), "bad-image")


def test_inference_pipeline_does_not_modify_input_array() -> None:
    from xh_detect.pipeline import InferencePipeline

    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    original = image.copy()
    pipeline = InferencePipeline(
        detector=RecordingDetector([[_prediction(class_id=1, score=0.9)]]),
        config=PipelineConfig(tile_size=4, overlap=0.0, batch_size=2, edge_margin=0),
    )

    pipeline.run(image, "immutable-scene")

    np.testing.assert_array_equal(image, original)


def test_inference_pipeline_does_not_write_cache_when_inference_fails(
    tmp_path: Path,
) -> None:
    from xh_detect.pipeline import InferencePipeline

    pipeline = InferencePipeline(
        detector=FailingDetector(),
        config=PipelineConfig(tile_size=4, overlap=0.5, batch_size=8, edge_margin=0),
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(RuntimeError, match="detector failed"):
        pipeline.run(np.zeros((4, 6, 3), dtype=np.uint8), "scene-fail")

    namespace_root = next((tmp_path / "cache").iterdir())
    assert list(namespace_root.iterdir()) == []
