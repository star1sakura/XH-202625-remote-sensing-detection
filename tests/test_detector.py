import math

import numpy as np
import pytest
import torch

from xh_detect.types import BoxPrediction

POLYGON = ((1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0))


class OomAboveTwoDetector:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        self.batch_sizes.append(len(images))
        assert confidence == 0.25
        if len(images) > 2:
            raise RuntimeError("CUDA out of memory while allocating tensor")
        return [
            [
                BoxPrediction(
                    class_id=int(image[0, 0, 0]),
                    score=float(image[0, 0, 0]),
                    polygon=POLYGON,
                )
            ]
            for image in images
        ]


class AlwaysCudaOomDetector:
    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        raise RuntimeError("CUDA out of memory")


class CpuOomDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        self.calls += 1
        raise RuntimeError("CPU out of memory")


class GenericRuntimeErrorDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        self.calls += 1
        raise RuntimeError("some other runtime failure")


class WrongLengthDetector:
    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        return []


class NeverCalledDetector:
    def __init__(self) -> None:
        self.called = False

    def predict(
        self, images: list[np.ndarray], confidence: float
    ) -> list[list[BoxPrediction]]:
        self.called = True
        return []


class FakeTensor:
    def __init__(self, values: object, name: str, calls: list[str]) -> None:
        self._values = np.asarray(values)
        self._name = name
        self._calls = calls

    def detach(self) -> "FakeTensor":
        self._calls.append(f"{self._name}.detach")
        return self

    def cpu(self) -> "FakeTensor":
        self._calls.append(f"{self._name}.cpu")
        return self

    def numpy(self) -> np.ndarray:
        self._calls.append(f"{self._name}.numpy")
        return self._values


class FakeObb:
    def __init__(self, polygons: object, classes: object, scores: object) -> None:
        self.tensor_calls: list[str] = []
        self.xyxyxyxy = FakeTensor(polygons, "polygon", self.tensor_calls)
        self.cls = FakeTensor(classes, "class", self.tensor_calls)
        self.conf = FakeTensor(scores, "score", self.tensor_calls)


class FakeResult:
    def __init__(self, obb: FakeObb | None) -> None:
        self.obb = obb


class FakeModel:
    def __init__(self, results: list[FakeResult]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return self.results


@pytest.mark.parametrize("model_path", ["", "   "])
def test_ultralytics_detector_rejects_empty_model_path(model_path: str) -> None:
    from xh_detect import detector as detector_module

    with pytest.raises(ValueError, match="model_path must be a non-empty string"):
        detector_module.UltralyticsOBBDetector(
            model_path=model_path,
            device="cpu",
            image_size=640,
            half=False,
        )


@pytest.mark.parametrize("device", ["", "   "])
def test_ultralytics_detector_rejects_empty_device(device: str) -> None:
    from xh_detect import detector as detector_module

    with pytest.raises(ValueError, match="device must be a non-empty string"):
        detector_module.UltralyticsOBBDetector(
            model_path="weights.pt",
            device=device,
            image_size=640,
            half=False,
        )


@pytest.mark.parametrize("image_size", [0, -1, True])
def test_ultralytics_detector_rejects_invalid_image_size(image_size: object) -> None:
    from xh_detect import detector as detector_module

    with pytest.raises(ValueError, match="image_size must be a positive integer"):
        detector_module.UltralyticsOBBDetector(
            model_path="weights.pt",
            device="cpu",
            image_size=image_size,  # type: ignore[arg-type]
            half=False,
        )


@pytest.mark.parametrize("half", [0, 1, "false"])
def test_ultralytics_detector_requires_bool_half(half: object) -> None:
    from xh_detect import detector as detector_module

    with pytest.raises(TypeError, match="half must be a bool"):
        detector_module.UltralyticsOBBDetector(
            model_path="weights.pt",
            device="cpu",
            image_size=640,
            half=half,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("initial_batch_size", [0, -1, True])
def test_predict_with_oom_backoff_rejects_invalid_batch_size(initial_batch_size: object) -> None:
    from xh_detect.detector import predict_with_oom_backoff

    with pytest.raises(ValueError, match="initial_batch_size must be a positive integer"):
        predict_with_oom_backoff(
            detector=NeverCalledDetector(),
            images=[np.zeros((4, 4, 3), dtype=np.uint8)],
            confidence=0.25,
            initial_batch_size=initial_batch_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1, math.inf, math.nan, True])
def test_predict_with_oom_backoff_rejects_invalid_confidence(confidence: object) -> None:
    from xh_detect.detector import predict_with_oom_backoff

    with pytest.raises(ValueError, match="confidence must be a finite real number in \\[0, 1\\]"):
        predict_with_oom_backoff(
            detector=NeverCalledDetector(),
            images=[np.zeros((4, 4, 3), dtype=np.uint8)],
            confidence=confidence,  # type: ignore[arg-type]
            initial_batch_size=1,
        )


def test_oom_backoff_retries_with_smaller_batches_and_preserves_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    empty_cache_calls: list[str] = []
    monkeypatch.setattr(
        detector_module.torch.cuda,
        "empty_cache",
        lambda: empty_cache_calls.append("called"),
    )

    detector = OomAboveTwoDetector()
    images = [
        np.full((8, 8, 3), fill_value=index, dtype=np.uint8)
        for index in range(5)
    ]

    results = detector_module.predict_with_oom_backoff(
        detector=detector,
        images=images,
        confidence=0.25,
        initial_batch_size=4,
    )

    assert len(results) == 5
    assert [len(item) for item in results] == [1, 1, 1, 1, 1]
    assert [item[0].class_id for item in results] == [0, 1, 2, 3, 4]
    assert detector.batch_sizes == [4, 2, 2, 1]
    assert empty_cache_calls == ["called"]


def test_oom_backoff_reraises_cuda_oom_when_batch_size_is_one() -> None:
    from xh_detect.detector import predict_with_oom_backoff

    images = [np.zeros((8, 8, 3), dtype=np.uint8)]

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        predict_with_oom_backoff(
            detector=AlwaysCudaOomDetector(),
            images=images,
            confidence=0.25,
            initial_batch_size=1,
        )


def test_oom_backoff_recognizes_torch_cuda_oom_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from xh_detect import detector as detector_module

    calls: list[str] = []
    monkeypatch.setattr(detector_module.torch.cuda, "empty_cache", lambda: calls.append("called"))

    class TorchCudaOomDetector:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def predict(
            self, images: list[np.ndarray], confidence: float
        ) -> list[list[BoxPrediction]]:
            self.batch_sizes.append(len(images))
            if len(images) > 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")
            return [[BoxPrediction(class_id=7, score=0.5, polygon=POLYGON)]]

    detector = TorchCudaOomDetector()

    results = detector_module.predict_with_oom_backoff(
        detector=detector,
        images=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
        confidence=0.25,
        initial_batch_size=2,
    )

    assert len(results) == 2
    assert detector.batch_sizes == [2, 1, 1]
    assert calls == ["called"]


def test_non_cuda_oom_runtime_error_does_not_backoff() -> None:
    from xh_detect.detector import predict_with_oom_backoff

    detector = CpuOomDetector()
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    with pytest.raises(RuntimeError, match="CPU out of memory"):
        predict_with_oom_backoff(
            detector=detector,
            images=images,
            confidence=0.25,
            initial_batch_size=3,
        )

    assert detector.calls == 1


def test_non_oom_runtime_error_is_reraised_without_backoff() -> None:
    from xh_detect.detector import predict_with_oom_backoff

    detector = GenericRuntimeErrorDetector()
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    with pytest.raises(RuntimeError, match="some other runtime failure"):
        predict_with_oom_backoff(
            detector=detector,
            images=images,
            confidence=0.25,
            initial_batch_size=3,
        )

    assert detector.calls == 1


def test_empty_input_returns_empty_without_calling_detector() -> None:
    from xh_detect.detector import predict_with_oom_backoff

    detector = NeverCalledDetector()

    assert predict_with_oom_backoff(
        detector=detector,
        images=[],
        confidence=0.25,
        initial_batch_size=4,
    ) == []
    assert detector.called is False


def test_predict_with_oom_backoff_rejects_result_length_mismatch() -> None:
    from xh_detect.detector import predict_with_oom_backoff

    with pytest.raises(
        ValueError,
        match="detector returned 0 results for a chunk of 2 images",
    ):
        predict_with_oom_backoff(
            detector=WrongLengthDetector(),
            images=[np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)],
            confidence=0.25,
            initial_batch_size=2,
        )


def test_ultralytics_detector_predict_passes_expected_kwargs_and_converts_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    images = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.ones((4, 4, 3), dtype=np.uint8),
    ]
    obb = FakeObb(
        polygons=[
            [[1, 2], [5, 2], [5, 6], [1, 6]],
            [[10.5, 20.25], [30.5, 20.25], [30.5, 40.75], [10.5, 40.75]],
        ],
        classes=[0.0, 2],
        scores=[0.75, 0.5],
    )
    fake_model = FakeModel([FakeResult(obb), FakeResult(None)])
    captured_paths: list[str] = []

    def fake_yolo(model_path: str) -> FakeModel:
        captured_paths.append(model_path)
        return fake_model

    monkeypatch.setattr(detector_module, "YOLO", fake_yolo)
    detector = detector_module.UltralyticsOBBDetector(
        model_path="weights.pt",
        device="cpu",
        image_size=768,
        half=False,
    )

    predictions = detector.predict(images, confidence=0.4)

    assert captured_paths == ["weights.pt"]
    assert fake_model.calls == [
        {
            "source": images,
            "imgsz": 768,
            "conf": 0.4,
            "device": "cpu",
            "half": False,
            "verbose": False,
        }
    ]
    assert obb.tensor_calls == [
        "polygon.detach",
        "polygon.cpu",
        "polygon.numpy",
        "class.detach",
        "class.cpu",
        "class.numpy",
        "score.detach",
        "score.cpu",
        "score.numpy",
    ]
    assert predictions == [
        [
            BoxPrediction(
                class_id=0,
                score=0.75,
                polygon=((1.0, 2.0), (5.0, 2.0), (5.0, 6.0), (1.0, 6.0)),
            ),
            BoxPrediction(
                class_id=2,
                score=0.5,
                polygon=((10.5, 20.25), (30.5, 20.25), (30.5, 40.75), (10.5, 40.75)),
            ),
        ],
        [],
    ]


def test_ultralytics_detector_empty_input_skips_model_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel([])
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    assert detector.predict([], confidence=0.25) == []
    assert fake_model.calls == []


@pytest.mark.parametrize("class_id", [1.9, -1, math.nan, math.inf, -math.inf])
def test_ultralytics_detector_rejects_invalid_class_ids(
    monkeypatch: pytest.MonkeyPatch,
    class_id: float,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel(
        [
            FakeResult(
                FakeObb(
                    polygons=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
                    classes=[class_id],
                    scores=[0.5],
                )
            )
        ]
    )
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(
        ValueError,
        match="result 0 has invalid OBB class at box 0",
    ):
        detector.predict([np.zeros((4, 4, 3), dtype=np.uint8)], confidence=0.25)


@pytest.mark.parametrize("score", [-0.1, 1.1, math.nan, math.inf, -math.inf])
def test_ultralytics_detector_rejects_invalid_scores(
    monkeypatch: pytest.MonkeyPatch,
    score: float,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel(
        [
            FakeResult(
                FakeObb(
                    polygons=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
                    classes=[0],
                    scores=[score],
                )
            )
        ]
    )
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(
        ValueError,
        match="result 0 has invalid OBB score at box 0",
    ):
        detector.predict([np.zeros((4, 4, 3), dtype=np.uint8)], confidence=0.25)


def test_ultralytics_detector_rejects_result_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel([FakeResult(None)])
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(
        ValueError,
        match="Ultralytics returned 1 results for 2 input images",
    ):
        detector.predict(
            [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)],
            confidence=0.25,
        )


def test_ultralytics_detector_rejects_invalid_polygon_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel(
        [FakeResult(FakeObb(polygons=[[[1, 2], [3, 4], [5, 6]]], classes=[0], scores=[0.5]))]
    )
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(ValueError, match="result 0 has invalid OBB polygon shape"):
        detector.predict([np.zeros((4, 4, 3), dtype=np.uint8)], confidence=0.25)


def test_ultralytics_detector_rejects_inconsistent_obb_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel(
        [
            FakeResult(
                FakeObb(
                    polygons=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
                    classes=[0, 1],
                    scores=[0.5],
                )
            )
        ]
    )
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(
        ValueError,
        match="result 0 has inconsistent OBB lengths: polygons=1, classes=2, scores=1",
    ):
        detector.predict([np.zeros((4, 4, 3), dtype=np.uint8)], confidence=0.25)


def test_ultralytics_detector_rejects_non_finite_polygon_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from xh_detect import detector as detector_module

    fake_model = FakeModel(
        [
            FakeResult(
                FakeObb(
                    polygons=[[[1, 2], [math.nan, 2], [3, 4], [1, 4]]],
                    classes=[0],
                    scores=[0.5],
                )
            )
        ]
    )
    monkeypatch.setattr(detector_module, "YOLO", lambda model_path: fake_model)
    detector = detector_module.UltralyticsOBBDetector("weights.pt", "cpu", 640, False)

    with pytest.raises(ValueError, match="result 0 contains non-finite OBB values"):
        detector.predict([np.zeros((4, 4, 3), dtype=np.uint8)], confidence=0.25)
