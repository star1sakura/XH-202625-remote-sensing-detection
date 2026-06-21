from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Point = tuple[float, float]
Polygon4 = tuple[Point, Point, Point, Point]
ImageArray = NDArray[np.uint8 | np.uint16 | np.int16 | np.float32 | np.float64]


@dataclass(frozen=True)
class ObjectAnnotation:
    image_id: str
    class_id: int
    polygon: Polygon4
    difficult: bool = False


@dataclass(frozen=True)
class BoxPrediction:
    class_id: int
    score: float
    polygon: Polygon4


@dataclass(frozen=True)
class Detection:
    image_id: str
    class_id: int
    score: float
    polygon: Polygon4


@dataclass(frozen=True)
class TileMeta:
    image_id: str
    tile_id: str
    x: int
    y: int
    width: int
    height: int
    valid_width: int
    valid_height: int


@dataclass(frozen=True)
class Tile:
    image: ImageArray
    meta: TileMeta


@dataclass(frozen=True)
class StageTimings:
    preprocess_s: float
    inference_s: float
    postprocess_s: float
    total_s: float


@dataclass(frozen=True)
class InferenceResult:
    detections: tuple[Detection, ...]
    timings: StageTimings
