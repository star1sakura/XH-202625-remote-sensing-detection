from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import yaml

from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.taxonomy import Taxonomy
from xh_detect.types import Detection


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return normalized


def _non_negative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


@dataclass(frozen=True)
class SameWeightMultiscalePolicy:
    class_thresholds: Mapping[int, float]
    aircraft_supplement_threshold: float = 0.75
    aircraft_duplicate_iou: float = 0.30
    ship_supplement_threshold: float = 0.56
    ship_duplicate_iou: float = 0.70
    vehicle_score_ceiling: float = 0.21
    vehicle_min_area: float = 700.0

    def __post_init__(self) -> None:
        if not isinstance(self.class_thresholds, Mapping):
            raise TypeError("class_thresholds must be a mapping")
        thresholds: dict[int, float] = {}
        for class_id, threshold in self.class_thresholds.items():
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise TypeError("class threshold IDs must be integers")
            thresholds[class_id] = _probability(
                threshold,
                f"class threshold {class_id}",
            )
        object.__setattr__(self, "class_thresholds", MappingProxyType(thresholds))
        for name in (
            "aircraft_supplement_threshold",
            "aircraft_duplicate_iou",
            "ship_supplement_threshold",
            "ship_duplicate_iou",
            "vehicle_score_ceiling",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        object.__setattr__(
            self,
            "vehicle_min_area",
            _non_negative(self.vehicle_min_area, "vehicle_min_area"),
        )


def load_same_weight_multiscale_policy(path: Path) -> SameWeightMultiscalePolicy:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("same-weight multiscale policy must be a mapping")
    thresholds = payload.get("class_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("same-weight multiscale policy must define class_thresholds")
    try:
        normalized_thresholds = {int(class_id): value for class_id, value in thresholds.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("class threshold IDs must be integers") from exc
    return SameWeightMultiscalePolicy(
        class_thresholds=normalized_thresholds,
        aircraft_supplement_threshold=payload.get(
            "aircraft_supplement_threshold",
            0.75,
        ),
        aircraft_duplicate_iou=payload.get("aircraft_duplicate_iou", 0.30),
        ship_supplement_threshold=payload.get("ship_supplement_threshold", 0.56),
        ship_duplicate_iou=payload.get("ship_duplicate_iou", 0.70),
        vehicle_score_ceiling=payload.get("vehicle_score_ceiling", 0.21),
        vehicle_min_area=payload.get("vehicle_min_area", 700.0),
    )


def _select_primary(
    detections: Iterable[Detection],
    *,
    coarse_name: str,
    taxonomy: Taxonomy,
    thresholds: Mapping[int, float],
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if taxonomy.coarse_name(detection.class_id) == coarse_name
        and detection.score >= thresholds[detection.class_id]
    ]


def _add_supplements(
    primary: list[Detection],
    supplements: Iterable[Detection],
    *,
    coarse_name: str,
    threshold: float,
    duplicate_iou: float,
    taxonomy: Taxonomy,
) -> list[Detection]:
    selected = list(primary)
    selected_by_image: dict[str, list[Detection]] = {}
    for detection in selected:
        selected_by_image.setdefault(detection.image_id, []).append(detection)

    candidates = sorted(
        (
            detection
            for detection in supplements
            if taxonomy.coarse_name(detection.class_id) == coarse_name
            and detection.score >= threshold
        ),
        key=lambda detection: -detection.score,
    )
    for candidate in candidates:
        candidate_hbb = obb_to_hbb(candidate.polygon)
        existing = selected_by_image.get(candidate.image_id, [])
        if any(
            hbb_iou(candidate_hbb, obb_to_hbb(detection.polygon)) >= duplicate_iou
            for detection in existing
        ):
            continue
        selected.append(candidate)
        selected_by_image.setdefault(candidate.image_id, []).append(candidate)
    return selected


def _filter_vehicle_area(
    detections: Iterable[Detection],
    *,
    score_ceiling: float,
    min_area: float,
) -> list[Detection]:
    selected: list[Detection] = []
    for detection in detections:
        xmin, ymin, xmax, ymax = obb_to_hbb(detection.polygon)
        area = (xmax - xmin) * (ymax - ymin)
        if detection.score < score_ceiling and area < min_area:
            continue
        selected.append(detection)
    return selected


def fuse_same_weight_multiscale(
    *,
    predictions_1024: Iterable[Detection],
    predictions_1280: Iterable[Detection],
    predictions_1536: Iterable[Detection],
    taxonomy: Taxonomy,
    policy: SameWeightMultiscalePolicy,
) -> list[Detection]:
    if not isinstance(policy, SameWeightMultiscalePolicy):
        raise TypeError("policy must be a SameWeightMultiscalePolicy")
    if set(policy.class_thresholds) != set(taxonomy.valid_ids):
        raise ValueError("class_thresholds must define exactly the taxonomy class IDs")

    detections_1024 = list(predictions_1024)
    detections_1280 = list(predictions_1280)
    detections_1536 = list(predictions_1536)

    aircraft = _select_primary(
        detections_1024,
        coarse_name="aircraft",
        taxonomy=taxonomy,
        thresholds=policy.class_thresholds,
    )
    aircraft = _add_supplements(
        aircraft,
        detections_1280,
        coarse_name="aircraft",
        threshold=policy.aircraft_supplement_threshold,
        duplicate_iou=policy.aircraft_duplicate_iou,
        taxonomy=taxonomy,
    )

    ship = _select_primary(
        detections_1536,
        coarse_name="ship",
        taxonomy=taxonomy,
        thresholds=policy.class_thresholds,
    )
    ship = _add_supplements(
        ship,
        detections_1024,
        coarse_name="ship",
        threshold=policy.ship_supplement_threshold,
        duplicate_iou=policy.ship_duplicate_iou,
        taxonomy=taxonomy,
    )

    vehicle = _select_primary(
        detections_1536,
        coarse_name="vehicle",
        taxonomy=taxonomy,
        thresholds=policy.class_thresholds,
    )
    vehicle = _filter_vehicle_area(
        vehicle,
        score_ceiling=policy.vehicle_score_ceiling,
        min_area=policy.vehicle_min_area,
    )

    return sorted(
        aircraft + ship + vehicle,
        key=lambda detection: (
            detection.image_id,
            detection.class_id,
            -detection.score,
            detection.polygon,
        ),
    )
