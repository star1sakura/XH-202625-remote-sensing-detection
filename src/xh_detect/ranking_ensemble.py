from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

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


@dataclass(frozen=True)
class RankingEnsemblePolicy:
    aircraft_threshold: float = 0.25
    ship_threshold: float = 0.31
    vehicle_primary_threshold: float = 0.25
    vehicle_supplement_threshold: float = 0.64
    vehicle_duplicate_iou: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "aircraft_threshold",
            "ship_threshold",
            "vehicle_primary_threshold",
            "vehicle_supplement_threshold",
            "vehicle_duplicate_iou",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))


def _select_coarse(
    detections: Iterable[Detection],
    *,
    coarse_name: str,
    threshold: float,
    taxonomy: Taxonomy,
) -> list[Detection]:
    return [
        detection
        for detection in detections
        if detection.score >= threshold and taxonomy.coarse_name(detection.class_id) == coarse_name
    ]


def _overlaps_selected(
    candidate: Detection,
    selected: Iterable[Detection],
    threshold: float,
) -> bool:
    candidate_hbb = obb_to_hbb(candidate.polygon)
    return any(
        candidate.image_id == existing.image_id
        and hbb_iou(candidate_hbb, obb_to_hbb(existing.polygon)) >= threshold
        for existing in selected
    )


def fuse_ranking_ensemble(
    *,
    aircraft_predictions: Iterable[Detection],
    ship_predictions: Iterable[Detection],
    vehicle_primary_predictions: Iterable[Detection],
    vehicle_supplement_predictions: Iterable[Detection],
    taxonomy: Taxonomy,
    policy: RankingEnsemblePolicy | None = None,
) -> list[Detection]:
    policy = RankingEnsemblePolicy() if policy is None else policy
    aircraft = _select_coarse(
        aircraft_predictions,
        coarse_name="aircraft",
        threshold=policy.aircraft_threshold,
        taxonomy=taxonomy,
    )
    ship = _select_coarse(
        ship_predictions,
        coarse_name="ship",
        threshold=policy.ship_threshold,
        taxonomy=taxonomy,
    )
    vehicle = _select_coarse(
        vehicle_primary_predictions,
        coarse_name="vehicle",
        threshold=policy.vehicle_primary_threshold,
        taxonomy=taxonomy,
    )
    supplements = _select_coarse(
        vehicle_supplement_predictions,
        coarse_name="vehicle",
        threshold=policy.vehicle_supplement_threshold,
        taxonomy=taxonomy,
    )
    for candidate in sorted(supplements, key=lambda item: -item.score):
        if not _overlaps_selected(candidate, vehicle, policy.vehicle_duplicate_iou):
            vehicle.append(candidate)

    combined = aircraft + ship + vehicle
    return sorted(
        combined,
        key=lambda item: (
            item.image_id,
            item.class_id,
            -item.score,
            item.polygon,
        ),
    )
