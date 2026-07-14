from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from xh_detect.types import Detection, ObjectAnnotation
from xh_detect.vehicle_confirmation.proposals import label_vehicle_proposals


@dataclass(frozen=True)
class VehicleExpertOperatingPoint:
    threshold: float
    added_tp: int
    added_fp: int
    fused_recall: float
    fused_fdr: float
    feasible: bool


@dataclass(frozen=True)
class VehicleExpertHoldoutReport:
    baseline_tp: int
    baseline_fp: int
    baseline_fn: int
    points: tuple[VehicleExpertOperatingPoint, ...]
    selected: VehicleExpertOperatingPoint | None


def _normalize_thresholds(values: Iterable[float]) -> tuple[float, ...]:
    thresholds = tuple(values)
    if not thresholds:
        raise ValueError("at least one threshold is required")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("thresholds must be unique")
    for value in thresholds:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError("thresholds must be finite and within [0, 1]")
    return tuple(float(value) for value in thresholds)


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def analyze_vehicle_expert_holdout(
    main_predictions: Iterable[Detection],
    expert_predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    image_ids: Iterable[str],
    thresholds: Iterable[float],
    fdr_ceiling: float = 0.202899,
    minimum_added_tp: int = 3,
) -> VehicleExpertHoldoutReport:
    normalized_ids = frozenset(image_ids)
    if not normalized_ids or any(
        not isinstance(image_id, str) or not image_id.strip() for image_id in normalized_ids
    ):
        raise ValueError("image_ids must contain non-empty strings")
    thresholds = _normalize_thresholds(thresholds)
    if (
        isinstance(fdr_ceiling, bool)
        or not isinstance(fdr_ceiling, (int, float))
        or not math.isfinite(fdr_ceiling)
        or not 0.0 <= fdr_ceiling <= 1.0
    ):
        raise ValueError("fdr_ceiling must be finite and within [0, 1]")
    minimum_added_tp = _non_negative_int(minimum_added_tp, "minimum_added_tp")

    main = [
        item for item in main_predictions if item.image_id in normalized_ids and item.class_id == 24
    ]
    expert: list[Detection] = []
    for item in expert_predictions:
        if item.class_id != 0:
            raise ValueError("vehicle expert predictions must use class 0")
        if item.image_id in normalized_ids:
            expert.append(Detection(item.image_id, 24, item.score, item.polygon))
    truth = [item for item in ground_truth if item.image_id in normalized_ids]
    labeled, proposal_report = label_vehicle_proposals(main, expert, truth)
    vehicle_truth = sum(item.class_id == 24 and not item.difficult for item in truth)
    baseline_fn = vehicle_truth - proposal_report.main_vehicle_tp
    points: list[VehicleExpertOperatingPoint] = []
    for threshold in thresholds:
        accepted = [
            item
            for item in labeled
            if not item.duplicate_main and item.detection.score >= threshold
        ]
        added_tp = sum(item.label == 1 for item in accepted)
        added_fp = len(accepted) - added_tp
        fused_tp = proposal_report.main_vehicle_tp + added_tp
        fused_fp = proposal_report.main_vehicle_fp + added_fp
        truth_denominator = proposal_report.main_vehicle_tp + baseline_fn
        detection_denominator = fused_tp + fused_fp
        fused_recall = fused_tp / truth_denominator if truth_denominator else 0.0
        fused_fdr = fused_fp / detection_denominator if detection_denominator else 0.0
        points.append(
            VehicleExpertOperatingPoint(
                threshold,
                added_tp,
                added_fp,
                fused_recall,
                fused_fdr,
                added_tp >= minimum_added_tp and fused_fdr <= fdr_ceiling,
            )
        )
    feasible = [point for point in points if point.feasible]
    selected = max(
        feasible,
        key=lambda point: (point.added_tp, -point.added_fp, point.threshold),
        default=None,
    )
    return VehicleExpertHoldoutReport(
        proposal_report.main_vehicle_tp,
        proposal_report.main_vehicle_fp,
        baseline_fn,
        tuple(points),
        selected,
    )


def vehicle_expert_report_to_dict(report: VehicleExpertHoldoutReport) -> dict[str, object]:
    def point_payload(point: VehicleExpertOperatingPoint) -> dict[str, object]:
        return {
            "threshold": point.threshold,
            "added_tp": point.added_tp,
            "added_fp": point.added_fp,
            "fused_recall": point.fused_recall,
            "fused_fdr": point.fused_fdr,
            "feasible": point.feasible,
        }

    return {
        "baseline": {
            "tp": report.baseline_tp,
            "fp": report.baseline_fp,
            "fn": report.baseline_fn,
        },
        "points": [point_payload(point) for point in report.points],
        "selected": point_payload(report.selected) if report.selected is not None else None,
        "decision": "PROMOTE" if report.selected is not None else "RETAIN MAIN",
    }
