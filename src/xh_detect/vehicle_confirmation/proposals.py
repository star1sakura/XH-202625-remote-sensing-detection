from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.types import Detection, ObjectAnnotation

_XH25_CLASS_COUNT = 25


@dataclass(frozen=True)
class LabeledVehicleProposal:
    proposal_index: int
    detection: Detection
    label: int
    reason: str
    matched_truth_index: int | None
    duplicate_main: bool


@dataclass(frozen=True)
class VehicleProposalReport:
    main_vehicle_tp: int
    main_vehicle_fp: int
    recoverable_tp: int
    proposal_fp: int
    duplicate_main: int
    duplicate_proposal: int


@dataclass(frozen=True)
class VehicleConsensusReport:
    sph: VehicleProposalReport
    mks: VehicleProposalReport
    consensus_recoverable_tp: int
    consensus_fp: int
    accepted_sph_indexes: tuple[int, ...]
    accepted_mks_indexes: tuple[int, ...]
    historical_fdr_constraint_passed: bool


def _validate_class_id(class_id: object) -> int:
    if isinstance(class_id, bool) or not isinstance(class_id, int):
        raise TypeError("class_id must be an integer")
    if not 0 <= class_id < _XH25_CLASS_COUNT:
        raise ValueError(f"class_id must be within [0, {_XH25_CLASS_COUNT - 1}]")
    return class_id


def _validate_threshold(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return result


def _best_unclaimed_truth(
    prediction: Detection,
    truth_by_image: dict[str, list[tuple[int, ObjectAnnotation]]],
    claimed_truth: set[int],
    iou_threshold: float,
) -> int | None:
    prediction_hbb = obb_to_hbb(prediction.polygon)
    best_index: int | None = None
    best_iou = -1.0
    for truth_index, truth in truth_by_image.get(prediction.image_id, []):
        if truth_index in claimed_truth:
            continue
        iou = hbb_iou(prediction_hbb, obb_to_hbb(truth.polygon))
        if iou >= iou_threshold and iou > best_iou:
            best_index = truth_index
            best_iou = iou
    return best_index


def _overlaps_detection(
    prediction: Detection,
    candidates: Iterable[Detection],
    iou_threshold: float,
) -> bool:
    prediction_hbb = obb_to_hbb(prediction.polygon)
    return any(
        candidate.image_id == prediction.image_id
        and hbb_iou(prediction_hbb, obb_to_hbb(candidate.polygon)) >= iou_threshold
        for candidate in candidates
    )


def _overlaps_claimed_truth(
    prediction: Detection,
    truth_by_index: dict[int, ObjectAnnotation],
    claimed_truth: set[int],
    iou_threshold: float,
) -> int | None:
    prediction_hbb = obb_to_hbb(prediction.polygon)
    for truth_index in sorted(claimed_truth):
        truth = truth_by_index[truth_index]
        if truth.image_id != prediction.image_id:
            continue
        if hbb_iou(prediction_hbb, obb_to_hbb(truth.polygon)) >= iou_threshold:
            return truth_index
    return None


def label_vehicle_proposals(
    main_predictions: Iterable[Detection],
    proposal_predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    vehicle_class_id: int = 24,
    iou_threshold: float = 0.35,
) -> tuple[tuple[LabeledVehicleProposal, ...], VehicleProposalReport]:
    vehicle_class_id = _validate_class_id(vehicle_class_id)
    iou_threshold = _validate_threshold(iou_threshold, name="iou_threshold")

    main_items = list(main_predictions)
    proposal_items = list(proposal_predictions)
    truth_items = list(ground_truth)
    for item in (*main_items, *proposal_items, *truth_items):
        _validate_class_id(item.class_id)

    vehicle_truth = [
        (index, item)
        for index, item in enumerate(truth_items)
        if item.class_id == vehicle_class_id and not item.difficult
    ]
    truth_by_image: dict[str, list[tuple[int, ObjectAnnotation]]] = defaultdict(list)
    truth_by_index: dict[int, ObjectAnnotation] = {}
    for truth_index, item in vehicle_truth:
        truth_by_image[item.image_id].append((truth_index, item))
        truth_by_index[truth_index] = item

    indexed_main = [
        (index, item)
        for index, item in enumerate(main_items)
        if item.class_id == vehicle_class_id
    ]
    indexed_main.sort(key=lambda pair: (-pair[1].score, pair[0]))
    main_vehicle = tuple(item for _, item in indexed_main)
    claimed_truth: set[int] = set()
    main_vehicle_tp = 0
    main_vehicle_fp = 0
    for _, prediction in indexed_main:
        truth_index = _best_unclaimed_truth(
            prediction,
            truth_by_image,
            claimed_truth,
            iou_threshold,
        )
        if truth_index is None:
            main_vehicle_fp += 1
        else:
            claimed_truth.add(truth_index)
            main_vehicle_tp += 1

    indexed_proposals = [
        (index, item)
        for index, item in enumerate(proposal_items)
        if item.class_id == vehicle_class_id
    ]
    indexed_proposals.sort(key=lambda pair: (-pair[1].score, pair[0]))
    labels: list[LabeledVehicleProposal] = []
    recoverable_tp = 0
    proposal_fp = 0
    duplicate_main = 0
    duplicate_proposal = 0
    for proposal_index, prediction in indexed_proposals:
        if _overlaps_detection(prediction, main_vehicle, iou_threshold):
            labels.append(
                LabeledVehicleProposal(
                    proposal_index,
                    prediction,
                    0,
                    "duplicate_main",
                    None,
                    True,
                )
            )
            proposal_fp += 1
            duplicate_main += 1
            continue

        truth_index = _best_unclaimed_truth(
            prediction,
            truth_by_image,
            claimed_truth,
            iou_threshold,
        )
        if truth_index is not None:
            claimed_truth.add(truth_index)
            labels.append(
                LabeledVehicleProposal(
                    proposal_index,
                    prediction,
                    1,
                    "recoverable_truth",
                    truth_index,
                    False,
                )
            )
            recoverable_tp += 1
            continue

        claimed_index = _overlaps_claimed_truth(
            prediction,
            truth_by_index,
            claimed_truth,
            iou_threshold,
        )
        reason = "duplicate_proposal" if claimed_index is not None else "background"
        labels.append(
            LabeledVehicleProposal(
                proposal_index,
                prediction,
                0,
                reason,
                claimed_index,
                False,
            )
        )
        proposal_fp += 1
        if reason == "duplicate_proposal":
            duplicate_proposal += 1

    return tuple(labels), VehicleProposalReport(
        main_vehicle_tp=main_vehicle_tp,
        main_vehicle_fp=main_vehicle_fp,
        recoverable_tp=recoverable_tp,
        proposal_fp=proposal_fp,
        duplicate_main=duplicate_main,
        duplicate_proposal=duplicate_proposal,
    )


def _validate_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def satisfies_vehicle_fdr(
    baseline_tp: int,
    baseline_fp: int,
    added_tp: int,
    added_fp: int,
    *,
    ceiling: float = 0.202899,
) -> bool:
    baseline_tp = _validate_count(baseline_tp, name="baseline_tp")
    baseline_fp = _validate_count(baseline_fp, name="baseline_fp")
    added_tp = _validate_count(added_tp, name="added_tp")
    added_fp = _validate_count(added_fp, name="added_fp")
    ceiling = _validate_threshold(ceiling, name="ceiling")

    false_positives = baseline_fp + added_fp
    detections = baseline_tp + added_tp + false_positives
    fused_fdr = false_positives / detections if detections else 0.0
    return fused_fdr <= ceiling


def analyze_vehicle_consensus(
    main_predictions: Iterable[Detection],
    sph_predictions: Iterable[Detection],
    mks_predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    vehicle_class_id: int = 24,
    iou_threshold: float = 0.35,
) -> VehicleConsensusReport:
    main_items = tuple(main_predictions)
    truth_items = tuple(ground_truth)
    sph_labels, sph_report = label_vehicle_proposals(
        main_items,
        sph_predictions,
        truth_items,
        vehicle_class_id=vehicle_class_id,
        iou_threshold=iou_threshold,
    )
    mks_labels, mks_report = label_vehicle_proposals(
        main_items,
        mks_predictions,
        truth_items,
        vehicle_class_id=vehicle_class_id,
        iou_threshold=iou_threshold,
    )

    sph_candidates = tuple(item for item in sph_labels if not item.duplicate_main)
    mks_candidates = tuple(item for item in mks_labels if not item.duplicate_main)
    matched_mks: set[int] = set()
    accepted_sph_indexes: list[int] = []
    accepted_mks_indexes: list[int] = []
    recoverable_tp = 0
    false_positives = 0
    for sph_item in sph_candidates:
        match_position: int | None = None
        sph_hbb = obb_to_hbb(sph_item.detection.polygon)
        for position, mks_item in enumerate(mks_candidates):
            if position in matched_mks:
                continue
            if mks_item.detection.image_id != sph_item.detection.image_id:
                continue
            if hbb_iou(sph_hbb, obb_to_hbb(mks_item.detection.polygon)) >= iou_threshold:
                match_position = position
                break
        if match_position is None:
            continue

        matched_mks.add(match_position)
        matched_item = mks_candidates[match_position]
        accepted_sph_indexes.append(sph_item.proposal_index)
        accepted_mks_indexes.append(matched_item.proposal_index)
        if sph_item.label == 1:
            recoverable_tp += 1
        else:
            false_positives += 1

    return VehicleConsensusReport(
        sph=sph_report,
        mks=mks_report,
        consensus_recoverable_tp=recoverable_tp,
        consensus_fp=false_positives,
        accepted_sph_indexes=tuple(accepted_sph_indexes),
        accepted_mks_indexes=tuple(accepted_mks_indexes),
        historical_fdr_constraint_passed=satisfies_vehicle_fdr(
            55,
            14,
            recoverable_tp,
            false_positives,
        ),
    )


def _proposal_report_to_dict(report: VehicleProposalReport) -> dict[str, int]:
    return {
        "main_vehicle_tp": report.main_vehicle_tp,
        "main_vehicle_fp": report.main_vehicle_fp,
        "recoverable_tp": report.recoverable_tp,
        "proposal_fp": report.proposal_fp,
        "duplicate_main": report.duplicate_main,
        "duplicate_proposal": report.duplicate_proposal,
    }


def vehicle_consensus_report_to_dict(report: VehicleConsensusReport) -> dict[str, object]:
    return {
        "sph": _proposal_report_to_dict(report.sph),
        "mks": _proposal_report_to_dict(report.mks),
        "consensus": {
            "recoverable_tp": report.consensus_recoverable_tp,
            "fp": report.consensus_fp,
            "accepted_sph_indexes": list(report.accepted_sph_indexes),
            "accepted_mks_indexes": list(report.accepted_mks_indexes),
            "historical_fdr_constraint_passed": report.historical_fdr_constraint_passed,
        },
    }
