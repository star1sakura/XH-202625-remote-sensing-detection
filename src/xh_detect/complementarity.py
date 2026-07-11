from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from xh_detect.evaluator import _iou_threshold, evaluate
from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.taxonomy import COARSE_NAMES, Taxonomy
from xh_detect.types import Detection, ObjectAnnotation


@dataclass(frozen=True)
class ModelClassOutcome:
    matched_truth_indices: frozenset[int]
    tp: int
    fp: int
    fn: int

    @property
    def recall(self) -> float:
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator else 0.0

    @property
    def fdr(self) -> float:
        denominator = self.tp + self.fp
        return self.fp / denominator if denominator else 0.0


@dataclass(frozen=True)
class PairwiseClassOutcome:
    shared_tp: int
    baseline_only_tp: int
    candidate_only_tp: int
    oracle_tp: int
    oracle_recall: float
    baseline_fp: int
    candidate_fp: int


@dataclass(frozen=True)
class ComplementarityReport:
    baseline_name: str
    models: dict[str, dict[str, ModelClassOutcome]]
    pairwise: dict[str, dict[str, PairwiseClassOutcome]]


def _model_outcomes(
    predictions: list[Detection],
    truth: list[ObjectAnnotation],
    taxonomy: Taxonomy,
) -> dict[str, ModelClassOutcome]:
    evaluate(predictions, truth, taxonomy=taxonomy)
    truth_by_group: dict[tuple[str, str], list[tuple[int, ObjectAnnotation]]] = defaultdict(list)
    truth_counts: dict[str, int] = dict.fromkeys(COARSE_NAMES, 0)
    for truth_index, item in enumerate(truth):
        if item.difficult:
            continue
        coarse_name = taxonomy.coarse_name(item.class_id)
        truth_by_group[(item.image_id, coarse_name)].append((truth_index, item))
        truth_counts[coarse_name] += 1

    matched_by_group: dict[tuple[str, str], set[int]] = defaultdict(set)
    matched_by_class: dict[str, set[int]] = {coarse_name: set() for coarse_name in COARSE_NAMES}
    false_positives: dict[str, int] = dict.fromkeys(COARSE_NAMES, 0)
    indexed_predictions = sorted(
        enumerate(predictions),
        key=lambda pair: (-pair[1].score, pair[0]),
    )
    for _, prediction in indexed_predictions:
        coarse_name = taxonomy.coarse_name(prediction.class_id)
        group_key = (prediction.image_id, coarse_name)
        prediction_hbb = obb_to_hbb(prediction.polygon)
        best_truth_index = -1
        best_iou = -1.0
        for truth_index, candidate in truth_by_group.get(group_key, []):
            if truth_index in matched_by_group[group_key]:
                continue
            iou = hbb_iou(prediction_hbb, obb_to_hbb(candidate.polygon))
            if iou >= _iou_threshold(candidate.class_id, taxonomy) and iou > best_iou:
                best_iou = iou
                best_truth_index = truth_index
        if best_truth_index >= 0:
            matched_by_group[group_key].add(best_truth_index)
            matched_by_class[coarse_name].add(best_truth_index)
        else:
            false_positives[coarse_name] += 1

    return {
        coarse_name: ModelClassOutcome(
            matched_truth_indices=frozenset(matched_by_class[coarse_name]),
            tp=len(matched_by_class[coarse_name]),
            fp=false_positives[coarse_name],
            fn=truth_counts[coarse_name] - len(matched_by_class[coarse_name]),
        )
        for coarse_name in sorted(COARSE_NAMES)
    }


def analyze_complementarity(
    predictions_by_model: Mapping[str, Iterable[Detection]],
    ground_truth: Iterable[ObjectAnnotation],
    taxonomy: Taxonomy,
    baseline_name: str,
) -> ComplementarityReport:
    if len(predictions_by_model) < 2:
        raise ValueError("at least two prediction models are required")
    if baseline_name not in predictions_by_model:
        raise ValueError(f"baseline model {baseline_name!r} is missing")
    if any(not isinstance(name, str) or not name.strip() for name in predictions_by_model):
        raise ValueError("model names must be non-empty strings")

    truth_items = list(ground_truth)
    model_outcomes = {
        name: _model_outcomes(list(predictions), truth_items, taxonomy)
        for name, predictions in predictions_by_model.items()
    }
    baseline = model_outcomes[baseline_name]
    truth_counts = {
        coarse_name: sum(
            not item.difficult and taxonomy.coarse_name(item.class_id) == coarse_name
            for item in truth_items
        )
        for coarse_name in COARSE_NAMES
    }
    pairwise: dict[str, dict[str, PairwiseClassOutcome]] = {}
    for name, candidate in model_outcomes.items():
        if name == baseline_name:
            continue
        pairwise[name] = {}
        for coarse_name in sorted(COARSE_NAMES):
            baseline_matches = baseline[coarse_name].matched_truth_indices
            candidate_matches = candidate[coarse_name].matched_truth_indices
            union = baseline_matches | candidate_matches
            denominator = truth_counts[coarse_name]
            pairwise[name][coarse_name] = PairwiseClassOutcome(
                shared_tp=len(baseline_matches & candidate_matches),
                baseline_only_tp=len(baseline_matches - candidate_matches),
                candidate_only_tp=len(candidate_matches - baseline_matches),
                oracle_tp=len(union),
                oracle_recall=len(union) / denominator if denominator else 0.0,
                baseline_fp=baseline[coarse_name].fp,
                candidate_fp=candidate[coarse_name].fp,
            )

    return ComplementarityReport(
        baseline_name=baseline_name,
        models=model_outcomes,
        pairwise=pairwise,
    )


def complementarity_report_to_dict(report: ComplementarityReport) -> dict[str, object]:
    return {
        "baseline_name": report.baseline_name,
        "models": {
            model_name: {
                coarse_name: {
                    "matched_truth_indices": sorted(outcome.matched_truth_indices),
                    "tp": outcome.tp,
                    "fp": outcome.fp,
                    "fn": outcome.fn,
                    "recall": outcome.recall,
                    "fdr": outcome.fdr,
                }
                for coarse_name, outcome in sorted(class_outcomes.items())
            }
            for model_name, class_outcomes in report.models.items()
        },
        "pairwise": {
            model_name: {
                coarse_name: {
                    "shared_tp": outcome.shared_tp,
                    "baseline_only_tp": outcome.baseline_only_tp,
                    "candidate_only_tp": outcome.candidate_only_tp,
                    "oracle_tp": outcome.oracle_tp,
                    "oracle_recall": outcome.oracle_recall,
                    "baseline_fp": outcome.baseline_fp,
                    "candidate_fp": outcome.candidate_fp,
                }
                for coarse_name, outcome in sorted(class_outcomes.items())
            }
            for model_name, class_outcomes in report.pairwise.items()
        },
    }
