from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from xh_detect.evaluator import EvaluationReport, Metrics, _match, evaluate, report_to_dict
from xh_detect.taxonomy import Taxonomy
from xh_detect.thresholds import filter_predictions_by_class_threshold, parse_threshold_grid
from xh_detect.types import Detection, ObjectAnnotation


@dataclass(frozen=True)
class RankingThresholdResult:
    thresholds: dict[int, float]
    report: EvaluationReport
    improved_accuracy_items: int
    group_scores: dict[str, dict[str, float | int | bool]]


def _normalized_score(candidate: Metrics, baseline: Metrics) -> tuple[float | int | bool, ...]:
    recall_gain = candidate.recall - baseline.recall
    fdr_gain = baseline.fdr - candidate.fdr
    recall_step = 1.0 / (baseline.tp + baseline.fn) if baseline.tp + baseline.fn else 1.0
    fdr_step = 1.0 / (baseline.tp + baseline.fp) if baseline.tp + baseline.fp else 1.0
    recall_units = recall_gain / recall_step
    fdr_units = fdr_gain / fdr_step
    recall_improved = recall_gain > 1e-12
    fdr_improved = fdr_gain > 1e-12
    improved = int(recall_improved) + int(fdr_improved)
    both = recall_improved and fdr_improved
    return (
        both,
        improved,
        min(recall_units, fdr_units) if both else -math.inf,
        recall_units + fdr_units,
        candidate.tp,
        -candidate.fp,
    )


def _evaluate_map(
    predictions: Sequence[Detection],
    truth: Sequence[ObjectAnnotation],
    thresholds: Mapping[int, float],
    taxonomy: Taxonomy,
) -> EvaluationReport:
    return evaluate(
        filter_predictions_by_class_threshold(predictions, thresholds, taxonomy),
        truth,
        taxonomy=taxonomy,
    )


def _evaluate_group(
    predictions: Sequence[Detection],
    truth: Sequence[ObjectAnnotation],
    thresholds: Mapping[int, float],
    taxonomy: Taxonomy,
    group: str,
) -> Metrics:
    group_predictions = [
        item
        for item in predictions
        if taxonomy.coarse_name(item.class_id) == group and item.score >= thresholds[item.class_id]
    ]
    group_truth = [item for item in truth if taxonomy.coarse_name(item.class_id) == group]
    metrics, _, _ = _match(
        group_predictions,
        group_truth,
        taxonomy,
        lambda _: "all",
    )
    return metrics


def optimize_ranking_thresholds(
    predictions: Iterable[Detection],
    ground_truth: Iterable[ObjectAnnotation],
    *,
    baseline: EvaluationReport,
    taxonomy: Taxonomy,
    thresholds: str | Sequence[float],
    passes: int = 2,
) -> RankingThresholdResult:
    if isinstance(passes, bool) or not isinstance(passes, int) or passes < 1:
        raise ValueError("passes must be a positive integer")
    grid = parse_threshold_grid(thresholds)
    prediction_items = list(predictions)
    truth_items = list(ground_truth)
    groups = ("aircraft", "ship", "vehicle")
    missing = [group for group in groups if group not in baseline.by_coarse_class]
    if missing:
        raise ValueError("baseline is missing coarse groups: " + ", ".join(missing))

    thresholds_by_class = dict.fromkeys(taxonomy.valid_ids, grid[0])
    classes_by_group = {
        group: sorted(
            class_id for class_id in taxonomy.valid_ids if taxonomy.coarse_name(class_id) == group
        )
        for group in groups
    }
    for group in groups:
        best_threshold = grid[0]
        best_score: tuple[float | int | bool, ...] | None = None
        for threshold in grid:
            candidate_map = dict(thresholds_by_class)
            for class_id in classes_by_group[group]:
                candidate_map[class_id] = threshold
            metrics = _evaluate_group(
                prediction_items,
                truth_items,
                candidate_map,
                taxonomy,
                group,
            )
            score = _normalized_score(
                metrics,
                baseline.by_coarse_class[group],
            )
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = threshold
        for class_id in classes_by_group[group]:
            thresholds_by_class[class_id] = best_threshold

    for _ in range(passes):
        for group in groups:
            for class_id in classes_by_group[group]:
                best_threshold = thresholds_by_class[class_id]
                current_metrics = _evaluate_group(
                    prediction_items,
                    truth_items,
                    thresholds_by_class,
                    taxonomy,
                    group,
                )
                best_score = _normalized_score(
                    current_metrics,
                    baseline.by_coarse_class[group],
                )
                for threshold in grid:
                    candidate_map = dict(thresholds_by_class)
                    candidate_map[class_id] = threshold
                    metrics = _evaluate_group(
                        prediction_items,
                        truth_items,
                        candidate_map,
                        taxonomy,
                        group,
                    )
                    score = _normalized_score(
                        metrics,
                        baseline.by_coarse_class[group],
                    )
                    if score > best_score:
                        best_score = score
                        best_threshold = threshold
                thresholds_by_class[class_id] = best_threshold

    report = _evaluate_map(
        prediction_items,
        truth_items,
        thresholds_by_class,
        taxonomy,
    )
    group_scores: dict[str, dict[str, float | int | bool]] = {}
    improved_items = 0
    for group in groups:
        candidate = report.by_coarse_class[group]
        reference = baseline.by_coarse_class[group]
        recall_gain = candidate.recall - reference.recall
        fdr_gain = reference.fdr - candidate.fdr
        recall_improved = recall_gain > 1e-12
        fdr_improved = fdr_gain > 1e-12
        improved_items += int(recall_improved) + int(fdr_improved)
        group_scores[group] = {
            "tp": candidate.tp,
            "fp": candidate.fp,
            "fn": candidate.fn,
            "recall": candidate.recall,
            "fdr": candidate.fdr,
            "recall_gain": recall_gain,
            "fdr_gain": fdr_gain,
            "recall_improved": recall_improved,
            "fdr_improved": fdr_improved,
        }
    return RankingThresholdResult(
        thresholds=thresholds_by_class,
        report=report,
        improved_accuracy_items=improved_items,
        group_scores=group_scores,
    )


def write_ranking_threshold_artifacts(
    result: RankingThresholdResult,
    *,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds_path = output_dir / "optimized-thresholds.yaml"
    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.json"
    thresholds_path.write_text(
        yaml.safe_dump(
            {
                "class_thresholds": {
                    class_id: result.thresholds[class_id] for class_id in sorted(result.thresholds)
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report_to_dict(result.report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "improved_accuracy_items": result.improved_accuracy_items,
                "single_weight_gate_passed": result.improved_accuracy_items >= 6,
                "group_scores": result.group_scores,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "thresholds": thresholds_path,
        "report": report_path,
        "summary": summary_path,
    }
