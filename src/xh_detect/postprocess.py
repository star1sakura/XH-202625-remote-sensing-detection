from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real

from xh_detect.geometry import hbb_iou, obb_to_hbb
from xh_detect.types import Detection

HBB = tuple[float, float, float, float]


@dataclass(frozen=True)
class SuppressionRule:
    method: str
    threshold: float

    def __post_init__(self) -> None:
        if self.method not in {"iou", "diou"}:
            raise ValueError("suppression method must be iou or diou")
        if isinstance(self.threshold, bool) or not isinstance(self.threshold, Real):
            raise TypeError("suppression threshold must be a finite real number")
        threshold = float(self.threshold)
        if not math.isfinite(threshold):
            raise ValueError("suppression threshold must be finite")
        lower, upper = (-1.0, 1.0) if self.method == "diou" else (0.0, 1.0)
        if not lower <= threshold <= upper:
            raise ValueError("suppression threshold is outside the method range")
        object.__setattr__(self, "threshold", threshold)


@dataclass(frozen=True)
class LowScoreAreaRule:
    score_ceiling: float
    min_area: float

    def __post_init__(self) -> None:
        if isinstance(self.score_ceiling, bool) or not isinstance(self.score_ceiling, Real):
            raise TypeError("score ceiling must be a finite real number")
        score_ceiling = float(self.score_ceiling)
        if not math.isfinite(score_ceiling) or not 0.0 <= score_ceiling <= 1.0:
            raise ValueError("score ceiling must be finite and in [0, 1]")
        if isinstance(self.min_area, bool) or not isinstance(self.min_area, Real):
            raise TypeError("minimum area must be a finite real number")
        min_area = float(self.min_area)
        if not math.isfinite(min_area) or min_area < 0.0:
            raise ValueError("minimum area must be finite and non-negative")
        object.__setattr__(self, "score_ceiling", score_ceiling)
        object.__setattr__(self, "min_area", min_area)


def diou(box_a: HBB, box_b: HBB) -> float:
    overlap = hbb_iou(box_a, box_b)
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    distance_sq = ((ax1 + ax2 - bx1 - bx2) / 2) ** 2 + ((ay1 + ay2 - by1 - by2) / 2) ** 2
    enclosing_x1 = min(ax1, bx1)
    enclosing_y1 = min(ay1, by1)
    enclosing_x2 = max(ax2, bx2)
    enclosing_y2 = max(ay2, by2)
    diagonal_sq = (enclosing_x2 - enclosing_x1) ** 2 + (enclosing_y2 - enclosing_y1) ** 2
    return overlap if diagonal_sq == 0.0 else overlap - distance_sq / diagonal_sq


def _validate_rules(rules: Mapping[int, SuppressionRule]) -> dict[int, SuppressionRule]:
    validated: dict[int, SuppressionRule] = {}
    for class_id, rule in rules.items():
        if isinstance(class_id, bool) or not isinstance(class_id, Integral):
            raise TypeError("suppression rule class IDs must be integers")
        if not isinstance(rule, SuppressionRule):
            raise TypeError("suppression rule values must be SuppressionRule instances")
        validated[int(class_id)] = rule
    return validated


def suppress_class_detections(
    detections: Iterable[Detection],
    rules: Mapping[int, SuppressionRule],
) -> list[Detection]:
    validated_rules = _validate_rules(rules)
    grouped: dict[tuple[str, int], list[tuple[int, Detection]]] = defaultdict(list)
    for original_index, detection in enumerate(detections):
        grouped[(detection.image_id, detection.class_id)].append((original_index, detection))

    kept: list[tuple[int, Detection]] = []
    for (_, class_id), group in grouped.items():
        rule = validated_rules.get(class_id)
        if rule is None:
            kept.extend(group)
            continue
        remaining = sorted(group, key=lambda item: (-item[1].score, item[0]))
        while remaining:
            selected_index, selected = remaining.pop(0)
            kept.append((selected_index, selected))
            selected_hbb = obb_to_hbb(selected.polygon)
            survivors: list[tuple[int, Detection]] = []
            for candidate_index, candidate in remaining:
                candidate_hbb = obb_to_hbb(candidate.polygon)
                overlap = (
                    hbb_iou(selected_hbb, candidate_hbb)
                    if rule.method == "iou"
                    else diou(selected_hbb, candidate_hbb)
                )
                if overlap < rule.threshold:
                    survivors.append((candidate_index, candidate))
            remaining = survivors

    kept.sort(key=lambda item: (-item[1].score, item[0]))
    return [detection for _, detection in kept]


def filter_low_score_area_detections(
    detections: Iterable[Detection],
    rules: Mapping[int, LowScoreAreaRule],
) -> list[Detection]:
    validated_rules: dict[int, LowScoreAreaRule] = {}
    for class_id, rule in rules.items():
        if isinstance(class_id, bool) or not isinstance(class_id, Integral):
            raise TypeError("low-score area rule class IDs must be integers")
        if not isinstance(rule, LowScoreAreaRule):
            raise TypeError("low-score area rule values must be LowScoreAreaRule instances")
        validated_rules[int(class_id)] = rule

    kept: list[Detection] = []
    for detection in detections:
        rule = validated_rules.get(detection.class_id)
        if rule is not None and detection.score < rule.score_ceiling:
            x1, y1, x2, y2 = obb_to_hbb(detection.polygon)
            if (x2 - x1) * (y2 - y1) < rule.min_area:
                continue
        kept.append(detection)
    return kept
