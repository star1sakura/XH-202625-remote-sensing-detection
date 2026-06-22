from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import ObjectAnnotation

_CLASS_COUNT = 25
_COARSE_NAMES = ("ship", "aircraft", "vehicle")
_SWAP_CANDIDATE_LIMIT = 48
_MAX_LOCAL_SEARCH_ITERATIONS = 512

SplitObjective = tuple[float, float, float, float]


class _Record(Protocol):
    group_id: str
    annotations: tuple[ObjectAnnotation, ...]


@dataclass(frozen=True)
class _GroupStats:
    image_count: int
    box_counts: tuple[int, ...]
    image_presence_counts: tuple[int, ...]
    coarse_box_counts: tuple[int, ...]
    classes: frozenset[int]


@dataclass
class _MutableGroupStats:
    image_count: int
    box_counts: Counter[int]
    image_presence_counts: Counter[int]
    coarse_box_counts: Counter[str]


def _required_val_group_counts(
    class_groups: Mapping[int, set[str]],
    val_ratio: float,
) -> dict[int, int]:
    required: dict[int, int] = {}
    for class_id in range(_CLASS_COUNT):
        group_count = len(class_groups[class_id])
        minimum = 2 if group_count >= 3 else 1
        required[class_id] = max(
            minimum,
            min(group_count - 1, round(group_count * val_ratio)),
        )
    return required


def _build_group_stats(
    records: Iterable[_Record],
) -> tuple[dict[str, _GroupStats], _GroupStats]:
    taxonomy = get_taxonomy("xh25")
    mutable: dict[str, _MutableGroupStats] = {}
    total_images = 0
    total_boxes: Counter[int] = Counter()
    total_image_presence: Counter[int] = Counter()
    total_coarse: Counter[str] = Counter()

    for record in records:
        group = mutable.setdefault(
            record.group_id,
            _MutableGroupStats(
                image_count=0,
                box_counts=Counter(),
                image_presence_counts=Counter(),
                coarse_box_counts=Counter(),
            ),
        )
        group.image_count += 1
        total_images += 1
        present_classes = {annotation.class_id for annotation in record.annotations}
        group.image_presence_counts.update(present_classes)
        total_image_presence.update(present_classes)
        for annotation in record.annotations:
            class_id = annotation.class_id
            coarse_name = taxonomy.coarse_name(class_id)
            group.box_counts[class_id] += 1
            group.coarse_box_counts[coarse_name] += 1
            total_boxes[class_id] += 1
            total_coarse[coarse_name] += 1

    def freeze(stats: _MutableGroupStats) -> _GroupStats:
        return _GroupStats(
            image_count=stats.image_count,
            box_counts=tuple(stats.box_counts[class_id] for class_id in range(_CLASS_COUNT)),
            image_presence_counts=tuple(
                stats.image_presence_counts[class_id] for class_id in range(_CLASS_COUNT)
            ),
            coarse_box_counts=tuple(
                stats.coarse_box_counts[coarse_name] for coarse_name in _COARSE_NAMES
            ),
            classes=frozenset(stats.box_counts),
        )

    group_stats = {group_id: freeze(stats) for group_id, stats in mutable.items()}
    totals = _GroupStats(
        image_count=total_images,
        box_counts=tuple(total_boxes[class_id] for class_id in range(_CLASS_COUNT)),
        image_presence_counts=tuple(
            total_image_presence[class_id] for class_id in range(_CLASS_COUNT)
        ),
        coarse_box_counts=tuple(total_coarse[name] for name in _COARSE_NAMES),
        classes=frozenset(total_boxes),
    )
    return group_stats, totals


def _objective(
    val_image_count: int,
    val_box_counts: list[int],
    val_image_presence_counts: list[int],
    val_coarse_counts: list[int],
    totals: _GroupStats,
    val_ratio: float,
) -> SplitObjective:
    fine_box_deviations = [
        abs(val_box_counts[index] / totals.box_counts[index] - val_ratio)
        for index in range(_CLASS_COUNT)
    ]
    remaining_deviations = [
        abs(val_image_presence_counts[index] / totals.image_presence_counts[index] - val_ratio)
        for index in range(_CLASS_COUNT)
    ] + [
        abs(val_coarse_counts[index] / totals.coarse_box_counts[index] - val_ratio)
        for index in range(len(_COARSE_NAMES))
    ]
    deviations = fine_box_deviations + remaining_deviations
    return (
        max(fine_box_deviations),
        sum(deviation * deviation for deviation in deviations),
        sum(deviations),
        abs(val_image_count / totals.image_count - val_ratio),
    )


def _add_delta(values: list[int], delta: tuple[int, ...], direction: int) -> None:
    for index, amount in enumerate(delta):
        values[index] += direction * amount


def optimize_validation_groups(
    records: tuple[_Record, ...],
    initial_selected: frozenset[str] | set[str],
    class_groups: Mapping[int, set[str]],
    val_ratio: float,
    seed: int,
    stable_rank: Callable[[int, str], str],
) -> frozenset[str]:
    group_stats, totals = _build_group_stats(records)
    required = _required_val_group_counts(class_groups, val_ratio)
    group_count_limits = {
        class_id: len(class_groups[class_id]) - 1 for class_id in range(_CLASS_COUNT)
    }
    selected = set(initial_selected)
    selected_group_counts = dict.fromkeys(range(_CLASS_COUNT), 0)
    val_image_count = 0
    val_box_counts = [0] * _CLASS_COUNT
    val_image_presence_counts = [0] * _CLASS_COUNT
    val_coarse_counts = [0] * len(_COARSE_NAMES)

    for group_id in selected:
        stats = group_stats[group_id]
        val_image_count += stats.image_count
        _add_delta(val_box_counts, stats.box_counts, 1)
        _add_delta(val_image_presence_counts, stats.image_presence_counts, 1)
        _add_delta(val_coarse_counts, stats.coarse_box_counts, 1)
        for class_id in stats.classes:
            selected_group_counts[class_id] += 1

    def move_is_safe(remove: str | None, add: str | None) -> bool:
        affected = set()
        if remove is not None:
            affected.update(group_stats[remove].classes)
        if add is not None:
            affected.update(group_stats[add].classes)
        for class_id in affected:
            count = selected_group_counts[class_id]
            if remove is not None and class_id in group_stats[remove].classes:
                count -= 1
            if add is not None and class_id in group_stats[add].classes:
                count += 1
            if count < required[class_id] or count > group_count_limits[class_id]:
                return False
        return True

    def move_objective(remove: str | None, add: str | None) -> SplitObjective:
        images = val_image_count
        boxes = val_box_counts.copy()
        image_presence = val_image_presence_counts.copy()
        coarse = val_coarse_counts.copy()
        if remove is not None:
            stats = group_stats[remove]
            images -= stats.image_count
            _add_delta(boxes, stats.box_counts, -1)
            _add_delta(image_presence, stats.image_presence_counts, -1)
            _add_delta(coarse, stats.coarse_box_counts, -1)
        if add is not None:
            stats = group_stats[add]
            images += stats.image_count
            _add_delta(boxes, stats.box_counts, 1)
            _add_delta(image_presence, stats.image_presence_counts, 1)
            _add_delta(coarse, stats.coarse_box_counts, 1)
        return _objective(
            images,
            boxes,
            image_presence,
            coarse,
            totals,
            val_ratio,
        )

    def apply_move(remove: str | None, add: str | None) -> None:
        nonlocal val_image_count
        if remove is not None:
            stats = group_stats[remove]
            selected.remove(remove)
            val_image_count -= stats.image_count
            _add_delta(val_box_counts, stats.box_counts, -1)
            _add_delta(val_image_presence_counts, stats.image_presence_counts, -1)
            _add_delta(val_coarse_counts, stats.coarse_box_counts, -1)
            for class_id in stats.classes:
                selected_group_counts[class_id] -= 1
        if add is not None:
            stats = group_stats[add]
            selected.add(add)
            val_image_count += stats.image_count
            _add_delta(val_box_counts, stats.box_counts, 1)
            _add_delta(val_image_presence_counts, stats.image_presence_counts, 1)
            _add_delta(val_coarse_counts, stats.coarse_box_counts, 1)
            for class_id in stats.classes:
                selected_group_counts[class_id] += 1

    current = _objective(
        val_image_count,
        val_box_counts,
        val_image_presence_counts,
        val_coarse_counts,
        totals,
        val_ratio,
    )
    all_group_ids = set(group_stats)
    for _ in range(_MAX_LOCAL_SEARCH_ITERATIONS):
        unselected = all_group_ids - selected
        best_objective = current
        best_tie: str | None = None
        best_move: tuple[str | None, str | None] | None = None

        def consider(
            remove: str | None,
            add: str | None,
            current_objective: SplitObjective,
        ) -> None:
            nonlocal best_move, best_objective, best_tie
            if not move_is_safe(remove, add):
                return
            objective = move_objective(remove, add)
            tie = stable_rank(seed, f"{remove or ''}->{add or ''}")
            if objective < best_objective or (
                objective == best_objective
                and objective < current_objective
                and (best_tie is None or tie < best_tie)
            ):
                best_objective = objective
                best_tie = tie
                best_move = (remove, add)

        for group_id in sorted(unselected):
            consider(None, group_id, current)
        for group_id in sorted(selected):
            consider(group_id, None, current)

        fine_deviations = [
            abs(val_box_counts[index] / totals.box_counts[index] - val_ratio)
            for index in range(_CLASS_COUNT)
        ]
        worst = max(fine_deviations)
        worst_classes = [
            class_id
            for class_id, deviation in enumerate(fine_deviations)
            if abs(deviation - worst) <= 1e-12
        ]
        if len(selected) * len(unselected) <= 20_000:
            remove_candidates = selected
            add_candidates = unselected
        else:
            remove_candidates: set[str] = set()
            add_candidates: set[str] = set()
            for class_id in worst_classes:
                over_target = val_box_counts[class_id] / totals.box_counts[class_id] > val_ratio
                remove_candidates.update(
                    sorted(
                        selected,
                        key=lambda group_id: (
                            (
                                -group_stats[group_id].box_counts[class_id]
                                if over_target
                                else group_stats[group_id].box_counts[class_id]
                            ),
                            stable_rank(seed, group_id),
                        ),
                    )[:_SWAP_CANDIDATE_LIMIT]
                )
                add_candidates.update(
                    sorted(
                        unselected,
                        key=lambda group_id: (
                            (
                                group_stats[group_id].box_counts[class_id]
                                if over_target
                                else -group_stats[group_id].box_counts[class_id]
                            ),
                            stable_rank(seed, group_id),
                        ),
                    )[:_SWAP_CANDIDATE_LIMIT]
                )
        for remove in sorted(remove_candidates):
            for add in sorted(add_candidates):
                consider(remove, add, current)

        if best_move is None:
            break
        apply_move(*best_move)
        current = best_objective

    return frozenset(selected)
