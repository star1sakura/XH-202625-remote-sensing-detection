from __future__ import annotations

import math

import numpy as np
import pytest

from xh_detect.types import InferenceResult, StageTimings
from xh_detect.vehicle_confirmation.benchmark import (
    benchmark_vehicle_proposal_pair,
    vehicle_latency_report_to_dict,
)


class _FakePipeline:
    def __init__(self, seconds: list[float]) -> None:
        self.seconds = iter(seconds)
        self.calls: list[str] = []

    def run(self, image: np.ndarray, image_id: str) -> InferenceResult:
        self.calls.append(image_id)
        total = next(self.seconds)
        return InferenceResult((), StageTimings(0.0, total, 0.0, total))


def test_benchmarks_warmup_and_five_sequential_pairs() -> None:
    main = _FakePipeline([100.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    sph = _FakePipeline([100.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    report = benchmark_vehicle_proposal_pair(  # type: ignore[arg-type]
        main,
        sph,
        image,
        "scene",
        repeats=5,
        reserve_seconds=1.0,
        limit_seconds=20.0,
    )

    assert main.calls == ["scene-main-warmup", *[f"scene-pair-{index}-main" for index in range(5)]]
    assert sph.calls == ["scene-sph-warmup", *[f"scene-pair-{index}-sph" for index in range(5)]]
    assert report.main_seconds == (5.0, 6.0, 7.0, 8.0, 9.0)
    assert report.sph_seconds == (4.0, 4.0, 4.0, 4.0, 4.0)
    assert report.combined_seconds == (9.0, 10.0, 11.0, 12.0, 13.0)
    assert report.proposal_gate_passed
    payload = vehicle_latency_report_to_dict(report)
    assert payload["combined"]["maximum_s"] == 13.0
    assert payload["combined"]["median_s"] == 11.0
    assert payload["gate"]["budget_seconds"] == 19.0


def test_gate_fails_when_any_paired_run_exceeds_reserved_budget() -> None:
    main = _FakePipeline([0.0, 10.0, 10.0])
    sph = _FakePipeline([0.0, 9.0, 9.01])

    report = benchmark_vehicle_proposal_pair(  # type: ignore[arg-type]
        main,
        sph,
        np.zeros((4, 4, 3), dtype=np.uint8),
        "scene",
        repeats=2,
        reserve_seconds=1.0,
        limit_seconds=20.0,
    )

    assert report.combined_seconds == (19.0, 19.009999999999998)
    assert not report.proposal_gate_passed


@pytest.mark.parametrize("repeats", [0, -1, True, 1.5])
def test_rejects_invalid_pair_repeats(repeats: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        benchmark_vehicle_proposal_pair(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            np.zeros((4, 4, 3), dtype=np.uint8),
            "scene",
            repeats=repeats,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("reserve", [-1.0, math.nan, math.inf, True, "1"])
def test_rejects_invalid_reserve(reserve: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        benchmark_vehicle_proposal_pair(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            np.zeros((4, 4, 3), dtype=np.uint8),
            "scene",
            reserve_seconds=reserve,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [0.0, -1.0, math.nan, math.inf, True, "20"])
def test_rejects_invalid_limit(limit: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        benchmark_vehicle_proposal_pair(
            object(),  # type: ignore[arg-type]
            object(),  # type: ignore[arg-type]
            np.zeros((4, 4, 3), dtype=np.uint8),
            "scene",
            limit_seconds=limit,  # type: ignore[arg-type]
        )
