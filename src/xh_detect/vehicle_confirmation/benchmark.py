from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

from xh_detect.benchmark import summarize_durations
from xh_detect.pipeline import InferencePipeline
from xh_detect.types import ImageArray


@dataclass(frozen=True)
class VehicleLatencyReport:
    main_seconds: tuple[float, ...]
    sph_seconds: tuple[float, ...]
    combined_seconds: tuple[float, ...]
    reserve_seconds: float
    limit_seconds: float
    proposal_gate_passed: bool


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _duration(value: object, *, name: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0 or (positive and normalized == 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return normalized


def benchmark_vehicle_proposal_pair(
    main: InferencePipeline,
    sph: InferencePipeline,
    image: ImageArray,
    image_id: str,
    repeats: int = 5,
    reserve_seconds: float = 1.0,
    limit_seconds: float = 20.0,
) -> VehicleLatencyReport:
    repeats = _positive_int(repeats, name="repeats")
    reserve_seconds = _duration(reserve_seconds, name="reserve_seconds", positive=False)
    limit_seconds = _duration(limit_seconds, name="limit_seconds", positive=True)
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty string")

    main.run(image, f"{image_id}-main-warmup")
    sph.run(image, f"{image_id}-sph-warmup")

    main_seconds: list[float] = []
    sph_seconds: list[float] = []
    combined_seconds: list[float] = []
    for index in range(repeats):
        main_result = main.run(image, f"{image_id}-pair-{index}-main")
        sph_result = sph.run(image, f"{image_id}-pair-{index}-sph")
        main_total = _duration(
            main_result.timings.total_s,
            name="main total_s",
            positive=False,
        )
        sph_total = _duration(
            sph_result.timings.total_s,
            name="sph total_s",
            positive=False,
        )
        main_seconds.append(main_total)
        sph_seconds.append(sph_total)
        combined_seconds.append(main_total + sph_total)

    budget = limit_seconds - reserve_seconds
    return VehicleLatencyReport(
        main_seconds=tuple(main_seconds),
        sph_seconds=tuple(sph_seconds),
        combined_seconds=tuple(combined_seconds),
        reserve_seconds=reserve_seconds,
        limit_seconds=limit_seconds,
        proposal_gate_passed=all(value <= budget for value in combined_seconds),
    )


def _timing_summary(values: tuple[float, ...]) -> dict[str, object]:
    summary = summarize_durations(list(values))
    return {
        "samples_s": list(values),
        "median_s": summary["median_s"],
        "p95_s": summary["p95_s"],
        "maximum_s": max(values),
    }


def vehicle_latency_report_to_dict(report: VehicleLatencyReport) -> dict[str, object]:
    return {
        "main": _timing_summary(report.main_seconds),
        "sph": _timing_summary(report.sph_seconds),
        "combined": _timing_summary(report.combined_seconds),
        "gate": {
            "reserve_seconds": report.reserve_seconds,
            "limit_seconds": report.limit_seconds,
            "budget_seconds": report.limit_seconds - report.reserve_seconds,
            "passed": report.proposal_gate_passed,
        },
    }
