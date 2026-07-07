from __future__ import annotations

import json
from pathlib import Path

import pytest

from xh_detect.competition import (
    build_competition_proxy,
    load_evaluation_report,
    render_competition_proxy_markdown,
    write_competition_proxy_artifacts,
)
from xh_detect.evaluator import EvaluationReport, Metrics, report_to_dict

DEFAULT_OVERALL = Metrics(90, 5, 10)
DEFAULT_SHIP = Metrics(20, 4, 5)
DEFAULT_AIRCRAFT = Metrics(60, 1, 3)
DEFAULT_VEHICLE = Metrics(10, 0, 2)


def _report(
    overall: Metrics = DEFAULT_OVERALL,
    ship: Metrics = DEFAULT_SHIP,
    aircraft: Metrics = DEFAULT_AIRCRAFT,
    vehicle: Metrics = DEFAULT_VEHICLE,
) -> EvaluationReport:
    return EvaluationReport(
        overall_class_agnostic=overall,
        by_coarse_class={
            "ship": ship,
            "aircraft": aircraft,
            "vehicle": vehicle,
        },
        by_fine_class={0: ship, 4: aircraft, 24: vehicle},
        by_image={},
    )


def test_build_competition_proxy_marks_pass_candidate_without_timing() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="unit")

    assert proxy["experiment_name"] == "unit"
    assert proxy["recommendation"] == "pass_candidate"
    assert proxy["hard_gates"] == {
        "overall_recall": {"value": 0.9, "threshold": 0.85, "passed": True},
        "overall_fdr": {"value": pytest.approx(5 / 95), "threshold": 0.2, "passed": True},
        "latency_seconds": {"value": None, "threshold": 20.0, "passed": None},
    }
    assert proxy["ranking_proxy"]["ship_recall"] == 0.8
    assert proxy["ranking_proxy"]["ship_fdr"] == pytest.approx(4 / 24)
    assert proxy["ranking_proxy"]["overall_timeliness_seconds"] is None


def test_build_competition_proxy_fails_accuracy_gate_before_timing_gate() -> None:
    proxy = build_competition_proxy(
        _report(overall=Metrics(80, 30, 30)),
        experiment_name="bad-accuracy",
        latency_seconds=25.0,
    )

    assert proxy["recommendation"] == "accuracy_gate_failed"
    assert proxy["hard_gates"]["overall_recall"]["passed"] is False
    assert proxy["hard_gates"]["overall_fdr"]["passed"] is False
    assert proxy["hard_gates"]["latency_seconds"]["passed"] is False


def test_build_competition_proxy_fails_timing_when_accuracy_passes() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="slow", latency_seconds=20.1)

    assert proxy["recommendation"] == "timing_gate_failed"
    assert proxy["hard_gates"]["latency_seconds"] == {
        "value": 20.1,
        "threshold": 20.0,
        "passed": False,
    }


def test_build_competition_proxy_rejects_negative_latency() -> None:
    with pytest.raises(ValueError, match="latency_seconds"):
        build_competition_proxy(_report(), experiment_name="bad", latency_seconds=-0.1)


def test_load_evaluation_report_round_trips_report_to_dict(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report_to_dict(_report())), encoding="utf-8")

    loaded = load_evaluation_report(path)

    assert loaded.overall_class_agnostic == Metrics(90, 5, 10)
    assert loaded.by_coarse_class["ship"] == Metrics(20, 4, 5)
    assert loaded.by_fine_class[24] == Metrics(10, 0, 2)


def test_write_competition_proxy_artifacts_writes_json_and_markdown(tmp_path: Path) -> None:
    artifacts = write_competition_proxy_artifacts(
        _report(),
        output_dir=tmp_path,
        experiment_name="unit",
        latency_seconds=12.5,
    )

    assert set(artifacts) == {"json", "markdown"}
    payload = json.loads((tmp_path / "competition-proxy.json").read_text(encoding="utf-8"))
    markdown = (tmp_path / "competition-proxy.md").read_text(encoding="utf-8")
    assert payload["ranking_proxy"]["overall_timeliness_seconds"] == 12.5
    assert "| Overall Recall |" in markdown
    assert "pass_candidate" in markdown


def test_render_competition_proxy_markdown_includes_all_ranking_signals() -> None:
    proxy = build_competition_proxy(_report(), experiment_name="unit", latency_seconds=10.0)

    markdown = render_competition_proxy_markdown(proxy)

    for label in (
        "Ship Recall",
        "Ship FDR",
        "Aircraft Recall",
        "Aircraft FDR",
        "Vehicle Recall",
        "Vehicle FDR",
        "Overall Timeliness",
    ):
        assert label in markdown
