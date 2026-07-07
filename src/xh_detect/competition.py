from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from xh_detect.evaluator import EvaluationReport, Metrics

RECALL_GATE = 0.85
FDR_GATE = 0.20
LATENCY_GATE_SECONDS = 20.0
COARSE_GROUPS = ("ship", "aircraft", "vehicle")


def _metric_from_mapping(payload: Mapping[str, object], label: str) -> Metrics:
    values: dict[str, int] = {}
    for key in ("tp", "fp", "fn"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} metric {key!r} must be a non-negative integer")
        values[key] = value
    return Metrics(values["tp"], values["fp"], values["fn"])


def _mapping_section(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    section = payload.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"evaluation report missing mapping section {key!r}")
    return section


def load_evaluation_report(path: Path | str) -> EvaluationReport:
    report_path = Path(path)
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation report JSON root must be an object")

    overall = _metric_from_mapping(
        _mapping_section(raw, "overall_class_agnostic"),
        "overall_class_agnostic",
    )
    coarse_raw = _mapping_section(raw, "by_coarse_class")
    fine_raw = _mapping_section(raw, "by_fine_class")
    image_raw = _mapping_section(raw, "by_image")
    by_coarse = {
        str(name): _metric_from_mapping(_mapping_section(coarse_raw, str(name)), f"coarse {name}")
        for name in sorted(coarse_raw)
    }
    by_fine = {
        int(class_id): _metric_from_mapping(
            _mapping_section(fine_raw, str(class_id)),
            f"fine {class_id}",
        )
        for class_id in sorted(fine_raw, key=lambda item: int(item))
    }
    by_image = {
        str(image_id): _metric_from_mapping(
            _mapping_section(image_raw, str(image_id)),
            f"image {image_id}",
        )
        for image_id in sorted(image_raw)
    }
    return EvaluationReport(
        overall_class_agnostic=overall,
        by_coarse_class=by_coarse,
        by_fine_class=by_fine,
        by_image=by_image,
    )


def _metric_payload(metrics: Metrics) -> dict[str, float | int]:
    return {
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "recall": metrics.recall,
        "fdr": metrics.fdr,
    }


def _gate(
    value: float | None,
    threshold: float,
    passed: bool | None,
) -> dict[str, float | bool | None]:
    return {"value": value, "threshold": threshold, "passed": passed}


def _validate_latency(latency_seconds: float | None) -> float | None:
    if latency_seconds is None:
        return None
    if (
        isinstance(latency_seconds, bool)
        or not isinstance(latency_seconds, int | float)
        or not math.isfinite(float(latency_seconds))
        or latency_seconds < 0.0
    ):
        raise ValueError("latency_seconds must be a non-negative finite number")
    return float(latency_seconds)


def build_competition_proxy(
    report: EvaluationReport,
    *,
    experiment_name: str,
    latency_seconds: float | None = None,
) -> dict[str, object]:
    if not experiment_name.strip():
        raise ValueError("experiment_name must be non-empty")
    latency = _validate_latency(latency_seconds)
    overall = report.overall_class_agnostic
    missing_groups = [group for group in COARSE_GROUPS if group not in report.by_coarse_class]
    if missing_groups:
        raise ValueError("missing coarse groups: " + ", ".join(missing_groups))

    recall_passed = overall.recall >= RECALL_GATE
    fdr_passed = overall.fdr <= FDR_GATE
    timing_passed = None if latency is None else latency <= LATENCY_GATE_SECONDS
    if not recall_passed or not fdr_passed:
        recommendation = "accuracy_gate_failed"
    elif timing_passed is False:
        recommendation = "timing_gate_failed"
    else:
        recommendation = "pass_candidate"

    ranking_proxy = {
        "ship_recall": report.by_coarse_class["ship"].recall,
        "ship_fdr": report.by_coarse_class["ship"].fdr,
        "aircraft_recall": report.by_coarse_class["aircraft"].recall,
        "aircraft_fdr": report.by_coarse_class["aircraft"].fdr,
        "vehicle_recall": report.by_coarse_class["vehicle"].recall,
        "vehicle_fdr": report.by_coarse_class["vehicle"].fdr,
        "overall_timeliness_seconds": latency,
    }
    return {
        "experiment_name": experiment_name,
        "recommendation": recommendation,
        "hard_gates": {
            "overall_recall": _gate(overall.recall, RECALL_GATE, recall_passed),
            "overall_fdr": _gate(overall.fdr, FDR_GATE, fdr_passed),
            "latency_seconds": _gate(latency, LATENCY_GATE_SECONDS, timing_passed),
        },
        "overall": _metric_payload(overall),
        "coarse": {
            group: _metric_payload(report.by_coarse_class[group])
            for group in COARSE_GROUPS
        },
        "ranking_proxy": ranking_proxy,
    }


def _fmt(value: object) -> str:
    if value is None:
        return "not measured"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def render_competition_proxy_markdown(proxy: Mapping[str, object]) -> str:
    hard_gates = proxy["hard_gates"]
    ranking = proxy["ranking_proxy"]
    if not isinstance(hard_gates, Mapping) or not isinstance(ranking, Mapping):
        raise ValueError("competition proxy is malformed")

    gate_rows = (
        ("Overall Recall", hard_gates["overall_recall"]),
        ("Overall FDR", hard_gates["overall_fdr"]),
        ("Latency Seconds", hard_gates["latency_seconds"]),
    )
    ranking_rows = (
        ("Ship Recall", ranking["ship_recall"]),
        ("Ship FDR", ranking["ship_fdr"]),
        ("Aircraft Recall", ranking["aircraft_recall"]),
        ("Aircraft FDR", ranking["aircraft_fdr"]),
        ("Vehicle Recall", ranking["vehicle_recall"]),
        ("Vehicle FDR", ranking["vehicle_fdr"]),
        ("Overall Timeliness", ranking["overall_timeliness_seconds"]),
    )
    lines = [
        f"# {proxy['experiment_name']} Competition Proxy",
        "",
        f"- Recommendation: `{proxy['recommendation']}`",
        "",
        "## Hard Gates",
        "",
        "| Gate | Value | Threshold | Passed |",
        "| --- | ---: | ---: | --- |",
    ]
    for label, gate in gate_rows:
        if not isinstance(gate, Mapping):
            raise ValueError("competition proxy gate is malformed")
        lines.append(
            f"| {label} | {_fmt(gate['value'])} | {_fmt(gate['threshold'])} | {gate['passed']} |"
        )
    lines.extend(
        [
            "",
            "## Ranking Proxy Signals",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
        ]
    )
    for label, value in ranking_rows:
        lines.append(f"| {label} | {_fmt(value)} |")
    lines.append("")
    return "\n".join(lines)


def write_competition_proxy_artifacts(
    report: EvaluationReport,
    *,
    output_dir: Path,
    experiment_name: str,
    latency_seconds: float | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    proxy = build_competition_proxy(
        report,
        experiment_name=experiment_name,
        latency_seconds=latency_seconds,
    )
    json_path = output_dir / "competition-proxy.json"
    markdown_path = output_dir / "competition-proxy.md"
    json_path.write_text(
        json.dumps(proxy, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path.write_text(render_competition_proxy_markdown(proxy), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
