from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

FINE_WATCHLIST = ("0", "1", "24")
REQUIRED_METRIC_KEYS = ("tp", "fp", "fn", "recall", "fdr")
ROUND_DIGITS = 6
ZERO_METRICS = {"tp": 0, "fp": 0, "fn": 0, "recall": 0.0, "fdr": 0.0}


def compare_experiments(
    *,
    baseline_report: Path,
    experiment_report: Path,
    output_dir: Path,
    baseline_name: str = "xh25-yolo26s-e80",
    experiment_name: str = "xh25-mksnet-lite",
    baseline_benchmark: Path | None = None,
    experiment_benchmark: Path | None = None,
) -> dict[str, object]:
    baseline = _load_json_object(baseline_report, "baseline report")
    experiment = _load_json_object(experiment_report, "experiment report")

    baseline_coarse = _section(baseline, "by_coarse_class", "baseline report")
    experiment_coarse = _section(experiment, "by_coarse_class", "experiment report")
    baseline_fine = _section(baseline, "by_fine_class", "baseline report")
    experiment_fine = _section(experiment, "by_fine_class", "experiment report")

    comparison: dict[str, object] = {
        "baseline_name": baseline_name,
        "experiment_name": experiment_name,
        "overall": _metric_block(
            _section(baseline, "overall_class_agnostic", "baseline report"),
            _section(experiment, "overall_class_agnostic", "experiment report"),
            baseline_label="baseline overall",
            experiment_label="experiment overall",
        ),
        "coarse": {
            group: _metric_block(
                _nested_metric(baseline_coarse, group, "baseline coarse group"),
                _nested_metric(experiment_coarse, group, "experiment coarse group"),
                baseline_label=f"baseline coarse group {group!r}",
                experiment_label=f"experiment coarse group {group!r}",
            )
            for group in sorted(set(baseline_coarse) | set(experiment_coarse))
        },
        "fine_watchlist": {
            class_id: _metric_block(
                _nested_metric(baseline_fine, class_id, "baseline fine class"),
                _nested_metric(experiment_fine, class_id, "experiment fine class"),
                baseline_label=f"baseline fine class {class_id!r}",
                experiment_label=f"experiment fine class {class_id!r}",
            )
            for class_id in FINE_WATCHLIST
        },
    }

    if baseline_benchmark is not None or experiment_benchmark is not None:
        if baseline_benchmark is None or experiment_benchmark is None:
            raise ValueError(
                "baseline_benchmark and experiment_benchmark must be provided together"
            )
        comparison["benchmark"] = _benchmark_block(
            _load_json_object(baseline_benchmark, "baseline benchmark"),
            _load_json_object(experiment_benchmark, "experiment benchmark"),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "comparison.md").write_text(
        _render_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return payload


def _section(report: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    section = report.get(key)
    if not isinstance(section, Mapping):
        raise ValueError(f"{label} must contain object field {key!r}")
    return section


def _nested_metric(
    section: Mapping[str, object],
    key: str,
    label: str,
) -> Mapping[str, object]:
    if key not in section:
        return ZERO_METRICS
    value = section[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} {key!r} must be an object")
    return value


def _metric_block(
    baseline: Mapping[str, object],
    experiment: Mapping[str, object],
    *,
    baseline_label: str,
    experiment_label: str,
) -> dict[str, int | float]:
    _validate_metric_object(baseline, baseline_label)
    _validate_metric_object(experiment, experiment_label)
    baseline_tp = _count(baseline, "tp")
    baseline_fp = _count(baseline, "fp")
    baseline_fn = _count(baseline, "fn")
    experiment_tp = _count(experiment, "tp")
    experiment_fp = _count(experiment, "fp")
    experiment_fn = _count(experiment, "fn")
    baseline_recall = _metric_value(baseline, "recall")
    experiment_recall = _metric_value(experiment, "recall")
    baseline_fdr = _metric_value(baseline, "fdr")
    experiment_fdr = _metric_value(experiment, "fdr")

    return {
        "baseline_tp": baseline_tp,
        "baseline_fp": baseline_fp,
        "baseline_fn": baseline_fn,
        "experiment_tp": experiment_tp,
        "experiment_fp": experiment_fp,
        "experiment_fn": experiment_fn,
        "baseline_recall": baseline_recall,
        "experiment_recall": experiment_recall,
        "recall_delta": _stable_float(experiment_recall - baseline_recall),
        "baseline_fdr": baseline_fdr,
        "experiment_fdr": experiment_fdr,
        "fdr_delta": _stable_float(experiment_fdr - baseline_fdr),
    }


def _validate_metric_object(metrics: Mapping[str, object], label: str) -> None:
    for key in REQUIRED_METRIC_KEYS:
        if key not in metrics:
            raise ValueError(f"{label} missing required metric {key!r}")


def _count(metrics: Mapping[str, object], key: str) -> int:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metric {key!r} must be an integer")
    return value


def _metric_value(metrics: Mapping[str, object], key: str) -> float:
    value = metrics[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metric {key!r} must be numeric")
    return _stable_float(value)


def _stable_float(value: int | float) -> float:
    return round(float(value), ROUND_DIGITS)


def _benchmark_block(
    baseline: Mapping[str, object],
    experiment: Mapping[str, object],
) -> dict[str, float]:
    baseline_keys = set(baseline)
    experiment_keys = set(experiment)
    if baseline_keys != experiment_keys:
        baseline_only = ", ".join(sorted(str(key) for key in baseline_keys - experiment_keys))
        experiment_only = ", ".join(sorted(str(key) for key in experiment_keys - baseline_keys))
        raise ValueError(
            "benchmark keys differ"
            f" (baseline only: {baseline_only or 'none'}; "
            f"experiment only: {experiment_only or 'none'})"
        )

    block: dict[str, float] = {}
    for key in sorted(baseline_keys):
        baseline_value = _number(baseline[key], f"baseline benchmark {key!r}")
        experiment_value = _number(experiment[key], f"experiment benchmark {key!r}")
        block[f"baseline_{key}"] = baseline_value
        block[f"experiment_{key}"] = experiment_value
        block[f"{key}_delta"] = _stable_float(experiment_value - baseline_value)
    return block


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    return _stable_float(value)


def _render_markdown(comparison: Mapping[str, object]) -> str:
    lines = [
        "# MKSNet-Lite Comparison",
        "",
        f"- Baseline: {comparison['baseline_name']}",
        f"- Experiment: {comparison['experiment_name']}",
        "",
        "## Overall",
        "",
    ]
    lines.extend(_metrics_table(["Overall"], {"Overall": comparison["overall"]}))
    lines.extend(["", "## Coarse Groups", ""])
    lines.extend(_metrics_table(["Group"], _as_mapping(comparison["coarse"])))
    lines.extend(["", "## Fine Watchlist", ""])
    lines.extend(_metrics_table(["Class"], _as_mapping(comparison["fine_watchlist"])))
    if "benchmark" in comparison:
        lines.extend(["", "## Benchmark", ""])
        lines.extend(_benchmark_table(_as_mapping(comparison["benchmark"])))
    lines.append("")
    return "\n".join(lines)


def _metrics_table(label_headers: list[str], rows: Mapping[str, object]) -> list[str]:
    label_header = label_headers[0]
    table = [
        (
            f"| {label_header} | Baseline TP | Baseline FP | Baseline FN | "
            "Experiment TP | Experiment FP | Experiment FN | Baseline Recall | "
            "Experiment Recall | Recall Delta | Baseline FDR | Experiment FDR | FDR Delta |"
        ),
        (
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: | ---: | ---: | ---: |"
        ),
    ]
    for name, block in rows.items():
        metrics = _as_mapping(block)
        table.append(
            f"| {name} | {metrics['baseline_tp']} | {metrics['baseline_fp']} | "
            f"{metrics['baseline_fn']} | {metrics['experiment_tp']} | "
            f"{metrics['experiment_fp']} | {metrics['experiment_fn']} | "
            f"{_format_number(metrics['baseline_recall'])} | "
            f"{_format_number(metrics['experiment_recall'])} | "
            f"{_format_number(metrics['recall_delta'])} | "
            f"{_format_number(metrics['baseline_fdr'])} | "
            f"{_format_number(metrics['experiment_fdr'])} | "
            f"{_format_number(metrics['fdr_delta'])} |"
        )
    return table


def _benchmark_table(benchmark: Mapping[str, object]) -> list[str]:
    rows = ["| Metric | Baseline | Experiment | Delta |", "| --- | ---: | ---: | ---: |"]
    metric_names = sorted(
        key[: -len("_delta")]
        for key in benchmark
        if key.endswith("_delta") and f"baseline_{key[: -len('_delta')]}" in benchmark
    )
    for name in metric_names:
        rows.append(
            f"| {name} | {_format_number(benchmark[f'baseline_{name}'])} | "
            f"{_format_number(benchmark[f'experiment_{name}'])} | "
            f"{_format_number(benchmark[f'{name}_delta'])} |"
        )
    return rows


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("expected mapping while rendering comparison")
    return value


def _format_number(value: object) -> str:
    if isinstance(value, float):
        text = f"{value:.{ROUND_DIGITS}f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)
