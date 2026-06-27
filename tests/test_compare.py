from __future__ import annotations

import json
from pathlib import Path

import pytest

from xh_detect.compare import compare_experiments


def _report(tp: int, fp: int, fn: int) -> dict[str, object]:
    return {
        "overall_class_agnostic": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / (tp + fn),
            "fdr": fp / (fp + tp),
        },
        "by_coarse_class": {
            "ship": {"tp": 2, "fp": 1, "fn": 3, "recall": 0.4, "fdr": 1 / 3},
            "aircraft": {"tp": 5, "fp": 2, "fn": 1, "recall": 5 / 6, "fdr": 2 / 7},
            "vehicle": {"tp": 1, "fp": 3, "fn": 4, "recall": 0.2, "fdr": 0.75},
        },
        "by_fine_class": {
            "0": {"tp": 1, "fp": 0, "fn": 1, "recall": 0.5, "fdr": 0.0},
            "1": {"tp": 0, "fp": 1, "fn": 2, "recall": 0.0, "fdr": 1.0},
            "24": {"tp": 1, "fp": 3, "fn": 4, "recall": 0.2, "fdr": 0.75},
        },
        "by_image": {},
    }


def test_compare_experiments_writes_json_and_markdown(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline_benchmark = tmp_path / "baseline-benchmark.json"
    experiment_benchmark = tmp_path / "experiment-benchmark.json"
    output_dir = tmp_path / "comparison"
    baseline_report.write_text(json.dumps(_report(10, 5, 10)), encoding="utf-8")
    experiment_report.write_text(json.dumps(_report(12, 6, 8)), encoding="utf-8")
    baseline_benchmark.write_text(json.dumps({"median_s": 10.0, "p95_s": 12.0}), encoding="utf-8")
    experiment_benchmark.write_text(
        json.dumps({"median_s": 11.0, "p95_s": 13.5}),
        encoding="utf-8",
    )

    comparison = compare_experiments(
        baseline_report=baseline_report,
        experiment_report=experiment_report,
        output_dir=output_dir,
        baseline_name="xh25-yolo26s-e80",
        experiment_name="xh25-mksnet-lite",
        baseline_benchmark=baseline_benchmark,
        experiment_benchmark=experiment_benchmark,
    )

    saved = json.loads((output_dir / "comparison.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "comparison.md").read_text(encoding="utf-8")

    assert comparison["overall"]["recall_delta"] == 0.1
    assert saved["overall"]["experiment_recall"] == 0.6
    assert "vehicle" in saved["coarse"]
    assert "coarse_groups" not in saved
    assert saved["benchmark"]["median_s_delta"] == 1.0
    assert "xh25-mksnet-lite" in markdown
    assert "| vehicle |" in markdown


@pytest.mark.parametrize(
    ("section", "metric_name", "missing_key"),
    [
        ("by_coarse_class", "vehicle", "recall"),
        ("by_fine_class", "1", "fn"),
    ],
)
def test_compare_experiments_rejects_present_metrics_missing_required_keys(
    tmp_path: Path,
    section: str,
    metric_name: str,
    missing_key: str,
) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline = _report(10, 5, 10)
    del baseline[section][metric_name][missing_key]  # type: ignore[index]
    baseline_report.write_text(json.dumps(baseline), encoding="utf-8")
    experiment_report.write_text(json.dumps(_report(12, 6, 8)), encoding="utf-8")

    with pytest.raises(ValueError, match=f"missing required metric {missing_key!r}"):
        compare_experiments(
            baseline_report=baseline_report,
            experiment_report=experiment_report,
            output_dir=tmp_path / "comparison",
        )


@pytest.mark.parametrize(
    ("metric_key", "bad_value", "message"),
    [
        ("tp", True, "metric 'tp' must be an integer"),
        ("recall", True, "metric 'recall' must be numeric"),
        ("fdr", "bad", "metric 'fdr' must be numeric"),
    ],
)
def test_compare_experiments_rejects_present_metrics_with_invalid_values(
    tmp_path: Path,
    metric_key: str,
    bad_value: object,
    message: str,
) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline = _report(10, 5, 10)
    baseline["by_coarse_class"]["vehicle"][metric_key] = bad_value  # type: ignore[index]
    baseline_report.write_text(json.dumps(baseline), encoding="utf-8")
    experiment_report.write_text(json.dumps(_report(12, 6, 8)), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compare_experiments(
            baseline_report=baseline_report,
            experiment_report=experiment_report,
            output_dir=tmp_path / "comparison",
        )


def test_compare_experiments_zero_fills_missing_coarse_and_watchlist_entries(
    tmp_path: Path,
) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline = _report(10, 5, 10)
    experiment = _report(12, 6, 8)
    del baseline["by_coarse_class"]["vehicle"]  # type: ignore[index]
    del experiment["by_fine_class"]["24"]  # type: ignore[index]
    baseline_report.write_text(json.dumps(baseline), encoding="utf-8")
    experiment_report.write_text(json.dumps(experiment), encoding="utf-8")

    comparison = compare_experiments(
        baseline_report=baseline_report,
        experiment_report=experiment_report,
        output_dir=tmp_path / "comparison",
    )

    assert comparison["coarse"]["vehicle"]["baseline_tp"] == 0
    assert comparison["coarse"]["vehicle"]["experiment_tp"] == 1
    assert comparison["fine_watchlist"]["24"]["baseline_tp"] == 1
    assert comparison["fine_watchlist"]["24"]["experiment_tp"] == 0


def test_compare_experiments_rejects_asymmetric_benchmark_keys(tmp_path: Path) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline_benchmark = tmp_path / "baseline-benchmark.json"
    experiment_benchmark = tmp_path / "experiment-benchmark.json"
    baseline_report.write_text(json.dumps(_report(10, 5, 10)), encoding="utf-8")
    experiment_report.write_text(json.dumps(_report(12, 6, 8)), encoding="utf-8")
    baseline_benchmark.write_text(json.dumps({"median_s": 10.0}), encoding="utf-8")
    experiment_benchmark.write_text(json.dumps({"p95_s": 13.5}), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark keys differ"):
        compare_experiments(
            baseline_report=baseline_report,
            experiment_report=experiment_report,
            output_dir=tmp_path / "comparison",
            baseline_benchmark=baseline_benchmark,
            experiment_benchmark=experiment_benchmark,
        )
