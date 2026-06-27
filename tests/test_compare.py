from __future__ import annotations

import json
from pathlib import Path

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
    assert saved["benchmark"]["median_s_delta"] == 1.0
    assert "xh25-mksnet-lite" in markdown
    assert "| vehicle |" in markdown
