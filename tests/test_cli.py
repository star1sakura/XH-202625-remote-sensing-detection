import json
from importlib import metadata, reload
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import typer
from typer.testing import CliRunner

import xh_detect
from xh_detect.cli import _load_image_id_map, app
from xh_detect.config import PipelineConfig
from xh_detect.data.dota import ConversionStats
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, InferenceResult, StageTimings


def test_package_version_comes_from_distribution_metadata(monkeypatch) -> None:
    def fake_version(distribution_name: str) -> str:
        assert distribution_name == "xh-detect"
        return "9.8.7"

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "version", fake_version)

        assert reload(xh_detect).__version__ == "9.8.7"

    reload(xh_detect)


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "xh-detect 0.1.0"


@patch("xh_detect.cli.write_dataset_yaml")
@patch("xh_detect.cli.convert_split")
def test_prepare_dota_command_converts_train_and_val(
    convert_split: Mock,
    write_dataset_yaml: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "converted"
    for relative in [
        Path("images/train"),
        Path("images/val"),
        Path("labelTxt/train"),
        Path("labelTxt/val"),
    ]:
        (source / relative).mkdir(parents=True)
    convert_split.side_effect = [
        ConversionStats(2, {0: 1, 1: 0, 2: 3}, 0, 0),
        ConversionStats(1, {0: 0, 1: 1, 2: 0}, 1, 0),
    ]
    write_dataset_yaml.return_value = output / "dataset.yaml"

    result = CliRunner().invoke(
        app,
        [
            "prepare-dota",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert [call.args[-1] for call in convert_split.call_args_list] == ["train", "val"]
    write_dataset_yaml.assert_called_once_with(output)
    payload = json.loads(result.stdout)
    assert payload["train"]["targets"] == {"0": 1, "1": 0, "2": 3}
    assert payload["dataset_yaml"].endswith("dataset.yaml")


def test_prepare_dota_command_rejects_incomplete_source_layout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()

    result = CliRunner().invoke(
        app,
        ["prepare-dota", "--source-root", str(source)],
    )

    assert result.exit_code != 0
    assert "missing directories" in result.output


@patch("xh_detect.cli.prepare_dataset")
def test_prepare_xh25_command_reports_output(
    prepare_dataset_mock: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "data"
    source.mkdir()
    output = tmp_path / "xh25"
    prepare_dataset_mock.return_value = SimpleNamespace(
        output_root=output,
        train_stems=frozenset({"a", "b"}),
        val_stems=frozenset({"c"}),
        train_class_counts={class_id: 1 for class_id in range(25)},
        val_class_counts={class_id: 1 for class_id in range(25)},
    )

    result = CliRunner().invoke(
        app,
        [
            "prepare-xh25",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--val-ratio",
            "0.15",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["train_images"] == 2
    assert json.loads(result.stdout)["val_images"] == 1
    prepare_dataset_mock.assert_called_once_with(source, output, val_ratio=0.15, seed=42)


@patch("xh_detect.cli.publish_train_mining_artifacts")
def test_publish_xh25_train_artifacts_command_reports_paths(
    publish_train_mining_artifacts: Mock,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "xh25"
    dataset.mkdir()
    image_map = dataset / "manifests" / "train-image-map.json"
    truth = dataset / "reports" / "train-ground-truth.json"
    publish_train_mining_artifacts.return_value = (image_map, truth)

    result = CliRunner().invoke(
        app,
        ["publish-xh25-train-artifacts", "--dataset-root", str(dataset)],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "train_image_map": str(image_map),
        "train_ground_truth": str(truth),
    }
    publish_train_mining_artifacts.assert_called_once_with(dataset)


@patch("xh_detect.cli.build_ship_balanced_dataset")
def test_build_ship_balanced_xh25_command_forwards_options(
    build_ship_balanced_dataset: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    output = tmp_path / "xh25-ship-balanced"
    source.mkdir()
    build_ship_balanced_dataset.return_value = SimpleNamespace(
        output_root=output,
        original_train_images=4,
        balanced_train_images=7,
        duplicated_train_images=3,
    )

    result = CliRunner().invoke(
        app,
        [
            "build-ship-balanced-xh25",
            "--source-root",
            str(source),
            "--output-root",
            str(output),
            "--qhs-factor",
            "3",
            "--ms-factor",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["output_root"] == str(output)
    assert payload["balanced_train_images"] == 7
    build_ship_balanced_dataset.assert_called_once_with(
        source,
        output,
        qhs_factor=3,
        ms_factor=2,
    )


@patch("xh_detect.cli.build_main_hn_dataset")
def test_build_main_hn_xh25_command_forwards_policy(
    build_main_hn_dataset: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    source.mkdir()
    predictions = tmp_path / "predictions.json"
    predictions.write_text("[]", encoding="utf-8")
    output = tmp_path / "xh25-main-hn"
    build_main_hn_dataset.return_value = SimpleNamespace(
        output_root=output,
        original_train_images=100,
        vehicle_upsampled_images=4,
        selected_hard_negatives=3,
        rejected_target_overlap=2,
        selected_by_coarse_class={"ship": 1, "vehicle": 2},
    )

    result = CliRunner().invoke(
        app,
        [
            "build-main-hn-xh25",
            "--source-root",
            str(source),
            "--predictions-json",
            str(predictions),
            "--output-root",
            str(output),
            "--confidence-floor",
            "0.60",
            "--crop-size",
            "512",
            "--object-margin",
            "16",
            "--max-crops-per-group",
            "2",
            "--vehicle-multiplier",
            "2",
            "--seed",
            "42",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["selected_hard_negatives"] == 3
    args = build_main_hn_dataset.call_args.args
    assert args[:3] == (source, predictions, output)
    assert args[3].confidence_floor == 0.60
    assert args[3].crop_size == 512
    assert args[3].object_margin == 16
    assert args[3].max_crops_per_group == 2
    assert args[3].vehicle_multiplier == 2
    assert args[3].seed == 42


@patch("xh_detect.cli.build_vehicle_confirmer_dataset")
def test_build_vehicle_confirmer_dataset_command_forwards_policy(
    build_dataset: Mock,
    tmp_path: Path,
) -> None:
    source = tmp_path / "xh25"
    source.mkdir()
    main = tmp_path / "main.json"
    sph = tmp_path / "sph.json"
    main.write_text("[]", encoding="utf-8")
    sph.write_text("[]", encoding="utf-8")
    output = tmp_path / "vehicle-confirmer"
    build_dataset.return_value = SimpleNamespace(
        output_root=output,
        train_examples=80,
        holdout_examples=20,
        train_positive=8,
        train_negative=72,
        holdout_positive=2,
        holdout_negative=18,
    )

    result = CliRunner().invoke(
        app,
        [
            "build-vehicle-confirmer-dataset",
            "--source-root",
            str(source),
            "--main-predictions-json",
            str(main),
            "--sph-predictions-json",
            str(sph),
            "--output-root",
            str(output),
            "--context-scale",
            "2.5",
            "--min-side",
            "48",
            "--max-side",
            "224",
            "--output-size",
            "160",
            "--holdout-ratio",
            "0.25",
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["holdout_positive"] == 2
    args = build_dataset.call_args.args
    assert args[:4] == (source, main, sph, output)
    assert args[4].context_scale == 2.5
    assert args[4].min_side == 48
    assert args[4].max_side == 224
    assert args[4].holdout_ratio == 0.25
    assert args[4].seed == 7


def test_analyze_complementarity_command_writes_pairwise_report(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    main_predictions = tmp_path / "main.json"
    candidate_predictions = tmp_path / "candidate.json"
    output = tmp_path / "complementarity.json"
    truth.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10]},
                    {"image_id": 1, "category_id": 24, "bbox": [20, 0, 10, 10]},
                ]
            }
        ),
        encoding="utf-8",
    )
    main_predictions.write_text(
        json.dumps([{"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9}]),
        encoding="utf-8",
    )
    candidate_predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 24, "bbox": [20, 0, 10, 10], "score": 0.8},
            ]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "analyze-complementarity",
            "--prediction",
            f"main={main_predictions}",
            "--prediction",
            f"candidate={candidate_predictions}",
            "--ground-truth-json",
            str(truth),
            "--baseline-name",
            "main",
            "--taxonomy",
            "xh25",
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["pairwise"]["candidate"]["vehicle"]["candidate_only_tp"] == 1


def test_analyze_vehicle_proposals_command_writes_consensus_report(tmp_path: Path) -> None:
    truth = tmp_path / "truth.json"
    main_predictions = tmp_path / "main.json"
    sph_predictions = tmp_path / "sph.json"
    mks_predictions = tmp_path / "mks.json"
    output = tmp_path / "vehicle-proposals.json"
    truth.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10]},
                    {"image_id": 1, "category_id": 24, "bbox": [30, 0, 10, 10]},
                ]
            }
        ),
        encoding="utf-8",
    )
    main_predictions.write_text(
        json.dumps(
            [{"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.9}]
        ),
        encoding="utf-8",
    )
    sph_predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 24, "bbox": [30, 0, 10, 10], "score": 0.8},
                {"image_id": 1, "category_id": 24, "bbox": [60, 0, 10, 10], "score": 0.7},
            ]
        ),
        encoding="utf-8",
    )
    mks_predictions.write_text(
        json.dumps(
            [{"image_id": 1, "category_id": 24, "bbox": [30, 0, 10, 10], "score": 0.8}]
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "analyze-vehicle-proposals",
            "--main-predictions",
            str(main_predictions),
            "--sph-predictions",
            str(sph_predictions),
            "--mks-predictions",
            str(mks_predictions),
            "--ground-truth-json",
            str(truth),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["consensus"]["recoverable_tp"] == 1


def test_apply_suppression_command_writes_filtered_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    image_map = tmp_path / "image-map.json"
    config = tmp_path / "config.yaml"
    output = tmp_path / "filtered.json"
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 3, "bbox": [6, 0, 10, 10], "score": 0.8},
            ]
        ),
        encoding="utf-8",
    )
    image_map.write_text(json.dumps({"image": 1}), encoding="utf-8")
    config.write_text(
        "task: detect\n"
        "taxonomy: xh25\n"
        "class_suppression:\n"
        "  3: {method: iou, threshold: 0.20}\n"
        "class_thresholds:\n" + "".join(f"  {class_id}: 0.25\n" for class_id in range(25)),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "apply-suppression",
            "--predictions-json",
            str(predictions),
            "--image-map-json",
            str(image_map),
            "--config-path",
            str(config),
            "--output-json",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(json.loads(output.read_text(encoding="utf-8"))) == 1


def test_audit_false_positives_command_writes_sources(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "audit.json"
    predictions.write_text(
        json.dumps(
            [
                {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.9},
                {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10], "score": 0.8},
                {"image_id": 1, "category_id": 3, "bbox": [30, 0, 10, 10], "score": 0.7},
            ]
        ),
        encoding="utf-8",
    )
    truth.write_text(
        json.dumps({"annotations": [{"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10]}]}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        app,
        [
            "audit-false-positives",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--taxonomy",
            "xh25",
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    ship = json.loads(output.read_text(encoding="utf-8"))["by_coarse_class"]["ship"]
    assert ship == {"overlap": 1, "background": 1, "total": 2}


@patch("xh_detect.cli.train_model")
def test_train_command_calls_wrapper(train_model: Mock, tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        ["train", "--dataset-yaml", str(dataset), "--epochs", "2"],
    )

    assert result.exit_code == 0, result.output
    train_model.assert_called_once_with(
        str(dataset),
        "yolo26s.pt",
        2,
        1024,
        "0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
        pretrained=None,
    )


@patch("xh_detect.cli.train_model")
def test_train_command_forwards_reproducible_options(
    train_model: Mock,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-yaml",
            str(dataset),
            "--model",
            "yolo26s-obb.pt",
            "--epochs",
            "1",
            "--image-size",
            "512",
            "--device",
            "cpu",
            "--batch",
            "2",
            "--workers",
            "0",
            "--amp",
            "--project",
            "runs/obb",
            "--name",
            "legacy-obb",
            "--resume",
        ],
    )

    assert result.exit_code == 0, result.output
    train_model.assert_called_once_with(
        str(dataset),
        "yolo26s-obb.pt",
        1,
        512,
        "cpu",
        batch=2,
        workers=0,
        amp=True,
        project="runs/obb",
        name="legacy-obb",
        resume=True,
        pretrained=None,
    )


@patch("xh_detect.cli.train_model")
def test_train_command_forwards_pretrained_option(
    train_model: Mock,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-yaml",
            str(dataset),
            "--model",
            "configs/models/xh25-mksnet-lite.yaml",
            "--pretrained",
            "yolo26s.pt",
            "--epochs",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    train_model.assert_called_once_with(
        str(dataset),
        "configs/models/xh25-mksnet-lite.yaml",
        2,
        1024,
        "0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
        pretrained="yolo26s.pt",
    )


@patch("xh_detect.cli.train_model")
def test_train_command_forwards_density_assignment_options(
    train_model: Mock,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.yaml"
    dataset.write_text("names: {}", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "train",
            "--dataset-yaml",
            str(dataset),
            "--epochs",
            "2",
            "--density-assignment",
            "--density-constant",
            "16",
            "--density-threshold",
            "0.3",
        ],
    )

    assert result.exit_code == 0, result.output
    train_model.assert_called_once_with(
        str(dataset),
        "yolo26s.pt",
        2,
        1024,
        "0",
        batch=8,
        workers=4,
        amp=False,
        project="runs/train",
        name="xh25-baseline",
        resume=False,
        pretrained=None,
        density_assignment=True,
        density_constant=16.0,
        density_threshold=0.3,
    )


@patch("xh_detect.cli.export_tensorrt", return_value="model.engine")
def test_export_engine_command_prints_path(export_tensorrt: Mock) -> None:
    result = CliRunner().invoke(
        app,
        ["export-engine", "--model-path", "best.pt", "--image-size", "640"],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == "model.engine"
    export_tensorrt.assert_called_once_with("best.pt", 640, "0")


@patch("xh_detect.cli.report_to_dict", return_value={"overall": {"recall": 1.0}})
@patch("xh_detect.cli.evaluate_detections")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=[])
@patch("xh_detect.cli.load_coco_predictions", return_value=[])
def test_evaluate_command_writes_report(
    load_predictions: Mock,
    load_truth: Mock,
    evaluate_detections: Mock,
    report_to_dict: Mock,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "report.json"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == {"overall": {"recall": 1.0}}
    taxonomy = get_taxonomy("legacy3")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    evaluate_detections.assert_called_once_with([], [], taxonomy=taxonomy)
    report_to_dict.assert_called_once()


@patch("xh_detect.cli.report_to_dict", return_value={"overall": {"recall": 1.0}})
@patch("xh_detect.cli.evaluate_detections")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=[])
@patch("xh_detect.cli.load_coco_predictions", return_value=[])
def test_evaluate_command_uses_xh25_taxonomy(
    load_predictions: Mock,
    load_truth: Mock,
    evaluate_detections: Mock,
    report_to_dict: Mock,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "report.json"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-path",
            str(output),
            "--taxonomy",
            "xh25",
        ],
    )

    assert result.exit_code == 0, result.output
    taxonomy = get_taxonomy("xh25")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    evaluate_detections.assert_called_once_with([], [], taxonomy=taxonomy)
    report_to_dict.assert_called_once()


@patch("xh_detect.cli.write_competition_proxy_artifacts")
@patch("xh_detect.cli.load_evaluation_report")
def test_competition_report_command_writes_artifacts(
    load_evaluation_report: Mock,
    write_competition_proxy_artifacts: Mock,
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    output = tmp_path / "competition"
    report.write_text("{}", encoding="utf-8")
    loaded = object()
    load_evaluation_report.return_value = loaded

    result = CliRunner().invoke(
        app,
        [
            "competition-report",
            "--report-json",
            str(report),
            "--output-dir",
            str(output),
            "--experiment-name",
            "unit",
            "--latency-seconds",
            "12.5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output / "competition-proxy.json")
    load_evaluation_report.assert_called_once_with(report)
    write_competition_proxy_artifacts.assert_called_once_with(
        loaded,
        output_dir=output,
        experiment_name="unit",
        latency_seconds=12.5,
    )


@patch("xh_detect.cli.threshold_sweep")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=[])
@patch("xh_detect.cli.load_coco_predictions", return_value=[])
def test_sweep_thresholds_command_writes_json(
    load_predictions: Mock,
    load_truth: Mock,
    threshold_sweep: Mock,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "sweep.json"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")
    threshold_sweep.return_value = []

    result = CliRunner().invoke(
        app,
        [
            "sweep-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-path",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(output.read_text(encoding="utf-8")) == []
    thresholds = threshold_sweep.call_args.args[2]
    assert thresholds[0] == 0.05
    assert thresholds[-1] == 0.95
    taxonomy = get_taxonomy("legacy3")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    threshold_sweep.assert_called_once_with([], [], thresholds, taxonomy=taxonomy)


@patch("xh_detect.cli.write_threshold_artifacts")
@patch("xh_detect.cli.optimize_thresholds_search")
@patch("xh_detect.cli.load_report_objective")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=["truth"])
@patch("xh_detect.cli.load_coco_predictions", return_value=["prediction"])
def test_optimize_thresholds_command_forwards_options(
    load_predictions: Mock,
    load_truth: Mock,
    load_report_objective: Mock,
    optimize_thresholds_search: Mock,
    write_threshold_artifacts: Mock,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    baseline = tmp_path / "baseline-report.json"
    output = tmp_path / "threshold-optimized"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")
    baseline.write_text("{}", encoding="utf-8")
    baseline_objective = object()
    optimized_result = object()
    load_report_objective.return_value = baseline_objective
    optimize_thresholds_search.return_value = optimized_result

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-dir",
            str(output),
            "--taxonomy",
            "xh25",
            "--baseline-report",
            str(baseline),
            "--experiment-name",
            "unit-thresholds",
            "--thresholds",
            "0.5,0.2,0.5",
            "--recall-floor-delta",
            "0.01",
            "--tie-epsilon",
            "0.002",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output / "report.json")
    taxonomy = get_taxonomy("xh25")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    load_report_objective.assert_called_once_with(baseline)
    optimize_thresholds_search.assert_called_once_with(
        ["prediction"],
        ["truth"],
        taxonomy=taxonomy,
        thresholds=[0.2, 0.5],
        baseline_objective=baseline_objective,
        recall_floor_delta=0.01,
        tie_epsilon=0.002,
    )
    write_threshold_artifacts.assert_called_once_with(
        optimized_result,
        output_dir=output,
        taxonomy=taxonomy,
        experiment_name="unit-thresholds",
        baseline_report=baseline,
    )


@patch("xh_detect.cli.write_threshold_artifacts")
@patch("xh_detect.cli.optimize_thresholds_search")
@patch("xh_detect.cli.load_report_objective")
@patch("xh_detect.cli.load_coco_ground_truth", return_value=["truth"])
@patch("xh_detect.cli.load_coco_predictions", return_value=["prediction"])
def test_optimize_thresholds_command_uses_default_baseline_report(
    load_predictions: Mock,
    load_truth: Mock,
    load_report_objective: Mock,
    optimize_thresholds_search: Mock,
    write_threshold_artifacts: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    default_baseline = Path("outputs/xh25/baseline/report.json")
    baseline_path = tmp_path / default_baseline
    output = tmp_path / "threshold-optimized"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")
    baseline_path.parent.mkdir(parents=True)
    baseline_path.write_text("{}", encoding="utf-8")
    baseline_objective = object()
    optimized_result = object()
    load_report_objective.return_value = baseline_objective
    optimize_thresholds_search.return_value = optimized_result

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-dir",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    taxonomy = get_taxonomy("xh25")
    load_predictions.assert_called_once_with(predictions, taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth, taxonomy=taxonomy)
    load_report_objective.assert_called_once_with(default_baseline)
    optimize_thresholds_search.assert_called_once()
    assert optimize_thresholds_search.call_args.kwargs["baseline_objective"] is baseline_objective
    write_threshold_artifacts.assert_called_once_with(
        optimized_result,
        output_dir=output,
        taxonomy=taxonomy,
        experiment_name="xh25-mksnet-lite-threshold-optimized",
        baseline_report=default_baseline,
    )


def test_optimize_thresholds_command_reports_invalid_threshold_grid(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    output = tmp_path / "threshold-optimized"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--output-dir",
            str(output),
            "--thresholds",
            "0.25,bad",
        ],
    )

    assert result.exit_code != 0
    assert "threshold grid value" in result.output or "invalid threshold value" in result.output
    assert "Traceback" not in result.output


def test_optimize_thresholds_command_reports_missing_baseline_report(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.json"
    truth = tmp_path / "truth.json"
    baseline = tmp_path / "missing-baseline.json"
    predictions.write_text("[]", encoding="utf-8")
    truth.write_text('{"annotations":[]}', encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "optimize-thresholds",
            "--predictions-json",
            str(predictions),
            "--ground-truth-json",
            str(truth),
            "--baseline-report",
            str(baseline),
        ],
    )

    assert result.exit_code != 0
    assert "baseline report does not exist" in result.output
    assert "Traceback" not in result.output


def _minimal_comparison_report() -> dict[str, object]:
    return {
        "overall_class_agnostic": {
            "tp": 1,
            "fp": 0,
            "fn": 1,
            "recall": 0.5,
            "fdr": 0.0,
        },
        "by_coarse_class": {},
        "by_fine_class": {},
        "by_image": {},
    }


def test_compare_experiments_command_reports_comparison_value_errors(
    tmp_path: Path,
) -> None:
    baseline_report = tmp_path / "baseline-report.json"
    experiment_report = tmp_path / "experiment-report.json"
    baseline_benchmark = tmp_path / "baseline-benchmark.json"
    baseline_report.write_text(json.dumps(_minimal_comparison_report()), encoding="utf-8")
    experiment_report.write_text(json.dumps(_minimal_comparison_report()), encoding="utf-8")
    baseline_benchmark.write_text(json.dumps({"median_s": 1.0}), encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "compare-experiments",
            "--baseline-report",
            str(baseline_report),
            "--experiment-report",
            str(experiment_report),
            "--baseline-benchmark",
            str(baseline_benchmark),
        ],
    )

    assert result.exit_code != 0
    compact_output = "".join(char for char in result.output if char.isalnum() or char == "_")
    assert "baseline_benchmarkandexperiment_benchmarkmustbeprovidedtogether" in compact_output
    assert "Traceback" not in result.output


@patch("xh_detect.cli.export_coco_results")
@patch("xh_detect.cli.draw_detections")
@patch("xh_detect.cli.InferencePipeline")
@patch("xh_detect.cli._build_detector", create=True)
@patch("xh_detect.cli.PipelineConfig.from_yaml")
@patch("xh_detect.cli.cv2.imwrite", return_value=True)
@patch("xh_detect.cli.cv2.imread")
def test_infer_command_uses_shared_pipeline_and_writes_outputs(
    imread: Mock,
    imwrite: Mock,
    from_yaml: Mock,
    build_detector: Mock,
    pipeline_class: Mock,
    draw_detections: Mock,
    export_results: Mock,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    image_path.write_bytes(b"image")
    config_path.write_text("config", encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    imread.return_value = image
    draw_detections.return_value = image
    config = PipelineConfig(model_path="model.pt", device="cpu", half=False)
    from_yaml.return_value = config
    pipeline_class.return_value.run.return_value = InferenceResult(
        detections=(),
        timings=StageTimings(0.1, 0.2, 0.3, 0.6),
    )

    result = CliRunner().invoke(
        app,
        [
            "infer",
            "--image-path",
            str(image_path),
            "--config-path",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    build_detector.assert_called_once_with(config)
    pipeline_class.return_value.run.assert_called_once_with(image, "scene")
    imwrite.assert_called_once()
    taxonomy = get_taxonomy("legacy3")
    draw_detections.assert_called_once_with(image, (), taxonomy=taxonomy)
    export_results.assert_called_once_with(
        (),
        {"scene": 1},
        output_dir / "scene.json",
        valid_class_ids=taxonomy.valid_ids,
    )
    assert json.loads(result.stdout)["total_s"] == 0.6


def test_infer_builds_detect_model_for_xh25(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    image_path.write_bytes(b"image")
    config_path.write_text("config", encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    with (
        patch("xh_detect.cli.cv2.imread", return_value=image),
        patch("xh_detect.cli.cv2.imwrite", return_value=True),
        patch("xh_detect.cli.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.cli.UltralyticsDetector", create=True) as detector_class,
        patch("xh_detect.cli.InferencePipeline") as pipeline_class,
        patch("xh_detect.cli.draw_detections", return_value=image),
        patch("xh_detect.cli.export_coco_results"),
    ):
        pipeline_class.return_value.run.return_value = InferenceResult(
            detections=(),
            timings=StageTimings(0.1, 0.2, 0.3, 0.6),
        )

        result = CliRunner().invoke(
            app,
            [
                "infer",
                "--image-path",
                str(image_path),
                "--config-path",
                str(config_path),
                "--output-dir",
                str(output_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    detector_class.assert_called_once_with("best.pt", "cpu", 1024, False, task="detect")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not-json", "invalid image map JSON:"),
        (json.dumps({"a": 1, "b": 1}), "image map values must be unique"),
        (json.dumps({"a": -1}), "image map values must be non-bool non-negative integers"),
        (json.dumps({"a": 1.2}), "image map values must be non-bool non-negative integers"),
        (json.dumps({"a": True}), "image map values must be non-bool non-negative integers"),
    ],
)
def test_load_image_id_map_rejects_invalid_payloads(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    path = tmp_path / "image-map.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(typer.BadParameter, match=message):
        _load_image_id_map(path)


def test_infer_dataset_exports_stable_image_ids(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "sample.jpg").write_bytes(b"image")
    image_map_json = tmp_path / "image-map.json"
    image_map_json.write_text(json.dumps({"sample": 7}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    output_json = tmp_path / "predictions.json"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    detection = Detection(
        "sample",
        24,
        0.9,
        ((0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)),
    )

    with (
        patch("xh_detect.cli.cv2.imread", return_value=image),
        patch("xh_detect.cli.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.cli._build_detector", create=True),
        patch("xh_detect.cli.InferencePipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.return_value = InferenceResult(
            detections=(detection,),
            timings=StageTimings(0.1, 0.2, 0.3, 0.6),
        )

        result = CliRunner().invoke(
            app,
            [
                "infer-dataset",
                "--images-dir",
                str(images_dir),
                "--image-map-json",
                str(image_map_json),
                "--config-path",
                str(config_path),
                "--output-json",
                str(output_json),
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(output_json.read_text(encoding="utf-8")) == [
        {"image_id": 7, "category_id": 24, "bbox": [0.0, 0.0, 4.0, 4.0], "score": 0.9}
    ]


def test_infer_dataset_preflights_mapped_images_before_model_load(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_map_json = tmp_path / "image-map.json"
    image_map_json.write_text(json.dumps({"missing": 7}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    with (
        patch("xh_detect.cli.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.cli._build_detector", create=True) as build_detector,
    ):
        result = CliRunner().invoke(
            app,
            [
                "infer-dataset",
                "--images-dir",
                str(images_dir),
                "--image-map-json",
                str(image_map_json),
                "--config-path",
                str(config_path),
                "--output-json",
                str(tmp_path / "predictions.json"),
            ],
        )

    assert result.exit_code != 0
    assert "missing image for stem 'missing'" in result.output
    build_detector.assert_not_called()


def test_infer_dataset_preflights_unreadable_images_before_model_load(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "corrupt.jpg").write_bytes(b"not a readable image")
    image_map_json = tmp_path / "image-map.json"
    image_map_json.write_text(json.dumps({"corrupt": 7}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    with (
        patch("xh_detect.cli.cv2.imread", return_value=None),
        patch("xh_detect.cli.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.cli._build_detector", create=True) as build_detector,
    ):
        result = CliRunner().invoke(
            app,
            [
                "infer-dataset",
                "--images-dir",
                str(images_dir),
                "--image-map-json",
                str(image_map_json),
                "--config-path",
                str(config_path),
                "--output-json",
                str(tmp_path / "predictions.json"),
            ],
        )

    assert result.exit_code != 0
    assert "cannot read image" in result.output
    assert "cannot read image corrupt.jpg" in result.output
    assert "corrupt.jpg" in result.output
    build_detector.assert_not_called()


def test_infer_dataset_runs_mapped_stems_in_sorted_order_and_ignores_extra_images(
    tmp_path: Path,
) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for stem in ["b", "a", "extra"]:
        (images_dir / f"{stem}.jpg").write_bytes(b"image")
    image_map_json = tmp_path / "image-map.json"
    image_map_json.write_text(json.dumps({"b": 2, "a": 1}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    output_json = tmp_path / "predictions.json"
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    with (
        patch("xh_detect.cli.cv2.imread", return_value=image),
        patch("xh_detect.cli.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.cli._build_detector", create=True),
        patch("xh_detect.cli.InferencePipeline") as pipeline_class,
    ):
        pipeline_class.return_value.run.return_value = InferenceResult(
            detections=(),
            timings=StageTimings(0.1, 0.2, 0.3, 0.6),
        )

        result = CliRunner().invoke(
            app,
            [
                "infer-dataset",
                "--images-dir",
                str(images_dir),
                "--image-map-json",
                str(image_map_json),
                "--config-path",
                str(config_path),
                "--output-json",
                str(output_json),
            ],
        )

    assert result.exit_code == 0, result.output
    assert [call.args[1] for call in pipeline_class.return_value.run.call_args_list] == [
        "a",
        "b",
    ]


@patch("xh_detect.cli.build_app")
def test_serve_command_launches_gradio(build_app: Mock, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("config", encoding="utf-8")

    result = CliRunner().invoke(
        app,
        [
            "serve",
            "--config-path",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            "7861",
        ],
    )

    assert result.exit_code == 0, result.output
    build_app.assert_called_once_with(config)
    build_app.return_value.launch.assert_called_once_with(
        server_name="127.0.0.1",
        server_port=7861,
    )


@patch(
    "xh_detect.cli.benchmark_pipeline",
    return_value={"median_s": 1.2, "p95_s": 1.5},
)
@patch("xh_detect.cli.create_synthetic_image")
@patch("xh_detect.cli.InferencePipeline")
@patch("xh_detect.cli._build_detector", create=True)
@patch("xh_detect.cli.PipelineConfig.from_yaml")
@patch("xh_detect.cli.cv2.imread")
def test_benchmark_command_creates_missing_image_and_prints_json(
    imread: Mock,
    from_yaml: Mock,
    build_detector: Mock,
    pipeline_class: Mock,
    create_image: Mock,
    benchmark_pipeline: Mock,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    image_path = tmp_path / "synthetic.png"
    config_path.write_text("config", encoding="utf-8")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    imread.return_value = image
    from_yaml.return_value = PipelineConfig(device="cpu", half=False)

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--config-path",
            str(config_path),
            "--image-path",
            str(image_path),
            "--repeats",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    create_image.assert_called_once_with(image_path)
    build_detector.assert_called_once_with(from_yaml.return_value)
    pipeline_class.assert_called_once()
    benchmark_pipeline.assert_called_once_with(
        pipeline_class.return_value,
        image,
        "synthetic",
        3,
    )
    assert json.loads(result.stdout) == {"median_s": 1.2, "p95_s": 1.5}


def test_benchmark_vehicle_proposals_command_writes_paired_report(tmp_path: Path) -> None:
    main_config_path = tmp_path / "main.yaml"
    sph_config_path = tmp_path / "sph.yaml"
    image_path = tmp_path / "scene.png"
    output_path = tmp_path / "paired-latency.json"
    main_config_path.write_text("main", encoding="utf-8")
    sph_config_path.write_text("sph", encoding="utf-8")
    image_path.write_bytes(b"image")
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    main_config = PipelineConfig(device="cpu", half=False)
    sph_config = PipelineConfig(device="cpu", half=False)
    latency_report = SimpleNamespace(proposal_gate_passed=True)
    payload = {"combined": {"maximum_s": 12.0}, "gate": {"passed": True}}

    with (
        patch("xh_detect.cli.cv2.imread", return_value=image),
        patch(
            "xh_detect.cli.PipelineConfig.from_yaml",
            side_effect=[main_config, sph_config],
        ),
        patch("xh_detect.cli._build_detector", side_effect=["main-detector", "sph-detector"]),
        patch("xh_detect.cli.InferencePipeline", side_effect=["main-pipeline", "sph-pipeline"]),
        patch(
            "xh_detect.cli.benchmark_vehicle_proposal_pair",
            return_value=latency_report,
        ) as benchmark_pair,
        patch("xh_detect.cli.vehicle_latency_report_to_dict", return_value=payload),
    ):
        result = CliRunner().invoke(
            app,
            [
                "benchmark-vehicle-proposals",
                "--main-config-path",
                str(main_config_path),
                "--sph-config-path",
                str(sph_config_path),
                "--image-path",
                str(image_path),
                "--repeats",
                "3",
                "--reserve-seconds",
                "1",
                "--limit-seconds",
                "20",
                "--output-path",
                str(output_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert result.stdout.strip() == str(output_path)
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written["image"] == {"path": str(image_path), "synthetic": False}
    assert written["gate"]["passed"] is True
    benchmark_pair.assert_called_once_with(
        "main-pipeline",
        "sph-pipeline",
        image,
        "scene",
        repeats=3,
        reserve_seconds=1.0,
        limit_seconds=20.0,
    )


@patch("xh_detect.cli.torch.cuda.get_device_name")
@patch("xh_detect.cli.torch.cuda.is_available", return_value=False)
def test_env_command_reports_cpu(
    cuda_available: Mock,
    get_device_name: Mock,
) -> None:
    result = CliRunner().invoke(app, ["env"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["cuda_available"] is False
    assert payload["gpu"] is None
    assert payload["python"]
    assert payload["torch"]
    assert payload["ultralytics"]
    cuda_available.assert_called_once_with()
    get_device_name.assert_not_called()
