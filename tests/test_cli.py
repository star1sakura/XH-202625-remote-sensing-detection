import json
from importlib import metadata, reload
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from typer.testing import CliRunner

import xh_detect
from xh_detect.cli import app
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
        "yolo26s-obb.pt",
        2,
        1024,
        "0",
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
