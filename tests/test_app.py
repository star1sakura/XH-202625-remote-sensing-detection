from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import gradio as gr
import numpy as np

from xh_detect.app import build_app, format_summary, run_prediction
from xh_detect.config import PipelineConfig
from xh_detect.taxonomy import get_taxonomy
from xh_detect.types import Detection, InferenceResult, StageTimings

POLYGON = ((4.0, 4.0), (20.0, 4.0), (20.0, 20.0), (4.0, 20.0))


def _detection(image_id: str = "img") -> Detection:
    return Detection(
        image_id,
        2,
        0.8,
        POLYGON,
    )


def test_format_summary_contains_counts_and_timings() -> None:
    summary = format_summary(
        [_detection()],
        StageTimings(0.1, 0.2, 0.3, 0.6),
    )

    assert summary == {
        "coarse": {"aircraft": 0, "ship": 0, "vehicle": 1},
        "fine": {"aircraft": 0, "ship": 0, "vehicle": 1},
        "preprocess_seconds": 0.1,
        "inference_seconds": 0.2,
        "postprocess_seconds": 0.3,
        "total_seconds": 0.6,
    }


def test_format_summary_contains_coarse_and_fine_counts() -> None:
    summary = format_summary(
        [
            Detection("image", 0, 0.9, POLYGON),
            Detection("image", 4, 0.8, POLYGON),
            Detection("image", 24, 0.7, POLYGON),
        ],
        StageTimings(0.1, 0.2, 0.3, 0.6),
        taxonomy=get_taxonomy("xh25"),
        official_counts=True,
    )

    assert summary["coarse_counts"] == {"aircraft": 1, "ship": 1, "vehicle": 1}
    assert summary["fine_counts"]["HM"] == 1
    assert summary["fine_counts"]["A1_SU-35"] == 1
    assert summary["fine_counts"]["FSC"] == 1


def test_run_prediction_uses_shared_pipeline_and_writes_outputs(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    pipeline = Mock()
    pipeline.run.return_value = InferenceResult(
        detections=(_detection("scene"),),
        timings=StageTimings(0.1, 0.2, 0.3, 0.6),
    )
    progress = Mock()

    image_output, summary, json_output = run_prediction(
        pipeline,
        str(image_path),
        "HBB",
        None,
        output_root=tmp_path / "outputs",
        progress=progress,
    )

    pipeline.run.assert_called_once()
    called_image, called_image_id = pipeline.run.call_args.args
    np.testing.assert_array_equal(called_image, image)
    assert called_image_id == "scene"
    assert Path(image_output).is_file()
    assert Path(json_output).is_file()
    assert summary["coarse"]["vehicle"] == 1
    assert progress.call_count == 4


def _components(demo: gr.Blocks) -> list[object]:
    return list(getattr(demo, "blocks", {}).values())


def test_build_app_constructs_detect_detector_for_xh25_without_real_model(
    tmp_path: Path,
) -> None:
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
        patch("xh_detect.app.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.app.UltralyticsDetector", create=True) as detector_class,
        patch("xh_detect.app.InferencePipeline") as pipeline_class,
    ):
        demo = build_app(config_path)

    assert isinstance(demo, gr.Blocks)
    detector_class.assert_called_once_with("best.pt", "cpu", 1024, False, task="detect")
    pipeline_class.assert_called_once()
    assert demo.title == "XH-202625 正式数据 25 类 HBB Demo"
    assert any(
        isinstance(component, gr.Markdown)
        and "XH-202625 正式数据 25 类 HBB Demo" in component.value
        for component in _components(demo)
    )
    assert not any(isinstance(component, gr.Radio) for component in _components(demo))


def test_build_app_retains_mode_radio_for_legacy_obb_without_real_model(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    config = PipelineConfig(
        task="obb",
        taxonomy="legacy3",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(3)},
    )

    with (
        patch("xh_detect.app.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.app.UltralyticsDetector", create=True) as detector_class,
        patch("xh_detect.app.InferencePipeline") as pipeline_class,
    ):
        demo = build_app(config_path)

    radios = [component for component in _components(demo) if isinstance(component, gr.Radio)]
    assert len(radios) == 1
    assert radios[0].choices == [("OBB", "OBB"), ("HBB", "HBB")]
    assert radios[0].value == "OBB"
    detector_class.assert_called_once_with("best.pt", "cpu", 1024, False, task="obb")
    pipeline_class.assert_called_once()


def test_build_app_loads_xh25_demo_examples_when_manifest_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    manifest_path = tmp_path / "datasets" / "xh25" / "manifests" / "demo-samples.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        '{"ship": "images/val/ship.jpg", "aircraft": "images/val/aircraft.jpg"}',
        encoding="utf-8",
    )
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        device="cpu",
        half=False,
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )
    monkeypatch.chdir(tmp_path)

    with (
        patch("xh_detect.app.PipelineConfig.from_yaml", return_value=config),
        patch("xh_detect.app.UltralyticsDetector", create=True),
        patch("xh_detect.app.InferencePipeline"),
        patch("xh_detect.app.gr.Examples") as examples_class,
    ):
        build_app(config_path)

    examples_class.assert_called_once()
    assert examples_class.call_args.kwargs["examples"] == [
        ["datasets/xh25/images/val/aircraft.jpg"],
        ["datasets/xh25/images/val/ship.jpg"],
    ]


@patch("xh_detect.app.report_to_dict", return_value={"overall": {"map50": 1.0}})
@patch("xh_detect.app.evaluate")
@patch("xh_detect.app.load_coco_ground_truth", return_value=[])
@patch("xh_detect.app.load_coco_predictions", return_value=[])
@patch("xh_detect.app.export_coco_results")
@patch("xh_detect.app.draw_detections")
def test_run_prediction_passes_taxonomy_to_summary_export_and_evaluation(
    draw_detections: Mock,
    export_results: Mock,
    load_predictions: Mock,
    load_truth: Mock,
    evaluate: Mock,
    report_to_dict: Mock,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "scene.png"
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    truth_path = tmp_path / "truth.json"
    truth_path.write_text('{"annotations":[]}', encoding="utf-8")
    pipeline = Mock()
    pipeline.run.return_value = InferenceResult(
        detections=(
            Detection(
                "scene",
                24,
                0.8,
                ((4.0, 4.0), (20.0, 4.0), (20.0, 20.0), (4.0, 20.0)),
            ),
        ),
        timings=StageTimings(0.1, 0.2, 0.3, 0.6),
    )
    draw_detections.return_value = image
    taxonomy = get_taxonomy("xh25")

    _, summary, json_output = run_prediction(
        pipeline,
        str(image_path),
        "HBB",
        str(truth_path),
        taxonomy=taxonomy,
        official_counts=True,
        output_root=tmp_path / "outputs",
    )

    assert summary["coarse_counts"]["vehicle"] == 1
    draw_detections.assert_called_once()
    draw_image, draw_detections_arg = draw_detections.call_args.args
    np.testing.assert_array_equal(draw_image, image)
    assert draw_detections_arg == pipeline.run.return_value.detections
    assert draw_detections.call_args.kwargs == {"mode": "hbb", "taxonomy": taxonomy}
    export_results.assert_called_once_with(
        pipeline.run.return_value.detections,
        {"scene": 1},
        Path(json_output),
        valid_class_ids=taxonomy.valid_ids,
    )
    load_predictions.assert_called_once_with(Path(json_output), taxonomy=taxonomy)
    load_truth.assert_called_once_with(truth_path, taxonomy=taxonomy)
    evaluate.assert_called_once_with([], [], taxonomy=taxonomy)
    report_to_dict.assert_called_once()
