from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import gradio as gr
import numpy as np

from xh_detect.app import build_app, format_summary, run_prediction
from xh_detect.config import PipelineConfig
from xh_detect.types import Detection, InferenceResult, StageTimings


def _detection(image_id: str = "img") -> Detection:
    return Detection(
        image_id,
        2,
        0.8,
        ((4.0, 4.0), (20.0, 4.0), (20.0, 20.0), (4.0, 20.0)),
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


@patch("xh_detect.app.InferencePipeline")
@patch("xh_detect.app.UltralyticsOBBDetector")
@patch("xh_detect.app.PipelineConfig.from_yaml")
def test_build_app_constructs_blocks_without_real_model(
    from_yaml: Mock,
    detector_class: Mock,
    pipeline_class: Mock,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config", encoding="utf-8")
    from_yaml.return_value = PipelineConfig(device="cpu", half=False)

    demo = build_app(config_path)

    assert isinstance(demo, gr.Blocks)
    detector_class.assert_called_once()
    pipeline_class.assert_called_once()
