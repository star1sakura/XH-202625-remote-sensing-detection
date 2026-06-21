from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xh_detect.config import PipelineConfig
from xh_detect.evaluator import evaluate, load_coco_predictions
from xh_detect.exporters import export_coco_results, validate_coco_results
from xh_detect.pipeline import InferencePipeline
from xh_detect.types import BoxPrediction, ObjectAnnotation
from xh_detect.visualize import draw_detections


class OneBoxDetector:
    def predict(self, images, confidence):
        prediction = BoxPrediction(
            class_id=0,
            score=0.9,
            polygon=((8.0, 8.0), (24.0, 8.0), (24.0, 24.0), (8.0, 24.0)),
        )
        return [[prediction] for _ in images]


def test_cpu_fake_detector_end_to_end_pipeline_export_evaluate_visualize(
    tmp_path: Path,
) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    pipeline = InferencePipeline(
        OneBoxDetector(),
        PipelineConfig(
            device="cpu",
            half=False,
            tile_size=32,
            image_size=32,
            batch_size=1,
            overlap=0.0,
            edge_margin=0,
        ),
    )

    result = pipeline.run(image, "scene")
    rendered = draw_detections(image, result.detections, mode="obb")
    image_output = tmp_path / "scene.jpg"
    json_output = tmp_path / "scene.json"
    assert cv2.imwrite(str(image_output), rendered)
    export_coco_results(result.detections, {"scene": 1}, json_output)

    records = json.loads(json_output.read_text(encoding="utf-8"))
    validate_coco_results(records)
    predictions = load_coco_predictions(json_output)
    report = evaluate(
        predictions,
        [
            ObjectAnnotation(
                image_id="1",
                class_id=0,
                polygon=((8.0, 8.0), (24.0, 8.0), (24.0, 24.0), (8.0, 24.0)),
            )
        ],
    )

    assert len(result.detections) == 1
    assert rendered.sum() > 0
    assert image_output.is_file()
    assert report.overall.tp == 1
    assert report.overall.fp == 0
    assert report.overall.fn == 0
