from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import cv2

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import gradio as gr

from xh_detect.config import PipelineConfig
from xh_detect.detector import UltralyticsDetector
from xh_detect.evaluator import (
    evaluate,
    load_coco_ground_truth,
    load_coco_predictions,
    report_to_dict,
)
from xh_detect.exporters import export_coco_results
from xh_detect.pipeline import InferencePipeline
from xh_detect.taxonomy import Taxonomy, get_taxonomy
from xh_detect.types import Detection, StageTimings
from xh_detect.visualize import class_counts, draw_detections

ProgressCallback = Callable[..., object]
_GRADIO_PROGRESS = gr.Progress()


def format_summary(
    detections: list[Detection] | tuple[Detection, ...],
    timings: StageTimings,
    *,
    taxonomy: Taxonomy | None = None,
) -> dict[str, object]:
    taxonomy = taxonomy or get_taxonomy("legacy3")
    return {
        **class_counts(detections, taxonomy=taxonomy),
        "preprocess_seconds": round(timings.preprocess_s, 4),
        "inference_seconds": round(timings.inference_s, 4),
        "postprocess_seconds": round(timings.postprocess_s, 4),
        "total_seconds": round(timings.total_s, 4),
    }


def _progress(
    callback: ProgressCallback | None,
    value: float,
    description: str,
) -> None:
    if callback is not None:
        callback(value, desc=description)


def run_prediction(
    pipeline: InferencePipeline,
    image_path: str,
    mode: str,
    truth_path: str | None,
    *,
    taxonomy: Taxonomy | None = None,
    output_root: Path = Path("outputs/gradio"),
    progress: ProgressCallback | None = None,
) -> tuple[str, dict[str, object], str]:
    if not isinstance(image_path, str) or not image_path:
        raise ValueError("image_path must be a non-empty string")
    normalized_mode = mode.lower() if isinstance(mode, str) else ""
    if normalized_mode not in {"obb", "hbb"}:
        raise ValueError("mode must be OBB or HBB")
    taxonomy = taxonomy or get_taxonomy("legacy3")

    _progress(progress, 0.0, "读取图像")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {image_path}")
    image_id = Path(image_path).stem

    _progress(progress, 0.2, "切片、推理与合并")
    result = pipeline.run(image, image_id)

    _progress(progress, 0.9, "生成可视化与 JSON")
    rendered = draw_detections(
        image,
        result.detections,
        mode=normalized_mode,
        taxonomy=taxonomy,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    image_output = output_root / f"{image_id}-{normalized_mode}-{run_id}.jpg"
    json_output = output_root / f"{image_id}-{run_id}.json"
    if not cv2.imwrite(str(image_output), rendered):
        raise OSError(f"failed to write rendered image: {image_output}")
    export_coco_results(
        result.detections,
        {image_id: 1},
        json_output,
        valid_class_ids=taxonomy.valid_ids,
    )

    summary = format_summary(result.detections, result.timings, taxonomy=taxonomy)
    if truth_path:
        report = evaluate(
            load_coco_predictions(json_output, taxonomy=taxonomy),
            load_coco_ground_truth(Path(truth_path), taxonomy=taxonomy),
            taxonomy=taxonomy,
        )
        summary["evaluation"] = report_to_dict(report)

    _progress(progress, 1.0, "完成")
    return str(image_output), summary, str(json_output)


def build_app(config_path: Path = Path("configs/baseline.yaml")) -> gr.Blocks:
    config = PipelineConfig.from_yaml(config_path)
    taxonomy = get_taxonomy(config.taxonomy)
    detector = UltralyticsDetector(
        config.model_path,
        config.device,
        config.image_size,
        config.half,
        task=config.task,
    )
    pipeline = InferencePipeline(detector, config, Path("cache/gradio"))

    def handle_prediction(
        image_path: str,
        mode: str,
        truth_path: str | None,
        progress: ProgressCallback = _GRADIO_PROGRESS,
    ) -> tuple[str, dict[str, object], str]:
        try:
            return run_prediction(
                pipeline,
                image_path,
                mode,
                truth_path,
                taxonomy=taxonomy,
                progress=progress,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="XH-202625 遥感目标检测 Demo") as demo:
        gr.Markdown("# XH-202625 遥感目标检测 Demo")
        gr.Markdown("上传单幅光学遥感图像，输出飞机、舰船和车辆检测结果。")
        with gr.Row():
            source = gr.Image(type="filepath", label="上传光学遥感图像")
            result_image = gr.Image(type="filepath", label="检测结果")
        mode = gr.Radio(["OBB", "HBB"], value="OBB", label="显示框类型")
        truth_file = gr.File(
            type="filepath",
            label="可选：COCO 真值 JSON（当前图像使用 image_id=1）",
        )
        run_button = gr.Button("开始检测", variant="primary")
        summary = gr.JSON(label="目标数量、耗时与评估")
        result_file = gr.File(label="COCO Detection JSON")
        run_button.click(
            handle_prediction,
            inputs=[source, mode, truth_file],
            outputs=[result_image, summary, result_file],
        )
    return demo.queue(default_concurrency_limit=1)
