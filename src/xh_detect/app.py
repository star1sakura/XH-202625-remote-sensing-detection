from __future__ import annotations

import json
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
    official_counts: bool = False,
    include_zero_fine_counts: bool = True,
) -> dict[str, object]:
    taxonomy = taxonomy or get_taxonomy("legacy3")
    counts = class_counts(detections, taxonomy=taxonomy)
    fine_counts = counts["fine"]
    if not include_zero_fine_counts:
        fine_counts = {name: count for name, count in fine_counts.items() if count > 0}
    summary = {
        "preprocess_seconds": round(timings.preprocess_s, 4),
        "inference_seconds": round(timings.inference_s, 4),
        "postprocess_seconds": round(timings.postprocess_s, 4),
        "total_seconds": round(timings.total_s, 4),
    }
    if official_counts:
        return {
            "coarse_counts": counts["coarse"],
            "fine_counts": fine_counts,
            **summary,
        }
    return {
        "coarse": counts["coarse"],
        "fine": fine_counts,
        **summary,
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
    official_counts: bool = False,
    include_zero_fine_counts: bool = True,
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

    summary = format_summary(
        result.detections,
        result.timings,
        taxonomy=taxonomy,
        official_counts=official_counts,
        include_zero_fine_counts=include_zero_fine_counts,
    )
    if truth_path:
        report = evaluate(
            load_coco_predictions(json_output, taxonomy=taxonomy),
            load_coco_ground_truth(Path(truth_path), taxonomy=taxonomy),
            taxonomy=taxonomy,
        )
        summary["evaluation"] = report_to_dict(report)

    _progress(progress, 1.0, "完成")
    return str(image_output), summary, str(json_output)


def _load_xh25_demo_examples(
    manifest_path: Path = Path("datasets/xh25/manifests/demo-samples.json"),
) -> list[list[str]] | None:
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None

    dataset_root = manifest_path.parents[1]
    examples: list[list[str]] = []
    for coarse_name in sorted(manifest):
        relative_path = manifest[coarse_name]
        if not isinstance(relative_path, str) or not relative_path:
            continue
        sample_path = dataset_root / relative_path
        examples.append([sample_path.as_posix()])
    return examples or None


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
    is_official_hbb = config.task == "detect"
    page_title = (
        "XH-202625 正式数据 25 类 HBB Demo" if is_official_hbb else "XH-202625 遥感目标检测 Demo"
    )
    demo_examples = _load_xh25_demo_examples() if is_official_hbb else None

    def handle_obb_prediction(
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

    def handle_hbb_prediction(
        image_path: str,
        truth_path: str | None,
        progress: ProgressCallback = _GRADIO_PROGRESS,
    ) -> tuple[str, dict[str, object], str]:
        try:
            return run_prediction(
                pipeline,
                image_path,
                "HBB",
                truth_path,
                taxonomy=taxonomy,
                official_counts=True,
                include_zero_fine_counts=False,
                progress=progress,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title=page_title) as demo:
        gr.Markdown(f"# {page_title}")
        gr.Markdown(
            "上传单幅光学遥感图像，输出正式数据集 25 类水平框检测结果。"
            if is_official_hbb
            else "上传单幅光学遥感图像，输出飞机、舰船和车辆检测结果。"
        )
        with gr.Row():
            source = gr.Image(type="filepath", label="上传光学遥感图像")
            result_image = gr.Image(type="filepath", label="检测结果")
        if is_official_hbb and demo_examples is not None:
            gr.Examples(examples=demo_examples, inputs=[source])
        mode = None
        if not is_official_hbb:
            mode = gr.Radio(["OBB", "HBB"], value="OBB", label="显示框类型")
        truth_file = gr.File(
            type="filepath",
            label="可选：COCO 真值 JSON（当前图像使用 image_id=1）",
        )
        run_button = gr.Button("开始检测", variant="primary")
        summary = gr.JSON(label="目标数量、耗时与评估")
        result_file = gr.File(label="COCO Detection JSON")
        if is_official_hbb:
            run_button.click(
                handle_hbb_prediction,
                inputs=[source, truth_file],
                outputs=[result_image, summary, result_file],
            )
        else:
            run_button.click(
                handle_obb_prediction,
                inputs=[source, mode, truth_file],
                outputs=[result_image, summary, result_file],
            )
    return demo.queue(default_concurrency_limit=1)
