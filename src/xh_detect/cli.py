from __future__ import annotations

import json
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import cv2
import torch
import typer
import ultralytics

from xh_detect import __version__
from xh_detect.config import PipelineConfig
from xh_detect.data.dota import ConversionStats, convert_split, write_dataset_yaml
from xh_detect.detector import UltralyticsOBBDetector
from xh_detect.evaluator import (
    evaluate as evaluate_detections,
)
from xh_detect.evaluator import (
    load_coco_ground_truth,
    load_coco_predictions,
    report_to_dict,
    threshold_sweep,
)
from xh_detect.exporters import export_coco_results
from xh_detect.pipeline import InferencePipeline
from xh_detect.training import export_tensorrt, train_model
from xh_detect.visualize import draw_detections

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(f"xh-detect {__version__}")


def _stats_payload(stats: ConversionStats) -> dict[str, object]:
    return {
        "images": stats.images,
        "targets": dict(stats.targets),
        "invalid_lines": stats.invalid_lines,
        "skipped_images": stats.skipped_images,
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def build_app(config_path: Path):
    from xh_detect.app import build_app as build_gradio_app

    return build_gradio_app(config_path)


def create_synthetic_image(destination: Path):
    from xh_detect.benchmark import create_synthetic_image as create_image

    return create_image(destination)


def benchmark_pipeline(
    pipeline: InferencePipeline,
    image,
    image_id: str,
    repeats: int,
):
    from xh_detect.benchmark import benchmark_pipeline as run_benchmark

    return run_benchmark(pipeline, image, image_id, repeats)


@app.command("prepare-dota")
def prepare_dota(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ],
    output_root: Annotated[Path, typer.Option()] = Path("datasets/dota3"),
) -> None:
    required_directories = [
        source_root / "images" / "train",
        source_root / "images" / "val",
        source_root / "labelTxt" / "train",
        source_root / "labelTxt" / "val",
    ]
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise typer.BadParameter(
            "DOTA source layout is incomplete; missing directories: " + ", ".join(missing)
        )
    train_stats = convert_split(
        source_root / "images" / "train",
        source_root / "labelTxt" / "train",
        output_root,
        "train",
    )
    val_stats = convert_split(
        source_root / "images" / "val",
        source_root / "labelTxt" / "val",
        output_root,
        "val",
    )
    dataset_yaml = write_dataset_yaml(output_root)
    typer.echo(
        json.dumps(
            {
                "train": _stats_payload(train_stats),
                "val": _stats_payload(val_stats),
                "dataset_yaml": str(dataset_yaml),
            },
            ensure_ascii=False,
        )
    )


@app.command()
def train(
    dataset_yaml: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    model: Annotated[str, typer.Option()] = "yolo26s-obb.pt",
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    image_size: Annotated[int, typer.Option(min=1)] = 1024,
    device: Annotated[str, typer.Option()] = "0",
) -> None:
    train_model(str(dataset_yaml), model, epochs, image_size, device)


@app.command()
def infer(
    image_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/baseline.yaml"),
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/infer"),
) -> None:
    config = PipelineConfig.from_yaml(config_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise typer.BadParameter(f"cannot read image: {image_path}")

    detector = UltralyticsOBBDetector(
        config.model_path,
        config.device,
        config.image_size,
        config.half,
    )
    pipeline = InferencePipeline(detector, config, output_dir / "cache")
    result = pipeline.run(image, image_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / f"{image_path.stem}.jpg"
    json_output = output_dir / f"{image_path.stem}.json"
    rendered = draw_detections(image, result.detections)
    if not cv2.imwrite(str(image_output), rendered):
        raise RuntimeError(f"failed to write rendered image: {image_output}")
    export_coco_results(result.detections, {image_path.stem: 1}, json_output)
    typer.echo(json.dumps(asdict(result.timings), allow_nan=False))


@app.command("export-engine")
def export_engine(
    model_path: Annotated[str, typer.Option()],
    image_size: Annotated[int, typer.Option(min=1)] = 1024,
    device: Annotated[str, typer.Option()] = "0",
) -> None:
    typer.echo(export_tensorrt(model_path, image_size, device))


@app.command("evaluate")
def evaluate_command(
    predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    ground_truth_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_path: Annotated[Path, typer.Option()] = Path("outputs/evaluation/report.json"),
) -> None:
    predictions = load_coco_predictions(predictions_json)
    truth = load_coco_ground_truth(ground_truth_json)
    report = evaluate_detections(predictions, truth)
    payload = report_to_dict(report)
    _write_json(output_path, payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))


@app.command("sweep-thresholds")
def sweep_thresholds_command(
    predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    ground_truth_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_path: Annotated[Path, typer.Option()] = Path("outputs/evaluation/threshold-sweep.json"),
) -> None:
    predictions = load_coco_predictions(predictions_json)
    truth = load_coco_ground_truth(ground_truth_json)
    thresholds = [round(index * 0.05, 2) for index in range(1, 20)]
    payload = [
        {"threshold": threshold, "report": report_to_dict(report)}
        for threshold, report in threshold_sweep(predictions, truth, thresholds)
    ]
    _write_json(output_path, payload)
    typer.echo(str(output_path))


@app.command()
def serve(
    config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/baseline.yaml"),
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 7860,
) -> None:
    build_app(config_path).launch(server_name=host, server_port=port)


@app.command()
def benchmark(
    config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/baseline.yaml"),
    image_path: Annotated[Path, typer.Option()] = Path("outputs/benchmark/synthetic-10000.png"),
    repeats: Annotated[int, typer.Option(min=1)] = 5,
) -> None:
    if not image_path.exists():
        create_synthetic_image(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise typer.BadParameter(f"cannot read benchmark image: {image_path}")
    config = PipelineConfig.from_yaml(config_path)
    detector = UltralyticsOBBDetector(
        config.model_path,
        config.device,
        config.image_size,
        config.half,
    )
    pipeline = InferencePipeline(detector, config, cache_root=None)
    summary = benchmark_pipeline(pipeline, image, image_path.stem, repeats)
    typer.echo(json.dumps(summary, allow_nan=False))


@app.command()
def env() -> None:
    cuda_available = bool(torch.cuda.is_available())
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "ultralytics": str(ultralytics.__version__),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    typer.echo(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    app()
