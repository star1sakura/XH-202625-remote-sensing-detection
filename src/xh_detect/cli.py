from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import cv2
import torch
import typer
import ultralytics

from xh_detect import __version__
from xh_detect.compare import compare_experiments
from xh_detect.competition import (
    load_evaluation_report,
    write_competition_proxy_artifacts,
)
from xh_detect.config import PipelineConfig
from xh_detect.data.dota import ConversionStats, convert_split, write_dataset_yaml
from xh_detect.data.ship_balance import build_ship_balanced_dataset
from xh_detect.data.xh25 import prepare_dataset
from xh_detect.detector import UltralyticsDetector
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
from xh_detect.taxonomy import get_taxonomy
from xh_detect.thresholds import (
    DEFAULT_THRESHOLD_GRID_TEXT,
    load_report_objective,
    parse_threshold_grid,
    write_threshold_artifacts,
)
from xh_detect.thresholds import (
    optimize_thresholds as optimize_thresholds_search,
)
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


def _unreadable_image_message(image_path: Path) -> str:
    return f"cannot read image {image_path.name}: {image_path}"


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


def _build_detector(config: PipelineConfig) -> UltralyticsDetector:
    return UltralyticsDetector(
        config.model_path,
        config.device,
        config.image_size,
        config.half,
        task=config.task,
    )


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


@app.command("prepare-xh25")
def prepare_xh25(
    source_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25"),
    val_ratio: Annotated[float, typer.Option(min=0.05, max=0.4)] = 0.15,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    prepared = prepare_dataset(source_root, output_root, val_ratio=val_ratio, seed=seed)
    typer.echo(
        json.dumps(
            {
                "output_root": str(prepared.output_root),
                "train_images": len(prepared.train_stems),
                "val_images": len(prepared.val_stems),
                "train_targets": dict(prepared.train_class_counts),
                "val_targets": dict(prepared.val_class_counts),
            },
            ensure_ascii=False,
        )
    )


@app.command("build-ship-balanced-xh25")
def build_ship_balanced_xh25_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-ship-balanced"),
    qhs_factor: Annotated[int, typer.Option(min=1)] = 2,
    ms_factor: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    try:
        result = build_ship_balanced_dataset(
            source_root,
            output_root,
            qhs_factor=qhs_factor,
            ms_factor=ms_factor,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "original_train_images": result.original_train_images,
                "balanced_train_images": result.balanced_train_images,
                "duplicated_train_images": result.duplicated_train_images,
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
    model: Annotated[str, typer.Option()] = "yolo26s.pt",
    pretrained: Annotated[str | None, typer.Option()] = None,
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    image_size: Annotated[int, typer.Option(min=1)] = 1024,
    device: Annotated[str, typer.Option()] = "0",
    batch: Annotated[int, typer.Option(min=1)] = 8,
    workers: Annotated[int, typer.Option(min=0)] = 4,
    amp: Annotated[bool, typer.Option()] = False,
    project: Annotated[str, typer.Option()] = "runs/train",
    name: Annotated[str, typer.Option()] = "xh25-baseline",
    resume: Annotated[bool, typer.Option()] = False,
) -> None:
    train_model(
        str(dataset_yaml),
        model,
        epochs,
        image_size,
        device,
        batch=batch,
        workers=workers,
        amp=amp,
        project=project,
        name=name,
        resume=resume,
        pretrained=pretrained,
    )


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
        raise typer.BadParameter(_unreadable_image_message(image_path))

    detector = _build_detector(config)
    pipeline = InferencePipeline(detector, config, output_dir / "cache")
    result = pipeline.run(image, image_path.stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / f"{image_path.stem}.jpg"
    json_output = output_dir / f"{image_path.stem}.json"
    taxonomy = get_taxonomy(config.taxonomy)
    rendered = draw_detections(image, result.detections, taxonomy=taxonomy)
    if not cv2.imwrite(str(image_output), rendered):
        raise RuntimeError(f"failed to write rendered image: {image_output}")
    export_coco_results(
        result.detections,
        {image_path.stem: 1},
        json_output,
        valid_class_ids=taxonomy.valid_ids,
    )
    typer.echo(json.dumps(asdict(result.timings), allow_nan=False))


def _load_image_id_map(path: Path) -> dict[str, int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"invalid image map JSON: {exc.msg}") from exc
    if not isinstance(payload, Mapping):
        raise typer.BadParameter("image map JSON must be a mapping")

    image_map: dict[str, int] = {}
    seen_ids: set[int] = set()
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise typer.BadParameter("image map keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise typer.BadParameter("image map values must be non-bool non-negative integers")
        if value < 0:
            raise typer.BadParameter("image map values must be non-bool non-negative integers")
        if value in seen_ids:
            raise typer.BadParameter("image map values must be unique")
        seen_ids.add(value)
        image_map[key] = value
    return image_map


@app.command("infer-dataset")
def infer_dataset(
    images_dir: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path(
        "configs/xh25-hbb.yaml"
    ),
    output_json: Annotated[Path, typer.Option()] = Path("outputs/xh25/val-predictions.json"),
) -> None:
    image_map = _load_image_id_map(image_map_json)
    for stem in sorted(image_map):
        image_path = images_dir / f"{stem}.jpg"
        if not image_path.is_file():
            raise typer.BadParameter(f"missing image for stem {stem!r}: {image_path}")
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise typer.BadParameter(_unreadable_image_message(image_path))
        del image

    config = PipelineConfig.from_yaml(config_path)
    taxonomy = get_taxonomy(config.taxonomy)
    detector = _build_detector(config)
    pipeline = InferencePipeline(detector, config, output_json.parent / "cache")

    all_detections = []
    for stem in sorted(image_map):
        image_path = images_dir / f"{stem}.jpg"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise typer.BadParameter(_unreadable_image_message(image_path))
        result = pipeline.run(image, stem)
        all_detections.extend(result.detections)

    export_coco_results(
        all_detections,
        image_map,
        output_json,
        valid_class_ids=taxonomy.valid_ids,
    )
    typer.echo(str(output_json))


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
    taxonomy: Annotated[str, typer.Option()] = "legacy3",
) -> None:
    taxonomy_object = get_taxonomy(taxonomy)
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy_object)
    truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
    report = evaluate_detections(predictions, truth, taxonomy=taxonomy_object)
    payload = report_to_dict(report)
    _write_json(output_path, payload)
    typer.echo(json.dumps(payload, ensure_ascii=False, allow_nan=False))


@app.command("competition-report")
def competition_report_command(
    report_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/competition-proxy"),
    experiment_name: Annotated[str, typer.Option()] = "xh25-experiment",
    latency_seconds: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    try:
        report = load_evaluation_report(report_json)
        write_competition_proxy_artifacts(
            report,
            output_dir=output_dir,
            experiment_name=experiment_name,
            latency_seconds=latency_seconds,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_dir / "competition-proxy.json"))


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
    taxonomy: Annotated[str, typer.Option()] = "legacy3",
) -> None:
    taxonomy_object = get_taxonomy(taxonomy)
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy_object)
    truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
    thresholds = [round(index * 0.05, 2) for index in range(1, 20)]
    payload = [
        {"threshold": threshold, "report": report_to_dict(report)}
        for threshold, report in threshold_sweep(
            predictions,
            truth,
            thresholds,
            taxonomy=taxonomy_object,
        )
    ]
    _write_json(output_path, payload)
    typer.echo(str(output_path))


@app.command("optimize-thresholds")
def optimize_thresholds_command(
    predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/mksnet-lite/threshold-optimized"
    ),
    taxonomy: Annotated[str, typer.Option()] = "xh25",
    baseline_report: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
    experiment_name: Annotated[str, typer.Option()] = "xh25-mksnet-lite-threshold-optimized",
    thresholds: Annotated[str, typer.Option()] = DEFAULT_THRESHOLD_GRID_TEXT,
    recall_floor_delta: Annotated[float, typer.Option(min=0.0)] = 0.003,
    tie_epsilon: Annotated[float, typer.Option(min=0.0)] = 0.0005,
) -> None:
    default_baseline = Path("outputs/xh25/baseline/report.json")
    resolved_baseline = (
        default_baseline
        if baseline_report is None and default_baseline.is_file()
        else baseline_report
    )
    if resolved_baseline is not None and not resolved_baseline.is_file():
        raise typer.BadParameter(f"baseline report does not exist: {resolved_baseline}")

    try:
        threshold_grid = parse_threshold_grid(thresholds)
        taxonomy_object = get_taxonomy(taxonomy)
        predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy_object)
        truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
        baseline_objective = (
            load_report_objective(resolved_baseline) if resolved_baseline is not None else None
        )
        result = optimize_thresholds_search(
            predictions,
            truth,
            taxonomy=taxonomy_object,
            thresholds=threshold_grid,
            baseline_objective=baseline_objective,
            recall_floor_delta=recall_floor_delta,
            tie_epsilon=tie_epsilon,
        )
        write_threshold_artifacts(
            result,
            output_dir=output_dir,
            taxonomy=taxonomy_object,
            experiment_name=experiment_name,
            baseline_report=resolved_baseline,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_dir / "report.json"))


@app.command("compare-experiments")
def compare_experiments_command(
    baseline_report: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    experiment_report: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/mksnet-lite"),
    baseline_name: Annotated[str, typer.Option()] = "xh25-yolo26s-e80",
    experiment_name: Annotated[str, typer.Option()] = "xh25-mksnet-lite",
    baseline_benchmark: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    experiment_benchmark: Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
) -> None:
    try:
        comparison = compare_experiments(
            baseline_report=baseline_report,
            experiment_report=experiment_report,
            output_dir=output_dir,
            baseline_name=baseline_name,
            experiment_name=experiment_name,
            baseline_benchmark=baseline_benchmark,
            experiment_benchmark=experiment_benchmark,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(json.dumps(comparison["overall"], ensure_ascii=False, allow_nan=False))


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
    detector = _build_detector(config)
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
