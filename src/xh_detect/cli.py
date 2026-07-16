from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Annotated

import cv2
import torch
import typer
import ultralytics

from xh_detect import __version__
from xh_detect.benchmark import summarize_durations
from xh_detect.calibration import (
    DEFAULT_CALIBRATION_GRID_TEXT,
    calibrate_ship_override_thresholds,
    calibrate_thresholds,
    load_image_group_mapping,
    write_calibration_artifacts,
    write_ship_override_calibration_artifacts,
)
from xh_detect.compare import compare_experiments
from xh_detect.competition import (
    load_evaluation_report,
    write_competition_proxy_artifacts,
    write_seven_metric_comparison_artifacts,
)
from xh_detect.complementarity import (
    analyze_complementarity,
    complementarity_report_to_dict,
)
from xh_detect.config import PipelineConfig
from xh_detect.data.dota import ConversionStats, convert_split, write_dataset_yaml
from xh_detect.data.hard_example import (
    HardExamplePolicy,
    build_hard_example_dataset,
)
from xh_detect.data.hard_negative import (
    HardNegativePolicy,
    build_main_hn_dataset,
)
from xh_detect.data.ship_balance import build_ship_balanced_dataset
from xh_detect.data.vehicle_expert import (
    VehicleExpertPolicy,
    build_vehicle_expert_dataset,
)
from xh_detect.data.xh25 import prepare_dataset, publish_train_mining_artifacts
from xh_detect.detector import UltralyticsDetector
from xh_detect.evaluator import (
    audit_false_positives,
    false_positive_audit_to_dict,
    load_coco_ground_truth,
    load_coco_predictions,
    report_to_dict,
    threshold_sweep,
)
from xh_detect.evaluator import (
    evaluate as evaluate_detections,
)
from xh_detect.exporters import export_coco_results
from xh_detect.mksnet_seed import initialize_mksnet_lite_from_main, interpolate_checkpoints
from xh_detect.pipeline import InferencePipeline
from xh_detect.postprocess import suppress_class_detections
from xh_detect.ranking_ensemble import RankingEnsemblePolicy, fuse_ranking_ensemble
from xh_detect.ranking_thresholds import (
    optimize_ranking_thresholds,
    write_ranking_threshold_artifacts,
)
from xh_detect.same_weight_multiscale import (
    fuse_same_weight_multiscale,
    load_same_weight_multiscale_policy,
)
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
from xh_detect.types import Detection
from xh_detect.vehicle_confirmation.benchmark import (
    benchmark_vehicle_proposal_pair,
    vehicle_latency_report_to_dict,
)
from xh_detect.vehicle_confirmation.data import (
    VehicleCropPolicy,
    build_vehicle_confirmer_dataset,
)
from xh_detect.vehicle_confirmation.model import (
    VehicleConfirmerTrainingConfig,
    export_vehicle_confirmer_engine,
    export_vehicle_confirmer_onnx,
    score_vehicle_confirmer,
    train_vehicle_confirmer,
)
from xh_detect.vehicle_confirmation.proposals import (
    analyze_vehicle_consensus,
    vehicle_consensus_report_to_dict,
)
from xh_detect.vehicle_expert import (
    analyze_vehicle_expert_holdout,
    vehicle_expert_report_to_dict,
)
from xh_detect.visualize import draw_detections

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(f"xh-detect {__version__}")


@app.command("init-mksnet-lite-from-main")
def init_mksnet_lite_from_main(
    main_checkpoint: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    model_yaml: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/models/xh25-yolo26s-mksnet-lite.yaml"),
    output_checkpoint: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/single-student/main-seeded-mksnet-lite.pt"
    ),
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    try:
        result = initialize_mksnet_lite_from_main(
            main_checkpoint,
            model_yaml,
            output_checkpoint,
            overwrite=overwrite,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report_path = result.output_checkpoint.with_suffix(".init.json")
    payload = asdict(result)
    payload.update(
        {
            "source_checkpoint": str(result.source_checkpoint),
            "target_model_yaml": str(result.target_model_yaml),
            "output_checkpoint": str(result.output_checkpoint),
        }
    )
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(str(report_path))


@app.command("interpolate-checkpoints")
def interpolate_checkpoints_command(
    base_checkpoint: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    tuned_checkpoint: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    output_checkpoint: Annotated[Path, typer.Option()],
    alpha: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.5,
    overwrite: Annotated[bool, typer.Option()] = False,
) -> None:
    try:
        result = interpolate_checkpoints(
            base_checkpoint,
            tuned_checkpoint,
            output_checkpoint,
            alpha,
            overwrite=overwrite,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    report_path = result.output_checkpoint.with_suffix(".interpolation.json")
    payload = asdict(result)
    for name in ("base_checkpoint", "tuned_checkpoint", "output_checkpoint"):
        payload[name] = str(payload[name])
    report_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    typer.echo(str(report_path))


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


@app.command("publish-xh25-train-artifacts")
def publish_xh25_train_artifacts_command(
    dataset_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
) -> None:
    try:
        image_map_path, truth_path = publish_train_mining_artifacts(dataset_root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "train_image_map": str(image_map_path),
                "train_ground_truth": str(truth_path),
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


@app.command("build-main-hn-xh25")
def build_main_hn_xh25_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("outputs/xh25/main-hn/train-predictions.json"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-main-hn"),
    confidence_floor: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.60,
    crop_size: Annotated[int, typer.Option(min=1)] = 512,
    object_margin: Annotated[int, typer.Option(min=0)] = 16,
    max_crops_per_group: Annotated[int, typer.Option(min=1)] = 2,
    vehicle_multiplier: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    try:
        policy = HardNegativePolicy(
            confidence_floor=confidence_floor,
            crop_size=crop_size,
            object_margin=object_margin,
            max_crops_per_group=max_crops_per_group,
            vehicle_multiplier=vehicle_multiplier,
            seed=seed,
        )
        result = build_main_hn_dataset(source_root, predictions_json, output_root, policy)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "original_train_images": result.original_train_images,
                "vehicle_upsampled_images": result.vehicle_upsampled_images,
                "selected_hard_negatives": result.selected_hard_negatives,
                "rejected_target_overlap": result.rejected_target_overlap,
                "selected_by_coarse_class": result.selected_by_coarse_class,
            },
            ensure_ascii=False,
        )
    )


@app.command("build-hard-example-xh25")
def build_hard_example_xh25_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("outputs/xh25/single-student/mks-train-predictions.json"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-hard-example"),
    crop_size: Annotated[int, typer.Option(min=1)] = 768,
    background_score_floor: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.60,
    max_positive_crops_per_group: Annotated[int, typer.Option(min=1)] = 8,
    max_negative_crops_per_group: Annotated[int, typer.Option(min=1)] = 2,
    vehicle_positive_multiplier: Annotated[int, typer.Option(min=1)] = 2,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    try:
        policy = HardExamplePolicy(
            crop_size=crop_size,
            background_score_floor=background_score_floor,
            max_positive_crops_per_group=max_positive_crops_per_group,
            max_negative_crops_per_group=max_negative_crops_per_group,
            vehicle_positive_multiplier=vehicle_positive_multiplier,
            seed=seed,
        )
        result = build_hard_example_dataset(
            source_root,
            predictions_json,
            output_root,
            policy,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "original_train_images": result.original_train_images,
                "hard_positive_crops": result.hard_positive_crops,
                "hard_negative_crops": result.hard_negative_crops,
                "missed_truth_by_coarse_class": result.missed_truth_by_coarse_class,
                "selected_positive_by_coarse_class": (result.selected_positive_by_coarse_class),
                "selected_negative_by_coarse_class": (result.selected_negative_by_coarse_class),
            },
            ensure_ascii=False,
        )
    )


@app.command("build-vehicle-confirmer-dataset")
def build_vehicle_confirmer_dataset_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    main_predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("outputs/xh25/vehicle-confirmation/train/main-predictions.json"),
    sph_predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("outputs/xh25/vehicle-confirmation/train/sph-predictions.json"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-vehicle-confirmer"),
    context_scale: Annotated[float, typer.Option(min=0.001)] = 2.0,
    min_side: Annotated[int, typer.Option(min=1)] = 64,
    max_side: Annotated[int, typer.Option(min=1)] = 256,
    output_size: Annotated[int, typer.Option(min=1)] = 160,
    holdout_ratio: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.20,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    try:
        policy = VehicleCropPolicy(
            context_scale=context_scale,
            min_side=min_side,
            max_side=max_side,
            output_size=output_size,
            holdout_ratio=holdout_ratio,
            seed=seed,
        )
        result = build_vehicle_confirmer_dataset(
            source_root,
            main_predictions_json,
            sph_predictions_json,
            output_root,
            policy,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "train_examples": result.train_examples,
                "holdout_examples": result.holdout_examples,
                "train_positive": result.train_positive,
                "train_negative": result.train_negative,
                "holdout_positive": result.holdout_positive,
                "holdout_negative": result.holdout_negative,
            },
            ensure_ascii=False,
        )
    )


@app.command("build-vehicle-expert-dataset")
def build_vehicle_expert_dataset_command(
    source_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25"),
    sph_predictions_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("outputs/xh25/vehicle-confirmation/train/sph-predictions.json"),
    output_root: Annotated[Path, typer.Option()] = Path("datasets/xh25-vehicle-expert"),
    crop_size: Annotated[int, typer.Option(min=1)] = 512,
    holdout_ratio: Annotated[float, typer.Option(min=0.001, max=0.999)] = 0.20,
    max_negatives_per_group: Annotated[int, typer.Option(min=1)] = 8,
    background_score_floor: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    negative_to_positive_ratio: Annotated[float, typer.Option(min=0.0)] = 1.0,
    seed: Annotated[int, typer.Option(min=0)] = 42,
) -> None:
    try:
        result = build_vehicle_expert_dataset(
            source_root,
            sph_predictions_json,
            output_root,
            VehicleExpertPolicy(
                crop_size=crop_size,
                holdout_ratio=holdout_ratio,
                max_negatives_per_group=max_negatives_per_group,
                background_score_floor=background_score_floor,
                negative_to_positive_ratio=negative_to_positive_ratio,
                seed=seed,
            ),
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        json.dumps(
            {
                "output_root": str(result.output_root),
                "positive_crops": result.positive_crops,
                "negative_crops": result.negative_crops,
                "train_crops": result.train_crops,
                "val_crops": result.val_crops,
                "train_positive": result.train_positive,
                "val_positive": result.val_positive,
            },
            ensure_ascii=False,
        )
    )


@app.command("train-vehicle-confirmer")
def train_vehicle_confirmer_command(
    dataset_root: Annotated[
        Path,
        typer.Option(exists=True, file_okay=False),
    ] = Path("datasets/xh25-vehicle-confirmer"),
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/vehicle-confirmation/model"),
    epochs: Annotated[int, typer.Option(min=1)] = 30,
    batch_size: Annotated[int, typer.Option(min=1)] = 64,
    learning_rate: Annotated[float, typer.Option(min=0.0000001)] = 1e-4,
    weight_decay: Annotated[float, typer.Option(min=0.0)] = 1e-4,
    workers: Annotated[int, typer.Option(min=0)] = 4,
    seed: Annotated[int, typer.Option(min=0)] = 42,
    pretrained: Annotated[bool, typer.Option()] = True,
    device: Annotated[str, typer.Option()] = "cuda:0",
) -> None:
    config = VehicleConfirmerTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        workers=workers,
        seed=seed,
        pretrained=pretrained,
    )
    try:
        path = train_vehicle_confirmer(dataset_root, output_dir, config, device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(path))


@app.command("score-vehicle-confirmer")
def score_vehicle_confirmer_command(
    dataset_root: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    manifest_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    checkpoint_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option()],
    device: Annotated[str, typer.Option()] = "cuda:0",
) -> None:
    try:
        score_vehicle_confirmer(dataset_root, manifest_path, checkpoint_path, output_path, device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_path))


@app.command("export-vehicle-confirmer-onnx")
def export_vehicle_confirmer_onnx_command(
    checkpoint_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option()],
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    try:
        path = export_vehicle_confirmer_onnx(checkpoint_path, output_path, device)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(path))


@app.command("export-vehicle-confirmer-engine")
def export_vehicle_confirmer_engine_command(
    onnx_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    engine_path: Annotated[Path, typer.Option()],
) -> None:
    try:
        path = export_vehicle_confirmer_engine(onnx_path, engine_path)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(path))


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
    seed: Annotated[int, typer.Option(min=0)] = 42,
    density_assignment: Annotated[bool, typer.Option()] = False,
    density_constant: Annotated[float, typer.Option(min=0.001)] = 12.0,
    density_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    gcd_loss: Annotated[bool, typer.Option()] = False,
    gcd_assignment: Annotated[bool, typer.Option()] = False,
    gcd_assignment_weight: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
    optimizer: Annotated[str | None, typer.Option()] = None,
    learning_rate: Annotated[float | None, typer.Option(min=0.0000001)] = None,
    freeze: Annotated[int | None, typer.Option(min=0)] = None,
    save_period: Annotated[int | None, typer.Option(min=1)] = None,
    warmup_epochs: Annotated[float | None, typer.Option(min=0.0)] = None,
    warmup_bias_lr: Annotated[float | None, typer.Option(min=0.0)] = None,
) -> None:
    if density_assignment and (gcd_loss or gcd_assignment):
        raise typer.BadParameter("density assignment cannot be combined with GCD training")
    if not gcd_assignment and gcd_assignment_weight != 1.0:
        raise typer.BadParameter("--gcd-assignment-weight requires --gcd-assignment")

    density_options: dict[str, object] = {}
    if density_assignment:
        density_options = {
            "density_assignment": True,
            "density_constant": density_constant,
            "density_threshold": density_threshold,
        }
    gcd_options: dict[str, object] = {}
    if gcd_loss or gcd_assignment:
        gcd_options = {
            "gcd_loss": gcd_loss,
            "gcd_assignment": gcd_assignment,
        }
        if gcd_assignment_weight != 1.0:
            gcd_options["gcd_assignment_weight"] = gcd_assignment_weight
    tuning_options: dict[str, object] = {}
    if optimizer is not None:
        tuning_options["optimizer"] = optimizer
    if learning_rate is not None:
        tuning_options["learning_rate"] = learning_rate
    if freeze is not None:
        tuning_options["freeze"] = freeze
    if save_period is not None:
        tuning_options["save_period"] = save_period
    if warmup_epochs is not None:
        tuning_options["warmup_epochs"] = warmup_epochs
    if warmup_bias_lr is not None:
        tuning_options["warmup_bias_lr"] = warmup_bias_lr
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
        seed=seed,
        pretrained=pretrained,
        **density_options,
        **gcd_options,
        **tuning_options,
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


def _load_named_predictions(
    prediction_specs: list[str],
    taxonomy_name: str,
) -> dict[str, list[Detection]]:
    taxonomy = get_taxonomy(taxonomy_name)
    predictions: dict[str, list[Detection]] = {}
    for spec in prediction_specs:
        name, separator, raw_path = spec.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise typer.BadParameter("prediction must use NAME=PATH")
        normalized_name = name.strip()
        if normalized_name in predictions:
            raise typer.BadParameter(f"duplicate prediction model name: {normalized_name}")
        path = Path(raw_path.strip())
        if not path.is_file():
            raise typer.BadParameter(f"prediction file does not exist: {path}")
        predictions[normalized_name] = load_coco_predictions(path, taxonomy=taxonomy)
    return predictions


@app.command("analyze-complementarity")
def analyze_complementarity_command(
    prediction: Annotated[list[str], typer.Option()],
    ground_truth_json: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ],
    baseline_name: Annotated[str, typer.Option()] = "main",
    taxonomy: Annotated[str, typer.Option()] = "xh25",
    output_path: Annotated[Path, typer.Option()] = Path("outputs/xh25/complementarity/report.json"),
) -> None:
    try:
        taxonomy_object = get_taxonomy(taxonomy)
        predictions = _load_named_predictions(prediction, taxonomy)
        truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
        report = analyze_complementarity(
            predictions,
            truth,
            taxonomy=taxonomy_object,
            baseline_name=baseline_name,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _write_json(output_path, complementarity_report_to_dict(report))
    typer.echo(str(output_path))


@app.command("analyze-vehicle-proposals")
def analyze_vehicle_proposals_command(
    main_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    sph_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    mks_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/vehicle-confirmation/proposal-report.json"
    ),
) -> None:
    taxonomy = get_taxonomy("xh25")
    try:
        main_items = load_coco_predictions(main_predictions, taxonomy=taxonomy)
        sph_items = load_coco_predictions(sph_predictions, taxonomy=taxonomy)
        mks_items = load_coco_predictions(mks_predictions, taxonomy=taxonomy)
        truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy)
        report = analyze_vehicle_consensus(main_items, sph_items, mks_items, truth)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _write_json(output_path, vehicle_consensus_report_to_dict(report))
    typer.echo(str(output_path))


@app.command("analyze-vehicle-expert")
def analyze_vehicle_expert_command(
    main_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    expert_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/vehicle-expert/holdout-report.json"
    ),
) -> None:
    xh25 = get_taxonomy("xh25")
    vehicle1 = get_taxonomy("vehicle1")
    try:
        image_map = _load_image_id_map(image_map_json)
        report = analyze_vehicle_expert_holdout(
            load_coco_predictions(main_predictions, taxonomy=xh25),
            load_coco_predictions(expert_predictions, taxonomy=vehicle1),
            load_coco_ground_truth(ground_truth_json, taxonomy=xh25),
            image_ids={str(image_id) for image_id in image_map.values()},
            thresholds=tuple(round(index * 0.05, 2) for index in range(1, 20)),
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    _write_json(output_path, vehicle_expert_report_to_dict(report))
    typer.echo(str(output_path))


@app.command("apply-suppression")
def apply_suppression_command(
    predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    config_path: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_json: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/postprocess/predictions.json"
    ),
) -> None:
    config = PipelineConfig.from_yaml(config_path)
    taxonomy = get_taxonomy(config.taxonomy)
    image_map = _load_image_id_map(image_map_json)
    stem_by_id = {str(image_id): stem for stem, image_id in image_map.items()}
    loaded = load_coco_predictions(predictions_json, taxonomy=taxonomy)
    try:
        predictions = [
            Detection(
                image_id=stem_by_id[item.image_id],
                class_id=item.class_id,
                score=item.score,
                polygon=item.polygon,
            )
            for item in loaded
        ]
    except KeyError as exc:
        message = f"prediction image_id is missing from image map: {exc.args[0]}"
        raise typer.BadParameter(message) from exc
    kept = suppress_class_detections(predictions, config.class_suppression)
    export_coco_results(
        kept,
        image_map,
        output_json,
        valid_class_ids=taxonomy.valid_ids,
    )
    typer.echo(str(output_json))


def _predictions_with_stem_ids(
    predictions_json: Path,
    *,
    image_map: Mapping[str, int],
    taxonomy_name: str,
) -> list[Detection]:
    taxonomy = get_taxonomy(taxonomy_name)
    stem_by_id = {str(image_id): stem for stem, image_id in image_map.items()}
    loaded = load_coco_predictions(predictions_json, taxonomy=taxonomy)
    try:
        return [
            Detection(
                image_id=stem_by_id[item.image_id],
                class_id=item.class_id,
                score=item.score,
                polygon=item.polygon,
            )
            for item in loaded
        ]
    except KeyError as exc:
        message = f"prediction image_id is missing from image map: {exc.args[0]}"
        raise typer.BadParameter(message) from exc


@app.command("fuse-ranking-ensemble")
def fuse_ranking_ensemble_command(
    aircraft_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ship_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    vehicle_primary_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    vehicle_supplement_predictions: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_json: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/ranking-ensemble/val-predictions.json"
    ),
    aircraft_threshold: Annotated[float, typer.Option()] = 0.25,
    ship_threshold: Annotated[float, typer.Option()] = 0.31,
    vehicle_primary_threshold: Annotated[float, typer.Option()] = 0.25,
    vehicle_supplement_threshold: Annotated[float, typer.Option()] = 0.74,
    vehicle_duplicate_iou: Annotated[float, typer.Option()] = 0.30,
) -> None:
    image_map = _load_image_id_map(image_map_json)
    taxonomy = get_taxonomy("xh25")
    try:
        policy = RankingEnsemblePolicy(
            aircraft_threshold=aircraft_threshold,
            ship_threshold=ship_threshold,
            vehicle_primary_threshold=vehicle_primary_threshold,
            vehicle_supplement_threshold=vehicle_supplement_threshold,
            vehicle_duplicate_iou=vehicle_duplicate_iou,
        )
        fused = fuse_ranking_ensemble(
            aircraft_predictions=_predictions_with_stem_ids(
                aircraft_predictions,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            ship_predictions=_predictions_with_stem_ids(
                ship_predictions,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            vehicle_primary_predictions=_predictions_with_stem_ids(
                vehicle_primary_predictions,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            vehicle_supplement_predictions=_predictions_with_stem_ids(
                vehicle_supplement_predictions,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            taxonomy=taxonomy,
            policy=policy,
        )
        export_coco_results(
            fused,
            image_map,
            output_json,
            valid_class_ids=taxonomy.valid_ids,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_json))


@app.command("fuse-same-weight-multiscale")
def fuse_same_weight_multiscale_command(
    predictions_1024: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    predictions_1280: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    predictions_1536: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    image_map_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    policy_yaml: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_json: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/same-weight-multiscale/val-predictions.json"
    ),
) -> None:
    image_map = _load_image_id_map(image_map_json)
    taxonomy = get_taxonomy("xh25")
    try:
        fused = fuse_same_weight_multiscale(
            predictions_1024=_predictions_with_stem_ids(
                predictions_1024,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            predictions_1280=_predictions_with_stem_ids(
                predictions_1280,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            predictions_1536=_predictions_with_stem_ids(
                predictions_1536,
                image_map=image_map,
                taxonomy_name="xh25",
            ),
            taxonomy=taxonomy,
            policy=load_same_weight_multiscale_policy(policy_yaml),
        )
        export_coco_results(
            fused,
            image_map,
            output_json,
            valid_class_ids=taxonomy.valid_ids,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(output_json))


@app.command("audit-false-positives")
def audit_false_positives_command(
    predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_path: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/postprocess/false-positive-audit.json"
    ),
    taxonomy: Annotated[str, typer.Option()] = "xh25",
) -> None:
    taxonomy_object = get_taxonomy(taxonomy)
    predictions = load_coco_predictions(predictions_json, taxonomy=taxonomy_object)
    truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
    audit = audit_false_positives(predictions, truth, taxonomy=taxonomy_object)
    _write_json(output_path, false_positive_audit_to_dict(audit))
    typer.echo(str(output_path))


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


@app.command("calibrate-thresholds")
def calibrate_thresholds_command(
    baseline_predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    candidate_predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    source_groups_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()],
    strategy: Annotated[str, typer.Option()] = "global",
    taxonomy: Annotated[str, typer.Option()] = "xh25",
    folds: Annotated[int, typer.Option(min=2)] = 5,
    seed: Annotated[int, typer.Option(min=0)] = 42,
    thresholds: Annotated[str, typer.Option()] = DEFAULT_CALIBRATION_GRID_TEXT,
    raw_threshold: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    recall_floor_delta: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.005,
    fdr_cap_delta: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.005,
    ship_recall_floor: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
    ship_calibration_fdr_cap: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.17,
    tie_epsilon: Annotated[float, typer.Option(min=0.0)] = 0.0001,
    acceptance_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.953772,
    acceptance_fdr: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.045037,
    acceptance_ship_recall: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.80,
    acceptance_ship_fdr: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.18,
    acceptance_worst_fold_ship_fdr: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.25,
    acceptance_threshold_range: Annotated[float, typer.Option(min=0.0, max=1.0)] = 0.05,
    base_config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    calibrated_config: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        if strategy not in {"global", "ship-override"}:
            raise ValueError("strategy must be 'global' or 'ship-override'")
        taxonomy_object = get_taxonomy(taxonomy)
        threshold_grid = parse_threshold_grid(thresholds)
        mapping = load_image_group_mapping(ground_truth_json, source_groups_json)
        baseline_predictions = load_coco_predictions(
            baseline_predictions_json, taxonomy=taxonomy_object
        )
        candidate_predictions = load_coco_predictions(
            candidate_predictions_json, taxonomy=taxonomy_object
        )
        ground_truth = load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy_object)
        if strategy == "ship-override":
            result = calibrate_ship_override_thresholds(
                baseline_predictions,
                candidate_predictions,
                ground_truth,
                mapping.image_to_group,
                taxonomy_object,
                folds=folds,
                seed=seed,
                thresholds=threshold_grid,
                raw_threshold=raw_threshold,
                recall_floor_delta=recall_floor_delta,
                fdr_cap_delta=fdr_cap_delta,
                ship_recall_floor=ship_recall_floor,
                ship_fdr_cap=ship_calibration_fdr_cap,
                tie_epsilon=tie_epsilon,
                acceptance_recall=acceptance_recall,
                acceptance_fdr=acceptance_fdr,
                acceptance_ship_recall=acceptance_ship_recall,
                acceptance_ship_fdr=acceptance_ship_fdr,
                acceptance_worst_fold_ship_fdr=acceptance_worst_fold_ship_fdr,
                acceptance_threshold_range=acceptance_threshold_range,
            )
            paths = write_ship_override_calibration_artifacts(
                result,
                output_dir,
                taxonomy_object,
                base_config=base_config,
                calibrated_config=calibrated_config,
            )
        else:
            result = calibrate_thresholds(
                baseline_predictions,
                candidate_predictions,
                ground_truth,
                mapping.image_to_group,
                taxonomy_object,
                folds=folds,
                seed=seed,
                thresholds=threshold_grid,
                raw_threshold=raw_threshold,
                recall_floor_delta=recall_floor_delta,
                fdr_cap_delta=fdr_cap_delta,
                tie_epsilon=tie_epsilon,
                acceptance_recall=acceptance_recall,
                acceptance_fdr=acceptance_fdr,
                acceptance_ship_recall=acceptance_ship_recall,
                acceptance_ship_fdr=acceptance_ship_fdr,
                acceptance_threshold_range=acceptance_threshold_range,
            )
            paths = write_calibration_artifacts(
                result,
                output_dir,
                taxonomy_object,
                base_config=base_config,
                calibrated_config=calibrated_config,
            )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(paths["summary"]))
    if not result.passed:
        raise typer.Exit(code=2)


@app.command("optimize-ranking-thresholds")
def optimize_ranking_thresholds_command(
    predictions_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    ground_truth_json: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    baseline_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    output_dir: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/single-student/ranking-thresholds"
    ),
    thresholds: Annotated[str, typer.Option()] = (
        "0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,"
        "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95"
    ),
    passes: Annotated[int, typer.Option(min=1)] = 2,
) -> None:
    taxonomy = get_taxonomy("xh25")
    try:
        result = optimize_ranking_thresholds(
            load_coco_predictions(predictions_json, taxonomy=taxonomy),
            load_coco_ground_truth(ground_truth_json, taxonomy=taxonomy),
            baseline=load_evaluation_report(baseline_report),
            taxonomy=taxonomy,
            thresholds=thresholds,
            passes=passes,
        )
        paths = write_ranking_threshold_artifacts(result, output_dir=output_dir)
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(paths["summary"]))


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


@app.command("compare-seven-metrics")
def compare_seven_metrics_command(
    baseline_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    experiment_report: Annotated[Path, typer.Option(exists=True, dir_okay=False)],
    baseline_latency_seconds: Annotated[float, typer.Option(min=0.0)],
    experiment_latency_seconds: Annotated[float, typer.Option(min=0.0)],
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/xh25/ranking-ensemble"),
) -> None:
    try:
        paths = write_seven_metric_comparison_artifacts(
            load_evaluation_report(baseline_report),
            load_evaluation_report(experiment_report),
            baseline_latency_seconds=baseline_latency_seconds,
            experiment_latency_seconds=experiment_latency_seconds,
            output_dir=output_dir,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(str(paths["json"]))


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


def _timing_payload(samples: list[float]) -> dict[str, object]:
    summary = summarize_durations(samples)
    return {
        "samples_s": samples,
        "median_s": summary["median_s"],
        "p95_s": summary["p95_s"],
        "maximum_s": max(samples),
    }


@app.command("benchmark-ranking-ensemble")
def benchmark_ranking_ensemble_command(
    primary_config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/xh25-mksnet-lite.yaml"),
    ship_config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/xh25-sph-p2-nam.yaml"),
    vehicle_supplement_config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/xh25-sph-p2.yaml"),
    image_path: Annotated[Path, typer.Option()] = Path("outputs/benchmark/synthetic-10000.png"),
    repeats: Annotated[int, typer.Option(min=1)] = 5,
    output_path: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/ranking-ensemble/benchmark.json"
    ),
) -> None:
    if not image_path.exists():
        create_synthetic_image(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise typer.BadParameter(f"cannot read benchmark image: {image_path}")

    primary_config = PipelineConfig.from_yaml(primary_config_path)
    ship_config = PipelineConfig.from_yaml(ship_config_path)
    supplement_config = PipelineConfig.from_yaml(vehicle_supplement_config_path)
    if {primary_config.taxonomy, ship_config.taxonomy, supplement_config.taxonomy} != {"xh25"}:
        raise typer.BadParameter("all ranking ensemble configs must use the xh25 taxonomy")
    primary = InferencePipeline(_build_detector(primary_config), primary_config, cache_root=None)
    ship = InferencePipeline(_build_detector(ship_config), ship_config, cache_root=None)
    supplement = InferencePipeline(
        _build_detector(supplement_config),
        supplement_config,
        cache_root=None,
    )
    policy = RankingEnsemblePolicy()
    taxonomy = get_taxonomy("xh25")
    for name, pipeline in (
        ("primary", primary),
        ("ship", ship),
        ("vehicle-supplement", supplement),
    ):
        pipeline.run(image, f"{image_path.stem}-{name}-warmup")

    primary_samples: list[float] = []
    ship_samples: list[float] = []
    supplement_samples: list[float] = []
    combined_samples: list[float] = []
    for index in range(repeats):
        image_id = f"{image_path.stem}-{index}"
        started = perf_counter()
        primary_result = primary.run(image, image_id)
        ship_result = ship.run(image, image_id)
        supplement_result = supplement.run(image, image_id)
        fuse_ranking_ensemble(
            aircraft_predictions=primary_result.detections,
            ship_predictions=ship_result.detections,
            vehicle_primary_predictions=primary_result.detections,
            vehicle_supplement_predictions=supplement_result.detections,
            taxonomy=taxonomy,
            policy=policy,
        )
        combined_samples.append(perf_counter() - started)
        primary_samples.append(primary_result.timings.total_s)
        ship_samples.append(ship_result.timings.total_s)
        supplement_samples.append(supplement_result.timings.total_s)

    payload = {
        "primary": _timing_payload(primary_samples),
        "ship": _timing_payload(ship_samples),
        "vehicle_supplement": _timing_payload(supplement_samples),
        "combined": _timing_payload(combined_samples),
        "gate": {
            "limit_seconds": 20.0,
            "passed": all(sample <= 20.0 for sample in combined_samples),
        },
    }
    _write_json(output_path, payload)
    typer.echo(str(output_path))


@app.command("benchmark-vehicle-proposals")
def benchmark_vehicle_proposals_command(
    main_config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/xh25-historical-main.yaml"),
    sph_config_path: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False),
    ] = Path("configs/xh25-sph-p2.yaml"),
    image_path: Annotated[Path, typer.Option()] = Path("outputs/benchmark/synthetic-10000.png"),
    repeats: Annotated[int, typer.Option(min=1)] = 5,
    reserve_seconds: Annotated[float, typer.Option(min=0.0)] = 1.0,
    limit_seconds: Annotated[float, typer.Option(min=0.001)] = 20.0,
    output_path: Annotated[Path, typer.Option()] = Path(
        "outputs/xh25/vehicle-confirmation/paired-latency.json"
    ),
) -> None:
    synthetic = not image_path.exists()
    if synthetic:
        create_synthetic_image(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise typer.BadParameter(f"cannot read benchmark image: {image_path}")

    main_config = PipelineConfig.from_yaml(main_config_path)
    sph_config = PipelineConfig.from_yaml(sph_config_path)
    main_pipeline = InferencePipeline(_build_detector(main_config), main_config, cache_root=None)
    sph_pipeline = InferencePipeline(_build_detector(sph_config), sph_config, cache_root=None)
    try:
        report = benchmark_vehicle_proposal_pair(
            main_pipeline,
            sph_pipeline,
            image,
            image_path.stem,
            repeats=repeats,
            reserve_seconds=reserve_seconds,
            limit_seconds=limit_seconds,
        )
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = {
        **vehicle_latency_report_to_dict(report),
        "image": {"path": str(image_path), "synthetic": synthetic},
    }
    _write_json(output_path, payload)
    typer.echo(str(output_path))


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
