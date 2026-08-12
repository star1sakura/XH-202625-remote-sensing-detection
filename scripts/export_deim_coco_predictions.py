#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from xh_detect.deim_integration import (  # noqa: E402
    initialize_deim_extensions,
    select_deim_checkpoint_state,
)
from xh_detect.mmdet_predictions import (  # noqa: E402
    instances_to_coco_predictions,
    positive_xyxy_mask,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export low-threshold DEIM predictions as COCO result JSON."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timing-output",
        type=Path,
        help="Optional JSON path for model and postprocessor timing.",
    )
    parser.add_argument(
        "--deim-root",
        type=Path,
        default=PROJECT_ROOT / ".third_party" / "DEIM",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confidence", type=float, default=0.001)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-state",
        choices=("auto", "ema", "model"),
        default="auto",
        help="Checkpoint state to export; auto prefers EMA for backward compatibility.",
    )
    return parser.parse_args()


def _load_annotation_metadata(path: Path) -> tuple[set[int], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("annotation JSON root must be a mapping")
    categories = payload.get("categories")
    images = payload.get("images")
    if not isinstance(categories, list) or not isinstance(images, list):
        raise ValueError("annotation JSON must contain categories and images lists")
    category_ids = {
        int(category["id"])
        for category in categories
        if isinstance(category, Mapping)
        and isinstance(category.get("id"), int)
        and not isinstance(category.get("id"), bool)
    }
    if len(category_ids) != len(categories):
        raise ValueError("every category must have a unique integer id")
    return category_ids, len(images)


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    if args.image_size <= 0 or args.image_size % 32 != 0:
        raise ValueError("image-size must be a positive multiple of 32")
    if args.batch_size <= 0 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers must be non-negative")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")

    args.deim_root = args.deim_root.resolve()
    sys.path.insert(0, str(args.deim_root))
    from engine.core import YAMLConfig

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    category_ids, annotation_images = _load_annotation_metadata(args.annotations)
    resize_ops = [
        {"type": "Resize", "size": [args.image_size, args.image_size]},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    ]
    cfg = YAMLConfig(
        str(args.config),
        device=str(device),
        eval_spatial_size=[args.image_size, args.image_size],
        HGNetv2={"pretrained": False},
        val_dataloader={
            "dataset": {
                "img_folder": str(args.data_root.resolve()),
                "ann_file": str(args.annotations.resolve()),
                "transforms": {"ops": resize_ops},
            },
            "total_batch_size": args.batch_size,
            "num_workers": args.workers,
        },
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_source, state = select_deim_checkpoint_state(
        checkpoint,
        preferred=args.checkpoint_state,
    )
    model = cfg.model
    initialize_deim_extensions(model)
    model.load_state_dict(state, strict=True)
    model = model.eval().to(device)
    postprocessor = cfg.postprocessor.eval().to(device)
    dataloader = cfg.val_dataloader

    use_amp = args.amp and device.type == "cuda"
    rows: list[dict[str, object]] = []
    processed = 0
    inference_seconds = 0.0
    started_at = time.perf_counter()

    with torch.inference_mode():
        for samples, targets in dataloader:
            remaining = None if args.limit is None else args.limit - processed
            if remaining is not None and remaining <= 0:
                break
            if remaining is not None and len(targets) > remaining:
                samples = samples[:remaining]
                targets = targets[:remaining]

            samples = samples.to(device, non_blocking=True)
            original_sizes = torch.stack([target["orig_size"] for target in targets]).to(
                device, non_blocking=True
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            _cuda_sync(device)
            batch_started_at = time.perf_counter()
            with autocast:
                outputs = model(samples)
                results = postprocessor(outputs, original_sizes)
            _cuda_sync(device)
            inference_seconds += time.perf_counter() - batch_started_at

            for target, result in zip(targets, results, strict=True):
                image_id = int(target["image_id"].reshape(-1)[0].item())
                bboxes = result["boxes"].detach().cpu().numpy()
                scores = result["scores"].detach().cpu().numpy()
                labels = result["labels"].detach().cpu().numpy()
                valid_boxes = positive_xyxy_mask(bboxes)
                rows.extend(
                    instances_to_coco_predictions(
                        image_id=image_id,
                        bboxes=bboxes[valid_boxes],
                        scores=scores[valid_boxes],
                        labels=labels[valid_boxes],
                        confidence=args.confidence,
                        valid_class_ids=category_ids,
                    )
                )
                processed += 1
            if processed % 50 == 0 or processed == annotation_images:
                expected = min(annotation_images, args.limit or annotation_images)
                print(f"DEIM inference: {processed}/{expected}")

    wall_seconds = time.perf_counter() - started_at
    if processed == 0:
        raise ValueError("annotation dataset contains no images")
    summary = {
        "model": "DEIM-D-FINE-L",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_state": state_source,
        "config": str(args.config.resolve()),
        "annotations": str(args.annotations.resolve()),
        "images": processed,
        "predictions": len(rows),
        "confidence": args.confidence,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "amp": use_amp,
        "device": str(device),
        "inference_seconds": inference_seconds,
        "inference_ms_per_image": 1000.0 * inference_seconds / processed,
        "wall_seconds": wall_seconds,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if args.timing_output is not None:
        args.timing_output.parent.mkdir(parents=True, exist_ok=True)
        args.timing_output.write_text(
            json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
