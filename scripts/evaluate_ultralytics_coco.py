#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from xh_detect.evaluator import load_coco_ground_truth, load_coco_predictions
from xh_detect.taxonomy import get_taxonomy
from xh_detect.ultralytics_evaluation import (
    evaluate_ultralytics,
    ultralytics_evaluation_to_dict,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate COCO predictions with the pinned Ultralytics metric implementation."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--taxonomy", default="xh25")
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU threads for the many small per-image IoU operations.",
    )
    parser.add_argument(
        "--keep-duplicate-ground-truth",
        action="store_true",
        help="Disable Ultralytics-style removal of exact duplicate labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.torch_threads <= 0:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    started_at = time.perf_counter()
    taxonomy = get_taxonomy(args.taxonomy)
    predictions = load_coco_predictions(args.predictions, taxonomy=taxonomy)
    ground_truth = load_coco_ground_truth(args.ground_truth, taxonomy=taxonomy)
    loaded_at = time.perf_counter()
    result = evaluate_ultralytics(
        predictions,
        ground_truth,
        taxonomy=taxonomy,
        max_detections=args.max_detections,
        deduplicate_ground_truth=not args.keep_duplicate_ground_truth,
    )
    payload = ultralytics_evaluation_to_dict(result)
    payload["runtime"] = {
        "load_seconds": loaded_at - started_at,
        "evaluate_seconds": time.perf_counter() - loaded_at,
        "torch_threads": args.torch_threads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
