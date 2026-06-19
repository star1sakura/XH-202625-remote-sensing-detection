from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def _default_thresholds() -> dict[int, float]:
    return {0: 0.25, 1: 0.25, 2: 0.25}


@dataclass(frozen=True)
class PipelineConfig:
    model_path: str = "yolo26s-obb.pt"
    device: str = "0"
    image_size: int = 1024
    tile_size: int = 1024
    overlap: float = 0.2
    batch: int = 8
    merge_iou: float = 0.3
    edge_margin: int = 16
    half: bool = True
    thresholds: dict[int, float] = field(default_factory=_default_thresholds)

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.batch <= 0:
            raise ValueError("batch must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 <= self.merge_iou <= 1:
            raise ValueError("merge_iou must be in [0, 1]")
        if self.edge_margin < 0:
            raise ValueError("edge_margin must be non-negative")
        if set(self.thresholds) != {0, 1, 2}:
            raise ValueError("thresholds must define class IDs 0, 1, and 2")
        for class_id, threshold in self.thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"threshold for class {class_id} must be in [0, 1]")

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PipelineConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("YAML root must be a mapping")

        raw_mapping = dict(raw)
        class_thresholds = raw_mapping.pop("class_thresholds", None)
        if not isinstance(class_thresholds, Mapping):
            raise ValueError("class_thresholds must be a mapping")

        raw_mapping["thresholds"] = {
            int(class_id): float(threshold) for class_id, threshold in class_thresholds.items()
        }
        return cls(**raw_mapping)
