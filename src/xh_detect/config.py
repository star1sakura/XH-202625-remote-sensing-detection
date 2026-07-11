from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import yaml

from xh_detect.postprocess import SuppressionRule
from xh_detect.taxonomy import get_taxonomy


def _default_class_thresholds() -> dict[int, float]:
    return {0: 0.25, 1: 0.25, 2: 0.25}


@dataclass(frozen=True)
class PipelineConfig:
    task: str = "obb"
    taxonomy: str = "legacy3"
    model_path: str = "yolo26s-obb.pt"
    device: str = "0"
    image_size: int = 1024
    tile_size: int = 1024
    overlap: float = 0.2
    batch_size: int = 8
    merge_iou: float = 0.3
    edge_margin: int = 16
    half: bool = True
    class_thresholds: Mapping[int, float] = field(default_factory=_default_class_thresholds)
    class_suppression: Mapping[int, SuppressionRule] = field(default_factory=dict)

    def __post_init__(self) -> None:
        class_thresholds = dict(self.class_thresholds)
        class_suppression = dict(self.class_suppression)
        taxonomy = get_taxonomy(self.taxonomy)

        if self.task not in {"detect", "obb"}:
            raise ValueError("task must be detect or obb")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 <= self.merge_iou <= 1:
            raise ValueError("merge_iou must be in [0, 1]")
        if self.edge_margin < 0:
            raise ValueError("edge_margin must be non-negative")
        if set(class_thresholds) != taxonomy.valid_ids:
            raise ValueError("class_thresholds must define exactly the taxonomy class IDs")
        for class_id, threshold in class_thresholds.items():
            if not 0 <= threshold <= 1:
                raise ValueError(f"threshold for class {class_id} must be in [0, 1]")
        for class_id, rule in class_suppression.items():
            if (
                isinstance(class_id, bool)
                or not isinstance(class_id, int)
                or class_id not in taxonomy.valid_ids
            ):
                raise ValueError("class_suppression IDs must belong to the taxonomy")
            if not isinstance(rule, SuppressionRule):
                raise TypeError("class_suppression values must be SuppressionRule instances")
        object.__setattr__(self, "class_thresholds", MappingProxyType(class_thresholds))
        object.__setattr__(self, "class_suppression", MappingProxyType(class_suppression))

    @property
    def valid_class_ids(self) -> frozenset[int]:
        return get_taxonomy(self.taxonomy).valid_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "taxonomy": self.taxonomy,
            "model_path": self.model_path,
            "device": self.device,
            "image_size": self.image_size,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "batch_size": self.batch_size,
            "merge_iou": self.merge_iou,
            "edge_margin": self.edge_margin,
            "half": self.half,
            "class_thresholds": dict(self.class_thresholds),
            "class_suppression": {
                class_id: {"method": rule.method, "threshold": rule.threshold}
                for class_id, rule in self.class_suppression.items()
            },
        }

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
        class_suppression = raw_mapping.pop("class_suppression", {})
        if not isinstance(class_suppression, Mapping):
            raise ValueError("class_suppression must be a mapping")

        valid_keys = {
            "task",
            "taxonomy",
            "model_path",
            "device",
            "image_size",
            "tile_size",
            "overlap",
            "batch_size",
            "merge_iou",
            "edge_margin",
            "half",
            "class_thresholds",
            "class_suppression",
        }
        unknown_keys = sorted(key for key in raw_mapping if key not in valid_keys)
        if unknown_keys:
            joined_keys = ", ".join(unknown_keys)
            raise ValueError(f"unknown configuration keys: {joined_keys}")

        raw_mapping["class_thresholds"] = {
            int(class_id): float(threshold) for class_id, threshold in class_thresholds.items()
        }
        parsed_suppression: dict[int, SuppressionRule] = {}
        for class_id, payload in class_suppression.items():
            if not isinstance(payload, Mapping):
                raise ValueError("class_suppression rules must be mappings")
            rule_fields = dict(payload)
            if set(rule_fields) != {"method", "threshold"}:
                raise ValueError("class_suppression rules require method and threshold")
            parsed_suppression[int(class_id)] = SuppressionRule(
                method=str(rule_fields["method"]),
                threshold=float(rule_fields["threshold"]),
            )
        raw_mapping["class_suppression"] = parsed_suppression
        return cls(**raw_mapping)
