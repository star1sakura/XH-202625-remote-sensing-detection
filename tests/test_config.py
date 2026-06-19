from pathlib import Path

import pytest

from xh_detect.config import PipelineConfig


def test_pipeline_config_defaults_match_expected_values() -> None:
    config = PipelineConfig()

    assert config.model_path == "yolo26s-obb.pt"
    assert config.device == "0"
    assert config.tile_size == 1024
    assert config.image_size == 1024
    assert config.overlap == 0.2
    assert config.batch == 8
    assert config.merge_iou == 0.3
    assert config.edge_margin == 16
    assert config.half is True
    assert config.thresholds == {0: 0.25, 1: 0.25, 2: 0.25}


def test_baseline_yaml_loads_expected_pipeline_values() -> None:
    config = PipelineConfig.from_yaml(Path("configs/baseline.yaml"))

    assert config.model_path == "yolo26s-obb.pt"
    assert config.tile_size == 1024
    assert config.thresholds == {0: 0.25, 1: 0.25, 2: 0.25}


def test_overlap_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PipelineConfig(overlap=1.0)


def test_tile_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tile_size"):
        PipelineConfig(tile_size=0)


def test_batch_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch"):
        PipelineConfig(batch=0)


def test_merge_iou_must_be_within_unit_interval() -> None:
    with pytest.raises(ValueError, match="merge_iou"):
        PipelineConfig(merge_iou=1.1)


def test_edge_margin_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="edge_margin"):
        PipelineConfig(edge_margin=-1)


def test_thresholds_must_cover_exact_three_class_ids() -> None:
    with pytest.raises(ValueError, match="class"):
        PipelineConfig(thresholds={0: 0.25, 1: 0.25})


def test_thresholds_must_stay_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold"):
        PipelineConfig(thresholds={0: -0.1, 1: 0.25, 2: 0.25})


def test_from_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        PipelineConfig.from_yaml(path)


def test_from_yaml_rejects_missing_class_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "missing-thresholds.yaml"
    path.write_text("model_path: yolo26s-obb.pt\n", encoding="utf-8")

    with pytest.raises(ValueError, match="class_thresholds"):
        PipelineConfig.from_yaml(path)


def test_from_yaml_rejects_non_mapping_class_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "bad-thresholds.yaml"
    path.write_text(
        "model_path: yolo26s-obb.pt\nclass_thresholds:\n  - 0.25\n  - 0.25\n  - 0.25\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="class_thresholds"):
        PipelineConfig.from_yaml(path)


def test_from_yaml_converts_threshold_keys_and_values(tmp_path: Path) -> None:
    path = tmp_path / "typed-thresholds.yaml"
    path.write_text(
        "model_path: yolo26s-obb.pt\nclass_thresholds:\n  '0': 1\n  '1': 0.5\n  '2': 0.25\n",
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(path)

    assert config.thresholds == {0: 1.0, 1: 0.5, 2: 0.25}
    assert all(isinstance(class_id, int) for class_id in config.thresholds)
    assert all(isinstance(threshold, float) for threshold in config.thresholds.values())
