from pathlib import Path

import pytest

from xh_detect.config import PipelineConfig


def test_pipeline_config_defaults_match_expected_values() -> None:
    config = PipelineConfig()

    assert config.task == "obb"
    assert config.taxonomy == "legacy3"
    assert config.model_path == "yolo26s-obb.pt"
    assert config.device == "0"
    assert config.tile_size == 1024
    assert config.image_size == 1024
    assert config.overlap == 0.2
    assert config.batch_size == 8
    assert config.merge_iou == 0.3
    assert config.edge_margin == 16
    assert config.half is True
    assert config.class_thresholds == {0: 0.25, 1: 0.25, 2: 0.25}
    assert config.class_suppression == {}
    assert config.valid_class_ids == frozenset({0, 1, 2})


def test_class_thresholds_are_immutable_and_copy_on_construct() -> None:
    source_thresholds = {0: 0.4, 1: 0.3, 2: 0.2}
    config = PipelineConfig(class_thresholds=source_thresholds)

    source_thresholds[0] = 0.9

    assert config.class_thresholds == {0: 0.4, 1: 0.3, 2: 0.2}
    assert min(config.class_thresholds.values()) == 0.2

    with pytest.raises(TypeError):
        config.class_thresholds[0] = 0.9


def test_pipeline_config_to_dict_returns_serializable_primitives() -> None:
    config = PipelineConfig()

    assert config.to_dict() == {
        "task": "obb",
        "taxonomy": "legacy3",
        "model_path": "yolo26s-obb.pt",
        "device": "0",
        "image_size": 1024,
        "tile_size": 1024,
        "overlap": 0.2,
        "batch_size": 8,
        "merge_iou": 0.3,
        "edge_margin": 16,
        "half": True,
        "class_thresholds": {0: 0.25, 1: 0.25, 2: 0.25},
        "class_suppression": {},
    }


def test_baseline_yaml_loads_expected_pipeline_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    baseline_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"

    config = PipelineConfig.from_yaml(baseline_path)

    assert config.task == "obb"
    assert config.taxonomy == "legacy3"
    assert config.model_path == "yolo26s-obb.pt"
    assert config.tile_size == 1024
    assert config.batch_size == 8
    assert config.class_thresholds == {0: 0.25, 1: 0.25, 2: 0.25}
    assert config.valid_class_ids == frozenset({0, 1, 2})


def test_xh25_detect_config_accepts_all_25_thresholds() -> None:
    config = PipelineConfig(
        task="detect",
        taxonomy="xh25",
        model_path="best.pt",
        class_thresholds={class_id: 0.25 for class_id in range(25)},
    )

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.valid_class_ids == frozenset(range(25))


def test_config_rejects_threshold_ids_not_matching_taxonomy() -> None:
    with pytest.raises(ValueError, match="class_thresholds"):
        PipelineConfig(
            task="detect",
            taxonomy="xh25",
            class_thresholds={0: 0.25, 1: 0.25, 2: 0.25},
        )


def test_config_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="task"):
        PipelineConfig(task="segment")  # type: ignore[arg-type]


def test_xh25_hbb_yaml_loads_detect_task_and_taxonomy() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "xh25-hbb.yaml"

    config = PipelineConfig.from_yaml(config_path)

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.valid_class_ids == frozenset(range(25))
    assert config.class_thresholds == {class_id: 0.25 for class_id in range(25)}


def test_xh25_mksnet_lite_thresholded_yaml_uses_optimized_thresholds() -> None:
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "xh25-mksnet-lite-thresholded.yaml"
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.task == "detect"
    assert config.taxonomy == "xh25"
    assert config.model_path == "runs/train/xh25-mksnet-lite/weights/best.pt"
    assert config.class_thresholds[2] == 0.40
    assert config.class_thresholds[4] == 0.55
    assert config.class_thresholds[5] == 0.50
    for class_id in set(range(25)) - {2, 4, 5}:
        assert config.class_thresholds[class_id] == 0.30


def test_overlap_one_raises_value_error() -> None:
    with pytest.raises(ValueError, match="overlap"):
        PipelineConfig(overlap=1.0)


def test_tile_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="tile_size"):
        PipelineConfig(tile_size=0)


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        PipelineConfig(batch_size=0)


def test_merge_iou_must_be_within_unit_interval() -> None:
    with pytest.raises(ValueError, match="merge_iou"):
        PipelineConfig(merge_iou=1.1)


def test_edge_margin_must_not_be_negative() -> None:
    with pytest.raises(ValueError, match="edge_margin"):
        PipelineConfig(edge_margin=-1)


def test_class_thresholds_must_cover_exact_three_class_ids() -> None:
    with pytest.raises(ValueError, match="class"):
        PipelineConfig(class_thresholds={0: 0.25, 1: 0.25})


def test_class_thresholds_must_stay_in_unit_interval() -> None:
    with pytest.raises(ValueError, match="threshold"):
        PipelineConfig(class_thresholds={0: -0.1, 1: 0.25, 2: 0.25})


def test_from_yaml_rejects_non_mapping_root(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        PipelineConfig.from_yaml(path)


def test_from_yaml_rejects_unknown_configuration_keys(tmp_path: Path) -> None:
    path = tmp_path / "extra-key.yaml"
    path.write_text(
        "model_path: yolo26s-obb.pt\n"
        "class_thresholds:\n"
        "  0: 0.25\n"
        "  1: 0.25\n"
        "  2: 0.25\n"
        "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown configuration keys"):
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

    assert config.class_thresholds == {0: 1.0, 1: 0.5, 2: 0.25}
    assert all(isinstance(class_id, int) for class_id in config.class_thresholds)
    assert all(isinstance(threshold, float) for threshold in config.class_thresholds.values())


def test_pipeline_config_loads_ship_only_suppression(tmp_path: Path) -> None:
    path = tmp_path / "ship.yaml"
    path.write_text(
        "task: detect\n"
        "taxonomy: xh25\n"
        "model_path: model.pt\n"
        "class_suppression:\n"
        "  3: {method: diou, threshold: 0.15}\n"
        "class_thresholds:\n" + "".join(f"  {class_id}: 0.25\n" for class_id in range(25)),
        encoding="utf-8",
    )

    config = PipelineConfig.from_yaml(path)

    assert config.class_suppression[3].method == "diou"
    assert config.class_suppression[3].threshold == 0.15


def test_pipeline_config_rejects_suppression_class_outside_taxonomy() -> None:
    from xh_detect.postprocess import SuppressionRule

    with pytest.raises(ValueError, match="class_suppression"):
        PipelineConfig(class_suppression={24: SuppressionRule("iou", 0.3)})


@pytest.mark.parametrize(
    ("name", "method", "threshold"),
    [
        ("xh25-main-ship-iou.yaml", "iou", 0.30),
        ("xh25-main-ship-diou.yaml", "diou", 0.15),
    ],
)
def test_main_ship_postprocess_configs_are_ship_only(
    name: str,
    method: str,
    threshold: float,
) -> None:
    config = PipelineConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / name)

    assert config.model_path == "runs/train/xh25-yolo26s-e80/weights/best.pt"
    assert set(config.class_suppression) == {0, 1, 2, 3}
    assert {rule.method for rule in config.class_suppression.values()} == {method}
    assert {rule.threshold for rule in config.class_suppression.values()} == {threshold}


@pytest.mark.parametrize(
    ("name", "run_name"),
    [
        ("xh25-main-hn.yaml", "xh25-main-hn"),
        ("xh25-main-hn-density.yaml", "xh25-main-hn-density"),
    ],
)
def test_main_hn_configs_only_change_candidate_weight_path(
    name: str,
    run_name: str,
) -> None:
    config = PipelineConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs" / name)

    assert config.model_path == f"runs/train/{run_name}/weights/best.pt"
    assert config.taxonomy == "xh25"
    assert config.image_size == 1024
    assert config.class_suppression == {}
    assert set(config.class_thresholds) == set(range(25))


def test_historical_main_config_uses_supplied_checkpoint() -> None:
    config = PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "xh25-historical-main.yaml"
    )

    assert config.model_path == "outputs/xh25/historical-main/best.pt"
    assert config.taxonomy == "xh25"
    assert config.image_size == 1024
    assert config.tile_size == 1024
    assert config.merge_iou == 0.3
    assert config.class_thresholds == {class_id: 0.25 for class_id in range(25)}


def test_single_student_config_uses_one_checkpoint() -> None:
    config = PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1] / "configs" / "xh25-single-student.yaml"
    )

    assert config.model_path == "runs/train/xh25-single-student-head/weights/best.pt"
    assert config.taxonomy == "xh25"
    assert config.image_size == 1024
    assert config.class_thresholds == {class_id: 0.25 for class_id in range(25)}


def test_single_student_search_config_keeps_low_score_predictions() -> None:
    config = PipelineConfig.from_yaml(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "xh25-single-student-search.yaml"
    )

    assert config.model_path == "runs/train/xh25-single-student-head/weights/best.pt"
    assert config.class_thresholds == {class_id: 0.05 for class_id in range(25)}
