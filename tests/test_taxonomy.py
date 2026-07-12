from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

import xh_detect.taxonomy as taxonomy_module
from xh_detect.taxonomy import Taxonomy, get_taxonomy


def test_xh25_taxonomy_matches_official_classes_and_coarse_groups() -> None:
    taxonomy = get_taxonomy("xh25")

    assert isinstance(taxonomy.valid_ids, frozenset)
    assert taxonomy.valid_ids == frozenset(range(25))
    assert taxonomy.names == {
        0: "HM",
        1: "LQS",
        2: "QHS",
        3: "MS",
        4: "A1_SU-35",
        5: "A2_C-130",
        6: "A3_C-17",
        7: "A4_C-5",
        8: "A5_F-16",
        9: "A6_TU-160",
        10: "A7_E-3",
        11: "A8_B-52",
        12: "A9_P-3C",
        13: "A10_B-1B",
        14: "A11_E-8",
        15: "A12_TU-22",
        16: "A13_F-15",
        17: "A14_KC-135",
        18: "A15_F-22",
        19: "A16_FA-18",
        20: "A17_TU-95",
        21: "A18_KC-10",
        22: "A19_SU-34",
        23: "A20_SU-24",
        24: "FSC",
    }
    assert taxonomy.coarse_name(0) == "ship"
    assert taxonomy.coarse_name(4) == "aircraft"
    assert taxonomy.coarse_name(24) == "vehicle"
    assert {taxonomy.coarse_name(class_id) for class_id in range(4)} == {"ship"}
    assert {taxonomy.coarse_name(class_id) for class_id in range(4, 24)} == {"aircraft"}


def test_legacy3_taxonomy_preserves_its_coarse_names() -> None:
    taxonomy = get_taxonomy("legacy3")

    assert taxonomy.names == {0: "aircraft", 1: "ship", 2: "vehicle"}
    assert {
        class_id: taxonomy.coarse_name(class_id) for class_id in taxonomy.valid_ids
    } == taxonomy.names


def test_vehicle_one_class_taxonomy_maps_class_zero_to_vehicle() -> None:
    taxonomy = get_taxonomy("vehicle1")

    assert taxonomy.names == {0: "vehicle"}
    assert taxonomy.coarse_name(0) == "vehicle"


def test_unknown_taxonomy_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown taxonomy"):
        get_taxonomy("unknown")


def test_canonical_taxonomy_registry_is_private_and_read_only() -> None:
    assert not hasattr(taxonomy_module, "TAXONOMIES")

    registry = taxonomy_module._TAXONOMIES
    with pytest.raises(TypeError):
        registry["xh25"] = get_taxonomy("legacy3")


def test_unknown_class_id_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_taxonomy("xh25").coarse_name(25)


@pytest.mark.parametrize("key", ["", " ", 0])
def test_taxonomy_rejects_invalid_keys(key: object) -> None:
    with pytest.raises(ValueError):
        Taxonomy(key, {0: "aircraft"}, {0: "aircraft"})


def test_taxonomy_rejects_empty_mappings() -> None:
    with pytest.raises(ValueError):
        Taxonomy("empty", {}, {})


@pytest.mark.parametrize("class_id", [False, 0.0])
def test_taxonomy_rejects_bool_and_non_integer_ids(class_id: object) -> None:
    with pytest.raises(ValueError):
        Taxonomy("invalid-id", {class_id: "aircraft"}, {class_id: "aircraft"})


@pytest.mark.parametrize("class_id", [True, 0.0])
def test_coarse_name_rejects_bool_and_non_integer_ids(class_id: object) -> None:
    with pytest.raises(ValueError):
        get_taxonomy("xh25").coarse_name(class_id)


def test_taxonomy_copies_input_mappings_into_read_only_proxies() -> None:
    names = {0: "aircraft"}
    coarse_by_id = {0: "aircraft"}

    taxonomy = Taxonomy("custom", names, coarse_by_id)
    names[0] = "changed"
    coarse_by_id[0] = "ship"

    assert isinstance(taxonomy.names, MappingProxyType)
    assert isinstance(taxonomy.coarse_by_id, MappingProxyType)
    assert taxonomy.names[0] == "aircraft"
    assert taxonomy.coarse_name(0) == "aircraft"


def test_taxonomy_fields_are_frozen() -> None:
    taxonomy = get_taxonomy("xh25")

    with pytest.raises(FrozenInstanceError):
        taxonomy.key = "changed"


@pytest.mark.parametrize(
    ("names", "coarse_by_id"),
    [
        ({0: "aircraft"}, {1: "aircraft"}),
        ({1: "aircraft"}, {1: "aircraft"}),
        ({0: ""}, {0: "aircraft"}),
        ({0: "aircraft"}, {0: "unknown"}),
    ],
)
def test_taxonomy_rejects_invalid_mappings(
    names: dict[int, str],
    coarse_by_id: dict[int, str],
) -> None:
    with pytest.raises(ValueError):
        Taxonomy("invalid", names, coarse_by_id)
