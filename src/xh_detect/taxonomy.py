from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

COARSE_NAMES = frozenset({"aircraft", "ship", "vehicle"})


@dataclass(frozen=True)
class Taxonomy:
    key: str
    names: Mapping[int, str]
    coarse_by_id: Mapping[int, str]

    def __post_init__(self) -> None:
        names = dict(self.names)
        coarse_by_id = dict(self.coarse_by_id)
        name_ids = set(names)
        coarse_ids = set(coarse_by_id)

        if name_ids != coarse_ids:
            raise ValueError("names and coarse IDs must match")
        if name_ids != set(range(len(names))):
            raise ValueError("taxonomy IDs must be contiguous from 0")
        if any(not isinstance(name, str) or not name.strip() for name in names.values()):
            raise ValueError("taxonomy names must be non-empty")
        if not set(coarse_by_id.values()) <= COARSE_NAMES:
            raise ValueError("coarse names must be aircraft, ship, or vehicle")

        object.__setattr__(self, "names", MappingProxyType(names))
        object.__setattr__(self, "coarse_by_id", MappingProxyType(coarse_by_id))

    @property
    def valid_ids(self) -> frozenset[int]:
        return frozenset(self.names)

    def coarse_name(self, class_id: int) -> str:
        try:
            return self.coarse_by_id[class_id]
        except KeyError:
            raise ValueError(f"unknown class ID: {class_id}") from None


XH25_NAMES = {
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

LEGACY3_NAMES = {0: "aircraft", 1: "ship", 2: "vehicle"}

TAXONOMIES = {
    "legacy3": Taxonomy("legacy3", LEGACY3_NAMES, LEGACY3_NAMES),
    "xh25": Taxonomy(
        "xh25",
        XH25_NAMES,
        {
            **dict.fromkeys(range(4), "ship"),
            **dict.fromkeys(range(4, 24), "aircraft"),
            24: "vehicle",
        },
    ),
}


def get_taxonomy(key: str) -> Taxonomy:
    try:
        return TAXONOMIES[key]
    except KeyError:
        raise ValueError(f"unknown taxonomy: {key}") from None
