import tomllib
from pathlib import Path


def test_hatch_editable_installs_use_dev_mode_exact() -> None:
    pyproject = tomllib.loads(
        Path(__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject["tool"]["hatch"]["build"]["dev-mode-exact"] is True
