from importlib import metadata, reload

from typer.testing import CliRunner

import xh_detect
from xh_detect.cli import app


def test_package_version_comes_from_distribution_metadata(monkeypatch) -> None:
    def fake_version(distribution_name: str) -> str:
        assert distribution_name == "xh-detect"
        return "9.8.7"

    with monkeypatch.context() as patch:
        patch.setattr(metadata, "version", fake_version)

        assert reload(xh_detect).__version__ == "9.8.7"

    reload(xh_detect)


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "xh-detect 0.1.0"
