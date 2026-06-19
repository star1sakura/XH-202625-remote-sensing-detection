from typer.testing import CliRunner

from xh_detect.cli import app


def test_version_command() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == "xh-detect 0.1.0"
