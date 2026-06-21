from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="requires a POSIX bash environment",
)
def test_bootstrap_reuses_existing_virtual_environment() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "bootstrap_gpu_server.sh"
    with tempfile.TemporaryDirectory(prefix=".bootstrap-test-", dir=project_root) as temp_dir:
        temp_root = Path(temp_dir)
        relative_root = temp_root.relative_to(project_root).as_posix()
        venv_dir = temp_root / "existing-venv"
        call_log = temp_root / "uv-calls.txt"
        bash_env = temp_root / "bash-env.sh"
        venv_bin = venv_dir / "bin"
        venv_bin.mkdir(parents=True)

        fake_python = venv_bin / "python"
        fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
        fake_python.chmod(0o755)
        bash_env.write_text(
            'uv() {\n  printf "%s\\n" "$*" >> "$UV_CALL_LOG"\n'
            '  if [[ "$1" == "venv" ]]; then return 42; fi\n  return 0\n}\n',
            encoding="utf-8",
            newline="\n",
        )

        env = os.environ.copy()
        env.update(
            {
                "BASH_ENV": f"{relative_root}/bash-env.sh",
                "PYTHON_BIN": f"{relative_root}/existing-venv/bin/python",
                "VENV_DIR": f"{relative_root}/existing-venv",
                "UV_CALL_LOG": f"{relative_root}/uv-calls.txt",
            }
        )

        result = subprocess.run(
            ["bash", script.relative_to(project_root).as_posix()],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert not any(line.startswith("venv ") for line in call_log.read_text().splitlines())
