<#
.SYNOPSIS
Runs uv with the repository's default settings.

.DESCRIPTION
Hatchling's exact dev mode keeps editable installs ASCII-safe on Windows paths,
so plain uv sync and uv run now work here without any wrapper-specific
environment variables.

.EXAMPLE
.\scripts\uv.ps1 sync --extra dev

.EXAMPLE
.\scripts\uv.ps1 run pytest
#>

$ErrorActionPreference = "Stop"

& uv @args
exit $LASTEXITCODE
