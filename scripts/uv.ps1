<#
.SYNOPSIS
Runs uv with non-editable project installs.

.DESCRIPTION
Hatchling's editable install writes an absolute UTF-8 path to a .pth file.
Python configured for Windows CP936 cannot read that file when this repository
is under its Chinese path. UV_NO_EDITABLE is uv's supported environment
equivalent to --no-editable and keeps both sync and run reproducible here.
Because the project is installed as a wheel, pass --reinstall-package xh-detect
after source changes so uv rebuilds the local package.

.EXAMPLE
.\scripts\uv.ps1 sync --extra dev

.EXAMPLE
.\scripts\uv.ps1 run --reinstall-package xh-detect pytest
#>

$ErrorActionPreference = "Stop"
$env:UV_NO_EDITABLE = "1"

& uv @args
exit $LASTEXITCODE
