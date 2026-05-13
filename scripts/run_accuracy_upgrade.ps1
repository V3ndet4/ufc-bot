$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

Invoke-PythonChecked -Arguments (@("scripts\run_accuracy_upgrade.py") + $args)
