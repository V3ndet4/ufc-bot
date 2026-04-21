$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$OutputFile = if ($env:FILTER_PERFORMANCE_FILE) { $env:FILTER_PERFORMANCE_FILE } else { "reports\filter_performance.csv" }

Invoke-PythonChecked -Arguments (@(
    "scripts\export_filter_performance.py",
    "--db", "data\ufc_betting.db",
    "--output", $OutputFile
) + $args)
