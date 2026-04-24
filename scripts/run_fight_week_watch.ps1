$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

Write-Host "Refreshing fight-week gym and news watch..."
Invoke-PythonChecked -Arguments (@(
    "scripts\run_fight_week_watch.py"
) + $args)
