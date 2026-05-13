$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()

Invoke-PythonChecked -Arguments (@(
    "scripts\run_event_pipeline.py",
    "--manifest", $ManifestPath,
    "--skip-odds",
    "--quiet-children"
) + $args)

Write-Host ""
Invoke-PythonChecked -Arguments @(
    "scripts\print_card_preview.py",
    "--manifest", $ManifestPath
)
