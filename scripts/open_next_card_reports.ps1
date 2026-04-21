$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ReportsDir = Invoke-PythonChecked -Arguments @("-c", "from pathlib import Path; from scripts.event_manifest import current_event_manifest, derived_paths; root = Path(r'$RootDir'); manifest = current_event_manifest(root); print(derived_paths(manifest)['reports_dir'])")

if (-not (Test-Path $ReportsDir)) {
    throw "Reports folder not found: $ReportsDir"
}

Write-Host "Opening reports folder: $ReportsDir"
Start-Process explorer.exe $ReportsDir
