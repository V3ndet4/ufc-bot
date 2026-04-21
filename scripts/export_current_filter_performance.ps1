$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()
$Manifest = Get-Content $ManifestPath -Raw | ConvertFrom-Json
$EventId = $Manifest.event_id
$Slug = $Manifest.slug
$OutputFile = if ($env:FILTER_PERFORMANCE_FILE) { $env:FILTER_PERFORMANCE_FILE } else { Join-Path "cards\$Slug" "reports\filter_performance.csv" }

Invoke-PythonChecked -Arguments (@(
    "scripts\export_filter_performance.py",
    "--db", "data\ufc_betting.db",
    "--event-id", $EventId,
    "--output", $OutputFile
) + $args)
