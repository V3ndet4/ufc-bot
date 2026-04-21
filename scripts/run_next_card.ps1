$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
. "$PSScriptRoot\_resolve_python.ps1"

$ManifestPath = (Get-Content "events\current_event.txt" -Raw).Trim()
$OddsSource = if ($env:ODDS_SOURCE) { $env:ODDS_SOURCE } else { "oddsapi" }
$OddsApiBookmaker = if ($env:ODDS_API_BOOKMAKER) { $env:ODDS_API_BOOKMAKER } else { "fanduel" }

Invoke-PythonChecked -Arguments (@(
    "scripts\run_event_pipeline.py",
    "--manifest", $ManifestPath,
    "--odds-source", $OddsSource,
    "--odds-api-bookmaker", $OddsApiBookmaker,
    "--quiet-children"
) + $args)
