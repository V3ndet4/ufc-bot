$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "UFC 327 files are static in-repo."
Write-Host "Context: cards\ufc_327_prochazka_ulberg\inputs\fighter_context.csv"
Write-Host "Map: cards\ufc_327_prochazka_ulberg\inputs\fighter_map.csv"
Write-Host "Template: cards\ufc_327_prochazka_ulberg\data\odds_template.csv"
