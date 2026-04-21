$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

Write-Host "Refreshing active-card FanDuel odds, saving a snapshot, and rebuilding line-movement charts..."
& "$PSScriptRoot\run_next_card.ps1" "--skip-stats" @args
