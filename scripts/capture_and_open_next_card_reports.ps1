$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& "$PSScriptRoot\capture_next_card_odds.ps1" @args
& "$PSScriptRoot\open_next_card_reports.ps1"
