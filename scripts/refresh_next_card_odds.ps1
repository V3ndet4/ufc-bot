$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& "$PSScriptRoot\run_next_card.ps1" "--skip-stats" @args
