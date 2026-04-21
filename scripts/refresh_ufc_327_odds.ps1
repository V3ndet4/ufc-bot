$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& "$PSScriptRoot\run_ufc_327_card.ps1" "--skip-stats" @args
