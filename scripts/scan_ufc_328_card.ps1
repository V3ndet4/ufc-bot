$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& "$PSScriptRoot\run_ufc_328_card.ps1" "--skip-stats" "--skip-odds" @args
