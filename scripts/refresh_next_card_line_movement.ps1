$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& ".\.venv-win\Scripts\python.exe" "scripts\build_line_movement_report.py" `
  --odds "cards\upcoming_card\data\oddsapi_odds.csv" `
  --db "data\ufc_betting.db" `
  --bookmaker "fanduel" `
  --output "cards\upcoming_card\reports\line_movement.svg" `
  --per-fight-dir "cards\upcoming_card\reports\line_movement_fights"
