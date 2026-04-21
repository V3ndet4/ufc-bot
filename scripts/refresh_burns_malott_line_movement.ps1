$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

& ".\.venv-win\Scripts\python.exe" "scripts\build_line_movement_report.py" `
  --odds "cards\ufc_fn_burns_malott\data\oddsapi_odds.csv" `
  --db "data\ufc_betting.db" `
  --bookmaker "fanduel" `
  --output "cards\ufc_fn_burns_malott\reports\line_movement.svg" `
  --per-fight-dir "cards\ufc_fn_burns_malott\reports\line_movement_fights"
