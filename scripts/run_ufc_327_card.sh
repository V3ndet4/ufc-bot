#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [[ -x "${HOME}/venvs/ufc-bot/bin/python" ]]; then
    PYTHON_BIN="${HOME}/venvs/ufc-bot/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

ODDS_SOURCE="${ODDS_SOURCE:-oddsapi}"
ODDS_API_BOOKMAKER="${ODDS_API_BOOKMAKER:-fanduel}"

exec "$PYTHON_BIN" scripts/run_event_pipeline.py \
  --manifest events/ufc_327_prochazka_ulberg.json \
  --stats-source espn \
  --odds-source "$ODDS_SOURCE" \
  --odds-api-bookmaker "$ODDS_API_BOOKMAKER" \
  --quiet-children \
  "$@"
