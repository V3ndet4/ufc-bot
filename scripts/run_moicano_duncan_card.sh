#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

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

CARD_DIR="cards/moicano_duncan_card"
MAPPING_FILE="${CARD_DIR}/inputs/fighter_map.csv"
CONTEXT_FILE="${CARD_DIR}/inputs/fighter_context.csv"
STATS_FILE="${CARD_DIR}/data/fighter_stats.csv"
ODDS_TEMPLATE="${CARD_DIR}/data/odds_template.csv"
ODDS_SOURCE="${ODDS_SOURCE:-fanduel}"
ODDS_API_BOOKMAKER="${ODDS_API_BOOKMAKER:-fanduel}"
ODDS_FILE="${CARD_DIR}/data/${ODDS_SOURCE}_odds.csv"
PROBS_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_projected_probs.csv"
REPORT_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_fight_week_report.csv"
VALUE_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_value_bets.csv"

echo "Using Python: $PYTHON_BIN"
echo "Odds source: $ODDS_SOURCE"

"$PYTHON_BIN" scripts/fetch_espn_stats.py \
  --mapping "$MAPPING_FILE" \
  --context "$CONTEXT_FILE" \
  --output "$STATS_FILE"

if [[ "$ODDS_SOURCE" != "fanduel" ]]; then
  echo "This runner currently supports ODDS_SOURCE=fanduel only." >&2
  exit 1
fi

"$PYTHON_BIN" scripts/fetch_the_odds_api_odds.py \
  --template "$ODDS_TEMPLATE" \
  --bookmaker "$ODDS_API_BOOKMAKER" \
  --output "$ODDS_FILE"

ODDS_ROW_COUNT=$("$PYTHON_BIN" - <<PY
import pandas as pd
from pathlib import Path

path = Path("$ODDS_FILE")
if not path.exists():
    print(0)
else:
    try:
        frame = pd.read_csv(path)
        print(len(frame))
    except Exception:
        print(0)
PY
)

if [[ "${ODDS_ROW_COUNT:-0}" -eq 0 ]]; then
  echo "No live odds available for this event yet; skipping projections, report, and value scan."
  exit 0
fi

"$PYTHON_BIN" scripts/project_fight_probs.py \
  --odds "$ODDS_FILE" \
  --fighter-stats "$STATS_FILE" \
  --output "$PROBS_FILE"

"$PYTHON_BIN" scripts/build_fight_week_report.py \
  --odds "$ODDS_FILE" \
  --fighter-stats "$STATS_FILE" \
  --output "$REPORT_FILE"

MIN_MODEL_CONFIDENCE="${MIN_MODEL_CONFIDENCE:-0.55}" \
MIN_STATS_COMPLETENESS="${MIN_STATS_COMPLETENESS:-0.80}" \
EXCLUDE_FALLBACK_ROWS="${EXCLUDE_FALLBACK_ROWS:-true}" \
"$PYTHON_BIN" scripts/run_value_scan.py \
  --input "$ODDS_FILE" \
  --fighter-stats "$STATS_FILE" \
  --fight-report "$REPORT_FILE" \
  --output "$VALUE_FILE"

echo
echo "Saved projected probabilities to $PROBS_FILE"
echo "Saved fight-week report to $REPORT_FILE"
echo "Saved value scan to $VALUE_FILE"
echo "Legacy runner note: prefer manifest-driven scripts for new cards."
