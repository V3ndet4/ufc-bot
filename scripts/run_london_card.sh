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

CARD_DIR="cards/upcoming_card"
CONTEXT_FILE="${CARD_DIR}/inputs/fighter_context.csv"
MAPPING_FILE="${CARD_DIR}/inputs/fighter_map.csv"
STATS_FILE="${CARD_DIR}/data/fighter_stats.csv"
ODDS_TEMPLATE="${CARD_DIR}/data/odds_template.csv"
ODDS_FILE="${CARD_DIR}/data/bfo_odds.csv"
REPORT_FILE="${CARD_DIR}/reports/fight_week_report.csv"
VALUE_FILE="${CARD_DIR}/reports/bfo_value_bets.csv"
EVENT_URL="https://www.bestfightodds.com/events/ufc-london-4081"

REFRESH_STATS=1
REFRESH_ODDS=1
INCLUDE_HISTORY=1

show_help() {
  cat <<'EOF'
Usage: ./scripts/run_london_card.sh [options]

Runs the UFC London pipeline:
1. ESPN fighter stats refresh
2. BestFightOdds odds refresh
3. Fight-week report build
4. Filtered value scan

Options:
  --skip-stats       Reuse existing fighter stats CSV
  --skip-odds        Reuse existing odds CSV
  --no-history       Skip BestFightOdds open/range enrichment
  --help             Show this help text

Environment overrides:
  PYTHON_BIN                 Python executable to use
  MIN_MODEL_CONFIDENCE       Default: 0.55
  MIN_STATS_COMPLETENESS     Default: 0.80
  EXCLUDE_FALLBACK_ROWS      Default: true
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stats)
      REFRESH_STATS=0
      shift
      ;;
    --skip-odds)
      REFRESH_ODDS=0
      shift
      ;;
    --no-history)
      INCLUDE_HISTORY=0
      shift
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      show_help >&2
      exit 1
      ;;
  esac
done

echo "Using Python: $PYTHON_BIN"

if [[ "$REFRESH_STATS" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/fetch_espn_stats.py \
    --mapping "$MAPPING_FILE" \
    --context "$CONTEXT_FILE" \
    --output "$STATS_FILE"
fi

if [[ "$REFRESH_ODDS" -eq 1 ]]; then
  odds_cmd=(
    "$PYTHON_BIN" scripts/fetch_bestfightodds_event_odds.py
    --template "$ODDS_TEMPLATE"
    --event-url "$EVENT_URL"
    --output "$ODDS_FILE"
  )
  if [[ "$INCLUDE_HISTORY" -eq 1 ]]; then
    odds_cmd+=(--include-history)
  fi
  "${odds_cmd[@]}"
fi

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
  --output "$VALUE_FILE"

echo
echo "Saved fight-week report to $REPORT_FILE"
echo "Saved value scan to $VALUE_FILE"
echo "Legacy runner note: prefer ./scripts/run_next_card.sh for the active manifest workflow."
