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

CARD_DIR="cards/seattle_card"
MAPPING_FILE="${CARD_DIR}/inputs/fighter_map.csv"
STATS_FILE="${CARD_DIR}/data/fighter_stats.csv"
ODDS_TEMPLATE="${CARD_DIR}/data/odds_template.csv"
ODDS_SOURCE="${ODDS_SOURCE:-bfo}"
ODDS_API_BOOKMAKER="${ODDS_API_BOOKMAKER:-fanduel}"
ODDS_FILE="${CARD_DIR}/data/${ODDS_SOURCE}_odds.csv"
PROBS_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_projected_probs.csv"
REPORT_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_fight_week_report.csv"
VALUE_FILE="${CARD_DIR}/reports/${ODDS_SOURCE}_value_bets.csv"

REFRESH_STATS=1
REFRESH_ODDS=1

show_help() {
  cat <<'EOF'
Usage: ./scripts/run_seattle_card.sh [options]

Runs the UFC Seattle pipeline:
1. ESPN fighter stats refresh
2. Odds refresh from BestFightOdds or The Odds API
3. Projection build
4. Fight-week report build
5. Filtered value scan

Options:
  --skip-stats       Reuse existing fighter stats CSV
  --skip-odds        Reuse existing odds CSV
  --help             Show this help text

Environment overrides:
  PYTHON_BIN                 Python executable to use
  ODDS_SOURCE                bfo or fanduel
  ODDS_API_BOOKMAKER         Default: fanduel
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
echo "Odds source: $ODDS_SOURCE"

if [[ "$REFRESH_STATS" -eq 1 ]]; then
  "$PYTHON_BIN" scripts/fetch_espn_stats.py \
    --mapping "$MAPPING_FILE" \
    --output "$STATS_FILE"
fi

if [[ "$REFRESH_ODDS" -eq 1 ]]; then
  if [[ "$ODDS_SOURCE" == "fanduel" ]]; then
    "$PYTHON_BIN" scripts/fetch_the_odds_api_odds.py \
      --template "$ODDS_TEMPLATE" \
      --bookmaker "$ODDS_API_BOOKMAKER" \
      --output "$ODDS_FILE"
  elif [[ "$ODDS_SOURCE" == "bfo" ]]; then
    "$PYTHON_BIN" - <<'PY'
import pandas as pd

from data_sources.bestfightodds import (
    enrich_odds_template_from_bestfightodds,
    enrich_with_bestfightodds_history,
    fetch_html,
    write_odds_csv,
)

template = pd.read_csv("cards/seattle_card/data/odds_template.csv")

lookup_name_overrides = {
    "Casey O'Neill": "Casey Oneill",
    "Lance Gibson Jr.": "Lance Gibson Jr",
}

page_4018_fighters = {
    "Israel Adesanya",
    "Joe Pyfer",
    "Alexa Grasso",
    "Maycee Barber",
    "Michael Chiesa",
    "Niko Price",
    "Julian Erosa",
    "Lerryan Douglas",
    "Mansur Abdul-Malik",
    "Yousri Belgaroui",
    "Kyle Nelson",
    "Terrance McKinney",
}

page_4095_fighters = {
    "Ignacio Bahamondes",
    "Tofiq Musayev",
    "Chase Hooper",
    "Lance Gibson Jr.",
    "Marcin Tybura",
    "Tyrell Fortune",
    "Casey O'Neill",
    "Gabriella Fernandes",
    "Bruno Lopes",
    "Navajo Stirling",
    "Adrian Yanez",
    "Ricky Simon",
    "Alexia Thainara",
    "Bruna Brasil",
}


def prep(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["selection_name"] = prepared.apply(
        lambda row: row["fighter_a"] if row["selection"] == "fighter_a" else row["fighter_b"],
        axis=1,
    )
    prepared["selection_name"] = prepared["selection_name"].replace(lookup_name_overrides)
    return prepared


mask_4018 = template["fighter_a"].isin(page_4018_fighters) & template["fighter_b"].isin(page_4018_fighters)
mask_4095 = template["fighter_a"].isin(page_4095_fighters) & template["fighter_b"].isin(page_4095_fighters)

frames = []
for url, frame in [
    ("https://www.bestfightodds.com/events/ufc-seattle-4018", prep(template.loc[mask_4018])),
    ("https://www.bestfightodds.com/events/ufc-seattle-4095", prep(template.loc[mask_4095])),
]:
    html = fetch_html(url)
    enriched = enrich_odds_template_from_bestfightodds(frame, html, book_preference="consensus")
    enriched = enrich_with_bestfightodds_history(enriched, html)
    frames.append(enriched)

out = pd.concat(frames, ignore_index=True)
out["selection_name"] = out.apply(
    lambda row: row["fighter_a"] if row["selection"] == "fighter_a" else row["fighter_b"],
    axis=1,
)
write_odds_csv(out, "cards/seattle_card/data/bfo_odds.csv")
print(f"Saved {len(out)} rows to cards/seattle_card/data/bfo_odds.csv")
PY
  else
    echo "Unsupported ODDS_SOURCE: $ODDS_SOURCE" >&2
    exit 1
  fi
fi

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
  echo "No live odds available for this completed event; skipping projections, report, and value scan."
  exit 0
fi

"$PYTHON_BIN" scripts/project_fight_probs.py \
  --odds "$ODDS_FILE" \
  --fighter-stats "$STATS_FILE" \
  --output "$PROBS_FILE"

"$PYTHON_BIN" scripts/build_fight_week_report.py \
  --odds "$ODDS_FILE" \
  --fighter-stats "$STATS_FILE" \
  --bestfightodds-event-url "https://www.bestfightodds.com/events/ufc-seattle-4018" \
  --bestfightodds-event-url "https://www.bestfightodds.com/events/ufc-seattle-4095" \
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
echo "Legacy runner note: prefer manifest-driven scripts for active cards."
