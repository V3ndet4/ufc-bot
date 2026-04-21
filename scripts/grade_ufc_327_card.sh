#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/venvs/ufc-bot/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MANIFEST_FILE="${ROOT_DIR}/events/ufc_327_prochazka_ulberg.json"
SLUG="$("$PYTHON_BIN" -c "import json, pathlib; data=json.loads(pathlib.Path(r'$MANIFEST_FILE').read_text(encoding='utf-8')); print(data['slug'])")"
EVENT_ID="$("$PYTHON_BIN" -c "import json, pathlib; data=json.loads(pathlib.Path(r'$MANIFEST_FILE').read_text(encoding='utf-8')); print(data['event_id'])")"
CARD_DIR="cards/${SLUG}"

RESULTS_FILE="${RESULTS_FILE:-${CARD_DIR}/data/results.csv}"
DB_FILE="${DB_FILE:-data/ufc_betting.db}"
GRADED_FILE="${GRADED_FILE:-${CARD_DIR}/reports/graded_picks.csv}"
LEARNING_FILE="${LEARNING_FILE:-${CARD_DIR}/reports/learning_report.csv}"
LEARNING_SUMMARY_FILE="${LEARNING_SUMMARY_FILE:-${CARD_DIR}/reports/learning_summary.csv}"
FILTER_PERFORMANCE_FILE="${FILTER_PERFORMANCE_FILE:-${CARD_DIR}/reports/filter_performance.csv}"

"$PYTHON_BIN" scripts/grade_tracked_picks.py \
  --results "$RESULTS_FILE" \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$GRADED_FILE"

"$PYTHON_BIN" scripts/export_learning_report.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$LEARNING_FILE"

"$PYTHON_BIN" scripts/export_learning_summary.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$LEARNING_SUMMARY_FILE"

"$PYTHON_BIN" scripts/export_filter_performance.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$FILTER_PERFORMANCE_FILE"

echo "Saved graded picks to $GRADED_FILE"
echo "Saved learning report to $LEARNING_FILE"
echo "Saved learning summary to $LEARNING_SUMMARY_FILE"
echo "Saved filter performance report to $FILTER_PERFORMANCE_FILE"
