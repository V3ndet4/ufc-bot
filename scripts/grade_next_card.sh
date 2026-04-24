#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/venvs/ufc-bot/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MANIFEST_PATH="$(tr -d '\r\n' < events/current_event.txt)"

EVENT_ID="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$MANIFEST_PATH").read_text(encoding="utf-8"))
print(manifest["event_id"])
PY
)"

SLUG="$("$PYTHON_BIN" - <<PY
import json
from pathlib import Path

manifest = json.loads(Path("$MANIFEST_PATH").read_text(encoding="utf-8"))
print(manifest["slug"])
PY
)"

readarray -t PATH_ROWS < <("$PYTHON_BIN" - <<PY
import sys
from pathlib import Path

ROOT = Path("$ROOT_DIR")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.event_manifest import derived_paths, load_manifest

manifest = load_manifest(ROOT / Path("$MANIFEST_PATH"))
paths = derived_paths(manifest)
print(f"RESULTS={paths['results']}")
print(f"GRADED={paths['graded']}")
print(f"LEARNING={paths['learning']}")
print(f"LEARNING_SUMMARY={paths['learning_summary']}")
print(f"LEARNING_POSTMORTEM={paths['learning_postmortem']}")
print(f"LEARNING_POSTMORTEM_SUMMARY={paths['learning_postmortem_summary']}")
print(f"FILTER_PERFORMANCE={paths['filter_performance']}")
PY
)

declare -A PATH_MAP
for row in "${PATH_ROWS[@]}"; do
  key="${row%%=*}"
  value="${row#*=}"
  PATH_MAP["$key"]="$value"
done

RESULTS_FILE="${RESULTS_FILE:-${PATH_MAP[RESULTS]}}"
DB_FILE="${DB_FILE:-data/ufc_betting.db}"
GRADED_FILE="${GRADED_FILE:-${PATH_MAP[GRADED]}}"
LEARNING_FILE="${LEARNING_FILE:-${PATH_MAP[LEARNING]}}"
LEARNING_SUMMARY_FILE="${LEARNING_SUMMARY_FILE:-${PATH_MAP[LEARNING_SUMMARY]}}"
LEARNING_POSTMORTEM_FILE="${LEARNING_POSTMORTEM_FILE:-${PATH_MAP[LEARNING_POSTMORTEM]}}"
LEARNING_POSTMORTEM_SUMMARY_FILE="${LEARNING_POSTMORTEM_SUMMARY_FILE:-${PATH_MAP[LEARNING_POSTMORTEM_SUMMARY]}}"
FILTER_PERFORMANCE_FILE="${FILTER_PERFORMANCE_FILE:-${PATH_MAP[FILTER_PERFORMANCE]}}"
SELECTIVE_MODEL_FILE="${SELECTIVE_MODEL_FILE:-models/selective_clv_model.pkl}"
AUTO_FETCH_RESULTS="${AUTO_FETCH_RESULTS:-1}"
REFRESH_RESULTS_FILE="${REFRESH_RESULTS_FILE:-0}"

if [[ "$AUTO_FETCH_RESULTS" != "0" && ( ! -f "$RESULTS_FILE" || "$REFRESH_RESULTS_FILE" == "1" ) ]]; then
  if "$PYTHON_BIN" scripts/fetch_event_results.py \
    --manifest "$MANIFEST_PATH" \
    --output "$RESULTS_FILE" \
    --db "$DB_FILE"; then
    echo "Saved auto-fetched results to $RESULTS_FILE"
  elif [[ ! -f "$RESULTS_FILE" ]]; then
    exit 1
  else
    echo "Auto results fetch failed; reusing existing results file: $RESULTS_FILE"
  fi
fi

"$PYTHON_BIN" scripts/grade_tracked_picks.py \
  --results "$RESULTS_FILE" \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$GRADED_FILE"

"$PYTHON_BIN" scripts/export_learning_report.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$LEARNING_FILE" \
  --postmortem-output "$LEARNING_POSTMORTEM_FILE" \
  --postmortem-summary-output "$LEARNING_POSTMORTEM_SUMMARY_FILE"

"$PYTHON_BIN" scripts/export_learning_summary.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$LEARNING_SUMMARY_FILE"

"$PYTHON_BIN" scripts/export_filter_performance.py \
  --db "$DB_FILE" \
  --event-id "$EVENT_ID" \
  --output "$FILTER_PERFORMANCE_FILE"

if "$PYTHON_BIN" scripts/train_selective_clv_model.py \
  --db "$DB_FILE" \
  --output "$SELECTIVE_MODEL_FILE" \
  --quiet; then
  echo "Refreshed selective CLV model to $SELECTIVE_MODEL_FILE"
else
  echo "Selective CLV model refresh skipped"
fi

echo "Saved graded picks to $GRADED_FILE"
echo "Saved learning report to $LEARNING_FILE"
echo "Saved learning postmortem to $LEARNING_POSTMORTEM_FILE"
echo "Saved learning postmortem summary to $LEARNING_POSTMORTEM_SUMMARY_FILE"
echo "Saved learning summary to $LEARNING_SUMMARY_FILE"
echo "Saved filter performance report to $FILTER_PERFORMANCE_FILE"
