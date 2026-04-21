#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/venvs/ufc-bot/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

echo "UFC 327 files are static in-repo."
echo "Context: cards/ufc_327_prochazka_ulberg/inputs/fighter_context.csv"
echo "Map: cards/ufc_327_prochazka_ulberg/inputs/fighter_map.csv"
echo "Template: cards/ufc_327_prochazka_ulberg/data/odds_template.csv"
