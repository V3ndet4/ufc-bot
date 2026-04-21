#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-${HOME}/venvs/ufc-bot/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

MANIFEST_PATH="$(tr -d '\r\n' < events/current_event.txt)"
"$PYTHON_BIN" scripts/prepare_event.py --manifest "$MANIFEST_PATH"
