#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Refreshing active-card FanDuel odds, saving a snapshot, and rebuilding line-movement charts..."
"$ROOT_DIR/scripts/run_next_card.sh" --skip-stats "$@"
