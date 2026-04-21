from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import load_odds_csv
from data_sources.storage import save_odds_snapshot
from normalization.odds import normalize_odds_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Store an odds snapshot in SQLite.")
    parser.add_argument("--input", required=True, help="Path to the odds CSV input.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_odds_csv(args.input)
    normalized = normalize_odds_frame(raw)
    inserted = save_odds_snapshot(normalized, args.db)
    print(f"Saved {inserted} odds rows to {args.db}")


if __name__ == "__main__":
    main()
