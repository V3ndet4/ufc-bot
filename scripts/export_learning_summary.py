from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks
from scripts.export_learning_report import build_learning_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an event-level learning summary from tracked picks.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path.")
    parser.add_argument("--event-id", help="Optional event id filter.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracked = load_tracked_picks(args.db, event_id=args.event_id)
    summary = build_learning_summary(tracked)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    if not args.quiet:
        print(f"Saved learning summary to {output_path}")


if __name__ == "__main__":
    main()
