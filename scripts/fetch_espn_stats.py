from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import (
    load_espn_fighter_map,
    merge_context_into_fighter_map,
    scrape_fighter_stats_from_map,
    write_fighter_stats_csv,
)
from data_sources.sherdog import merge_fighter_gym_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch fighter stats from ESPN pages listed in a mapping CSV.")
    parser.add_argument(
        "--mapping",
        required=True,
        help="CSV with fighter_name and espn_url columns.",
    )
    parser.add_argument(
        "--output",
        default="data/espn_fighter_stats.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--context",
        help="Optional CSV with fighter_name plus manual context flags and context_notes.",
    )
    parser.add_argument(
        "--fighter-gyms",
        help="Optional Sherdog-derived fighter gym CSV to merge into the output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_espn_fighter_map(args.mapping)
    mapping = merge_context_into_fighter_map(mapping, args.context)
    frame = scrape_fighter_stats_from_map(mapping)
    frame = merge_fighter_gym_data(frame, args.fighter_gyms)
    output_path = write_fighter_stats_csv(frame, args.output)
    if not args.quiet:
        print(f"Fetched {len(frame)} fighters from ESPN")
        print(f"Saved fighter stats to {output_path}")


if __name__ == "__main__":
    main()
