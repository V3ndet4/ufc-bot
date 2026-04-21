from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.sherdog import merge_fighter_gym_data
from data_sources.ufc_stats import scrape_fighter_stats, scrape_fighter_stats_for_names, write_fighter_stats_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch fighter stats from UFC Stats and export them as CSV.")
    parser.add_argument(
        "--letters",
        nargs="*",
        default=None,
        help="Optional fighter surname initials to fetch, for example: a b c",
    )
    parser.add_argument(
        "--output",
        default="data/ufc_stats_fighter_stats.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fighters-csv",
        help="Optional CSV with a fighter_name column used to fetch only the listed fighters.",
    )
    parser.add_argument(
        "--context",
        help="Optional CSV with fighter_name plus manual context flags to merge into the output.",
    )
    parser.add_argument(
        "--fighter-gyms",
        help="Optional Sherdog-derived fighter gym CSV to merge into the output.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def merge_context(frame: pd.DataFrame, context_path: str | None) -> pd.DataFrame:
    if not context_path:
        return frame
    context = pd.read_csv(context_path)
    if "fighter_name" not in context.columns:
        raise ValueError("Context CSV must contain fighter_name")
    merged = frame.merge(context, on="fighter_name", how="left")
    for column in [
        "short_notice_flag",
        "short_notice_acceptance_flag",
        "short_notice_success_flag",
        "new_gym_flag",
        "new_contract_flag",
        "cardio_fade_flag",
        "injury_concern_flag",
        "weight_cut_concern_flag",
        "replacement_fighter_flag",
        "travel_disadvantage_flag",
        "camp_change_flag",
    ]:
        if column not in merged.columns:
            merged[column] = 0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0).astype(int)
    if "context_notes" not in merged.columns:
        merged["context_notes"] = ""
    else:
        merged["context_notes"] = merged["context_notes"].fillna("")
    return merged


def main() -> None:
    args = parse_args()
    if args.fighters_csv:
        fighters = pd.read_csv(args.fighters_csv)
        if "fighter_name" not in fighters.columns:
            raise ValueError("fighters CSV must contain fighter_name")
        frame = scrape_fighter_stats_for_names(fighters["fighter_name"].tolist())
    else:
        frame = scrape_fighter_stats(letters=args.letters)
    frame = merge_context(frame, args.context)
    frame = merge_fighter_gym_data(frame, args.fighter_gyms)
    output_path = write_fighter_stats_csv(frame, args.output)
    if not args.quiet:
        print(f"Fetched {len(frame)} fighters from UFC Stats")
        print(f"Saved fighter stats to {output_path}")


if __name__ == "__main__":
    main()
