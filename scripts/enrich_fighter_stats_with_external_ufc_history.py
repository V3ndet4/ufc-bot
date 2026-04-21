from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.external_ufc_history import (
    HISTORY_NUMERIC_COLUMNS,
    PROFILE_NUMERIC_COLUMNS,
    PROFILE_STRING_COLUMNS,
    ROLLING_NUMERIC_COLUMNS,
    STYLE_PROFILE_NUMERIC_COLUMNS,
    STYLE_PROFILE_STRING_COLUMNS,
    TECHNIQUE_NUMERIC_COLUMNS,
    build_external_ufc_history_features,
    load_external_ufc_history_datasets,
    merge_external_ufc_history_into_fighter_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Best-effort enrichment of fighter_stats.csv using weekly UFC history from Greco1899/scrape_ufc_stats."
    )
    parser.add_argument("--input", required=True, help="Base fighter stats CSV.")
    parser.add_argument("--output", help="Output CSV path. Defaults to updating --input in place.")
    parser.add_argument(
        "--cache-dir",
        default=".tmp/external_ufc_history",
        help="Directory used to cache downloaded repository CSVs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output) if args.output else Path(args.input)

    fighter_stats = pd.read_csv(args.input)
    datasets = load_external_ufc_history_datasets(args.cache_dir)
    external_features = build_external_ufc_history_features(
        fight_results=datasets["fight_results"],
        fight_stats=datasets["fight_stats"],
        event_details=datasets["event_details"],
        fighter_tott=datasets["fighter_tott"],
    )
    enriched = merge_external_ufc_history_into_fighter_stats(fighter_stats, external_features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)

    if not args.quiet:
        base_columns = set(fighter_stats.columns)
        added_columns = [
            column
            for column in [
                *PROFILE_NUMERIC_COLUMNS,
                *PROFILE_STRING_COLUMNS,
                *HISTORY_NUMERIC_COLUMNS,
                *TECHNIQUE_NUMERIC_COLUMNS,
                *STYLE_PROFILE_NUMERIC_COLUMNS,
                *ROLLING_NUMERIC_COLUMNS,
                *STYLE_PROFILE_STRING_COLUMNS,
            ]
            if column in enriched.columns and column not in base_columns
        ]
        print(f"Enriched fighter stats with external UFC history for {len(enriched)} fighters")
        if added_columns:
            print(f"Added columns: {', '.join(added_columns)}")
        print(f"Saved enriched fighter stats to {output_path}")


if __name__ == "__main__":
    main()
