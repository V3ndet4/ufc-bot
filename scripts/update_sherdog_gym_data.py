from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.sherdog import (
    load_fighter_gym_cache,
    refresh_fighter_gym_data,
    write_fighter_gym_csv,
    write_gym_registry_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh Sherdog-derived fighter gym and gym registry data.")
    parser.add_argument("--fighters-csv", required=True, help="CSV with a fighter_name column.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path for the current event fighter gym snapshot.",
    )
    parser.add_argument(
        "--event-registry-output",
        help="Optional output CSV path for the current gym registry snapshot.",
    )
    parser.add_argument(
        "--global-cache",
        default=str(ROOT / "data" / "sherdog_fighter_gyms.csv"),
        help="Persistent fighter gym cache path.",
    )
    parser.add_argument(
        "--global-registry",
        default=str(ROOT / "data" / "sherdog_gym_registry.csv"),
        help="Persistent gym registry path.",
    )
    parser.add_argument(
        "--refresh-days",
        type=int,
        default=7,
        help="Days before a fighter profile is refreshed again.",
    )
    parser.add_argument(
        "--association-refresh-days",
        type=int,
        default=30,
        help="Days before association roster fighters are refreshed again.",
    )
    parser.add_argument(
        "--max-association-pages",
        type=int,
        default=5,
        help="Max Sherdog Fight Finder pages to scan per association.",
    )
    parser.add_argument(
        "--no-association-expansion",
        action="store_true",
        help="Only refresh the requested fighters, without expanding gym rosters.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _load_fighter_names(path: str | Path) -> list[str]:
    fighters = pd.read_csv(path)
    if "fighter_name" not in fighters.columns:
        raise ValueError("fighters CSV must contain fighter_name")
    cleaned = fighters["fighter_name"].astype(str).str.strip()
    return [fighter_name for fighter_name in cleaned.drop_duplicates().tolist() if fighter_name]


def main() -> None:
    args = parse_args()
    fighter_names = _load_fighter_names(args.fighters_csv)
    cache = load_fighter_gym_cache(args.global_cache)
    current_fighters, updated_cache, gym_registry = refresh_fighter_gym_data(
        fighter_names,
        cache_frame=cache,
        refresh_days=args.refresh_days,
        association_refresh_days=args.association_refresh_days,
        expand_associations=not args.no_association_expansion,
        max_association_pages=args.max_association_pages,
    )

    output_path = write_fighter_gym_csv(current_fighters, args.output)
    cache_path = write_fighter_gym_csv(updated_cache, args.global_cache)
    registry_path = write_gym_registry_csv(gym_registry, args.global_registry)
    event_registry_path = None
    if args.event_registry_output:
        event_registry_path = write_gym_registry_csv(gym_registry, args.event_registry_output)

    if not args.quiet:
        covered_fighters = int(current_fighters["gym_name"].fillna("").astype(str).str.strip().ne("").sum()) if not current_fighters.empty and "gym_name" in current_fighters.columns else 0
        print(
            "Sherdog gym refresh: "
            f"{len(current_fighters)} current fighters | "
            f"{covered_fighters} with camps | "
            f"{len(updated_cache)} cached fighters | "
            f"{len(gym_registry)} gyms"
        )
        print(f"Saved event fighter gyms to {output_path}")
        if event_registry_path is not None:
            print(f"Saved event gym registry to {event_registry_path}")
        print(f"Saved global fighter gym cache to {cache_path}")
        print(f"Saved global gym registry to {registry_path}")


if __name__ == "__main__":
    main()
