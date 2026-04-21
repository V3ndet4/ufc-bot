from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import merge_espn_url_maps
from scripts.event_manifest import (
    build_context_frame,
    build_fighter_map_frame,
    build_fighter_list_frame,
    build_modeled_market_template_frame,
    build_odds_template_frame,
    derived_paths,
    load_manifest,
    merge_existing_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fighter list, context, and odds template files from an event manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _write_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)

    fighter_list = build_fighter_list_frame(manifest)
    fighter_map_template = build_fighter_map_frame(manifest)
    context_template = build_context_frame(manifest)
    existing_context = None
    fighter_map_sources: list[pd.DataFrame] = [fighter_map_template]
    global_espn_cache = ROOT / "data" / "espn_fighter_map.csv"
    if global_espn_cache.exists():
        fighter_map_sources.append(pd.read_csv(global_espn_cache))
    if paths["fighter_map"].exists():
        fighter_map_sources.append(pd.read_csv(paths["fighter_map"]))
    if paths["context"].exists():
        existing_context = pd.read_csv(paths["context"])
    merged_fighter_map = merge_espn_url_maps(*fighter_map_sources)
    merged_context = merge_existing_context(context_template, existing_context)

    _write_csv(fighter_list, paths["fighter_list"])
    _write_csv(merged_fighter_map, paths["fighter_map"])
    _write_csv(merged_context, paths["context"])
    _write_csv(build_odds_template_frame(manifest), paths["odds_template"])
    _write_csv(build_modeled_market_template_frame(manifest), paths["modeled_market_template"])

    if not args.quiet:
        print(f"Saved fighter list to {paths['fighter_list']}")
        print(f"Saved fighter map template to {paths['fighter_map']}")
        print(f"Saved context template to {paths['context']}")
        print(f"Saved odds template to {paths['odds_template']}")
        print(f"Saved modeled-market template to {paths['modeled_market_template']}")


if __name__ == "__main__":
    main()
