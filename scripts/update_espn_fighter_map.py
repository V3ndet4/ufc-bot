from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import merge_espn_url_maps, resolve_espn_fighter_url
from data_sources.fighter_aliases import (
    build_fighter_alias_lookup,
    load_fighter_alias_overrides,
    resolve_fighter_alias,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-fill missing ESPN fighter profile URLs.")
    parser.add_argument(
        "--mapping",
        required=True,
        help="Event fighter map CSV with fighter_name and espn_url columns.",
    )
    parser.add_argument(
        "--fighters-csv",
        help="Optional fighter list CSV used to create a mapping when one does not exist yet.",
    )
    parser.add_argument(
        "--global-cache",
        help="Optional cache CSV storing fighter_name and espn_url across events.",
    )
    parser.add_argument(
        "--alias-overrides",
        default=str(ROOT / "data" / "fighter_alias_overrides.csv"),
        help="Optional CSV mapping alternate fighter names to canonical names.",
    )
    parser.add_argument(
        "--output",
        help="Optional output CSV path. Defaults to --mapping.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _load_map_frame(mapping_path: Path, fighters_csv: Path | None) -> pd.DataFrame:
    if mapping_path.exists():
        frame = pd.read_csv(mapping_path)
    elif fighters_csv and fighters_csv.exists():
        fighter_list = pd.read_csv(fighters_csv)
        if "fighter_name" not in fighter_list.columns:
            raise ValueError("fighters CSV must contain fighter_name")
        frame = pd.DataFrame({"fighter_name": fighter_list["fighter_name"], "espn_url": ""})
    else:
        raise FileNotFoundError(f"Could not locate mapping file: {mapping_path}")

    if "fighter_name" not in frame.columns:
        raise ValueError("mapping CSV must contain fighter_name")
    if "espn_url" not in frame.columns:
        frame["espn_url"] = ""
    frame["fighter_name"] = frame["fighter_name"].fillna("").astype(str).str.strip()
    frame["espn_url"] = frame["espn_url"].fillna("").astype(str).str.strip()
    frame = frame.loc[frame["fighter_name"] != "", ["fighter_name", "espn_url"]].copy()
    return frame


def _load_cache_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    frame = pd.read_csv(path)
    if "fighter_name" not in frame.columns:
        return None
    if "espn_url" not in frame.columns:
        frame["espn_url"] = ""
    frame["fighter_name"] = frame["fighter_name"].fillna("").astype(str).str.strip()
    frame["espn_url"] = frame["espn_url"].fillna("").astype(str).str.strip()
    return frame.loc[frame["fighter_name"] != "", ["fighter_name", "espn_url"]].copy()


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _resolve_missing_urls(
    fighter_names: list[str],
    alias_lookup: dict[str, str],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for fighter_name in fighter_names:
        search_names = [fighter_name]
        canonical_name = resolve_fighter_alias(fighter_name, alias_lookup)
        if canonical_name and canonical_name.lower() != fighter_name.lower():
            search_names.append(canonical_name)

        for search_name in search_names:
            url = resolve_espn_fighter_url(search_name)
            if url:
                resolved[fighter_name] = url
                break
    return resolved


def _filter_to_event_fighters(merged: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    event_fighters = mapping.loc[:, ["fighter_name"]].copy()
    event_fighters["fighter_name"] = event_fighters["fighter_name"].fillna("").astype(str).str.strip()
    event_fighters = event_fighters.loc[event_fighters["fighter_name"] != ""].drop_duplicates()

    filtered = event_fighters.merge(merged, on="fighter_name", how="left")
    if "espn_url" not in filtered.columns:
        filtered["espn_url"] = ""
    filtered["espn_url"] = filtered["espn_url"].fillna("").astype(str).str.strip()
    return filtered.loc[:, ["fighter_name", "espn_url"]].copy()


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping)
    fighters_csv = Path(args.fighters_csv) if args.fighters_csv else None
    cache_path = Path(args.global_cache) if args.global_cache else None
    output_path = Path(args.output) if args.output else mapping_path

    mapping = _load_map_frame(mapping_path, fighters_csv)
    cache = _load_cache_frame(cache_path)
    alias_overrides = load_fighter_alias_overrides(args.alias_overrides)
    alias_lookup = build_fighter_alias_lookup(alias_overrides)
    merged = merge_espn_url_maps(mapping, cache, alias_lookup=alias_lookup)
    merged = _filter_to_event_fighters(merged, mapping)

    missing_names = merged.loc[merged["espn_url"].eq(""), "fighter_name"].tolist()
    resolved_urls = _resolve_missing_urls(missing_names, alias_lookup) if missing_names else {}
    if resolved_urls:
        resolved_frame = pd.DataFrame(
            [{"fighter_name": fighter_name, "espn_url": espn_url} for fighter_name, espn_url in resolved_urls.items()]
        )
        merged = merge_espn_url_maps(merged, resolved_frame, alias_lookup=alias_lookup)
        merged = _filter_to_event_fighters(merged, mapping)

    output_path = _write_csv(merged, output_path)
    unresolved_names = merged.loc[merged["espn_url"].eq(""), "fighter_name"].tolist()

    if cache_path:
        cache_frame = merge_espn_url_maps(cache, merged, alias_lookup=alias_lookup)
        cache_output = _write_csv(cache_frame, cache_path)
    else:
        cache_output = None

    if not args.quiet:
        print(
            "ESPN fighter map refresh: "
            f"{len(merged) - len(unresolved_names)} resolved | {len(unresolved_names)} unresolved"
        )
        print(f"Saved fighter map to {output_path}")
        if cache_output is not None:
            print(f"Saved global ESPN cache to {cache_output}")
        if unresolved_names:
            print("Unresolved fighters: " + ", ".join(unresolved_names))


if __name__ == "__main__":
    main()
