from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_sources.odds_api import (
    DEFAULT_BOOKMAKER,
    DEFAULT_MARKETS,
    DEFAULT_REGIONS,
    DEFAULT_SPORT,
    enrich_odds_template_from_the_odds_api,
    fetch_modeled_market_snapshots,
    fetch_the_odds_api_events,
    load_api_key,
    write_odds_csv,
)
from data_sources.storage import save_odds_snapshot
from normalization.odds import normalize_odds_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill an odds template from The Odds API.")
    parser.add_argument("--template", required=True, help="Path to the existing odds template CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--api-key", help="The Odds API key. Falls back to ODDS_API_KEY.")
    parser.add_argument("--sport", default=DEFAULT_SPORT, help="The Odds API sport key.")
    parser.add_argument("--regions", default=DEFAULT_REGIONS, help="Comma-delimited bookmaker regions.")
    parser.add_argument("--markets", default=DEFAULT_MARKETS, help="Comma-delimited market keys.")
    parser.add_argument("--bookmaker", default=DEFAULT_BOOKMAKER, help="Bookmaker key, for example fanduel.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path for automatic odds snapshots.")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip saving the fetched odds into SQLite snapshot history.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()

    api_key = load_api_key(args.api_key)
    template = pd.read_csv(args.template)
    events = fetch_the_odds_api_events(
        api_key=api_key,
        sport=args.sport,
        regions=args.regions,
        markets=args.markets,
    )
    enriched = enrich_odds_template_from_the_odds_api(template, events, bookmaker_key=args.bookmaker, strict=False)
    output_path = write_odds_csv(enriched, args.output)
    snapshot_rows = 0
    if not args.no_snapshot:
        normalized = normalize_odds_frame(enriched)
        snapshot_frames = [normalized]
        modeled_market_snapshots = fetch_modeled_market_snapshots(
            api_key=api_key,
            event_rows=enriched,
            bookmaker_key=args.bookmaker,
        )
        if not modeled_market_snapshots.empty:
            snapshot_frames.append(normalize_odds_frame(modeled_market_snapshots))
        snapshot_frame = pd.concat(snapshot_frames, ignore_index=True).drop_duplicates(
            subset=["event_id", "fighter_a", "fighter_b", "market", "selection", "book", "american_odds"]
        )
        snapshot_rows = save_odds_snapshot(snapshot_frame, args.db)
    if not args.quiet:
        print(f"Filled {len(enriched)} rows from The Odds API")
        print(f"Saved odds to {output_path}")
        if args.no_snapshot:
            print("Skipped snapshot save")
        else:
            print(f"Saved {snapshot_rows} odds rows to {args.db}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from data_sources.odds_api import OddsApiError

        if isinstance(exc, OddsApiError):
            print(f"The Odds API refresh failed: {exc}", file=sys.stderr)
            raise SystemExit(1)
        raise
