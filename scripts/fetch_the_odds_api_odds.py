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
    selection_name_for_row,
    write_odds_csv,
)
from data_sources.storage import save_odds_snapshot
from normalization.odds import normalize_odds_frame

TEMPLATE_KEY_COLUMNS = ["event_id", "fighter_a", "fighter_b", "market", "selection"]


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


def _build_event_context_rows(template: pd.DataFrame) -> pd.DataFrame:
    if template.empty:
        return pd.DataFrame()

    fights = (
        template[["event_id", "event_name", "start_time", "fighter_a", "fighter_b"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    rows: list[dict[str, object]] = []
    for fight in fights.to_dict("records"):
        fighter_a = str(fight["fighter_a"])
        fighter_b = str(fight["fighter_b"])
        rows.append(
            {
                **fight,
                "market": "moneyline",
                "selection": "fighter_a",
                "selection_name": fighter_a,
                "book": "manual",
                "american_odds": 0,
            }
        )
        rows.append(
            {
                **fight,
                "market": "moneyline",
                "selection": "fighter_b",
                "selection_name": fighter_b,
                "book": "manual",
                "american_odds": 0,
            }
        )
    return pd.DataFrame(rows)


def _fill_template_from_priced_rows(template: pd.DataFrame, priced_rows: pd.DataFrame) -> pd.DataFrame:
    if template.empty or priced_rows.empty:
        return template.iloc[0:0].copy()

    enriched = template.copy()
    if "selection_name" not in enriched.columns:
        enriched["selection_name"] = enriched.apply(selection_name_for_row, axis=1)

    fetched = priced_rows.copy()
    if "selection_name" not in fetched.columns:
        fetched["selection_name"] = fetched.apply(selection_name_for_row, axis=1)
    fetched = fetched.loc[:, TEMPLATE_KEY_COLUMNS + ["selection_name", "book", "american_odds", "odds_api_event_id"]]
    fetched = fetched.drop_duplicates(subset=TEMPLATE_KEY_COLUMNS, keep="last")

    merged = enriched.merge(fetched, on=TEMPLATE_KEY_COLUMNS, how="left", suffixes=("", "_fetched"))
    merged["selection_name"] = merged["selection_name_fetched"].fillna(merged["selection_name"])
    merged["book"] = merged["book_fetched"].fillna(merged["book"])
    merged["american_odds"] = merged["american_odds_fetched"].fillna(merged["american_odds"])
    merged["odds_api_event_id"] = merged["odds_api_event_id"].fillna("")

    merged = merged.drop(columns=["selection_name_fetched", "book_fetched", "american_odds_fetched"])
    merged = merged.loc[pd.to_numeric(merged["american_odds"], errors="coerce").notna()].reset_index(drop=True)
    return merged


def _enrich_template_with_supported_markets(
    template: pd.DataFrame,
    events: list[dict[str, object]],
    *,
    api_key: str,
    bookmaker_key: str,
    fetch_modeled_markets: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_rows = _build_event_context_rows(template)
    event_rows = enrich_odds_template_from_the_odds_api(
        context_rows,
        events,
        bookmaker_key=bookmaker_key,
        strict=False,
    )
    if event_rows.empty:
        return template.iloc[0:0].copy(), pd.DataFrame()

    modeled_snapshots = (
        fetch_modeled_market_snapshots(
            api_key=api_key,
            event_rows=event_rows,
            bookmaker_key=bookmaker_key,
        )
        if fetch_modeled_markets
        else pd.DataFrame()
    )
    priced_rows = pd.concat([event_rows, modeled_snapshots], ignore_index=True) if not modeled_snapshots.empty else event_rows
    return _fill_template_from_priced_rows(template, priced_rows), modeled_snapshots


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
    requested_markets = set(template.get("market", pd.Series(dtype=str)).astype(str).str.strip())
    template_requests_modeled_markets = any(market and market != "moneyline" for market in requested_markets)
    enriched, modeled_market_snapshots = _enrich_template_with_supported_markets(
        template,
        events,
        api_key=api_key,
        bookmaker_key=args.bookmaker,
        fetch_modeled_markets=template_requests_modeled_markets or not args.no_snapshot,
    )
    output_path = write_odds_csv(enriched, args.output)
    snapshot_rows = 0
    if not args.no_snapshot:
        normalized = normalize_odds_frame(enriched)
        snapshot_frames = [normalized]
        if not template_requests_modeled_markets and not modeled_market_snapshots.empty:
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
