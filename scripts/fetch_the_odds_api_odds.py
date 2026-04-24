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
from data_sources.storage import load_snapshot_history, save_odds_snapshot
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
    if template.empty:
        return template.copy()

    enriched = template.copy()
    if "selection_name" not in enriched.columns:
        enriched["selection_name"] = enriched.apply(selection_name_for_row, axis=1)
    enriched["american_odds"] = pd.to_numeric(enriched.get("american_odds"), errors="coerce")
    enriched.loc[enriched["american_odds"] == 0, "american_odds"] = pd.NA

    if priced_rows.empty:
        enriched["odds_api_event_id"] = ""
        enriched["odds_source"] = "missing"
        enriched["snapshot_fallback_used"] = 0
        enriched["last_snapshot_time"] = ""
        return enriched.reset_index(drop=True)

    fetched = priced_rows.copy()
    if "selection_name" not in fetched.columns:
        fetched["selection_name"] = fetched.apply(selection_name_for_row, axis=1)
    fetched = fetched.loc[:, TEMPLATE_KEY_COLUMNS + ["selection_name", "book", "american_odds", "odds_api_event_id"]]
    fetched = fetched.drop_duplicates(subset=TEMPLATE_KEY_COLUMNS, keep="last")
    fetched["american_odds"] = pd.to_numeric(fetched["american_odds"], errors="coerce")
    fetched = fetched.rename(
        columns={
            "selection_name": "selection_name_fetched",
            "book": "book_fetched",
            "american_odds": "american_odds_fetched",
        }
    )

    merged = enriched.merge(fetched, on=TEMPLATE_KEY_COLUMNS, how="left")
    live_mask = merged["american_odds_fetched"].notna()
    merged["selection_name"] = merged["selection_name_fetched"].fillna(merged["selection_name"])
    merged["book"] = merged["book_fetched"].fillna(merged["book"])
    merged["american_odds"] = merged["american_odds_fetched"].fillna(merged["american_odds"])
    merged["odds_api_event_id"] = merged["odds_api_event_id"].fillna("")
    merged["odds_source"] = live_mask.map({True: "live_api", False: "missing"})
    merged["snapshot_fallback_used"] = 0
    merged["last_snapshot_time"] = ""

    merged = merged.drop(columns=["selection_name_fetched", "book_fetched", "american_odds_fetched"])
    merged["american_odds"] = pd.to_numeric(merged["american_odds"], errors="coerce")
    merged.loc[merged["american_odds"] == 0, "american_odds"] = pd.NA
    merged = merged.reset_index(drop=True)
    return merged


def _load_template_snapshot_history(template: pd.DataFrame, db_path: str | Path) -> pd.DataFrame:
    path = Path(db_path)
    if template.empty or not path.exists() or "event_id" not in template.columns:
        return pd.DataFrame()

    history_frames: list[pd.DataFrame] = []
    for event_id in template["event_id"].dropna().astype(str).str.strip().unique().tolist():
        if not event_id:
            continue
        history = load_snapshot_history(path, event_id=event_id)
        if not history.empty:
            history_frames.append(history)
    if not history_frames:
        return pd.DataFrame()
    return pd.concat(history_frames, ignore_index=True)


def _latest_snapshot_rows(snapshot_history: pd.DataFrame) -> pd.DataFrame:
    required_columns = set(TEMPLATE_KEY_COLUMNS) | {"book", "american_odds"}
    if snapshot_history.empty or not required_columns.issubset(snapshot_history.columns):
        return pd.DataFrame()

    working = snapshot_history.copy()
    working["american_odds"] = pd.to_numeric(working["american_odds"], errors="coerce")
    working = working.loc[working["american_odds"].notna()].copy()
    if working.empty:
        return pd.DataFrame()

    if "selection_name" not in working.columns:
        working["selection_name"] = working.apply(selection_name_for_row, axis=1)
    if "snapshot_time" in working.columns:
        working["snapshot_sort_key"] = pd.to_datetime(working["snapshot_time"], errors="coerce")
    else:
        working["snapshot_sort_key"] = pd.NaT
        working["snapshot_time"] = ""
    if "snapshot_id" in working.columns:
        working["snapshot_id"] = pd.to_numeric(working["snapshot_id"], errors="coerce").fillna(-1)
    else:
        working["snapshot_id"] = pd.RangeIndex(start=0, stop=len(working), step=1)

    latest = (
        working.sort_values(["snapshot_sort_key", "snapshot_id"], na_position="last")
        .groupby(TEMPLATE_KEY_COLUMNS, dropna=False, as_index=False)
        .tail(1)
        .loc[:, TEMPLATE_KEY_COLUMNS + ["selection_name", "book", "american_odds", "snapshot_time"]]
        .rename(
            columns={
                "selection_name": "selection_name_snapshot",
                "book": "book_snapshot",
                "american_odds": "american_odds_snapshot",
                "snapshot_time": "last_snapshot_time_snapshot",
            }
        )
        .reset_index(drop=True)
    )
    return latest


def _apply_snapshot_fallback(enriched: pd.DataFrame, snapshot_history: pd.DataFrame) -> pd.DataFrame:
    if enriched.empty:
        return enriched.copy()

    latest_snapshot = _latest_snapshot_rows(snapshot_history)
    if latest_snapshot.empty:
        output = enriched.copy()
        if "odds_source" not in output.columns:
            output["odds_source"] = output["american_odds"].notna().map({True: "live_api", False: "missing"})
        if "snapshot_fallback_used" not in output.columns:
            output["snapshot_fallback_used"] = 0
        if "last_snapshot_time" not in output.columns:
            output["last_snapshot_time"] = ""
        return output

    merged = enriched.merge(latest_snapshot, on=TEMPLATE_KEY_COLUMNS, how="left")
    current_prices = pd.to_numeric(merged["american_odds"], errors="coerce")
    snapshot_prices = pd.to_numeric(merged["american_odds_snapshot"], errors="coerce")
    snapshot_mask = current_prices.isna() & snapshot_prices.notna()

    merged.loc[snapshot_mask, "selection_name"] = merged.loc[snapshot_mask, "selection_name_snapshot"].fillna(
        merged.loc[snapshot_mask, "selection_name"]
    )
    merged.loc[snapshot_mask, "book"] = merged.loc[snapshot_mask, "book_snapshot"].fillna(
        merged.loc[snapshot_mask, "book"]
    )
    merged.loc[snapshot_mask, "american_odds"] = snapshot_prices.loc[snapshot_mask]
    merged.loc[snapshot_mask, "odds_source"] = "snapshot_history"
    merged.loc[snapshot_mask, "snapshot_fallback_used"] = 1
    merged.loc[snapshot_mask, "last_snapshot_time"] = (
        merged.loc[snapshot_mask, "last_snapshot_time_snapshot"].fillna("").astype(str)
    )

    merged["odds_source"] = merged["odds_source"].fillna("missing")
    merged["snapshot_fallback_used"] = pd.to_numeric(
        merged["snapshot_fallback_used"], errors="coerce"
    ).fillna(0).astype(int)
    merged["last_snapshot_time"] = merged["last_snapshot_time"].fillna("").astype(str)

    merged = merged.drop(
        columns=[
            "selection_name_snapshot",
            "book_snapshot",
            "american_odds_snapshot",
            "last_snapshot_time_snapshot",
        ]
    )
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
    snapshot_history = _load_template_snapshot_history(template, args.db)
    enriched = _apply_snapshot_fallback(enriched, snapshot_history)
    output_path = write_odds_csv(enriched, args.output)
    snapshot_rows = 0
    if not args.no_snapshot:
        live_snapshot_candidates = enriched.loc[
            enriched.get("snapshot_fallback_used", pd.Series(0, index=enriched.index)).astype(int) == 0
        ].copy()
        live_snapshot_candidates = live_snapshot_candidates.loc[
            live_snapshot_candidates.get("odds_source", pd.Series("", index=live_snapshot_candidates.index)).astype(str) != "missing"
        ].copy()
        snapshot_frames: list[pd.DataFrame] = []
        if not live_snapshot_candidates.empty:
            snapshot_frames.append(normalize_odds_frame(live_snapshot_candidates))
        if not template_requests_modeled_markets and not modeled_market_snapshots.empty:
            snapshot_frames.append(normalize_odds_frame(modeled_market_snapshots))
        if snapshot_frames:
            snapshot_frame = pd.concat(snapshot_frames, ignore_index=True).drop_duplicates(
                subset=["event_id", "fighter_a", "fighter_b", "market", "selection", "book", "american_odds"]
            )
            snapshot_rows = save_odds_snapshot(snapshot_frame, args.db)
    if not args.quiet:
        print(f"Filled {len(enriched)} rows from The Odds API")
        print(f"Saved odds to {output_path}")
        snapshot_fallback_rows = int(
            pd.to_numeric(enriched.get("snapshot_fallback_used", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        )
        missing_rows = int(pd.to_numeric(enriched.get("american_odds", pd.Series(dtype=float)), errors="coerce").isna().sum())
        if snapshot_fallback_rows > 0:
            print(f"Reused {snapshot_fallback_rows} template rows from snapshot history")
        if missing_rows > 0:
            print(f"Left {missing_rows} template rows unpriced after live + snapshot fallback")
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
