from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.grading import fight_key, normalize_name
from data_sources.storage import load_snapshot_history
from data_sources.ufc_stats_scraper import UFCStatsScraper
from scripts.event_manifest import load_manifest


CLOSING_ODDS_COLUMN_MAP = {
    ("moneyline", "fighter_a"): "closing_fighter_a_odds",
    ("moneyline", "fighter_b"): "closing_fighter_b_odds",
    ("fight_goes_to_decision", "fight_goes_to_decision"): "closing_fight_goes_to_decision_odds",
    ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision"): "closing_fight_doesnt_go_to_decision_odds",
    ("inside_distance", "fighter_a"): "closing_fighter_a_inside_distance_odds",
    ("inside_distance", "fighter_b"): "closing_fighter_b_inside_distance_odds",
    ("by_decision", "fighter_a"): "closing_fighter_a_by_decision_odds",
    ("by_decision", "fighter_b"): "closing_fighter_b_by_decision_odds",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch completed event results from UFC Stats and export them as results.csv.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path used for closing-odds backfill.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between UFC Stats requests.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def fetch_results_for_manifest(
    manifest: dict[str, object],
    *,
    db_path: str | Path | None = None,
    scraper: UFCStatsScraper | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = scraper or UFCStatsScraper(delay=1.0)
    event, event_fights = select_matching_event(manifest, client)
    snapshot_history = load_snapshot_history(db_path, event_id=str(manifest["event_id"])) if db_path else pd.DataFrame()
    results = build_results_frame(manifest, event_fights, snapshot_history=snapshot_history)
    metadata = {
        "ufcstats_event_name": event.get("name", ""),
        "ufcstats_event_url": event.get("url", ""),
        "matched_fights": int(len(results)),
    }
    return results, metadata


def select_matching_event(manifest: dict[str, object], scraper: UFCStatsScraper) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    configured_event_url = str(manifest.get("ufcstats_event_url") or "").strip()
    if configured_event_url:
        event = {
            "name": str(manifest.get("event_name", "")).strip(),
            "url": configured_event_url,
            "date": "",
        }
        return event, scraper.get_event_fights(configured_event_url)

    target_date = _manifest_event_date(manifest)
    today = date.today()
    if target_date > today:
        raise ValueError(
            f"Manifest event date {target_date.isoformat()} is in the future relative to local date {today.isoformat()}; "
            "UFC Stats completed results are only available after the event finishes."
        )

    events = scraper.get_event_list()
    manifest_keys = {
        fight_key(fight["fighter_a"], fight["fighter_b"])
        for fight in manifest.get("fights", [])
        if str(fight.get("fighter_a", "")).strip() and str(fight.get("fighter_b", "")).strip()
    }
    candidates = _candidate_events(events, target_date)
    if not candidates:
        nearest = _nearest_events(events, target_date)
        nearest_summary = ", ".join(
            f"{event.get('name', 'unknown event')} ({event['event_date'].isoformat()})"
            for event in nearest
        )
        suffix = f" Nearest completed events: {nearest_summary}" if nearest_summary else ""
        raise ValueError(f"Could not find UFC Stats event candidates near {target_date.isoformat()}.{suffix}")

    best_event: dict[str, Any] | None = None
    best_fights: list[dict[str, Any]] = []
    best_score: tuple[int, int, int] | None = None

    for event in candidates:
        fights = scraper.get_event_fights(str(event["url"]))
        overlap = len(
            manifest_keys
            & {
                fight_key(fight.get("fighter_a", ""), fight.get("fighter_b", ""))
                for fight in fights
            }
        )
        exact_name_match = int(normalize_name(event.get("name", "")) == normalize_name(manifest.get("event_name", "")))
        date_penalty = -abs((event["event_date"] - target_date).days)
        score = (overlap, exact_name_match, date_penalty)
        if best_score is None or score > best_score:
            best_event = event
            best_fights = fights
            best_score = score

    if best_event is None or best_score is None or best_score[0] == 0:
        raise ValueError(f"Could not match UFC Stats event for {manifest.get('event_name', 'unknown event')}")

    return best_event, best_fights


def build_results_frame(
    manifest: dict[str, object],
    event_fights: list[dict[str, Any]],
    *,
    snapshot_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    fetched_by_key = {
        fight_key(fight.get("fighter_a", ""), fight.get("fighter_b", "")): fight
        for fight in event_fights
    }
    closing_odds_lookup = build_closing_odds_lookup(snapshot_history, start_time=manifest.get("start_time"))
    rows: list[dict[str, Any]] = []

    for fight in manifest.get("fights", []):
        fighter_a = str(fight.get("fighter_a", "")).strip()
        fighter_b = str(fight.get("fighter_b", "")).strip()
        if not fighter_a or not fighter_b:
            continue

        key = fight_key(fighter_a, fighter_b)
        fetched = fetched_by_key.get(key)
        if fetched is None:
            continue

        winner_name = _winner_name_for_manifest_order(fetched.get("winner"), fighter_a, fighter_b)
        winner_side = _winner_side_for_manifest_order(fetched.get("winner"), fighter_a, fighter_b)
        result_status = _result_status(fetched.get("winner"))
        method = str(fetched.get("method", "") or "").strip()
        went_decision = int("decision" in method.lower())
        ended_inside_distance = int(result_status == "official" and went_decision == 0)

        row = {
            "event_id": str(manifest.get("event_id", "")).strip(),
            "event_name": str(manifest.get("event_name", "")).strip(),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "winner_name": winner_name,
            "winner_side": winner_side,
            "result_status": result_status,
            "went_decision": went_decision,
            "ended_inside_distance": ended_inside_distance,
            "method": method,
        }
        row.update(closing_odds_lookup.get(key, {}))
        rows.append(row)

    columns = [
        "event_id",
        "event_name",
        "fighter_a",
        "fighter_b",
        "winner_name",
        "winner_side",
        "result_status",
        "went_decision",
        "ended_inside_distance",
        "method",
        "closing_fighter_a_odds",
        "closing_fighter_b_odds",
        "closing_fight_goes_to_decision_odds",
        "closing_fight_doesnt_go_to_decision_odds",
        "closing_fighter_a_inside_distance_odds",
        "closing_fighter_b_inside_distance_odds",
        "closing_fighter_a_by_decision_odds",
        "closing_fighter_b_by_decision_odds",
    ]
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def build_closing_odds_lookup(snapshot_history: pd.DataFrame | None, *, start_time: object) -> dict[str, dict[str, int]]:
    if snapshot_history is None or snapshot_history.empty:
        return {}

    working = snapshot_history.copy()
    if not {"fighter_a", "fighter_b", "market", "selection", "american_odds"}.issubset(working.columns):
        return {}

    working["fight_key"] = working.apply(lambda row: fight_key(row["fighter_a"], row["fighter_b"]), axis=1)
    working["snapshot_time"] = pd.to_datetime(working.get("snapshot_time"), errors="coerce", utc=True)
    start_timestamp = pd.to_datetime(start_time, errors="coerce", utc=True)
    if pd.notna(start_timestamp) and working["snapshot_time"].notna().any():
        eligible = working.loc[working["snapshot_time"] <= start_timestamp].copy()
        if not eligible.empty:
            working = eligible

    working["american_odds"] = pd.to_numeric(working["american_odds"], errors="coerce")
    working = working.loc[working["american_odds"].notna()].copy()
    if working.empty:
        return {}

    working = working.sort_values("snapshot_time")
    lookup: dict[str, dict[str, int]] = {}
    for (fight_id, market, selection), rows in working.groupby(["fight_key", "market", "selection"], dropna=False):
        column = CLOSING_ODDS_COLUMN_MAP.get((str(market).strip(), str(selection).strip()))
        if not column:
            continue
        lookup.setdefault(str(fight_id), {})[column] = int(float(rows.iloc[-1]["american_odds"]))
    return lookup


def write_results_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _candidate_events(events: list[dict[str, Any]], target_date: date) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for event in events:
        event_date = _parse_event_date(event.get("date"))
        if event_date is None:
            continue
        parsed.append({**event, "event_date": event_date})

    same_day = [event for event in parsed if event["event_date"] == target_date]
    if same_day:
        return same_day

    nearby = [event for event in parsed if abs((event["event_date"] - target_date).days) <= 3]
    nearby.sort(key=lambda event: abs((event["event_date"] - target_date).days))
    return nearby[:8]


def _nearest_events(events: list[dict[str, Any]], target_date: date, *, limit: int = 3) -> list[dict[str, Any]]:
    parsed: list[dict[str, Any]] = []
    for event in events:
        event_date = _parse_event_date(event.get("date"))
        if event_date is None:
            continue
        parsed.append({**event, "event_date": event_date})

    parsed.sort(key=lambda event: abs((event["event_date"] - target_date).days))
    return parsed[:limit]


def _manifest_event_date(manifest: dict[str, object]) -> date:
    return pd.to_datetime(manifest["start_time"], errors="raise").date()


def _parse_event_date(value: object) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _winner_name_for_manifest_order(winner: object, fighter_a: str, fighter_b: str) -> str:
    winner_side = _winner_side_for_manifest_order(winner, fighter_a, fighter_b)
    if winner_side == "fighter_a":
        return fighter_a
    if winner_side == "fighter_b":
        return fighter_b
    return ""


def _winner_side_for_manifest_order(winner: object, fighter_a: str, fighter_b: str) -> str:
    normalized_winner = normalize_name(winner)
    if normalized_winner in {"draw"}:
        return "draw"
    if normalized_winner in {"nc", "no contest", "no_contest"}:
        return "no_contest"
    if normalized_winner == normalize_name(fighter_a):
        return "fighter_a"
    if normalized_winner == normalize_name(fighter_b):
        return "fighter_b"
    return ""


def _result_status(winner: object) -> str:
    normalized_winner = normalize_name(winner)
    if normalized_winner in {"draw"}:
        return "draw"
    if normalized_winner in {"nc", "no contest", "no_contest"}:
        return "no contest"
    return "official"


def main() -> None:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        results, metadata = fetch_results_for_manifest(
            manifest,
            db_path=args.db,
            scraper=UFCStatsScraper(delay=args.delay),
        )
        output_path = write_results_csv(results, args.output)
    except ValueError as exc:
        if not args.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.quiet:
        print(f"Matched UFC Stats event: {metadata['ufcstats_event_name']}")
        print(f"Saved {len(results)} result rows to {output_path}")


if __name__ == "__main__":
    main()
