from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.grading import fight_key, normalize_name
from data_sources.espn import fetch_html as fetch_espn_html, parse_fight_history
from data_sources.storage import load_snapshot_history
from data_sources.ufc_stats_scraper import UFCStatsScraper
from scripts.event_manifest import derived_paths, load_manifest


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
REQUEST_TIMEOUT_SECONDS = 30
TAPOLOGY_BASE_URL = "https://www.tapology.com"
TAPOLOGY_SEARCH_URL = f"{TAPOLOGY_BASE_URL}/search"
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch completed event results from UFC Stats and export them as results.csv.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path used for closing-odds backfill.")
    parser.add_argument("--fighter-stats", help="Optional fighter stats CSV used for ESPN fallback when UFC Stats is unavailable.")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between UFC Stats requests.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def fetch_results_for_manifest(
    manifest: dict[str, object],
    *,
    db_path: str | Path | None = None,
    scraper: UFCStatsScraper | None = None,
    fighter_stats_path: str | Path | None = None,
    ufc_html_fetcher: Callable[[str], str] | None = None,
    tapology_html_fetcher: Callable[[str], str] | None = None,
    espn_history_fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    client = scraper or UFCStatsScraper(delay=1.0)
    snapshot_history = load_snapshot_history(db_path, event_id=str(manifest["event_id"])) if db_path else pd.DataFrame()
    try:
        event, event_fights = select_matching_event(manifest, client)
    except Exception as exc:
        if _manifest_event_date(manifest) > date.today():
            raise
        ufc_error: Exception | None = None
        if _has_configured_official_ufc_event(manifest) or ufc_html_fetcher is not None:
            try:
                ufc_event, event_fights = _build_event_fights_from_official_ufc(
                    manifest,
                    html_fetcher=ufc_html_fetcher,
                )
            except Exception as official_ufc_exc:
                ufc_error = official_ufc_exc
            else:
                results = build_results_frame(manifest, event_fights, snapshot_history=snapshot_history)
                metadata = {
                    "event_name": str(ufc_event.get("name", "")).strip() or str(manifest.get("event_name", "")).strip(),
                    "event_url": str(ufc_event.get("url", "")).strip(),
                    "ufc_event_url": str(ufc_event.get("url", "")).strip(),
                    "matched_fights": int(len(results)),
                    "results_source": "ufc_official",
                    "fallback_reason": str(exc),
                }
                return results, metadata
        else:
            ufc_error = ValueError("official UFC fallback skipped because no event URL was configured")
        tapology_error: Exception | None = None
        try:
            tapology_event, event_fights = _build_event_fights_from_tapology(
                manifest,
                html_fetcher=tapology_html_fetcher,
            )
        except Exception as tapology_exc:
            tapology_error = tapology_exc
        else:
            results = build_results_frame(manifest, event_fights, snapshot_history=snapshot_history)
            metadata = {
                "event_name": str(tapology_event.get("name", "")).strip() or str(manifest.get("event_name", "")).strip(),
                "event_url": str(tapology_event.get("url", "")).strip(),
                "tapology_event_url": str(tapology_event.get("url", "")).strip(),
                "matched_fights": int(len(results)),
                "results_source": "tapology",
                "fallback_reason": str(exc),
            }
            return results, metadata
        try:
            event_fights = _build_event_fights_from_espn_history(
                manifest,
                fighter_stats_path=fighter_stats_path,
                history_fetcher=espn_history_fetcher,
            )
        except Exception as fallback_exc:
            raise RuntimeError(
                "UFC Stats results lookup failed "
                f"({exc.__class__.__name__}: {exc}) and official UFC fallback failed "
                f"({ufc_error.__class__.__name__}: {ufc_error}) and Tapology fallback failed "
                f"({tapology_error.__class__.__name__}: {tapology_error}) and ESPN fallback failed "
                f"({fallback_exc.__class__.__name__}: {fallback_exc})"
            ) from exc

        results = build_results_frame(manifest, event_fights, snapshot_history=snapshot_history)
        metadata = {
            "event_name": str(manifest.get("event_name", "")).strip(),
            "event_url": str(manifest.get("ufcstats_event_url", "")).strip(),
            "matched_fights": int(len(results)),
            "results_source": "espn_fighter_history",
            "fallback_reason": str(exc),
        }
        return results, metadata

    results = build_results_frame(manifest, event_fights, snapshot_history=snapshot_history)
    metadata = {
        "event_name": str(event.get("name", "")).strip(),
        "event_url": str(event.get("url", "")).strip(),
        "matched_fights": int(len(results)),
        "results_source": "ufc_stats",
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
    closing_odds_lookup = build_closing_odds_lookup(snapshot_history, start_time=manifest.get("start_time"))
    rows: list[dict[str, Any]] = []

    for fight in manifest.get("fights", []):
        fighter_a = str(fight.get("fighter_a", "")).strip()
        fighter_b = str(fight.get("fighter_b", "")).strip()
        if not fighter_a or not fighter_b:
            continue

        key = fight_key(fighter_a, fighter_b)
        fetched, result_match_status = _match_manifest_fight(fighter_a, fighter_b, event_fights)
        if fetched is None:
            continue

        actual_fighter_a = str(fetched.get("fighter_a", "") or "").strip()
        actual_fighter_b = str(fetched.get("fighter_b", "") or "").strip()
        actual_winner_name = str(fetched.get("winner", "") or "").strip()
        winner_name = _winner_name_for_manifest_order(actual_winner_name, fighter_a, fighter_b)
        winner_side = _winner_side_for_manifest_order(actual_winner_name, fighter_a, fighter_b)
        result_status = "replacement_opponent" if result_match_status == "replacement_opponent" else _result_status(actual_winner_name)
        method = str(fetched.get("method", "") or "").strip()
        went_decision = int(_method_is_decision(method))
        ended_inside_distance = int(result_status == "official" and went_decision == 0)

        row = {
            "event_id": str(manifest.get("event_id", "")).strip(),
            "event_name": str(manifest.get("event_name", "")).strip(),
            "fighter_a": fighter_a,
            "fighter_b": fighter_b,
            "actual_fighter_a": actual_fighter_a,
            "actual_fighter_b": actual_fighter_b,
            "winner_name": winner_name,
            "actual_winner_name": actual_winner_name,
            "winner_side": winner_side,
            "result_status": result_status,
            "result_match_status": result_match_status,
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
        "actual_fighter_a",
        "actual_fighter_b",
        "winner_name",
        "actual_winner_name",
        "winner_side",
        "result_status",
        "result_match_status",
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


def _match_manifest_fight(
    fighter_a: str,
    fighter_b: str,
    event_fights: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    manifest_key = fight_key(fighter_a, fighter_b)
    manifest_a = normalize_name(fighter_a)
    manifest_b = normalize_name(fighter_b)

    for fight in event_fights:
        if fight_key(fight.get("fighter_a", ""), fight.get("fighter_b", "")) == manifest_key:
            return fight, "exact"

    candidates: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for fight in event_fights:
        actual_a = str(fight.get("fighter_a", "") or "").strip()
        actual_b = str(fight.get("fighter_b", "") or "").strip()
        actual_names = {normalize_name(actual_a), normalize_name(actual_b)}
        if "" in actual_names:
            actual_names.discard("")
        shared = {manifest_a, manifest_b} & actual_names
        if not shared:
            continue
        score = (
            len(shared),
            int(manifest_a in actual_names),
            int(manifest_b in actual_names),
        )
        candidates.append((score, fight))

    if not candidates:
        return None, "unmatched"

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score = candidates[0][0]
    best = [fight for score, fight in candidates if score == best_score]
    if len(best) != 1 or best_score[0] <= 0:
        return None, "unmatched"

    return best[0], "replacement_opponent"


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


def _build_event_fights_from_official_ufc(
    manifest: dict[str, object],
    *,
    html_fetcher: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    event = _resolve_official_ufc_event(manifest)
    fetch_html = html_fetcher or _fetch_html
    html = fetch_html(str(event["url"]))
    event_fights = _parse_official_ufc_event_fights(html)
    if not event_fights:
        raise ValueError(f"no finished fights were parsed from official UFC event page {event['url']}")
    return event, event_fights


def _resolve_official_ufc_event(manifest: dict[str, object]) -> dict[str, str]:
    configured_url = str(
        manifest.get("ufc_event_url")
        or manifest.get("official_ufc_event_url")
        or manifest.get("ufc_results_url")
        or ""
    ).strip()
    if configured_url:
        return {
            "name": str(manifest.get("event_name", "")).strip(),
            "url": configured_url,
        }

    target_date = _manifest_event_date(manifest)
    event_name = str(manifest.get("event_name", "")).strip()
    candidates = _candidate_official_ufc_event_urls(event_name, target_date)
    if not candidates:
        raise ValueError(f"could not derive official UFC event URL for {event_name}")
    return {
        "name": event_name,
        "url": candidates[0],
    }


def _has_configured_official_ufc_event(manifest: dict[str, object]) -> bool:
    return bool(
        str(
            manifest.get("ufc_event_url")
            or manifest.get("official_ufc_event_url")
            or manifest.get("ufc_results_url")
            or ""
        ).strip()
    )


def _candidate_official_ufc_event_urls(event_name: str, target_date: date) -> list[str]:
    normalized_name = normalize_name(event_name)
    month_slug = target_date.strftime("%B").lower()
    candidates: list[str] = []

    if "ufc fight night" in normalized_name:
        candidates.append(f"https://www.ufc.com/event/ufc-fight-night-{month_slug}-{target_date.day}-{target_date.year}")

    numbered_match = re.search(r"\bufc\s+(\d+)\b", normalized_name)
    if numbered_match:
        candidates.append(f"https://www.ufc.com/event/ufc-{numbered_match.group(1)}")

    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate)
    return deduped


def _parse_official_ufc_event_fights(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    fights: list[dict[str, Any]] = []

    for node in soup.select(".c-listing-fight"):
        red_name_node = node.select_one(".c-listing-fight__corner-name--red")
        blue_name_node = node.select_one(".c-listing-fight__corner-name--blue")
        if red_name_node is None or blue_name_node is None:
            continue

        fighter_a = _normalize_space(red_name_node.get_text(" ", strip=True))
        fighter_b = _normalize_space(blue_name_node.get_text(" ", strip=True))
        if not fighter_a or not fighter_b:
            continue

        red_outcome = _normalize_space(
            (node.select_one(".c-listing-fight__corner--red .c-listing-fight__outcome-wrapper") or node).get_text(" ", strip=True)
        )
        blue_outcome = _normalize_space(
            (node.select_one(".c-listing-fight__corner--blue .c-listing-fight__outcome-wrapper") or node).get_text(" ", strip=True)
        )
        winner = _winner_from_official_ufc_outcomes(
            fighter_a=fighter_a,
            fighter_b=fighter_b,
            red_outcome=red_outcome,
            blue_outcome=blue_outcome,
        )

        method = ""
        for result_node in node.select(".c-listing-fight__result"):
            label_node = result_node.select_one(".c-listing-fight__result-label")
            value_node = result_node.select_one(".c-listing-fight__result-text")
            label = _normalize_space(label_node.get_text(" ", strip=True)) if label_node else ""
            value = _normalize_space(value_node.get_text(" ", strip=True)) if value_node else ""
            if label.lower() == "method":
                method = value
                break

        if not winner or not method:
            continue

        fights.append(
            {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "winner": winner,
                "method": method,
            }
        )

    return fights


def _winner_from_official_ufc_outcomes(
    *,
    fighter_a: str,
    fighter_b: str,
    red_outcome: str,
    blue_outcome: str,
) -> str:
    red_normalized = red_outcome.lower()
    blue_normalized = blue_outcome.lower()
    if "win" in red_normalized and "loss" in blue_normalized:
        return fighter_a
    if "win" in blue_normalized and "loss" in red_normalized:
        return fighter_b
    if "draw" in red_normalized or "draw" in blue_normalized:
        return "DRAW"
    if "no contest" in red_normalized or "no contest" in blue_normalized:
        return "NC"
    return ""


def _build_event_fights_from_tapology(
    manifest: dict[str, object],
    *,
    html_fetcher: Callable[[str], str] | None = None,
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    event = _resolve_tapology_event(manifest, html_fetcher=html_fetcher)
    fetch_html = html_fetcher or _fetch_html
    html = fetch_html(str(event["url"]))
    event_fights = _parse_tapology_event_fights(html)
    if not event_fights:
        raise ValueError(f"no finished fights were parsed from Tapology event page {event['url']}")
    return event, event_fights


def _resolve_tapology_event(
    manifest: dict[str, object],
    *,
    html_fetcher: Callable[[str], str] | None = None,
) -> dict[str, str]:
    configured_url = str(manifest.get("tapology_event_url") or manifest.get("tapology_url") or "").strip()
    if configured_url:
        return {
            "name": str(manifest.get("event_name", "")).strip(),
            "url": configured_url,
        }

    fetch_html = html_fetcher or _fetch_html
    target_date = _manifest_event_date(manifest)
    queries = _tapology_search_terms(manifest)
    candidates_by_url: dict[str, dict[str, Any]] = {}

    for query in queries:
        search_url = f"{TAPOLOGY_SEARCH_URL}?{urlencode({'model[events]': 'eventsSearch', 'search': 'Submit', 'term': query})}"
        html = fetch_html(search_url)
        for candidate in _parse_tapology_search_results(html):
            url = str(candidate.get("url", "")).strip()
            if not url:
                continue
            candidates_by_url[url] = candidate

    if not candidates_by_url:
        raise ValueError(f"could not find Tapology search candidates for {manifest.get('event_name', 'unknown event')}")

    manifest_name = normalize_name(manifest.get("event_name", ""))
    manifest_tokens = set(manifest_name.split())
    if ":" in str(manifest.get("event_name", "")):
        manifest_tokens.update(normalize_name(str(manifest.get("event_name", "")).split(":", 1)[1]).split())
    main_event_tokens: set[str] = set()
    if manifest.get("fights"):
        main_fight = manifest["fights"][0]
        main_event_tokens.update(normalize_name(main_fight.get("fighter_a", "")).split())
        main_event_tokens.update(normalize_name(main_fight.get("fighter_b", "")).split())

    def score(candidate: dict[str, Any]) -> tuple[int, int, int, int]:
        candidate_date = candidate.get("event_date")
        row_text = normalize_name(candidate.get("row_text", "") or candidate.get("name", ""))
        row_tokens = set(row_text.split())
        exact_date = int(candidate_date == target_date)
        main_overlap = len(main_event_tokens & row_tokens)
        name_overlap = len(manifest_tokens & row_tokens)
        date_penalty = -abs((candidate_date - target_date).days) if isinstance(candidate_date, date) else -999
        return (exact_date, main_overlap, name_overlap, date_penalty)

    best = max(candidates_by_url.values(), key=score, default=None)
    if best is None:
        raise ValueError(f"could not resolve Tapology event for {manifest.get('event_name', 'unknown event')}")

    best_score = score(best)
    if not (best_score[0] or best_score[1] >= 2 or best_score[2] >= 3):
        raise ValueError(
            f"Tapology search candidates did not confidently match {manifest.get('event_name', 'unknown event')}"
        )

    return {
        "name": str(best.get("name", "")).strip() or str(manifest.get("event_name", "")).strip(),
        "url": str(best["url"]).strip(),
    }


def _tapology_search_terms(manifest: dict[str, object]) -> list[str]:
    event_name = str(manifest.get("event_name", "")).strip()
    terms: list[str] = []
    for candidate in (
        event_name,
        event_name.replace(":", " "),
        event_name.split(":", 1)[1].strip() if ":" in event_name else "",
    ):
        cleaned = " ".join(candidate.split())
        if cleaned and cleaned not in terms:
            terms.append(cleaned)

    fights = manifest.get("fights", [])
    if fights:
        main_fight = fights[0]
        main_event_term = " ".join(
            part for part in (
                str(main_fight.get("fighter_a", "")).strip(),
                str(main_fight.get("fighter_b", "")).strip(),
                "UFC Fight Night",
            )
            if part
        )
        if main_event_term and main_event_term not in terms:
            terms.append(main_event_term)
    return terms


def _parse_tapology_search_results(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for row in soup.select("tr"):
        link = row.select_one("a[href*='/fightcenter/events/']")
        if link is None:
            continue
        href = str(link.get("href", "") or "").strip()
        if not href:
            continue
        url = urljoin(TAPOLOGY_BASE_URL, href)
        if url in seen_urls:
            continue
        row_text = _normalize_space(row.get_text(" ", strip=True))
        event_date = _parse_tapology_search_date(row_text)
        candidates.append(
            {
                "name": _normalize_space(link.get_text(" ", strip=True)),
                "url": url,
                "row_text": row_text,
                "event_date": event_date,
            }
        )
        seen_urls.add(url)
    return candidates


def _parse_tapology_search_date(text: str) -> date | None:
    for pattern in (r"\b\d{4}\.\d{2}\.\d{2}\b", r"\b\d{2}\.\d{2}\.\d{4}\b"):
        match = re.search(pattern, text)
        if not match:
            continue
        value = match.group(0)
        if value.count(".") == 2 and len(value.split(".")[0]) == 4:
            return _parse_event_date(value.replace(".", "-"))
        month, day, year = value.split(".")
        return _parse_event_date(f"{year}-{month}-{day}")
    return None


def _parse_tapology_event_fights(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    raw_lines = [_clean_tapology_line(line) for line in soup.get_text("\n").splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return []

    try:
        start_index = lines.index("Fight Card")
        working_lines = lines[start_index + 1 :]
    except ValueError:
        working_lines = lines

    fights: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(working_lines):
        if not _looks_like_tapology_summary(line):
            continue
        winner_name = _extract_tapology_marked_name(working_lines, index + 1, "W")
        loser_name = _extract_tapology_marked_name(working_lines, index + 1, "L")
        if not winner_name or not loser_name:
            continue
        fight = {
            "fighter_a": winner_name,
            "fighter_b": loser_name,
            "winner": winner_name,
            "method": _tapology_method_from_summary(line),
        }
        fights[fight_key(winner_name, loser_name)] = fight
    return list(fights.values())


def _extract_tapology_marked_name(lines: list[str], start_index: int, marker: str) -> str:
    for index in range(start_index, min(len(lines), start_index + 60)):
        line = lines[index]
        if index > start_index and _looks_like_tapology_summary(line):
            break
        if line != marker:
            continue
        name = _next_tapology_name(lines[index + 1 : index + 8])
        if name:
            return name
    return ""


def _next_tapology_name(lines: list[str]) -> str:
    for line in lines:
        if not line or not re.search(r"[A-Za-z]", line):
            continue
        if _is_tapology_name_noise(line):
            continue
        return line
    return ""


def _is_tapology_name_noise(line: str) -> bool:
    normalized = line.strip().lower()
    if normalized in {
        "professional mma",
        "betting odds",
        "weigh-in result",
        "height",
        "reach",
        "main card",
        "prelim",
        "main event",
        "co-main",
        "matchup page",
    }:
        return True
    if normalized.startswith(("up to", "down to")):
        return True
    if "nickname" in normalized:
        return True
    if re.fullmatch(r"#?\d+", line):
        return True
    if re.fullmatch(r"#?\d+\s+add_circle", line):
        return True
    if re.fullmatch(r"\d+(\.\d+)?", line):
        return True
    if re.fullmatch(r"\d+\s*x\s*\d+", line.lower()):
        return True
    if re.fullmatch(r"\d+(\.\d+)?\s*lbs.*", normalized):
        return True
    if "favorite" in normalized or "underdog" in normalized:
        return True
    return False


def _looks_like_tapology_summary(line: str) -> bool:
    normalized = _normalize_space(line)
    if not normalized:
        return False
    if not any(token in normalized for token in ("Decision", "Submission", "KO", "TKO", "Disqualification")):
        return False
    return "Round" in normalized or "Total" in normalized


def _tapology_method_from_summary(summary: str) -> str:
    normalized = _normalize_space(summary)
    match = re.match(r"^(.*?)(?:\s+\d+:\d+\s+Round|\s+\d+\s+Rounds?,)", normalized)
    if match:
        return match.group(1).strip()
    return normalized


def _clean_tapology_line(line: str) -> str:
    return _normalize_space(line.replace("\xa0", " "))


def _normalize_space(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _fetch_html(url: str) -> str:
    response = requests.get(url, headers=HTTP_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def _build_event_fights_from_espn_history(
    manifest: dict[str, object],
    *,
    fighter_stats_path: str | Path | None = None,
    history_fetcher: Callable[[str], pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    resolved_stats_path = Path(fighter_stats_path) if fighter_stats_path else derived_paths(manifest)["fighter_stats"]
    if not resolved_stats_path.exists():
        raise ValueError(f"fighter stats file missing at {resolved_stats_path}")

    fighter_stats = pd.read_csv(resolved_stats_path)
    espn_url_map = _build_espn_url_map(fighter_stats)
    if not espn_url_map:
        raise ValueError(f"no ESPN fighter URLs found in {resolved_stats_path}")

    fetch_history = history_fetcher or _fetch_espn_history
    target_date = _manifest_event_date(manifest)
    target_event_name = str(manifest.get("event_name", "")).strip()
    history_cache: dict[str, pd.DataFrame] = {}
    event_fights: list[dict[str, Any]] = []

    for fight in manifest.get("fights", []):
        fighter_a = str(fight.get("fighter_a", "")).strip()
        fighter_b = str(fight.get("fighter_b", "")).strip()
        if not fighter_a or not fighter_b:
            continue

        matched_fight: dict[str, Any] | None = None
        for page_fighter, opponent_name in ((fighter_a, fighter_b), (fighter_b, fighter_a)):
            fighter_url = espn_url_map.get(normalize_name(page_fighter), "")
            if not fighter_url:
                continue
            if fighter_url not in history_cache:
                history_cache[fighter_url] = fetch_history(fighter_url)
            matched_row = _match_espn_history_row(
                history_cache[fighter_url],
                opponent_name=opponent_name,
                target_date=target_date,
                event_name=target_event_name,
            )
            if matched_row is None:
                continue
            matched_fight = {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "winner": _winner_from_espn_history_row(
                    matched_row,
                    page_fighter=page_fighter,
                    opponent_name=opponent_name,
                ),
                "method": _method_from_espn_history_row(matched_row),
            }
            break

        if matched_fight is not None:
            event_fights.append(matched_fight)

    if not event_fights:
        raise ValueError(f"no manifest fights were matched from ESPN history in {resolved_stats_path}")
    return event_fights


def _build_espn_url_map(fighter_stats: pd.DataFrame) -> dict[str, str]:
    if fighter_stats.empty or "fighter_name" not in fighter_stats.columns:
        return {}

    url_map: dict[str, str] = {}
    for row in fighter_stats.to_dict("records"):
        fighter_name = str(row.get("fighter_name", "") or "").strip()
        if not fighter_name:
            continue
        normalized_name = normalize_name(fighter_name)
        for column in ("espn_url", "source_url"):
            candidate_url = str(row.get(column, "") or "").strip()
            if candidate_url and "espn.com" in candidate_url.lower():
                url_map[normalized_name] = candidate_url
                break
    return url_map


def _fetch_espn_history(url: str) -> pd.DataFrame:
    return parse_fight_history(fetch_espn_html(url))


def _match_espn_history_row(
    history: pd.DataFrame,
    *,
    opponent_name: str,
    target_date: date,
    event_name: str,
) -> dict[str, Any] | None:
    if history is None or history.empty:
        return None

    working = history.copy()
    if "opponent" not in working.columns or "date" not in working.columns:
        return None
    working["opponent_normalized"] = working["opponent"].astype(str).map(normalize_name)
    working = working.loc[working["opponent_normalized"] == normalize_name(opponent_name)].copy()
    if working.empty:
        return None

    working["event_date"] = pd.to_datetime(working["date"], errors="coerce").dt.date
    target_event_name_normalized = normalize_name(event_name)
    target_tokens = set(target_event_name_normalized.split())

    def score(row: dict[str, Any]) -> tuple[int, int, int, int]:
        candidate_date = row.get("event_date")
        event_normalized = normalize_name(row.get("event", ""))
        event_tokens = set(event_normalized.split())
        exact_date = int(candidate_date == target_date)
        exact_event_name = int(event_normalized == target_event_name_normalized)
        token_overlap = len(target_tokens & event_tokens)
        date_penalty = -abs((candidate_date - target_date).days) if isinstance(candidate_date, date) else -999
        return (exact_date, exact_event_name, token_overlap, date_penalty)

    candidates = working.to_dict("records")
    best = max(candidates, key=score, default=None)
    if best is None:
        return None

    best_score = score(best)
    candidate_date = best.get("event_date")
    close_in_time = isinstance(candidate_date, date) and abs((candidate_date - target_date).days) <= 3
    if not (best_score[0] or best_score[1] or best_score[2] >= 2 or close_in_time):
        return None
    return best


def _winner_from_espn_history_row(row: dict[str, Any], *, page_fighter: str, opponent_name: str) -> str:
    result_code = str(row.get("result_code", row.get("result", "")) or "").strip().upper()[:1]
    if result_code == "W":
        return page_fighter
    if result_code == "L":
        return opponent_name
    if result_code == "D":
        return "DRAW"
    if result_code == "N":
        return "NC"
    return ""


def _method_from_espn_history_row(row: dict[str, Any]) -> str:
    method = str(row.get("decision_type", "") or "").strip()
    if method:
        return method
    result_text = str(row.get("result", "") or "").strip()
    return result_text


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


def _method_is_decision(method: object) -> bool:
    normalized_method = str(method or "").strip().upper()
    return "DEC" in normalized_method or "DECISION" in normalized_method


def main() -> None:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest)
        results, metadata = fetch_results_for_manifest(
            manifest,
            db_path=args.db,
            scraper=UFCStatsScraper(delay=args.delay),
            fighter_stats_path=args.fighter_stats,
        )
        output_path = write_results_csv(results, args.output)
    except (ValueError, requests.RequestException, RuntimeError) as exc:
        if not args.quiet:
            print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not args.quiet:
        event_name = str(metadata.get("event_name", "")).strip() or str(manifest.get("event_name", "")).strip()
        if metadata.get("results_source") == "ufc_official":
            print(f"Used official UFC event-page fallback for {event_name}")
        elif metadata.get("results_source") == "espn_fighter_history":
            print(f"Used ESPN fighter-history fallback for {event_name}")
        elif metadata.get("results_source") == "tapology":
            print(f"Used Tapology fallback for {event_name}")
        else:
            print(f"Matched UFC Stats event: {event_name}")
        print(f"Saved {len(results)} result rows to {output_path}")


if __name__ == "__main__":
    main()
