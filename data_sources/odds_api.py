"""The Odds API integration for UFC live odds and modeled market snapshots."""

from __future__ import annotations

import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from models.ev import market_overround, no_vig_two_way_probabilities, probability_to_american


DEFAULT_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_SPORT = "mma_mixed_martial_arts"
DEFAULT_REGIONS = "us"
DEFAULT_MARKETS = "h2h"
DEFAULT_BOOKMAKER = "fanduel"
REQUEST_TIMEOUT_SECONDS = 30
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
RETRY_BACKOFF_SECONDS = (1.0, 2.0, 4.0)
FEATURED_MARKET_KEYS = {"h2h", "spreads", "totals", "outrights", "h2h_lay", "outrights_lay"}
MODELED_MARKET_ORDER = (
    "h2h",
    "fight_goes_distance",
    "fight_goes_the_distance",
    "fight_ends_inside_distance",
    "fight_does_not_go_the_distance",
    "fight_doesnt_go_the_distance",
    "fighter_by_decision",
)


class OddsApiError(RuntimeError):
    """Raised when The Odds API response cannot satisfy the request."""


def load_odds_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_api_key(explicit_api_key: str | None = None) -> str:
    api_key = (explicit_api_key or os.getenv("ODDS_API_KEY") or "").strip()
    if not api_key:
        raise OddsApiError("Missing ODDS_API_KEY. Set it in .env or pass --api-key.")
    return api_key


def selection_name_for_row(row: pd.Series | dict[str, Any]) -> str:
    selection_name = str(row.get("selection_name", "") or "").strip()
    if selection_name:
        return selection_name

    selection = str(row.get("selection", "") or "").strip()
    fighter_a = str(row.get("fighter_a", "") or "").strip()
    fighter_b = str(row.get("fighter_b", "") or "").strip()
    if selection == "fighter_a":
        return fighter_a
    if selection == "fighter_b":
        return fighter_b
    if selection == "fight_goes_to_decision":
        return "Fight goes to decision"
    if selection == "fight_doesnt_go_to_decision":
        return "Fight doesn't go to decision"
    return selection


def _request_json(
    url: str,
    *,
    params: dict[str, Any],
    session: requests.Session | None = None,
) -> Any:
    client = session or requests.Session()
    attempts = len(RETRY_BACKOFF_SECONDS) + 1
    for attempt in range(attempts):
        try:
            response = client.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            should_retry = status_code in RETRYABLE_STATUS_CODES and attempt < attempts - 1
            if not should_retry:
                if status_code is not None:
                    raise OddsApiError(f"The Odds API returned HTTP {status_code} for {url}") from exc
                raise OddsApiError(f"The Odds API request failed for {url}") from exc
        except requests.RequestException as exc:
            if attempt >= attempts - 1:
                raise OddsApiError(f"The Odds API request failed for {url}: {exc}") from exc
        except ValueError as exc:
            raise OddsApiError(f"The Odds API returned invalid JSON for {url}") from exc

        time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    raise OddsApiError(f"The Odds API request failed for {url}")


def fetch_the_odds_api_events(
    *,
    api_key: str,
    sport: str = DEFAULT_SPORT,
    regions: str = DEFAULT_REGIONS,
    markets: str = DEFAULT_MARKETS,
    bookmakers: str | None = None,
    odds_format: str = "american",
    date_format: str = "iso",
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "apiKey": api_key,
        "regions": regions,
        "markets": markets,
        "oddsFormat": odds_format,
        "dateFormat": date_format,
    }
    if bookmakers:
        params["bookmakers"] = bookmakers
    payload = _request_json(
        f"{DEFAULT_BASE_URL}/sports/{sport}/odds",
        params=params,
        session=session,
    )
    if not isinstance(payload, list):
        raise OddsApiError("Unexpected The Odds API payload shape.")
    return payload


def fetch_the_odds_api_event_markets(
    *,
    api_key: str,
    event_id: str,
    sport: str = DEFAULT_SPORT,
    regions: str = DEFAULT_REGIONS,
    date_format: str = "iso",
    session: requests.Session | None = None,
) -> dict[str, Any]:
    payload = _request_json(
        f"{DEFAULT_BASE_URL}/sports/{sport}/events/{event_id}/markets",
        params={
            "apiKey": api_key,
            "regions": regions,
            "dateFormat": date_format,
        },
        session=session,
    )
    if not isinstance(payload, dict):
        raise OddsApiError("Unexpected The Odds API event-markets payload shape.")
    return payload


def fetch_the_odds_api_event_odds(
    *,
    api_key: str,
    event_id: str,
    markets: str,
    sport: str = DEFAULT_SPORT,
    regions: str = DEFAULT_REGIONS,
    bookmakers: str = DEFAULT_BOOKMAKER,
    odds_format: str = "american",
    date_format: str = "iso",
    session: requests.Session | None = None,
) -> dict[str, Any]:
    payload = _request_json(
        f"{DEFAULT_BASE_URL}/sports/{sport}/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": regions,
            "markets": markets,
            "bookmakers": bookmakers,
            "oddsFormat": odds_format,
            "dateFormat": date_format,
        },
        session=session,
    )
    if not isinstance(payload, dict):
        raise OddsApiError("Unexpected The Odds API event-odds payload shape.")
    return payload


def enrich_odds_template_from_the_odds_api(
    odds_frame: pd.DataFrame,
    events: list[dict[str, Any]],
    *,
    bookmaker_key: str = DEFAULT_BOOKMAKER,
    strict: bool = True,
) -> pd.DataFrame:
    enriched = odds_frame.copy()
    if "selection_name" not in enriched.columns:
        enriched["selection_name"] = enriched.apply(selection_name_for_row, axis=1)

    event_lookup = build_event_lookup(events, bookmaker_key=bookmaker_key)
    rows_to_drop: set[int] = set()
    unmatched_fights: set[str] = set()

    for idx, row in enriched.iterrows():
        fighter_a = str(row["fighter_a"]).strip()
        fighter_b = str(row["fighter_b"]).strip()
        event = event_lookup.get(frozenset({_normalize_name(fighter_a), _normalize_name(fighter_b)}))
        if not event:
            if strict:
                raise OddsApiError(f"Could not locate The Odds API event for fight: {fighter_a} vs {fighter_b}")
            rows_to_drop.add(idx)
            unmatched_fights.add(f"{fighter_a} vs {fighter_b}")
            continue

        outcome_name = str(row["selection_name"]).strip()
        try:
            outcome_price = lookup_outcome_price(event, outcome_name)
        except OddsApiError:
            if strict:
                raise
            rows_to_drop.add(idx)
            unmatched_fights.add(f"{fighter_a} vs {fighter_b}")
            continue
        enriched.at[idx, "american_odds"] = outcome_price
        enriched.at[idx, "book"] = bookmaker_key
        enriched.at[idx, "odds_api_event_id"] = event["id"]

    if rows_to_drop:
        enriched = enriched.drop(index=sorted(rows_to_drop)).reset_index(drop=True)
        print(
            "Skipped fights missing from the live The Odds API feed: "
            + ", ".join(sorted(unmatched_fights))
        )

    events_by_id = {
        str(event.get("id", "")).strip(): event
        for event in events
        if str(event.get("id", "")).strip()
    }
    return attach_moneyline_market_context(enriched, events_by_id, bookmaker_key=bookmaker_key)


def write_odds_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def extract_bookmaker_market_keys(event_markets_payload: dict[str, Any], bookmaker_key: str) -> list[str]:
    keys: set[str] = set()
    for bookmaker in event_markets_payload.get("bookmakers", []):
        if str(bookmaker.get("key", "")).strip() != bookmaker_key:
            continue
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", "")).strip()
            if market_key:
                keys.add(market_key)
    return sorted(keys)


def extract_alternative_market_keys(event_markets_payload: dict[str, Any], bookmaker_key: str) -> list[str]:
    return [
        key
        for key in extract_bookmaker_market_keys(event_markets_payload, bookmaker_key)
        if key not in FEATURED_MARKET_KEYS
    ]


def fetch_modeled_market_snapshots(
    *,
    api_key: str,
    event_rows: pd.DataFrame,
    bookmaker_key: str,
) -> pd.DataFrame:
    if event_rows.empty or "odds_api_event_id" not in event_rows.columns:
        return pd.DataFrame()

    snapshot_frames: list[pd.DataFrame] = []
    base_rows = event_rows.copy()
    base_rows["selection_name"] = base_rows.apply(selection_name_for_row, axis=1)
    base_rows = base_rows.loc[base_rows["odds_api_event_id"].astype(str).str.strip() != ""].copy()
    if base_rows.empty:
        return pd.DataFrame()

    for event_id, rows in base_rows.groupby("odds_api_event_id", dropna=False):
        event_id_text = str(event_id).strip()
        if not event_id_text:
            continue
        available_payload = fetch_the_odds_api_event_markets(api_key=api_key, event_id=event_id_text)
        available_keys = extract_bookmaker_market_keys(available_payload, bookmaker_key)
        modeled_keys = _select_modeled_market_keys(available_keys)
        if not modeled_keys:
            continue
        odds_payload = fetch_the_odds_api_event_odds(
            api_key=api_key,
            event_id=event_id_text,
            markets=",".join(modeled_keys),
            bookmakers=bookmaker_key,
        )
        snapshot_frame = extract_modeled_market_rows(
            odds_payload,
            bookmaker_key=bookmaker_key,
            event_rows=rows,
        )
        if not snapshot_frame.empty:
            snapshot_frames.append(snapshot_frame)

    if not snapshot_frames:
        return pd.DataFrame()
    return pd.concat(snapshot_frames, ignore_index=True)


def build_event_lookup(events: list[dict[str, Any]], *, bookmaker_key: str) -> dict[frozenset[str], dict[str, Any]]:
    lookup: dict[frozenset[str], dict[str, Any]] = {}
    for event in events:
        if not event_has_bookmaker(event, bookmaker_key):
            continue
        fighters = extract_event_fighters(event)
        if len(fighters) != 2:
            continue
        lookup[frozenset({_normalize_name(fighters[0]), _normalize_name(fighters[1])})] = event
    return lookup


def extract_event_fighters(event: dict[str, Any]) -> tuple[str, str]:
    home_team = str(event.get("home_team", "")).strip()
    away_team = str(event.get("away_team", "")).strip()
    return home_team, away_team


def event_has_bookmaker(event: dict[str, Any], bookmaker_key: str) -> bool:
    bookmakers = event.get("bookmakers", [])
    return any(str(bookmaker.get("key", "")).strip() == bookmaker_key for bookmaker in bookmakers)


def lookup_outcome_price(event: dict[str, Any], outcome_name: str) -> int:
    target = _normalize_name(outcome_name)
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if str(market.get("key", "")).strip() != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                if _normalize_name(str(outcome.get("name", "")).strip()) == target:
                    price = outcome.get("price")
                    if price is None:
                        raise OddsApiError(f"Missing price for outcome: {outcome_name}")
                    return int(price)
    raise OddsApiError(f"Could not locate outcome price for fighter: {outcome_name}")


def attach_moneyline_market_context(
    frame: pd.DataFrame,
    events_by_id: dict[str, dict[str, Any]],
    *,
    bookmaker_key: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    enriched["market_target_fair_prob"] = pd.NA
    enriched["market_target_overround"] = pd.NA
    enriched["market_consensus_prob"] = pd.NA
    enriched["market_consensus_american_odds"] = pd.NA
    enriched["market_consensus_bookmaker_count"] = 0
    enriched["market_overround"] = pd.NA

    summaries: dict[str, dict[str, Any]] = {}
    for event_id, event in events_by_id.items():
        summary = summarize_moneyline_market(event, bookmaker_key=bookmaker_key)
        if summary:
            summaries[event_id] = summary

    for idx, row in enriched.iterrows():
        event_id = str(row.get("odds_api_event_id", "")).strip()
        if not event_id or event_id not in summaries:
            continue
        selection_name = str(row.get("selection_name", "")).strip()
        if not selection_name:
            continue
        outcome_summary = summaries[event_id]["outcomes"].get(_normalize_name(selection_name))
        if not outcome_summary:
            continue
        enriched.at[idx, "market_target_fair_prob"] = outcome_summary.get("target_fair_prob", pd.NA)
        enriched.at[idx, "market_target_overround"] = summaries[event_id].get("target_overround", pd.NA)
        enriched.at[idx, "market_consensus_prob"] = outcome_summary.get("consensus_prob", pd.NA)
        enriched.at[idx, "market_consensus_american_odds"] = outcome_summary.get("consensus_american_odds", pd.NA)
        enriched.at[idx, "market_consensus_bookmaker_count"] = summaries[event_id].get("bookmaker_count", 0)
        enriched.at[idx, "market_overround"] = summaries[event_id].get("average_overround", pd.NA)

    return enriched


def summarize_moneyline_market(
    event: dict[str, Any],
    *,
    bookmaker_key: str,
) -> dict[str, Any]:
    outcome_prob_lists: dict[str, list[float]] = {}
    target_fair_probs: dict[str, float] = {}
    bookmaker_count = 0
    overrounds: list[float] = []
    target_overround: float | None = None

    for bookmaker in event.get("bookmakers", []):
        h2h_market = next(
            (
                market
                for market in bookmaker.get("markets", [])
                if str(market.get("key", "")).strip() == "h2h"
            ),
            None,
        )
        if not h2h_market:
            continue

        prices_by_name = _extract_two_way_prices(h2h_market)
        if len(prices_by_name) != 2:
            continue

        normalized_names = list(prices_by_name.keys())
        price_a = int(prices_by_name[normalized_names[0]])
        price_b = int(prices_by_name[normalized_names[1]])
        fair_a, fair_b = no_vig_two_way_probabilities(price_a, price_b)
        book_overround = market_overround(price_a, price_b)
        bookmaker_count += 1
        overrounds.append(book_overround)

        outcome_prob_lists.setdefault(normalized_names[0], []).append(fair_a)
        outcome_prob_lists.setdefault(normalized_names[1], []).append(fair_b)

        if str(bookmaker.get("key", "")).strip() == bookmaker_key:
            target_fair_probs[normalized_names[0]] = fair_a
            target_fair_probs[normalized_names[1]] = fair_b
            target_overround = book_overround

    if bookmaker_count == 0:
        return {}

    outcomes: dict[str, dict[str, Any]] = {}
    for normalized_name, fair_probs in outcome_prob_lists.items():
        consensus_prob = float(sum(fair_probs) / len(fair_probs))
        outcomes[normalized_name] = {
            "target_fair_prob": target_fair_probs.get(normalized_name, pd.NA),
            "consensus_prob": consensus_prob,
            "consensus_american_odds": probability_to_american(consensus_prob),
        }

    return {
        "bookmaker_count": bookmaker_count,
        "average_overround": float(sum(overrounds) / len(overrounds)),
        "target_overround": target_overround if target_overround is not None else pd.NA,
        "outcomes": outcomes,
    }


def extract_modeled_market_rows(
    payload: dict[str, Any],
    *,
    bookmaker_key: str,
    event_rows: pd.DataFrame,
) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()

    sample_row = event_rows.iloc[0]
    fighter_a = str(sample_row.get("fighter_a", "") or "").strip()
    fighter_b = str(sample_row.get("fighter_b", "") or "").strip()
    if not fighter_a or not fighter_b:
        return pd.DataFrame()

    outcome_lookup = _extract_modeled_market_lookup(payload, fighter_a=fighter_a, fighter_b=fighter_b, bookmaker_key=bookmaker_key)
    if not outcome_lookup:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    event_id = str(sample_row.get("event_id", "") or "").strip()
    event_name = str(sample_row.get("event_name", "") or "").strip()
    start_time = str(sample_row.get("start_time", "") or "").strip()
    book = bookmaker_key
    for market_key, selection_key, selection_name in modeled_market_template_rows(fighter_a, fighter_b):
        price = outcome_lookup.get((market_key, selection_key))
        if price is None:
            continue
        rows.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "start_time": start_time,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "market": market_key,
                "selection": selection_key,
                "selection_name": selection_name,
                "book": book,
                "american_odds": int(price),
                "odds_api_event_id": str(sample_row.get("odds_api_event_id", event_id) or "").strip(),
            }
        )
    return pd.DataFrame(rows)


def _extract_two_way_prices(market: dict[str, Any]) -> dict[str, int]:
    prices_by_name: dict[str, int] = {}
    for outcome in market.get("outcomes", []):
        normalized_name = _normalize_name(str(outcome.get("name", "")).strip())
        price = outcome.get("price")
        if not normalized_name or price is None:
            continue
        prices_by_name[normalized_name] = int(price)
    return prices_by_name if len(prices_by_name) == 2 else {}


def _select_modeled_market_keys(available_keys: list[str]) -> list[str]:
    available_set = {str(key).strip() for key in available_keys if str(key).strip()}
    return [key for key in MODELED_MARKET_ORDER if key in available_set]


def modeled_market_template_rows(fighter_a: str, fighter_b: str) -> list[tuple[str, str, str]]:
    return [
        ("moneyline", "fighter_a", fighter_a),
        ("moneyline", "fighter_b", fighter_b),
        ("fight_goes_to_decision", "fight_goes_to_decision", "Fight goes to decision"),
        ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision", "Fight doesn't go to decision"),
        ("inside_distance", "fighter_a", fighter_a),
        ("inside_distance", "fighter_b", fighter_b),
        ("by_decision", "fighter_a", fighter_a),
        ("by_decision", "fighter_b", fighter_b),
    ]


def _extract_modeled_market_lookup(
    payload: dict[str, Any],
    *,
    fighter_a: str,
    fighter_b: str,
    bookmaker_key: str,
) -> dict[tuple[str, str], int]:
    fighter_a_normalized = _normalize_name(fighter_a)
    fighter_b_normalized = _normalize_name(fighter_b)
    lookup: dict[tuple[str, str], int] = {}

    for bookmaker in payload.get("bookmakers", []):
        if str(bookmaker.get("key", "")).strip() != bookmaker_key:
            continue
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", "")).strip()
            normalized_market_key = _normalize_market_key(market_key)
            for outcome in market.get("outcomes", []):
                descriptor = " ".join(
                    str(value).strip()
                    for value in [outcome.get("name", ""), outcome.get("description", "")]
                    if str(value).strip()
                ).strip()
                canonical_key = _classify_market_outcome(
                    normalized_market_key,
                    descriptor,
                    fighter_a_normalized=fighter_a_normalized,
                    fighter_b_normalized=fighter_b_normalized,
                )
                price = outcome.get("price")
                if canonical_key is None or price is None:
                    continue
                lookup[canonical_key] = int(price)
    return lookup


def _classify_market_outcome(
    normalized_market_key: str,
    descriptor: str,
    *,
    fighter_a_normalized: str,
    fighter_b_normalized: str,
) -> tuple[str, str] | None:
    normalized_descriptor = _normalize_market_key(descriptor)
    if any(token in normalized_market_key for token in ["h2h", "moneyline"]):
        if fighter_a_normalized and fighter_a_normalized in normalized_descriptor:
            return ("moneyline", "fighter_a")
        if fighter_b_normalized and fighter_b_normalized in normalized_descriptor:
            return ("moneyline", "fighter_b")
    if any(token in normalized_market_key for token in ["goes_distance", "goes_the_distance", "to_go_the_distance", "decision"]):
        if normalized_descriptor in {"yes", "over", "fight goes to decision"}:
            return ("fight_goes_to_decision", "fight_goes_to_decision")
        if normalized_descriptor in {"no", "under", "fight doesnt go to decision", "fight does not go to decision"}:
            return ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision")
    if any(token in normalized_market_key for token in ["inside_distance", "ends_inside_distance", "distance"]):
        if fighter_a_normalized and fighter_a_normalized in normalized_descriptor:
            return ("inside_distance", "fighter_a")
        if fighter_b_normalized and fighter_b_normalized in normalized_descriptor:
            return ("inside_distance", "fighter_b")
    if "by_decision" in normalized_market_key:
        if fighter_a_normalized and fighter_a_normalized in normalized_descriptor:
            return ("by_decision", "fighter_a")
        if fighter_b_normalized and fighter_b_normalized in normalized_descriptor:
            return ("by_decision", "fighter_b")
    return None


def _normalize_market_key(value: str) -> str:
    return _normalize_name(value).replace(" ", "_")


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", normalized.lower())
    cleaned = " ".join(cleaned.split())
    alias_map = {
        "joseph": "joe",
        "mckinney": "mckinney",
        "jr": "",
        "junior": "",
        "sr": "",
        "senior": "",
        "ii": "",
        "iii": "",
        "iv": "",
    }
    tokens = [alias_map.get(token, token) for token in cleaned.split()]
    return " ".join(token for token in tokens if token)
