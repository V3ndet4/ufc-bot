from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from statistics import median
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = "ufc-bot/1.0 (+https://www.bestfightodds.com/)"
REQUEST_TIMEOUT_SECONDS = 30
SUPPORTED_BOOKS = [
    "FanDuel",
    "Caesars",
    "BetMGM",
    "BetRivers",
    "BetWay",
    "Unibet",
    "DraftKings",
    "Bet365",
    "PointsBet",
]


def fetch_html(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def enrich_odds_template_from_bestfightodds(
    odds_frame: pd.DataFrame,
    event_html: str,
    *,
    book_preference: str = "consensus",
    strict: bool = True,
) -> pd.DataFrame:
    enriched = odds_frame.copy()
    if "selection_name" not in enriched.columns:
        enriched["selection_name"] = enriched.apply(
            lambda row: row["fighter_a"] if row["selection"] == "fighter_a" else row["fighter_b"],
            axis=1,
        )
    lines = extract_text_lines(event_html)
    books = parse_book_order(lines)

    current_odds: dict[str, dict[str, int]] = {}
    for fighter_name in sorted(set(enriched["selection_name"].astype(str).tolist())):
        try:
            current_odds[fighter_name] = parse_fighter_moneyline(lines, fighter_name, books)
        except ValueError:
            if strict:
                raise
            current_odds[fighter_name] = {}

    for idx, row in enriched.iterrows():
        selection_name = str(row["selection_name"])
        fighter_odds = current_odds.get(selection_name, {})
        if not fighter_odds:
            continue
        selected_odds, selected_book = select_moneyline(fighter_odds, book_preference)
        enriched.at[idx, "american_odds"] = selected_odds
        enriched.at[idx, "book"] = selected_book

    return enriched


def enrich_with_bestfightodds_history(
    odds_frame: pd.DataFrame,
    event_html: str,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    enriched = odds_frame.copy()
    if "selection_name" not in enriched.columns:
        enriched["selection_name"] = enriched.apply(
            lambda row: row["fighter_a"] if row["selection"] == "fighter_a" else row["fighter_b"],
            axis=1,
        )
    client = session or requests.Session()
    fighter_urls = extract_fighter_urls(event_html)

    for fighter_name in sorted(set(enriched["selection_name"].astype(str).tolist())):
        fighter_url = fighter_urls.get(_normalize_name(fighter_name))
        if not fighter_url:
            continue
        history_html = fetch_html(fighter_url, session=client)
        sample_row = enriched.loc[enriched["selection_name"].astype(str) == fighter_name].iloc[0]
        opponent_name = str(sample_row["fighter_b"]) if sample_row["selection"] == "fighter_a" else str(sample_row["fighter_a"])
        history_row = parse_history_row_for_event(
            history_html,
            event_date=str(enriched["start_time"].iloc[0]),
            fighter_name=fighter_name,
            opponent_name=opponent_name,
        )
        if not history_row:
            continue

        selection_mask = enriched["selection_name"].astype(str) == fighter_name
        enriched.loc[selection_mask, "open_american_odds"] = history_row.get("open_american_odds")
        enriched.loc[selection_mask, "current_best_range_low"] = history_row.get("current_range_low")
        enriched.loc[selection_mask, "current_best_range_high"] = history_row.get("current_range_high")
        enriched.loc[selection_mask, "bfo_fighter_url"] = fighter_url

    return enriched


def write_odds_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def extract_text_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    return [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]


def extract_fighter_urls(event_html: str) -> dict[str, str]:
    soup = BeautifulSoup(event_html, "html.parser")
    mapping: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.stripped_strings)
        href = anchor["href"]
        if not text or not href.startswith("/fighters/"):
            continue
        mapping[_normalize_name(text)] = f"https://www.bestfightodds.com{href}"
    return mapping


def parse_history_row_for_event(
    history_html: str,
    *,
    event_date: str,
    fighter_name: str,
    opponent_name: str,
) -> dict[str, int | str] | None:
    lines = extract_text_lines(history_html)
    event_year = str(pd.to_datetime(event_date).year)

    for index, line in enumerate(lines):
        if _normalize_name(line) != _normalize_name(fighter_name):
            continue
        window = lines[max(0, index - 3): min(len(lines), index + 14)]
        window_text = " ".join(window)
        normalized_window = _normalize_name(window_text)
        if _normalize_name(opponent_name) not in normalized_window:
            continue
        if event_year not in window_text:
            continue

        tokens = []
        for candidate in lines[index + 1: index + 8]:
            tokens.extend(_extract_odds_tokens(candidate))
        if len(tokens) < 3:
            continue

        open_odds = tokens[0]
        range_values = tokens[1:3]
        return {
            "open_american_odds": open_odds,
            "current_range_low": min(range_values),
            "current_range_high": max(range_values),
        }
    return None


def parse_book_order(lines: Iterable[str]) -> list[str]:
    line_list = list(lines)
    for line in line_list:
        if "FanDuel" in line and "Caesars" in line:
            return [book for book in SUPPORTED_BOOKS if book in line]
    for index, line in enumerate(line_list):
        if line == "FanDuel":
            books: list[str] = []
            cursor = index
            while cursor < len(line_list) and line_list[cursor] != "Props":
                if line_list[cursor] in SUPPORTED_BOOKS:
                    books.append(line_list[cursor])
                cursor += 1
            if books:
                return books
    raise ValueError("Could not locate sportsbook header on BestFightOdds page")


def parse_fighter_moneyline(lines: Iterable[str], fighter_name: str, books: list[str]) -> dict[str, int]:
    line_list = list(lines)
    candidate_names = _candidate_name_variants(fighter_name)
    matching_indices = [
        idx
        for idx, line in enumerate(line_list)
        if any(_normalize_name(candidate) in _normalize_name(line) for candidate in candidate_names)
    ]
    if not matching_indices:
        raise ValueError(f"Could not locate odds row for fighter: {fighter_name}")

    line_index = max(matching_indices)
    tokens = _extract_odds_tokens(line_list[line_index])
    if not tokens:
        for offset in range(1, len(books) + 3):
            candidate_index = line_index + offset
            if candidate_index >= len(line_list):
                break
            tokens.extend(_extract_odds_tokens(line_list[candidate_index]))
    if len(tokens) > len(books):
        tokens = tokens[: len(books)]
    return {book: odds for book, odds in zip(books, tokens)}


def select_moneyline(fighter_odds: dict[str, int], book_preference: str) -> tuple[int, str]:
    if not fighter_odds:
        raise ValueError("No odds available for fighter")
    if book_preference != "consensus" and book_preference in fighter_odds:
        return fighter_odds[book_preference], book_preference

    implied_probabilities = [_american_to_implied_probability(odds) for odds in fighter_odds.values()]
    consensus_probability = median(implied_probabilities)
    return implied_probability_to_american(consensus_probability), "bestfightodds_consensus"


def implied_probability_to_american(probability: float) -> int:
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")
    if probability >= 0.5:
        return int(round(-(probability / (1 - probability)) * 100))
    return int(round(((1 - probability) / probability) * 100))


def _extract_odds_tokens(text: str) -> list[int]:
    tokens = [int(token) for token in re.findall(r"(?<!\d)([+-]\d{2,4})(?!\d)", text)]
    return [token for token in tokens if abs(token) <= 2000]


def _american_to_implied_probability(american_odds: int) -> float:
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return abs(american_odds) / (abs(american_odds) + 100)


def _normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(normalized.lower().replace(".", "").split())


def _candidate_name_variants(fighter_name: str) -> list[str]:
    variants = {fighter_name}
    variants.add(fighter_name.replace(" Jr.", "").replace(" Jr", ""))
    return [variant for variant in variants if variant]
