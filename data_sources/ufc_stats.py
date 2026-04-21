from __future__ import annotations

import string
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://ufcstats.com"
FIGHTER_DIRECTORY_TEMPLATE = f"{BASE_URL}/statistics/fighters?char={{letter}}&page=all"
USER_AGENT = "ufc-bot/1.0 (+https://ufcstats.com)"
REQUEST_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class FighterIndexEntry:
    fighter_name: str
    fighter_url: str


def build_fighter_directory_urls(letters: Iterable[str] | None = None) -> list[str]:
    alphabet = letters or string.ascii_lowercase
    return [FIGHTER_DIRECTORY_TEMPLATE.format(letter=letter.lower()) for letter in alphabet]


def fetch_html(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def parse_fighter_directory(html: str, base_url: str = BASE_URL) -> list[FighterIndexEntry]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.b-statistics__table-row")
    fighters: list[FighterIndexEntry] = []

    for row in rows:
        cells = row.select("td")
        if len(cells) < 2:
            continue

        link = cells[0].find("a", href=True)
        if link is None:
            continue

        first_name = cells[0].get_text(" ", strip=True)
        last_name = cells[1].get_text(" ", strip=True)
        fighter_name = " ".join(part for part in [first_name, last_name] if part).strip()
        if not fighter_name:
            continue

        fighters.append(
            FighterIndexEntry(
                fighter_name=fighter_name,
                fighter_url=urljoin(base_url, link["href"]),
            )
        )

    deduped: dict[str, FighterIndexEntry] = {}
    for fighter in fighters:
        deduped[fighter.fighter_url] = fighter
    return list(deduped.values())


def parse_fighter_details(html: str, source_url: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    stats = _extract_label_value_map(soup)

    fighter_name = _extract_fighter_name(soup)
    wins, losses, draws = _parse_record(_extract_record_text(soup))

    return {
        "fighter_name": fighter_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_in": _parse_height_to_inches(stats.get("height", "")),
        "reach_in": _parse_reach_to_inches(stats.get("reach", "")),
        "sig_strikes_landed_per_min": _parse_float(stats.get("slpm", "")),
        "sig_strikes_absorbed_per_min": _parse_float(stats.get("sapm", "")),
        "strike_accuracy_pct": _parse_percentage(stats.get("str. acc.", "")),
        "strike_defense_pct": _parse_percentage(stats.get("str. def", "")),
        "takedown_avg": _parse_float(stats.get("td avg.", "")),
        "takedown_accuracy_pct": _parse_percentage(stats.get("td acc.", "")),
        "takedown_defense_pct": _parse_percentage(stats.get("td def.", "")),
        "submission_avg": _parse_float(stats.get("sub. avg.", "")),
        "stance": _normalize_text(stats.get("stance", "")),
        "date_of_birth": _normalize_text(stats.get("dob", "")),
        "source_url": source_url,
    }


def scrape_fighter_stats(
    letters: Iterable[str] | None = None,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    directory_entries: list[FighterIndexEntry] = []
    for url in build_fighter_directory_urls(letters):
        directory_entries.extend(parse_fighter_directory(fetch_html(url, session=client)))

    fighters: list[dict[str, object]] = []
    for entry in directory_entries:
        fighters.append(parse_fighter_details(fetch_html(entry.fighter_url, session=client), entry.fighter_url))

    frame = pd.DataFrame(fighters)
    if frame.empty:
        return frame

    frame = frame.sort_values("fighter_name").drop_duplicates(subset=["fighter_name"], keep="first")
    numeric_columns = [
        "wins",
        "losses",
        "draws",
        "height_in",
        "reach_in",
        "sig_strikes_landed_per_min",
        "sig_strikes_absorbed_per_min",
        "strike_accuracy_pct",
        "strike_defense_pct",
        "takedown_avg",
        "takedown_accuracy_pct",
        "takedown_defense_pct",
        "submission_avg",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_columns = [
        "fighter_name",
        "wins",
        "losses",
        "draws",
        "height_in",
        "reach_in",
        "sig_strikes_landed_per_min",
        "sig_strikes_absorbed_per_min",
        "strike_accuracy_pct",
        "strike_defense_pct",
        "takedown_avg",
        "takedown_accuracy_pct",
        "takedown_defense_pct",
        "submission_avg",
        "stance",
        "date_of_birth",
        "source_url",
    ]
    return frame[ordered_columns]


def scrape_fighter_stats_for_names(
    fighter_names: Iterable[str],
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    requested_names = [str(name).strip() for name in fighter_names if str(name).strip()]
    if not requested_names:
        raise ValueError("At least one fighter name is required")

    requested_by_normalized = {_normalize_fighter_name(name): name for name in requested_names}
    letters = sorted({_fighter_last_name_initial(name) for name in requested_names if _fighter_last_name_initial(name)})
    client = session or requests.Session()

    directory_entries: list[FighterIndexEntry] = []
    for url in build_fighter_directory_urls(letters):
        directory_entries.extend(parse_fighter_directory(fetch_html(url, session=client)))

    entries_by_name = {
        _normalize_fighter_name(entry.fighter_name): entry
        for entry in directory_entries
    }

    missing = [
        original_name
        for normalized_name, original_name in requested_by_normalized.items()
        if normalized_name not in entries_by_name
    ]
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Could not locate UFC Stats pages for: {missing_list}")

    fighters: list[dict[str, object]] = []
    for normalized_name, original_name in requested_by_normalized.items():
        entry = entries_by_name[normalized_name]
        fighter = parse_fighter_details(fetch_html(entry.fighter_url, session=client), entry.fighter_url)
        fighter["fighter_name"] = original_name
        fighters.append(fighter)

    frame = pd.DataFrame(fighters)
    numeric_columns = [
        "wins",
        "losses",
        "draws",
        "height_in",
        "reach_in",
        "sig_strikes_landed_per_min",
        "sig_strikes_absorbed_per_min",
        "strike_accuracy_pct",
        "strike_defense_pct",
        "takedown_avg",
        "takedown_accuracy_pct",
        "takedown_defense_pct",
        "submission_avg",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    ordered_columns = [
        "fighter_name",
        "wins",
        "losses",
        "draws",
        "height_in",
        "reach_in",
        "sig_strikes_landed_per_min",
        "sig_strikes_absorbed_per_min",
        "strike_accuracy_pct",
        "strike_defense_pct",
        "takedown_avg",
        "takedown_accuracy_pct",
        "takedown_defense_pct",
        "submission_avg",
        "stance",
        "date_of_birth",
        "source_url",
    ]
    return frame[ordered_columns]


def write_fighter_stats_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _extract_fighter_name(soup: BeautifulSoup) -> str:
    title = soup.select_one(".b-content__title-highlight")
    if title is not None:
        return title.get_text(" ", strip=True)
    raise ValueError("Could not locate fighter name on UFC Stats page")


def _extract_label_value_map(soup: BeautifulSoup) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = " ".join(item.stripped_strings)
        if ":" not in text:
            continue
        label, raw_value = text.split(":", 1)
        values[_normalize_label(label)] = _normalize_text(raw_value)
    return values


def _extract_record_text(soup: BeautifulSoup) -> str:
    record = soup.select_one(".b-content__title-record")
    if record is not None:
        return record.get_text(" ", strip=True)
    return ""


def _normalize_label(value: str) -> str:
    return " ".join(value.lower().replace("\xa0", " ").split())


def _normalize_text(value: str) -> str:
    cleaned = " ".join(str(value).replace("\xa0", " ").split())
    return "" if cleaned in {"--", "---", "N/A"} else cleaned


def _normalize_fighter_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(normalized.lower().replace(".", "").split())


def _fighter_last_name_initial(value: str) -> str:
    normalized = _normalize_fighter_name(value)
    if not normalized:
        return ""
    parts = normalized.split()
    return parts[-1][0] if parts and parts[-1] else ""


def _parse_record(raw_value: str) -> tuple[int, int, int]:
    cleaned = _normalize_text(raw_value).lower().removeprefix("record").strip(": ").split(" ", 1)[0]
    parts = [part for part in cleaned.split("-") if part]
    if len(parts) < 2:
        return 0, 0, 0
    wins = int(parts[0])
    losses = int(parts[1])
    draws = int(parts[2]) if len(parts) > 2 else 0
    return wins, losses, draws


def _parse_height_to_inches(raw_value: str) -> int | None:
    cleaned = _normalize_text(raw_value)
    if not cleaned:
        return None
    feet_part, _, inch_part = cleaned.partition("'")
    inches_part = inch_part.replace('"', "").strip()
    return (int(feet_part.strip()) * 12) + int(inches_part)


def _parse_reach_to_inches(raw_value: str) -> float | None:
    cleaned = _normalize_text(raw_value).replace('"', "")
    if not cleaned:
        return None
    return float(cleaned)


def _parse_percentage(raw_value: str) -> float | None:
    cleaned = _normalize_text(raw_value).replace("%", "")
    if not cleaned:
        return None
    return float(cleaned)


def _parse_float(raw_value: str) -> float | None:
    cleaned = _normalize_text(raw_value)
    if not cleaned:
        return None
    return float(cleaned)


def parse_fighter_directory_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(StringIO(html))
    if not tables:
        raise ValueError("No fighter directory tables found")
    return tables[0]
