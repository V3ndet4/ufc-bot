from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://www.sherdog.com"
SEARCH_API_URL = f"{BASE_URL}/search/fightfinder/"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = "ufc-bot/1.0 (+https://www.sherdog.com/)"
UTC = timezone.utc

FIGHTER_GYM_NUMERIC_COLUMNS = [
    "gym_score",
    "gym_fighter_count",
    "gym_total_wins",
    "gym_total_losses",
    "gym_total_draws",
    "gym_win_rate",
    "gym_elite_fighter_count",
    "gym_changed_flag",
    "fighter_wins",
    "fighter_losses",
    "fighter_draws",
    "fighter_win_rate",
    "fighter_elite_flag",
]

FIGHTER_GYM_STRING_COLUMNS = [
    "sherdog_url",
    "gym_source",
    "gym_name",
    "gym_name_normalized",
    "gym_page_url",
    "gym_tier",
    "gym_record",
    "weight_class",
    "previous_gym_name",
    "last_changed_at",
    "profile_last_refreshed_at",
    "last_seen_at",
]


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    cleaned = str(value).strip()
    return "" if cleaned.lower() in {"nan", "none", "<na>"} else cleaned


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        numeric = float("nan")
    return default if pd.isna(numeric) else int(float(numeric))


def normalize_name(value: object) -> str:
    normalized = _clean_text(value).lower().replace("’", "'")
    normalized = re.sub(r"\b(jr|jr\.|sr|sr\.|iii|iv|v)\b", "", normalized)
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return " ".join(normalized.split())


def normalize_gym_name(value: object) -> str:
    normalized = _clean_text(value).lower().replace("’", "'").replace("&", " and ")
    normalized = normalized.replace("/", " / ")
    normalized = re.sub(r"[^a-z0-9/+' -]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def build_fightfinder_url(*, search_text: str = "", association: str = "", page: int = 1, weightclass: str = "") -> str:
    search_text_value = _clean_text(search_text)
    association_value = _clean_text(association)
    weightclass_value = _clean_text(weightclass)
    return (
        f"{BASE_URL}/stats/fightfinder?"
        f"SearchTxt={quote_plus(search_text_value)}&association={quote_plus(association_value)}&page={int(page)}&weightclass={quote_plus(weightclass_value)}"
    )


def fetch_html(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def fetch_search_results(query: str, session: requests.Session | None = None) -> dict[str, Any]:
    client = session or requests.Session()
    response = client.get(
        SEARCH_API_URL,
        params={"q": _clean_text(query)},
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "X-Requested-With": "XMLHttpRequest",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def parse_search_results(payload: dict[str, Any]) -> list[dict[str, str]]:
    collection = payload.get("collection", [])
    if not isinstance(collection, list):
        return []

    results: list[dict[str, str]] = []
    for item in collection:
        if not isinstance(item, dict):
            continue
        if _clean_text(item.get("source", "")).lower() not in {"fighter", ""}:
            continue
        fighter_name = " ".join(
            part for part in [_clean_text(item.get("firstname", "")), _clean_text(item.get("lastname", ""))] if part
        )
        if not fighter_name:
            fighter_name = _clean_text(item.get("name", ""))
        if not fighter_name:
            continue

        association = item.get("association", "")
        if isinstance(association, list):
            gym_name = _clean_text(association[0]) if association else ""
        else:
            gym_name = _clean_text(association)

        url = _clean_text(item.get("url", ""))
        sherdog_url = urljoin(BASE_URL, url) if url else ""
        results.append(
            {
                "fighter_name": fighter_name,
                "fighter_name_normalized": normalize_name(fighter_name),
                "gym_source": "sherdog_search",
                "gym_name": gym_name,
                "gym_name_normalized": normalize_gym_name(gym_name),
                "sherdog_url": sherdog_url,
            }
        )
    return results


def parse_fightfinder_results_page(html: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    fighter_rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        header_cells = table.find_all("th")
        normalized_columns = [_normalize_table_column(cell.get_text(" ", strip=True)) for cell in header_cells]
        if "fighter" not in normalized_columns or "association" not in normalized_columns:
            continue
        fighter_index = normalized_columns.index("fighter")
        association_index = normalized_columns.index("association")
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue
            values = [" ".join(cell.get_text(" ", strip=True).split()) for cell in cells]
            if fighter_index >= len(values) or association_index >= len(values):
                continue
            fighter_name = values[fighter_index]
            association = values[association_index]
            if not fighter_name:
                continue
            fighter_cell = cells[fighter_index]
            fighter_anchor = fighter_cell.find("a", href=True)
            sherdog_url = (
                urljoin(BASE_URL, str(fighter_anchor["href"]).strip())
                if fighter_anchor is not None and "/fighter/" in str(fighter_anchor.get("href", ""))
                else ""
            )
            fighter_rows.append(
                {
                    "fighter_name": fighter_name,
                    "fighter_name_normalized": normalize_name(fighter_name),
                    "gym_name": association,
                    "gym_name_normalized": normalize_gym_name(association),
                    "sherdog_url": sherdog_url,
                }
            )
        if fighter_rows:
            break
    if not fighter_rows:
        return []

    link_map: dict[str, list[str]] = {}
    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href", "")).strip()
        text = " ".join(anchor.get_text(" ", strip=True).split())
        if not href or "/fighter/" not in href or not text:
            continue
        link_map.setdefault(normalize_name(text), []).append(urljoin(BASE_URL, href))

    used_counts: dict[str, int] = {}
    results: list[dict[str, str]] = []
    for row in fighter_rows:
        fighter_name = row["fighter_name"]
        normalized_fighter_name = row["fighter_name_normalized"]
        association = row["gym_name"]
        urls = link_map.get(normalized_fighter_name, [])
        url_index = used_counts.get(normalized_fighter_name, 0)
        sherdog_url = row["sherdog_url"] or (urls[url_index] if url_index < len(urls) else "")
        used_counts[normalized_fighter_name] = url_index + 1
        results.append(
            {
                "fighter_name": fighter_name,
                "fighter_name_normalized": normalized_fighter_name,
                "gym_name": association,
                "gym_name_normalized": normalize_gym_name(association),
                "sherdog_url": sherdog_url,
            }
        )
    return results


def parse_fighter_profile(html: str, source_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    lines = [" ".join(line.split()) for line in soup.get_text("\n", strip=True).splitlines() if " ".join(line.split())]
    fighter_name = ""
    heading = soup.find("h1")
    if heading is not None:
        fighter_name = " ".join(heading.get_text(" ", strip=True).split())
    if not fighter_name:
        fighter_name = _line_after_label(lines, "#", fallback="")

    association = _line_after_label(lines, "ASSOCIATION", fallback="")
    weight_class = _line_after_label(lines, "CLASS", fallback="")
    association_anchor = soup.find(
        "a",
        href=lambda href: isinstance(href, str) and "/stats/fightfinder?association=" in href,
        string=lambda text: normalize_gym_name(text) == normalize_gym_name(association) if text and association else False,
    )
    association_url = urljoin(BASE_URL, association_anchor["href"]) if association_anchor and association_anchor.get("href") else build_fightfinder_url(association=association)

    raw_text = soup.get_text(" ", strip=True)
    wins = _extract_record_value(raw_text, "Wins")
    draws = _extract_record_value(raw_text, "Draws")
    losses = _extract_record_value(raw_text, "Losses")
    total_fights = wins + losses + draws
    fighter_win_rate = round(wins / total_fights, 4) if total_fights > 0 else 0.0
    fighter_elite_flag = int(total_fights >= 12 and wins >= 10 and fighter_win_rate >= 0.70)

    return {
        "fighter_name": fighter_name,
        "fighter_name_normalized": normalize_name(fighter_name),
        "sherdog_url": source_url,
        "gym_source": "sherdog_profile",
        "gym_name": association,
        "gym_name_normalized": normalize_gym_name(association),
        "gym_page_url": association_url,
        "weight_class": weight_class,
        "fighter_wins": wins,
        "fighter_losses": losses,
        "fighter_draws": draws,
        "fighter_win_rate": fighter_win_rate,
        "fighter_elite_flag": fighter_elite_flag,
    }


def search_fighter_profiles(fighter_name: str, session: requests.Session | None = None) -> list[dict[str, str]]:
    try:
        payload = fetch_search_results(fighter_name, session=session)
    except (requests.RequestException, ValueError):
        payload = {}

    results = parse_search_results(payload)
    if results:
        return results

    html = fetch_html(build_fightfinder_url(search_text=fighter_name), session=session)
    return parse_fightfinder_results_page(html)


def select_best_search_result(fighter_name: str, results: list[dict[str, str]]) -> dict[str, str] | None:
    if not results:
        return None
    normalized_target = normalize_name(fighter_name)
    exact_matches = [row for row in results if row.get("fighter_name_normalized") == normalized_target]
    if exact_matches:
        return exact_matches[0]

    def score(result: dict[str, str]) -> tuple[int, int]:
        candidate = result.get("fighter_name_normalized", "")
        target_parts = normalized_target.split()
        candidate_parts = candidate.split()
        shared_tokens = len(set(target_parts) & set(candidate_parts))
        return (shared_tokens, -abs(len(candidate_parts) - len(target_parts)))

    return max(results, key=score, default=None)


def fetch_association_roster(
    association_name: str,
    *,
    session: requests.Session | None = None,
    max_pages: int = 25,
) -> list[dict[str, str]]:
    roster: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    normalized_association = normalize_gym_name(association_name)

    for page in range(1, max_pages + 1):
        html = fetch_html(build_fightfinder_url(association=association_name, page=page), session=session)
        results = parse_fightfinder_results_page(html)
        page_rows = [
            row
            for row in results
            if row.get("gym_name_normalized") == normalized_association
        ]
        if not page_rows:
            break

        new_rows = [row for row in page_rows if row.get("sherdog_url") and row["sherdog_url"] not in seen_urls]
        if not new_rows:
            break
        roster.extend(new_rows)
        seen_urls.update(row["sherdog_url"] for row in new_rows if row.get("sherdog_url"))
    return roster


def load_fighter_gym_cache(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    cache_path = Path(path)
    if not cache_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(cache_path)
    if "fighter_name" in frame.columns:
        frame["fighter_name"] = frame["fighter_name"].astype(str).str.strip()
        frame["fighter_name_normalized"] = frame["fighter_name"].map(normalize_name)
    return frame


def merge_fighter_gym_data(frame: pd.DataFrame, fighter_gyms_path: str | Path | None) -> pd.DataFrame:
    if frame.empty or not fighter_gyms_path:
        return frame.copy()
    fighter_gyms = load_fighter_gym_cache(fighter_gyms_path)
    if fighter_gyms.empty:
        return frame.copy()
    if "fighter_name" not in fighter_gyms.columns:
        raise ValueError("fighter gyms CSV must contain fighter_name")

    merged = frame.merge(fighter_gyms, on="fighter_name", how="left", suffixes=("", "_gym"))
    for column in FIGHTER_GYM_NUMERIC_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    for column in FIGHTER_GYM_STRING_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("").astype(str)

    if "new_gym_flag" in merged.columns:
        merged["new_gym_flag"] = pd.to_numeric(merged["new_gym_flag"], errors="coerce").fillna(0).astype(int)
        merged["new_gym_flag"] = merged[["new_gym_flag", "gym_changed_flag"]].max(axis=1).astype(int)
    if "camp_change_flag" in merged.columns:
        merged["camp_change_flag"] = pd.to_numeric(merged["camp_change_flag"], errors="coerce").fillna(0).astype(int)
        merged["camp_change_flag"] = merged[["camp_change_flag", "gym_changed_flag"]].max(axis=1).astype(int)
    return merged


def refresh_fighter_gym_data(
    fighter_names: list[str],
    *,
    cache_frame: pd.DataFrame | None = None,
    session: requests.Session | None = None,
    refresh_days: int = 7,
    association_refresh_days: int = 30,
    expand_associations: bool = True,
    max_association_pages: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = cache_frame.copy() if cache_frame is not None else pd.DataFrame()
    if not cache.empty and "fighter_name_normalized" not in cache.columns:
        cache["fighter_name_normalized"] = cache["fighter_name"].map(normalize_name)

    client = session or requests.Session()
    now = datetime.now(UTC).replace(microsecond=0).isoformat()
    refreshed_rows: list[dict[str, Any]] = []
    refreshed_names: set[str] = set()

    for fighter_name in fighter_names:
        normalized_fighter_name = normalize_name(fighter_name)
        existing = _existing_cache_row(cache, normalized_fighter_name)
        if _is_cache_fresh(existing, refresh_days):
            reused = existing.to_dict()
            reused["fighter_name"] = fighter_name
            reused["fighter_name_normalized"] = normalized_fighter_name
            reused["last_seen_at"] = now
            refreshed_rows.append(reused)
            refreshed_names.add(normalized_fighter_name)
            continue

        resolved = _resolve_fighter_profile(fighter_name, session=client)
        if resolved is None:
            fallback = existing.to_dict() if existing is not None else {}
            fallback.update(
                {
                    "fighter_name": fighter_name,
                    "fighter_name_normalized": normalized_fighter_name,
                    "last_seen_at": now,
                    "profile_last_refreshed_at": fallback.get("profile_last_refreshed_at", ""),
                }
            )
            refreshed_rows.append(_finalize_fighter_row(fallback))
            refreshed_names.add(normalized_fighter_name)
            continue

        row = _profile_row_from_result(resolved["sherdog_url"], resolved["fighter_name"], client)
        row["fighter_name"] = fighter_name
        row["fighter_name_normalized"] = normalized_fighter_name
        row["last_seen_at"] = now
        row["profile_last_refreshed_at"] = now
        row = _apply_change_tracking(row, existing, now)
        refreshed_rows.append(_finalize_fighter_row(row))
        refreshed_names.add(normalized_fighter_name)

    if expand_associations:
        association_rows = pd.DataFrame(refreshed_rows)
        for association_name in sorted({str(value).strip() for value in association_rows.get("gym_name", []) if str(value).strip()}):
            roster_entries = fetch_association_roster(
                association_name,
                session=client,
                max_pages=max_association_pages,
            )
            for roster_entry in roster_entries:
                normalized_fighter_name = roster_entry["fighter_name_normalized"]
                if normalized_fighter_name in refreshed_names:
                    continue
                existing = _existing_cache_row(cache, normalized_fighter_name)
                if _is_cache_fresh(existing, association_refresh_days):
                    continue
                if not roster_entry.get("sherdog_url"):
                    continue
                row = _profile_row_from_result(roster_entry["sherdog_url"], roster_entry["fighter_name"], client)
                row["profile_last_refreshed_at"] = now
                row["last_seen_at"] = existing.get("last_seen_at", "") if existing is not None else ""
                row = _apply_change_tracking(row, existing, now)
                refreshed_rows.append(_finalize_fighter_row(row))

    refreshed_frame = pd.DataFrame(refreshed_rows)
    updated_cache = _merge_cache_rows(cache, refreshed_frame)
    gym_registry = build_gym_registry(updated_cache)
    current_fighters = updated_cache.loc[
        updated_cache["fighter_name_normalized"].isin([normalize_name(name) for name in fighter_names])
    ].copy()
    current_fighters = current_fighters.merge(
        gym_registry[
            [
                "gym_name_normalized",
                "gym_name",
                "gym_tier",
                "gym_score",
                "gym_record",
                "gym_fighter_count",
                "gym_total_wins",
                "gym_total_losses",
                "gym_total_draws",
                "gym_win_rate",
                "gym_elite_fighter_count",
            ]
        ],
        on="gym_name_normalized",
        how="left",
        suffixes=("", "_registry"),
    )
    for column in ["gym_name", "gym_tier", "gym_record"]:
        if f"{column}_registry" in current_fighters.columns:
            current_fighters[column] = current_fighters[f"{column}_registry"].where(
                current_fighters[f"{column}_registry"].astype(str).str.strip() != "",
                current_fighters[column],
            )
            current_fighters = current_fighters.drop(columns=[f"{column}_registry"])
    for column in ["gym_score", "gym_fighter_count", "gym_total_wins", "gym_total_losses", "gym_total_draws", "gym_win_rate", "gym_elite_fighter_count"]:
        if f"{column}_registry" in current_fighters.columns:
            current_fighters[column] = pd.to_numeric(current_fighters[f"{column}_registry"], errors="coerce").fillna(
                pd.to_numeric(current_fighters[column], errors="coerce")
            )
            current_fighters = current_fighters.drop(columns=[f"{column}_registry"])
    ordered_columns = _ordered_fighter_gym_columns(current_fighters)
    return current_fighters[ordered_columns].sort_values(by=["fighter_name"]).reset_index(drop=True), updated_cache, gym_registry


def build_gym_registry(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "gym_name_normalized" not in frame.columns:
        return pd.DataFrame(
            columns=[
                "gym_name_normalized",
                "gym_name",
                "gym_page_url",
                "gym_fighter_count",
                "gym_total_wins",
                "gym_total_losses",
                "gym_total_draws",
                "gym_record",
                "gym_win_rate",
                "gym_elite_fighter_count",
                "gym_score",
                "gym_tier",
                "last_refreshed_at",
            ]
        )

    working = frame.copy()
    working["gym_name_normalized"] = working["gym_name_normalized"].map(normalize_gym_name)
    working = working.loc[working["gym_name_normalized"] != ""].copy()
    if working.empty:
        return pd.DataFrame(
            columns=[
                "gym_name_normalized",
                "gym_name",
                "gym_page_url",
                "gym_fighter_count",
                "gym_total_wins",
                "gym_total_losses",
                "gym_total_draws",
                "gym_record",
                "gym_win_rate",
                "gym_elite_fighter_count",
                "gym_score",
                "gym_tier",
                "last_refreshed_at",
            ]
        )

    grouped_rows: list[dict[str, Any]] = []
    for gym_key, gym_rows in working.groupby("gym_name_normalized", dropna=False):
        fighter_count = int(gym_rows["fighter_name_normalized"].nunique())
        total_wins = int(pd.to_numeric(gym_rows["fighter_wins"], errors="coerce").fillna(0).sum())
        total_losses = int(pd.to_numeric(gym_rows["fighter_losses"], errors="coerce").fillna(0).sum())
        total_draws = int(pd.to_numeric(gym_rows["fighter_draws"], errors="coerce").fillna(0).sum())
        total_fights = total_wins + total_losses + total_draws
        gym_win_rate = round(total_wins / total_fights, 4) if total_fights > 0 else 0.0
        elite_fighter_count = int(pd.to_numeric(gym_rows["fighter_elite_flag"], errors="coerce").fillna(0).sum())
        depth_score = _clamp(math.log1p(fighter_count) / math.log1p(75), 0.0, 1.0)
        elite_score = _clamp(elite_fighter_count / 8, 0.0, 1.0)
        experience_score = _clamp(
            (
                pd.to_numeric(gym_rows["fighter_wins"], errors="coerce").fillna(0)
                + pd.to_numeric(gym_rows["fighter_losses"], errors="coerce").fillna(0)
                + pd.to_numeric(gym_rows["fighter_draws"], errors="coerce").fillna(0)
            ).mean()
            / 20,
            0.0,
            1.0,
        )
        gym_score = round((gym_win_rate * 0.45) + (depth_score * 0.20) + (elite_score * 0.20) + (experience_score * 0.15), 4)
        grouped_rows.append(
            {
                "gym_name_normalized": gym_key,
                "gym_name": _preferred_label(gym_rows["gym_name"]),
                "gym_page_url": _preferred_label(gym_rows["gym_page_url"]),
                "gym_fighter_count": fighter_count,
                "gym_total_wins": total_wins,
                "gym_total_losses": total_losses,
                "gym_total_draws": total_draws,
                "gym_record": f"{total_wins}-{total_losses}-{total_draws}",
                "gym_win_rate": gym_win_rate,
                "gym_elite_fighter_count": elite_fighter_count,
                "gym_score": gym_score,
                "gym_tier": _gym_tier(gym_score),
                "last_refreshed_at": _preferred_label(gym_rows["profile_last_refreshed_at"]),
            }
        )

    return pd.DataFrame(grouped_rows).sort_values(by=["gym_score", "gym_fighter_count"], ascending=[False, False]).reset_index(drop=True)


def write_fighter_gym_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_gym_registry_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _existing_cache_row(cache: pd.DataFrame, fighter_name_normalized: str) -> pd.Series | None:
    if cache.empty or "fighter_name_normalized" not in cache.columns:
        return None
    matches = cache.loc[cache["fighter_name_normalized"] == fighter_name_normalized]
    if matches.empty:
        return None
    return matches.iloc[0]


def _is_cache_fresh(row: pd.Series | None, refresh_days: int) -> bool:
    if row is None:
        return False
    refreshed_at = _clean_text(row.get("profile_last_refreshed_at", ""))
    if not refreshed_at:
        return False
    refreshed_timestamp = pd.to_datetime(refreshed_at, errors="coerce", utc=True)
    if pd.isna(refreshed_timestamp):
        return False
    age_days = (pd.Timestamp.now(tz=UTC) - refreshed_timestamp).days
    return age_days < refresh_days


def _resolve_fighter_profile(fighter_name: str, session: requests.Session) -> dict[str, str] | None:
    results = search_fighter_profiles(fighter_name, session=session)
    return select_best_search_result(fighter_name, results)


def _profile_row_from_result(sherdog_url: str, fallback_name: str, session: requests.Session) -> dict[str, Any]:
    profile_html = fetch_html(sherdog_url, session=session)
    parsed = parse_fighter_profile(profile_html, sherdog_url)
    if not parsed.get("fighter_name"):
        parsed["fighter_name"] = fallback_name
        parsed["fighter_name_normalized"] = normalize_name(fallback_name)
    return parsed


def _apply_change_tracking(row: dict[str, Any], existing: pd.Series | None, now: str) -> dict[str, Any]:
    previous_gym_name = ""
    last_changed_at = ""
    gym_changed_flag = 0
    if existing is not None:
        previous_gym_name = _clean_text(existing.get("previous_gym_name", ""))
        last_changed_at = _clean_text(existing.get("last_changed_at", ""))
        prior_gym_name = _clean_text(existing.get("gym_name", ""))
        if prior_gym_name and normalize_gym_name(prior_gym_name) != normalize_gym_name(row.get("gym_name", "")):
            previous_gym_name = prior_gym_name
            last_changed_at = now
            gym_changed_flag = 1
    row["previous_gym_name"] = previous_gym_name
    row["last_changed_at"] = last_changed_at
    row["gym_changed_flag"] = gym_changed_flag
    return row


def _finalize_fighter_row(row: dict[str, Any]) -> dict[str, Any]:
    fighter_wins = _coerce_int(row.get("fighter_wins", 0), 0)
    fighter_losses = _coerce_int(row.get("fighter_losses", 0), 0)
    fighter_draws = _coerce_int(row.get("fighter_draws", 0), 0)
    total_fights = fighter_wins + fighter_losses + fighter_draws
    fighter_win_rate = round(fighter_wins / total_fights, 4) if total_fights > 0 else 0.0
    existing_elite_flag = _coerce_int(row.get("fighter_elite_flag", 0), 0)
    fighter_elite_flag = int(existing_elite_flag or (total_fights >= 12 and fighter_wins >= 10 and fighter_win_rate >= 0.70))
    row["fighter_wins"] = fighter_wins
    row["fighter_losses"] = fighter_losses
    row["fighter_draws"] = fighter_draws
    row["fighter_win_rate"] = fighter_win_rate
    row["fighter_elite_flag"] = fighter_elite_flag
    gym_name = _clean_text(row.get("gym_name", ""))
    row["gym_source"] = _clean_text(row.get("gym_source", ""))
    row["gym_name"] = gym_name
    row["gym_name_normalized"] = normalize_gym_name(gym_name)
    gym_page_url = _clean_text(row.get("gym_page_url", ""))
    row["gym_page_url"] = gym_page_url or (build_fightfinder_url(association=gym_name) if gym_name else "")
    row.setdefault("gym_tier", "")
    row.setdefault("gym_score", 0.0)
    row.setdefault("gym_record", "")
    row.setdefault("gym_fighter_count", 0)
    row.setdefault("gym_total_wins", 0)
    row.setdefault("gym_total_losses", 0)
    row.setdefault("gym_total_draws", 0)
    row.setdefault("gym_win_rate", 0.0)
    row.setdefault("gym_elite_fighter_count", 0)
    return row


def _merge_cache_rows(cache: pd.DataFrame, refreshed_frame: pd.DataFrame) -> pd.DataFrame:
    if cache.empty:
        merged = refreshed_frame.copy()
    else:
        merged = cache.copy()
        if not refreshed_frame.empty:
            merged = merged.loc[
                ~merged["fighter_name_normalized"].isin(refreshed_frame["fighter_name_normalized"])
            ].copy()
            merged = pd.concat([merged, refreshed_frame], ignore_index=True)
    ordered_columns = _ordered_fighter_gym_columns(merged)
    return merged[ordered_columns].sort_values(by=["fighter_name"]).reset_index(drop=True)


def _ordered_fighter_gym_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "fighter_name",
        "fighter_name_normalized",
        "sherdog_url",
        "gym_source",
        "gym_name",
        "gym_name_normalized",
        "gym_page_url",
        "gym_tier",
        "gym_score",
        "gym_record",
        "gym_fighter_count",
        "gym_total_wins",
        "gym_total_losses",
        "gym_total_draws",
        "gym_win_rate",
        "gym_elite_fighter_count",
        "weight_class",
        "fighter_wins",
        "fighter_losses",
        "fighter_draws",
        "fighter_win_rate",
        "fighter_elite_flag",
        "gym_changed_flag",
        "previous_gym_name",
        "last_changed_at",
        "profile_last_refreshed_at",
        "last_seen_at",
    ]
    return [column for column in preferred if column in frame.columns] + [column for column in frame.columns if column not in preferred]


def _normalize_table_column(column: object) -> str:
    if isinstance(column, tuple):
        value = " ".join(str(part) for part in column if str(part).strip())
    else:
        value = str(column)
    return " ".join(value.strip().lower().split())


def _preferred_label(series: pd.Series) -> str:
    values = [str(value).strip() for value in series if str(value).strip() and str(value).strip().lower() != "nan"]
    if not values:
        return ""
    return max(set(values), key=values.count)


def _line_after_label(lines: list[str], label: str, *, fallback: str = "") -> str:
    for index, line in enumerate(lines):
        if line == label and index + 1 < len(lines):
            return lines[index + 1]
    return fallback


def _extract_record_value(raw_text: str, label: str) -> int:
    match = re.search(rf"\b{re.escape(label)}\s+(\d+)\b", raw_text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else 0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _gym_tier(score: float) -> str:
    if score >= 0.85:
        return "S"
    if score >= 0.72:
        return "A"
    if score >= 0.60:
        return "B"
    if score >= 0.48:
        return "C"
    return "D"
