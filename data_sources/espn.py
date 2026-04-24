from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re
import unicodedata

import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = "ufc-bot/1.0 (+https://www.espn.com/mma/)"
REQUEST_TIMEOUT_SECONDS = 30
UTC = timezone.utc
SEARCH_URL = "https://site.web.api.espn.com/apis/common/v3/search"
FIGHTER_MAP_COLUMNS = ["fighter_name", "espn_url"]


def load_espn_fighter_map(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"fighter_name", "espn_url"}
    missing = required - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required mapping columns: {missing_list}")
    normalized = frame.copy()
    normalized["fighter_name"] = normalized["fighter_name"].fillna("").astype(str).str.strip()
    normalized["espn_url"] = normalized["espn_url"].fillna("").astype(str).str.strip()
    for column in [
        "short_notice_flag",
        "short_notice_acceptance_flag",
        "short_notice_success_flag",
        "new_gym_flag",
        "new_contract_flag",
        "cardio_fade_flag",
        "injury_concern_flag",
        "weight_cut_concern_flag",
        "replacement_fighter_flag",
        "travel_disadvantage_flag",
        "camp_change_flag",
    ]:
        if column not in normalized.columns:
            normalized[column] = 0
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)
    if "context_notes" not in normalized.columns:
        normalized["context_notes"] = ""
    return normalized


def merge_context_into_fighter_map(
    mapping: pd.DataFrame,
    context_path: str | Path | None,
) -> pd.DataFrame:
    if not context_path:
        return mapping
    context = pd.read_csv(context_path)
    if "fighter_name" not in context.columns:
        raise ValueError("Context CSV must contain fighter_name")
    normalized_context = context.copy()
    normalized_context["fighter_name"] = normalized_context["fighter_name"].astype(str).str.strip()
    merged = mapping.merge(normalized_context, on="fighter_name", how="left", suffixes=("", "_context"))
    for column in [
        "short_notice_flag",
        "short_notice_acceptance_flag",
        "short_notice_success_flag",
        "new_gym_flag",
        "new_contract_flag",
        "cardio_fade_flag",
        "injury_concern_flag",
        "weight_cut_concern_flag",
        "replacement_fighter_flag",
        "travel_disadvantage_flag",
        "camp_change_flag",
    ]:
        source_column = f"{column}_context" if f"{column}_context" in merged.columns else column
        merged[column] = pd.to_numeric(merged[source_column], errors="coerce").fillna(merged[column]).fillna(0).astype(int)
        if source_column != column and source_column in merged.columns:
            merged = merged.drop(columns=[source_column])
    if "context_notes_context" in merged.columns:
        merged["context_notes"] = merged["context_notes_context"].fillna("")
        merged = merged.drop(columns=["context_notes_context"])
    elif "context_notes" not in merged.columns:
        merged["context_notes"] = ""
    return merged


def fetch_html(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def merge_espn_url_maps(*frames: pd.DataFrame | None) -> pd.DataFrame:
    merged_urls: dict[str, str] = {}
    fighter_order: list[str] = []

    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "fighter_name" not in frame.columns:
            continue

        normalized = frame.copy()
        normalized["fighter_name"] = normalized["fighter_name"].fillna("").astype(str).str.strip()
        if "espn_url" not in normalized.columns:
            normalized["espn_url"] = ""
        normalized["espn_url"] = normalized["espn_url"].fillna("").astype(str).str.strip()

        for row in normalized.itertuples(index=False):
            fighter_name = str(getattr(row, "fighter_name", "") or "").strip()
            if not fighter_name:
                continue
            if fighter_name not in merged_urls:
                fighter_order.append(fighter_name)
                merged_urls[fighter_name] = ""
            espn_url = str(getattr(row, "espn_url", "") or "").strip()
            if espn_url:
                merged_urls[fighter_name] = espn_url

    return pd.DataFrame(
        [{"fighter_name": fighter_name, "espn_url": merged_urls[fighter_name]} for fighter_name in fighter_order],
        columns=FIGHTER_MAP_COLUMNS,
    )


def search_espn_fighters(
    query: str,
    *,
    session: requests.Session | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    cleaned_query = str(query or "").strip()
    if not cleaned_query:
        return []

    client = session or requests.Session()
    response = client.get(
        SEARCH_URL,
        params={
            "region": "us",
            "lang": "en",
            "query": cleaned_query,
            "limit": int(limit),
            "mode": "prefix",
            "type": "player",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    items = payload.get("items")
    return items if isinstance(items, list) else []


def resolve_espn_fighter_url(
    fighter_name: str,
    *,
    session: requests.Session | None = None,
) -> str:
    requested_name = str(fighter_name or "").strip()
    if not requested_name:
        return ""

    requested_normalized = _normalize_person_name(requested_name)
    requested_tokens = requested_normalized.split()
    requested_last_name = requested_tokens[-1] if requested_tokens else ""
    client = session or requests.Session()

    best_url = ""
    best_score = float("-inf")
    fallback_queries = _candidate_search_queries(requested_name)

    for query_index, query in enumerate(fallback_queries):
        items = search_espn_fighters(query, session=client)
        mma_candidates: list[tuple[str, str]] = []

        for item in items:
            if str(item.get("sport", "")).strip().lower() != "mma":
                continue
            display_name = str(item.get("displayName", "") or "").strip()
            overview_url = _extract_espn_overview_url(item)
            if not display_name or not overview_url:
                continue
            candidate_normalized = _normalize_person_name(display_name)
            if candidate_normalized == requested_normalized:
                return overview_url
            mma_candidates.append((display_name, overview_url))

        if len(mma_candidates) == 1:
            candidate_name, candidate_url = mma_candidates[0]
            candidate_tokens = _normalize_person_name(candidate_name).split()
            candidate_last_name = candidate_tokens[-1] if candidate_tokens else ""
            if requested_last_name and candidate_last_name == requested_last_name:
                return candidate_url

        for candidate_name, candidate_url in mma_candidates:
            score = _espn_candidate_score(
                requested_name=requested_name,
                candidate_name=candidate_name,
                result_count=len(mma_candidates),
                query_index=query_index,
            )
            if score > best_score:
                best_score = score
                best_url = candidate_url

    return best_url if best_score >= 60.0 else ""


def resolve_espn_fighter_urls(
    fighter_names: list[str],
    *,
    session: requests.Session | None = None,
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    client = session or requests.Session()
    for fighter_name in fighter_names:
        cleaned_name = str(fighter_name or "").strip()
        if not cleaned_name:
            continue
        url = resolve_espn_fighter_url(cleaned_name, session=client)
        if url:
            resolved[cleaned_name] = url
    return resolved


def normalize_espn_fighter_url(url: str, page: str = "overview") -> str:
    cleaned = url.strip().replace("/fighter//", "/fighter/_/")
    if "/fighter/stats/" in cleaned:
        cleaned = cleaned.replace("/fighter/stats/", "/fighter/_/")
    elif "/fighter/bio/" in cleaned:
        cleaned = cleaned.replace("/fighter/bio/", "/fighter/_/")

    if page == "overview":
        return cleaned
    if page == "stats":
        return cleaned.replace("/fighter/_/", "/fighter/stats/_/")
    if page == "bio":
        return cleaned.replace("/fighter/_/", "/fighter/bio/_/")
    raise ValueError(f"Unsupported ESPN page type: {page}")


def scrape_fighter_stats_from_map(
    mapping: pd.DataFrame,
    *,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    client = session or requests.Session()
    fighters: list[dict[str, Any]] = []

    for row in mapping.itertuples(index=False):
        overview_url = normalize_espn_fighter_url(row.espn_url, page="overview")
        stats_url = normalize_espn_fighter_url(row.espn_url, page="stats")
        bio_url = normalize_espn_fighter_url(row.espn_url, page="bio")

        overview_html = fetch_html(overview_url, session=client)
        stats_html = fetch_html(stats_url, session=client)
        bio_html = fetch_html(bio_url, session=client)

        fighter = parse_espn_fighter_pages(
            overview_html=overview_html,
            stats_html=stats_html,
            bio_html=bio_html,
            source_url=overview_url,
        )
        fighter["fighter_name"] = row.fighter_name
        fighter["short_notice_flag"] = int(getattr(row, "short_notice_flag", 0))
        fighter["short_notice_acceptance_flag"] = int(getattr(row, "short_notice_acceptance_flag", 0))
        fighter["short_notice_success_flag"] = int(getattr(row, "short_notice_success_flag", 0))
        fighter["new_gym_flag"] = int(getattr(row, "new_gym_flag", 0))
        fighter["new_contract_flag"] = int(getattr(row, "new_contract_flag", 0))
        fighter["cardio_fade_flag"] = int(getattr(row, "cardio_fade_flag", 0))
        fighter["injury_concern_flag"] = int(getattr(row, "injury_concern_flag", 0))
        fighter["weight_cut_concern_flag"] = int(getattr(row, "weight_cut_concern_flag", 0))
        fighter["replacement_fighter_flag"] = int(getattr(row, "replacement_fighter_flag", 0))
        fighter["travel_disadvantage_flag"] = int(getattr(row, "travel_disadvantage_flag", 0))
        fighter["camp_change_flag"] = int(getattr(row, "camp_change_flag", 0))
        fighter["news_alert_count"] = int(pd.to_numeric(pd.Series([getattr(row, "news_alert_count", 0)]), errors="coerce").fillna(0).iloc[0])
        fighter["news_radar_score"] = float(pd.to_numeric(pd.Series([getattr(row, "news_radar_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        fighter["news_radar_label"] = str(getattr(row, "news_radar_label", "") or "")
        fighter["news_radar_summary"] = str(getattr(row, "news_radar_summary", "") or "")
        fighter["context_notes"] = str(getattr(row, "context_notes", "") or "")
        fighters.append(fighter)

    return pd.DataFrame(fighters)


def parse_espn_fighter_pages(
    *,
    overview_html: str,
    stats_html: str,
    bio_html: str,
    source_url: str,
) -> dict[str, Any]:
    bio_fields = parse_bio_page(bio_html)
    fight_history = parse_fight_history(overview_html)
    sig_landed_per_min = 0.0
    sig_absorbed_per_min = 0.0
    takedown_avg = 0.0
    takedown_accuracy_pct = 0.0
    data_notes = ""
    fallback_used = 0
    stats_completeness = 1.0
    recent_strike_margin_per_min = 0.0
    recent_grappling_rate = 0.0

    try:
        striking = parse_stats_tables(stats_html)
        merged = striking.merge(
            fight_history[["date", "minutes"]].drop_duplicates(subset=["date"], keep="first"),
            on="date",
            how="left",
        )
        merged = merged.dropna(subset=["minutes"])

        total_minutes = merged["minutes"].sum()
        if total_minutes <= 0:
            raise ValueError("Could not compute fight durations from ESPN fight history")

        sig_landed_per_min = round(merged["SSL"].sum() / total_minutes, 3)
        sig_absorbed_per_min = round(merged["SSA"].sum() / total_minutes, 3)
        takedown_avg = round((merged["TDL"].sum() / total_minutes) * 15, 3)
        recent_slice = merged.head(3)
        recent_minutes = recent_slice["minutes"].sum()
        if recent_minutes > 0:
            recent_strike_margin_per_min = round(
                (recent_slice["SSL"].sum() - recent_slice["SSA"].sum()) / recent_minutes,
                3,
            )
            recent_grappling_rate = round((recent_slice["TDL"].sum() / recent_minutes) * 15, 3)

        valid_attempts = merged["TDA"].sum()
        if valid_attempts > 0:
            takedown_accuracy_pct = round((merged["TDL"].sum() / valid_attempts) * 100, 2)
        data_notes = "ESPN does not expose takedown defense directly; takedown_accuracy_pct is used as a proxy."
    except ValueError as exc:
        fallback_used = 1
        stats_completeness = 0.55
        data_notes = (
            "Detailed ESPN rate stats were unavailable or incomplete for this fighter; "
            "rate-based columns were defaulted to 0. "
            f"Parser note: {exc}"
        )

    last_fight_date = _last_fight_date(fight_history)
    recent_result_score = _recent_result_score(fight_history)
    losses_in_row = _loss_streak(fight_history)
    ufc_fight_count = _ufc_fight_count(fight_history)
    first_round_finish_wins, first_round_finish_rate = _first_round_finish_metrics(fight_history)
    (
        finish_win_rate,
        finish_loss_rate,
        decision_rate,
        ko_win_rate,
        submission_win_rate,
        ko_loss_rate,
        submission_loss_rate,
    ) = _outcome_profile_metrics(fight_history)
    (
        recent_finish_loss_count,
        recent_ko_loss_count,
        recent_finish_loss_365d,
        recent_ko_loss_365d,
        recent_damage_score,
    ) = _recent_damage_metrics(fight_history)
    career_total = bio_fields["wins"] + bio_fields["losses"] + bio_fields["draws"]
    if career_total == 0:
        stats_completeness = min(stats_completeness, 0.4)

    return {
        "fighter_name": bio_fields["fighter_name"],
        "wins": bio_fields["wins"],
        "losses": bio_fields["losses"],
        "draws": bio_fields["draws"],
        "height_in": bio_fields["height_in"],
        "reach_in": bio_fields["reach_in"],
        "age_years": bio_fields["age_years"],
        "sig_strikes_landed_per_min": sig_landed_per_min,
        "sig_strikes_absorbed_per_min": sig_absorbed_per_min,
        "takedown_avg": takedown_avg,
        "takedown_defense_pct": takedown_accuracy_pct,
        "recent_strike_margin_per_min": recent_strike_margin_per_min,
        "recent_grappling_rate": recent_grappling_rate,
        "recent_result_score": recent_result_score,
        "losses_in_row": losses_in_row,
        "first_round_finish_wins": first_round_finish_wins,
        "first_round_finish_rate": first_round_finish_rate,
        "finish_win_rate": finish_win_rate,
        "finish_loss_rate": finish_loss_rate,
        "decision_rate": decision_rate,
        "ko_win_rate": ko_win_rate,
        "submission_win_rate": submission_win_rate,
        "ko_loss_rate": ko_loss_rate,
        "submission_loss_rate": submission_loss_rate,
        "recent_finish_loss_count": recent_finish_loss_count,
        "recent_ko_loss_count": recent_ko_loss_count,
        "recent_finish_loss_365d": recent_finish_loss_365d,
        "recent_ko_loss_365d": recent_ko_loss_365d,
        "recent_damage_score": recent_damage_score,
        "days_since_last_fight": _days_since(last_fight_date),
        "ufc_fight_count": ufc_fight_count,
        "ufc_debut_flag": 1 if ufc_fight_count == 0 else 0,
        "stats_completeness": round(stats_completeness, 2),
        "fallback_used": fallback_used,
        "stance": bio_fields["stance"],
        "weight_class": bio_fields["weight_class"],
        "source_url": source_url,
        "data_notes": data_notes,
    }


def _candidate_search_queries(fighter_name: str) -> list[str]:
    normalized_name = _normalize_person_name(fighter_name)
    tokens = normalized_name.split()
    if not tokens:
        return []

    queries: list[str] = [" ".join(tokens)]
    if len(tokens) >= 3:
        queries.append(" ".join(tokens[-2:]))
    if len(tokens[-1]) >= 6:
        queries.append(tokens[-1])

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        cleaned_query = " ".join(str(query or "").split())
        if cleaned_query and cleaned_query not in seen:
            deduped.append(cleaned_query)
            seen.add(cleaned_query)
    return deduped


def _extract_espn_overview_url(item: dict[str, Any]) -> str:
    links = item.get("links")
    if not isinstance(links, list):
        return ""

    fallback_url = ""
    for link in links:
        rel = link.get("rel")
        href = str(link.get("href", "") or "").strip()
        if not href or not isinstance(rel, list):
            continue
        if "athlete" not in rel:
            continue
        if "playercard" in rel or "overview" in rel:
            return normalize_espn_fighter_url(href, page="overview")
        if "/mma/fighter/" in href and not fallback_url:
            fallback_url = normalize_espn_fighter_url(href, page="overview")
    return fallback_url


def _espn_candidate_score(
    *,
    requested_name: str,
    candidate_name: str,
    result_count: int,
    query_index: int,
) -> float:
    requested_tokens = _normalize_person_name(requested_name).split()
    candidate_tokens = _normalize_person_name(candidate_name).split()
    if not requested_tokens or not candidate_tokens:
        return float("-inf")

    requested_last_name = requested_tokens[-1]
    candidate_last_name = candidate_tokens[-1]
    token_overlap = len(set(requested_tokens) & set(candidate_tokens))
    score = 0.0
    if requested_last_name == candidate_last_name:
        score += 40.0
    score += 20.0 * token_overlap
    if len(requested_tokens) >= 2 and set(requested_tokens[-2:]).issubset(candidate_tokens):
        score += 15.0
    if result_count == 1:
        score += 10.0
    score -= float(query_index) * 5.0
    return score


def _normalize_person_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_only.lower()).strip()


def parse_bio_page(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")
    fighter_name = _first_text(soup.select_one("h1"))
    if not fighter_name:
        raise ValueError("Could not locate fighter name on ESPN page")

    raw_text = soup.get_text("\n", strip=True)
    wins, losses, draws = _extract_record(raw_text)
    height_in = _extract_inches(raw_text, "HT/WT")
    reach_in = _extract_simple_inches(raw_text, "Reach")
    stance = _extract_field_after_label(raw_text, "Stance")
    weight_class = _extract_weight_class(soup, raw_text)
    birthdate = _extract_birthdate(raw_text)

    return {
        "fighter_name": fighter_name,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "height_in": height_in,
        "reach_in": reach_in,
        "age_years": _age_years_from_birthdate(birthdate),
        "stance": stance,
        "weight_class": weight_class,
    }


def parse_stats_tables(html: str) -> pd.DataFrame:
    tables = _parse_html_tables(html)
    if len(tables) < 2:
        raise ValueError("Expected striking and clinch tables on ESPN stats page")

    striking = _normalize_columns(tables[0]).rename(columns={"res.": "result"})
    clinch = _normalize_columns(tables[1]).rename(columns={"res.": "result"})

    required_striking = {"date", "opponent", "event", "result", "ssl", "ssa"}
    required_clinch = {"date", "opponent", "event", "result", "tdl", "tda"}
    if not required_striking.issubset(striking.columns):
        raise ValueError("Missing expected striking columns on ESPN stats page")
    if not required_clinch.issubset(clinch.columns):
        raise ValueError("Missing expected clinch columns on ESPN stats page")

    merged = striking[list(required_striking)].merge(
        clinch[list(required_clinch)],
        on=["date", "opponent", "event", "result"],
        how="outer",
    )
    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["SSL"] = pd.to_numeric(merged["ssl"], errors="coerce").fillna(0)
    merged["SSA"] = pd.to_numeric(merged["ssa"], errors="coerce").fillna(0)
    merged["TDL"] = pd.to_numeric(merged["tdl"], errors="coerce").fillna(0)
    merged["TDA"] = pd.to_numeric(merged["tda"], errors="coerce").fillna(0)
    return merged.dropna(subset=["date"])[["date", "opponent", "event", "result", "SSL", "SSA", "TDL", "TDA"]]


def parse_fight_history(html: str) -> pd.DataFrame:
    tables = _parse_html_tables(html)
    for table in tables:
        normalized = _normalize_columns(table)
        if {"date", "opponent", "res.", "rnd", "time", "event"}.issubset(normalized.columns):
            history = normalized.rename(columns={"res.": "result"})
            history["result_code"] = history["result"].astype(str).str.strip().str.upper().str[0]
            history["date"] = pd.to_datetime(history["date"], errors="coerce")
            history["round_number"] = pd.to_numeric(history["rnd"], errors="coerce")
            if "decision" in history.columns:
                history["decision_type"] = history["decision"].astype(str).str.strip()
            else:
                history["decision_type"] = ""
            history["minutes"] = history.apply(
                lambda row: _fight_duration_minutes(row["rnd"], row["time"]),
                axis=1,
            )
            history = history.dropna(subset=["minutes", "date"]).reset_index(drop=True)
            return history[["date", "opponent", "event", "result", "result_code", "decision_type", "round_number", "minutes"]]
    raise ValueError("Could not locate fight history table on ESPN overview page")


def write_fighter_stats_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        " ".join(str(column).strip().lower().split())
        for column in normalized.columns
    ]
    return normalized


def _parse_html_tables(html: str) -> list[pd.DataFrame]:
    soup = BeautifulSoup(html, "html.parser")
    frames: list[pd.DataFrame] = []

    for table in soup.select("table"):
        rows = table.select("tr")
        if not rows:
            continue

        header_cells = rows[0].select("th, td")
        headers = [" ".join(cell.stripped_strings) for cell in header_cells]
        body_rows: list[list[str]] = []
        for row in rows[1:]:
            cells = row.select("td, th")
            values = [" ".join(cell.stripped_strings) for cell in cells]
            if values:
                body_rows.append(values)

        if headers and body_rows:
            frames.append(pd.DataFrame(body_rows, columns=headers))

    return frames


def _extract_record(raw_text: str) -> tuple[int, int, int]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == "W-L-D" and index + 1 < len(lines):
            parts = lines[index + 1].split("-")
            if len(parts) >= 2:
                wins = int(parts[0])
                losses = int(parts[1])
                draws = int(parts[2]) if len(parts) > 2 else 0
                return wins, losses, draws
    raise ValueError("Could not locate W-L-D record on ESPN page")


def _extract_inches(raw_text: str, label: str) -> float | None:
    value = _extract_field_after_label(raw_text, label)
    if not value:
        return None
    height = value.split(",", 1)[0].strip()
    feet, _, inches = height.partition("'")
    if not feet or not inches:
        return None
    return (int(feet.strip()) * 12) + float(inches.replace('"', "").strip())


def _extract_simple_inches(raw_text: str, label: str) -> float | None:
    value = _extract_field_after_label(raw_text, label)
    if not value:
        return None
    return float(value.replace('"', "").strip())


def _extract_field_after_label(raw_text: str, label: str) -> str:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == label and index + 1 < len(lines):
            return lines[index + 1]
        if line.startswith(f"{label} "):
            return line.removeprefix(f"{label} ").strip()
    return ""


def _extract_weight_class(soup: BeautifulSoup, raw_text: str) -> str:
    items = [item.get_text(" ", strip=True) for item in soup.select("li")]
    for index, item in enumerate(items):
        if item in {"USA", "Brazil", "England"} and index + 1 < len(items):
            return items[index + 1]

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if line == "WT Class" and index + 1 < len(lines):
            return lines[index + 1]
    return ""


def _extract_birthdate(raw_text: str) -> str:
    return _extract_field_after_label(raw_text, "Birthdate")


def _age_years_from_birthdate(raw_value: str) -> float:
    cleaned = raw_value.split("(", 1)[0].strip()
    if not cleaned:
        return 0.0
    birthdate = pd.to_datetime(cleaned, errors="coerce")
    if pd.isna(birthdate):
        return 0.0
    today = datetime.now(UTC).date()
    delta_days = (today - birthdate.date()).days
    return round(delta_days / 365.25, 2)


def _last_fight_date(history: pd.DataFrame) -> pd.Timestamp | None:
    if history.empty:
        return None
    return history["date"].max()


def _days_since(last_fight_date: pd.Timestamp | None) -> int:
    if last_fight_date is None or pd.isna(last_fight_date):
        return 999
    today = datetime.now(UTC).date()
    return int((today - last_fight_date.date()).days)


def _recent_result_score(history: pd.DataFrame) -> float:
    if history.empty:
        return 0.0
    recent = history.head(3).copy()
    weights = [3, 2, 1][: len(recent)]
    mapped = recent["result"].astype(str).str.upper().map({"W": 1.0, "L": -1.0, "D": 0.0}).fillna(0.0)
    weighted = sum(value * weight for value, weight in zip(mapped.tolist(), weights))
    return round(weighted / sum(weights), 3)


def _loss_streak(history: pd.DataFrame) -> int:
    streak = 0
    for result in history["result"].astype(str).str.upper().tolist():
        if result.startswith("L"):
            streak += 1
        else:
            break
    return streak


def _ufc_fight_count(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    return int(history["event"].astype(str).str.contains("UFC", case=False, na=False).sum())


def _first_round_finish_metrics(history: pd.DataFrame) -> tuple[int, float]:
    if history.empty:
        return 0, 0.0

    first_round_wins = history.loc[
        (history["result_code"] == "W")
        & (pd.to_numeric(history["round_number"], errors="coerce") == 1)
    ].copy()
    if first_round_wins.empty:
        return 0, 0.0

    decision_text = first_round_wins["decision_type"].astype(str).str.upper()
    finish_wins = first_round_wins.loc[~decision_text.str.contains("DEC", na=False)]
    count = int(len(finish_wins))
    rate = round(count / len(history), 3)
    return count, rate


def _outcome_profile_metrics(history: pd.DataFrame) -> tuple[float, float, float, float, float, float, float]:
    if history.empty:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    decision_text = history["decision_type"].astype(str).str.upper()
    is_decision = decision_text.str.contains("DEC", na=False)
    is_submission = decision_text.str.contains("SUB", na=False)
    is_ko_tko = decision_text.str.contains("KO|TKO", na=False) & ~is_submission
    is_finish = ~is_decision
    total_fights = len(history)
    finish_wins = int(((history["result_code"] == "W") & is_finish).sum())
    finish_losses = int(((history["result_code"] == "L") & is_finish).sum())
    decisions = int(is_decision.sum())
    ko_wins = int(((history["result_code"] == "W") & is_ko_tko).sum())
    submission_wins = int(((history["result_code"] == "W") & is_submission).sum())
    ko_losses = int(((history["result_code"] == "L") & is_ko_tko).sum())
    submission_losses = int(((history["result_code"] == "L") & is_submission).sum())
    return (
        round(finish_wins / total_fights, 3),
        round(finish_losses / total_fights, 3),
        round(decisions / total_fights, 3),
        round(ko_wins / total_fights, 3),
        round(submission_wins / total_fights, 3),
        round(ko_losses / total_fights, 3),
        round(submission_losses / total_fights, 3),
    )


def _recent_damage_metrics(history: pd.DataFrame) -> tuple[int, int, int, int, float]:
    if history.empty:
        return 0, 0, 0, 0, 0.0

    decision_text = history["decision_type"].astype(str).str.upper()
    is_decision = decision_text.str.contains("DEC", na=False)
    is_submission = decision_text.str.contains("SUB", na=False)
    is_ko_tko = decision_text.str.contains("KO|TKO", na=False) & ~is_submission
    is_finish_loss = (history["result_code"] == "L") & ~is_decision
    is_ko_loss = (history["result_code"] == "L") & is_ko_tko

    recent = history.head(3)
    cutoff = pd.Timestamp(datetime.now(UTC).date()) - pd.Timedelta(days=365)
    recent_window = history["date"] >= cutoff

    recent_finish_loss_count = int(is_finish_loss.loc[recent.index].sum())
    recent_ko_loss_count = int(is_ko_loss.loc[recent.index].sum())
    recent_finish_loss_365d = int((is_finish_loss & recent_window).sum())
    recent_ko_loss_365d = int((is_ko_loss & recent_window).sum())
    recent_damage_score = round(
        (recent_finish_loss_count * 0.45)
        + (recent_ko_loss_count * 0.75)
        + (recent_finish_loss_365d * 0.25)
        + (recent_ko_loss_365d * 0.45),
        3,
    )
    return (
        recent_finish_loss_count,
        recent_ko_loss_count,
        recent_finish_loss_365d,
        recent_ko_loss_365d,
        recent_damage_score,
    )


def _fight_duration_minutes(round_value: Any, time_value: Any) -> float | None:
    raw_time = str(time_value).strip()
    if ":" not in raw_time:
        return None
    round_number = int(round_value)
    minutes_part, seconds_part = raw_time.split(":")
    elapsed = int(minutes_part) + (int(seconds_part) / 60)
    return ((round_number - 1) * 5) + elapsed


def _first_text(node: Any) -> str:
    if node is None:
        return ""
    return str(node.get_text(" ", strip=True))
