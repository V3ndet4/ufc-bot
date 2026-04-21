from __future__ import annotations

from datetime import timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urlparse
import re
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from bs4 import BeautifulSoup


USER_AGENT = "ufc-bot/1.0 (+https://news.google.com/)"
REQUEST_TIMEOUT_SECONDS = 30
UTC = timezone.utc
GOOGLE_NEWS_SEARCH_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
WATCH_FLAG_COLUMNS = [
    "short_notice_flag",
    "new_gym_flag",
    "injury_concern_flag",
    "weight_cut_concern_flag",
    "replacement_fighter_flag",
    "camp_change_flag",
]

TRUSTED_SOURCE_SCORES = {
    "sherdog.com": 0.92,
    "mmajunkie.usatoday.com": 0.90,
    "usatoday.com": 0.80,
    "mmafighting.com": 0.88,
    "espn.com": 0.86,
    "ufc.com": 0.85,
    "tapology.com": 0.82,
    "cagesidepress.com": 0.78,
    "mmamania.com": 0.72,
    "bbc.com": 0.74,
}


def build_google_news_search_url(fighter_name: str, gym_name: str = "") -> str:
    query_parts = [
        f'"{fighter_name}"',
        "(UFC OR MMA)",
        '("short notice" OR replacement OR injury OR injured OR withdraws OR withdrawn OR "weight cut" OR "missed weight" OR camp OR gym OR coach OR training)',
    ]
    if gym_name:
        query_parts.append(f'"{gym_name}"')
    return GOOGLE_NEWS_SEARCH_URL.format(query=quote_plus(" ".join(query_parts)))


def fetch_text(url: str, session: requests.Session | None = None) -> str:
    client = session or requests.Session()
    response = client.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def parse_google_news_rss(xml_text: str) -> list[dict[str, str]]:
    root = ET.fromstring(xml_text)
    entries: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        description = item.findtext("description", default="")
        source_node = item.find("source")
        entries.append(
            {
                "title": (item.findtext("title", default="") or "").strip(),
                "link": (item.findtext("link", default="") or "").strip(),
                "published_at": _published_at_to_iso(item.findtext("pubDate", default="")),
                "description": _clean_html_text(description),
                "source_name": ((source_node.text or "").strip() if source_node is not None and source_node.text else ""),
                "source_url": ((source_node.get("url") or "").strip() if source_node is not None else ""),
            }
        )
    return entries


def collect_fight_week_alerts(
    fighters: pd.DataFrame,
    *,
    session: requests.Session | None = None,
    lookback_days: int = 10,
    max_results_per_fighter: int = 5,
) -> pd.DataFrame:
    if fighters.empty or "fighter_name" not in fighters.columns:
        return _empty_alert_frame()

    client = session or requests.Session()
    cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(days=max(1, lookback_days))
    alert_rows: list[dict[str, Any]] = []

    for fighter in fighters.itertuples(index=False):
        fighter_name = str(getattr(fighter, "fighter_name", "") or "").strip()
        gym_name = str(getattr(fighter, "gym_name", "") or "").strip()
        if not fighter_name:
            continue

        query_url = build_google_news_search_url(fighter_name, gym_name)
        try:
            rss_text = fetch_text(query_url, session=client)
        except requests.RequestException:
            continue
        entries = parse_google_news_rss(rss_text)
        matches_for_fighter = 0
        seen_keys: set[tuple[str, str]] = set()

        for entry in entries:
            published_at = pd.to_datetime(entry.get("published_at", ""), errors="coerce", utc=True)
            if pd.isna(published_at) or published_at < cutoff:
                continue
            classified = classify_fight_week_entry(
                fighter_name=fighter_name,
                gym_name=gym_name,
                title=entry.get("title", ""),
                summary=entry.get("description", ""),
                published_at=published_at,
                source_name=entry.get("source_name", ""),
                source_url=entry.get("source_url", ""),
                article_url=entry.get("link", ""),
            )
            if classified is None:
                continue
            dedupe_key = (classified["fighter_name"], classified["title"])
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            alert_rows.append(classified)
            matches_for_fighter += 1
            if matches_for_fighter >= max_results_per_fighter:
                break

    if not alert_rows:
        return _empty_alert_frame()
    alerts = pd.DataFrame(alert_rows)
    alerts = alerts.sort_values(
        by=["fighter_name", "confidence_score", "published_at"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return alerts


def classify_fight_week_entry(
    *,
    fighter_name: str,
    gym_name: str,
    title: str,
    summary: str,
    published_at: pd.Timestamp,
    source_name: str,
    source_url: str,
    article_url: str,
) -> dict[str, Any] | None:
    combined_text = _normalize_text(f"{title} {summary}")
    fighter_text = _normalize_text(fighter_name)
    fighter_tokens = [token for token in fighter_text.split() if token]
    if not fighter_tokens:
        return None
    if fighter_text not in combined_text and not all(token in combined_text for token in fighter_tokens):
        return None

    flags, keywords = _infer_flags(combined_text, gym_name=gym_name)
    if not any(flags.values()):
        return None

    source_score = _source_score(source_name=source_name, source_url=source_url, article_url=article_url)
    keyword_score = min(1.0, 0.35 + (0.12 * len(keywords)))
    recency_score = _recency_score(published_at)
    confidence_score = round(min(1.0, (source_score * 0.45) + (keyword_score * 0.35) + (recency_score * 0.20)), 4)

    return {
        "fighter_name": fighter_name,
        "gym_name": gym_name,
        "title": title.strip(),
        "summary": summary.strip(),
        "published_at": published_at.isoformat(),
        "published_date": published_at.date().isoformat(),
        "source_type": "google_news_rss",
        "source_name": source_name.strip(),
        "source_url": source_url.strip(),
        "article_url": article_url.strip(),
        "matched_keywords": ", ".join(keywords),
        "confidence_score": confidence_score,
        "confidence_label": _confidence_label(confidence_score),
        "alert_summary": _build_alert_summary(
            keywords=keywords,
            source_name=source_name,
            published_at=published_at,
            title=title,
        ),
        **flags,
    }


def merge_alerts_into_context(
    context_frame: pd.DataFrame,
    alerts_frame: pd.DataFrame,
) -> pd.DataFrame:
    context = _normalize_context_frame(context_frame)
    if alerts_frame.empty:
        return context

    for fighter_name, fighter_alerts in alerts_frame.groupby("fighter_name", dropna=False):
        if not str(fighter_name).strip():
            continue
        if fighter_name not in context["fighter_name"].astype(str).tolist():
            new_row = {column: 0 for column in context.columns if column != "fighter_name" and column != "context_notes"}
            new_row["fighter_name"] = fighter_name
            new_row["context_notes"] = ""
            context = pd.concat([context, pd.DataFrame([new_row])], ignore_index=True)

        row_index = context.index[context["fighter_name"].astype(str) == fighter_name][0]
        top_alerts = fighter_alerts.sort_values(
            by=["confidence_score", "published_at"],
            ascending=[False, False],
        ).head(2)

        for column in WATCH_FLAG_COLUMNS:
            source_series = (
                fighter_alerts[column]
                if column in fighter_alerts.columns
                else pd.Series(0, index=fighter_alerts.index, dtype=float)
            )
            detected = int(pd.to_numeric(source_series, errors="coerce").fillna(0).max())
            context.at[row_index, column] = max(int(context.at[row_index, column]), detected)

        existing_note = str(context.at[row_index, "context_notes"] or "").strip()
        new_notes = [
            str(value).strip()
            for value in top_alerts.get("alert_summary", pd.Series(dtype="object")).tolist()
            if str(value).strip()
        ]
        context.at[row_index, "context_notes"] = _merge_notes(existing_note, new_notes)

    return _normalize_context_frame(context)


def write_alerts_csv(frame: pd.DataFrame, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _normalize_context_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "fighter_name" not in normalized.columns:
        normalized["fighter_name"] = ""
    normalized["fighter_name"] = normalized["fighter_name"].astype(str).str.strip()
    for column in WATCH_FLAG_COLUMNS + ["short_notice_acceptance_flag", "short_notice_success_flag", "new_contract_flag", "cardio_fade_flag", "travel_disadvantage_flag"]:
        if column not in normalized.columns:
            normalized[column] = 0
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce").fillna(0).astype(int)
    if "context_notes" not in normalized.columns:
        normalized["context_notes"] = ""
    else:
        normalized["context_notes"] = normalized["context_notes"].fillna("").astype(str)
    return normalized


def _infer_flags(text: str, *, gym_name: str) -> tuple[dict[str, int], list[str]]:
    flags = {column: 0 for column in WATCH_FLAG_COLUMNS}
    matched_keywords: list[str] = []

    keyword_rules = {
        "injury_concern_flag": [
            "injury",
            "injured",
            "hurt",
            "withdraws",
            "withdrawn",
            "medical issue",
            "illness",
            "infection",
        ],
        "weight_cut_concern_flag": [
            "weight cut",
            "weight-cut",
            "missed weight",
            "misses weight",
            "scale issue",
            "weigh-in",
            "weigh in",
        ],
        "replacement_fighter_flag": [
            "replacement",
            "replaces",
            "steps in",
            "late notice",
            "late-notice",
            "fill in",
        ],
        "short_notice_flag": [
            "short notice",
            "late notice",
            "late-notice",
            "steps in",
            "replacement",
        ],
    }
    for column, phrases in keyword_rules.items():
        for phrase in phrases:
            if phrase in text:
                flags[column] = 1
                matched_keywords.append(phrase)

    camp_terms = ["gym", "camp", "coach", "academy", "training"]
    camp_change_terms = [
        "new gym",
        "new coach",
        "new camp",
        "switch camps",
        "switched camps",
        "switches camp",
        "joined",
        "joins",
        "moved to",
        "now training",
        "now trains",
        "working with",
    ]
    if any(term in text for term in camp_terms) and any(term in text for term in camp_change_terms):
        flags["camp_change_flag"] = 1
        flags["new_gym_flag"] = 1
        matched_keywords.extend(term for term in camp_change_terms if term in text)
    elif gym_name and _normalize_text(gym_name) in text and any(term in text for term in ["training at", "working with", "joined", "joins"]):
        flags["camp_change_flag"] = 1
        flags["new_gym_flag"] = 1
        matched_keywords.append(gym_name.strip())

    deduped_keywords = list(dict.fromkeys(matched_keywords))
    return flags, deduped_keywords


def _clean_html_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    return " ".join(soup.get_text(" ", strip=True).split())


def _published_at_to_iso(raw_value: str) -> str:
    if not raw_value:
        return ""
    try:
        parsed = parsedate_to_datetime(raw_value)
    except (TypeError, ValueError, IndexError):
        return ""
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0).isoformat()


def _normalize_text(value: str) -> str:
    lowered = str(value or "").lower().replace("’", "'")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return " ".join(lowered.split())


def _source_score(*, source_name: str, source_url: str, article_url: str) -> float:
    candidates = [source_url, article_url]
    domains = [_extract_domain(value) for value in candidates if value]
    for domain in domains:
        for known_domain, score in TRUSTED_SOURCE_SCORES.items():
            if domain == known_domain or domain.endswith(f".{known_domain}"):
                return score
    source_name_normalized = _normalize_text(source_name)
    if "sherdog" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["sherdog.com"]
    if "mma junkie" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["mmajunkie.usatoday.com"]
    if "mma fighting" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["mmafighting.com"]
    if "espn" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["espn.com"]
    if "ufc" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["ufc.com"]
    if "tapology" in source_name_normalized:
        return TRUSTED_SOURCE_SCORES["tapology.com"]
    return 0.55


def _extract_domain(url: str) -> str:
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower().strip()
    return domain.removeprefix("www.")


def _recency_score(published_at: pd.Timestamp) -> float:
    age = pd.Timestamp.now(tz=UTC) - published_at
    if age <= timedelta(days=2):
        return 1.0
    if age <= timedelta(days=5):
        return 0.8
    if age <= timedelta(days=10):
        return 0.6
    return 0.4


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.60:
        return "medium"
    return "low"


def _build_alert_summary(
    *,
    keywords: list[str],
    source_name: str,
    published_at: pd.Timestamp,
    title: str,
) -> str:
    labels = []
    keyword_set = {keyword.lower() for keyword in keywords}
    if keyword_set & {"short notice", "late notice", "late-notice", "replacement", "replaces", "steps in"}:
        labels.append("short-notice watch")
    if keyword_set & {"injury", "injured", "hurt", "withdraws", "withdrawn", "medical issue", "illness", "infection"}:
        labels.append("injury watch")
    if keyword_set & {"weight cut", "weight-cut", "missed weight", "misses weight", "scale issue", "weigh-in", "weigh in"}:
        labels.append("weight watch")
    if not labels and keyword_set:
        labels.append("camp watch")
    label_text = "/".join(labels[:2]) if labels else "fight-week watch"
    source_label = source_name.strip() or "News"
    published_label = published_at.date().isoformat()
    headline = title.strip()
    return f"{label_text}: {headline} ({source_label} {published_label})"


def _merge_notes(existing_note: str, new_notes: list[str]) -> str:
    parts = [part.strip() for part in str(existing_note or "").split(" | ") if part.strip()]
    for note in new_notes:
        if note and note not in parts:
            parts.append(note)
    return " | ".join(parts[:4])


def _empty_alert_frame() -> pd.DataFrame:
    columns = [
        "fighter_name",
        "gym_name",
        "title",
        "summary",
        "published_at",
        "published_date",
        "source_type",
        "source_name",
        "source_url",
        "article_url",
        "matched_keywords",
        "confidence_score",
        "confidence_label",
        "alert_summary",
        *WATCH_FLAG_COLUMNS,
    ]
    return pd.DataFrame(columns=columns)
