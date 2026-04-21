from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_sources.odds_api import modeled_market_template_rows


CONTEXT_COLUMNS = [
    "fighter_name",
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
    "context_notes",
]

FIGHTER_MAP_COLUMNS = [
    "fighter_name",
    "espn_url",
]

MODEL_CONTEXT_FLAG_COLUMNS = [
    "short_notice_flag",
    "short_notice_acceptance_flag",
    "short_notice_success_flag",
    "cardio_fade_flag",
    "injury_concern_flag",
    "weight_cut_concern_flag",
    "replacement_fighter_flag",
    "travel_disadvantage_flag",
    "camp_change_flag",
]

OPERATOR_CONTEXT_FLAG_COLUMNS = [
    "new_gym_flag",
    "new_contract_flag",
]


def load_manifest(path: str | Path) -> dict[str, object]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_manifest_path"] = str(manifest_path)
    return data


def manifest_slug(manifest: dict[str, object]) -> str:
    return str(manifest["slug"])


def manifest_root(manifest: dict[str, object]) -> Path:
    manifest_path = Path(str(manifest["_manifest_path"])).resolve()
    return manifest_path.parents[1]


def event_workspace_root(manifest: dict[str, object]) -> Path:
    return manifest_root(manifest) / "cards" / manifest_slug(manifest)


def unique_fighters(manifest: dict[str, object]) -> list[str]:
    fighters: list[str] = []
    seen: set[str] = set()
    for fight in manifest["fights"]:
        fighter_a = str(fight["fighter_a"]).strip()
        fighter_b = str(fight["fighter_b"]).strip()
        for fighter_name in [fighter_a, fighter_b]:
            if fighter_name and fighter_name not in seen:
                fighters.append(fighter_name)
                seen.add(fighter_name)
    return fighters


def derived_paths(manifest: dict[str, object]) -> dict[str, Path]:
    root = event_workspace_root(manifest)
    return {
        "workspace": root,
        "inputs_dir": root / "inputs",
        "data_dir": root / "data",
        "reports_dir": root / "reports",
        "fighter_list": root / "inputs" / "fighter_list.csv",
        "fighter_map": root / "inputs" / "fighter_map.csv",
        "context": root / "inputs" / "fighter_context.csv",
        "odds_template": root / "data" / "odds_template.csv",
        "modeled_market_template": root / "data" / "modeled_market_template.csv",
        "fighter_gyms": root / "data" / "fighter_gyms.csv",
        "gym_registry": root / "data" / "gym_registry.csv",
        "fighter_stats": root / "data" / "fighter_stats.csv",
        "bfo_odds": root / "data" / "bfo_odds.csv",
        "oddsapi_odds": root / "data" / "oddsapi_odds.csv",
        "modeled_market_odds": root / "data" / "modeled_market_odds.csv",
        "report": root / "reports" / "fight_week_report.csv",
        "lean_board": root / "reports" / "lean_board.csv",
        "fight_week_alerts": root / "reports" / "fight_week_alerts.csv",
        "line_movement": root / "reports" / "line_movement.svg",
        "line_movement_fights_dir": root / "reports" / "line_movement_fights",
        "skipped": root / "reports" / "skipped_fights.csv",
        "value": root / "reports" / "value_bets.csv",
        "shortlist": root / "reports" / "value_bets_shortlist.csv",
        "board": root / "reports" / "betting_board.csv",
        "passes": root / "reports" / "pass_reasons.csv",
        "parlays": root / "reports" / "parlay_board.csv",
        "operator_dashboard": root / "reports" / "operator_dashboard.html",
        "graded": root / "reports" / "graded_picks.csv",
        "learning": root / "reports" / "learning_report.csv",
        "learning_summary": root / "reports" / "learning_summary.csv",
        "filter_performance": root / "reports" / "filter_performance.csv",
        "results": root / "data" / "results.csv",
    }


def current_event_manifest_path(root: str | Path) -> Path:
    project_root = Path(root)
    pointer = project_root / "events" / "current_event.txt"
    manifest_relative = pointer.read_text(encoding="utf-8").strip()
    if not manifest_relative:
        raise ValueError("events/current_event.txt is empty")
    return (project_root / manifest_relative).resolve()


def current_event_manifest(root: str | Path) -> dict[str, object]:
    return load_manifest(current_event_manifest_path(root))


def bestfightodds_refresh_url(manifest: dict[str, object]) -> str:
    value = str(
        manifest.get("bestfightodds_refresh_url")
        or manifest.get("bestfightodds_url")
        or ""
    ).strip()
    return value


def bestfightodds_event_urls(manifest: dict[str, object]) -> list[str]:
    urls = manifest.get("bestfightodds_event_urls")
    if isinstance(urls, list):
        cleaned = [str(url).strip() for url in urls if str(url).strip()]
        if cleaned:
            return cleaned
    fallback = str(manifest.get("bestfightodds_url") or "").strip()
    return [fallback] if is_verified_bestfightodds_event_url(fallback) else []


def is_verified_bestfightodds_event_url(url: str) -> bool:
    value = str(url).strip().lower()
    if not value.startswith("https://www.bestfightodds.com/"):
        return False
    blocked = {
        "https://www.bestfightodds.com/",
        "https://www.bestfightodds.com/?desktop=on",
    }
    return value not in blocked


def manifest_status_rows(manifest: dict[str, object]) -> list[tuple[str, str]]:
    paths = derived_paths(manifest)
    refresh_url = bestfightodds_refresh_url(manifest)
    event_urls = bestfightodds_event_urls(manifest)
    rows = [
        ("manifest", str(manifest["_manifest_path"])),
        ("event_id", str(manifest["event_id"])),
        ("event_name", str(manifest["event_name"])),
        ("slug", manifest_slug(manifest)),
        ("bfo_refresh_url", refresh_url or "missing"),
        ("bfo_alt_market_urls", str(len(event_urls))),
        ("bfo_alt_market_status", "verified" if event_urls else "unverified_or_missing"),
    ]
    for label, path in [
        ("fighter_list", paths["fighter_list"]),
        ("fighter_map", paths["fighter_map"]),
        ("context", paths["context"]),
        ("odds_template", paths["odds_template"]),
        ("modeled_market_template", paths["modeled_market_template"]),
        ("fighter_gyms", paths["fighter_gyms"]),
        ("gym_registry", paths["gym_registry"]),
        ("fighter_stats", paths["fighter_stats"]),
        ("bfo_odds", paths["bfo_odds"]),
        ("oddsapi_odds", paths["oddsapi_odds"]),
        ("modeled_market_odds", paths["modeled_market_odds"]),
        ("report", paths["report"]),
        ("lean_board", paths["lean_board"]),
        ("fight_week_alerts", paths["fight_week_alerts"]),
        ("line_movement", paths["line_movement"]),
        ("line_movement_fights_dir", paths["line_movement_fights_dir"]),
        ("board", paths["board"]),
        ("passes", paths["passes"]),
        ("parlays", paths["parlays"]),
        ("operator_dashboard", paths["operator_dashboard"]),
        ("learning", paths["learning"]),
        ("learning_summary", paths["learning_summary"]),
        ("filter_performance", paths["filter_performance"]),
    ]:
        rows.append((label, "ready" if path.exists() else "missing"))
    return rows


def build_fighter_list_frame(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({"fighter_name": unique_fighters(manifest)})


def build_fighter_map_frame(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"fighter_name": fighter_name, "espn_url": ""} for fighter_name in unique_fighters(manifest)],
        columns=FIGHTER_MAP_COLUMNS,
    )


def build_context_frame(manifest: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fighter_name": fighter_name,
                "short_notice_flag": 0,
                "short_notice_acceptance_flag": 0,
                "short_notice_success_flag": 0,
                "new_gym_flag": 0,
                "new_contract_flag": 0,
                "cardio_fade_flag": 0,
                "injury_concern_flag": 0,
                "weight_cut_concern_flag": 0,
                "replacement_fighter_flag": 0,
                "travel_disadvantage_flag": 0,
                "camp_change_flag": 0,
                "context_notes": "",
            }
            for fighter_name in unique_fighters(manifest)
        ],
        columns=CONTEXT_COLUMNS,
    )


def merge_existing_context(
    template: pd.DataFrame,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return template.copy()
    if "fighter_name" not in existing.columns:
        return template.copy()

    normalized_existing = existing.copy()
    normalized_existing["fighter_name"] = normalized_existing["fighter_name"].astype(str).str.strip()
    merged = template.merge(normalized_existing, on="fighter_name", how="left", suffixes=("", "_existing"))
    for column in CONTEXT_COLUMNS:
        if column == "fighter_name":
            continue
        existing_column = f"{column}_existing"
        if existing_column not in merged.columns:
            continue
        if column == "context_notes":
            merged[column] = merged[existing_column].fillna(merged[column]).fillna("")
        else:
            merged[column] = (
                pd.to_numeric(merged[existing_column], errors="coerce")
                .fillna(pd.to_numeric(merged[column], errors="coerce"))
                .fillna(0)
                .astype(int)
            )
        merged = merged.drop(columns=[existing_column])
    return merged[CONTEXT_COLUMNS].copy()


def merge_existing_fighter_map(
    template: pd.DataFrame,
    existing: pd.DataFrame | None,
) -> pd.DataFrame:
    if existing is None or existing.empty:
        return template.copy()
    if "fighter_name" not in existing.columns:
        return template.copy()

    normalized_existing = existing.copy()
    normalized_existing["fighter_name"] = normalized_existing["fighter_name"].astype(str).str.strip()
    if "espn_url" not in normalized_existing.columns:
        normalized_existing["espn_url"] = ""
    normalized_existing["espn_url"] = normalized_existing["espn_url"].fillna("").astype(str).str.strip()

    merged = template.merge(
        normalized_existing[FIGHTER_MAP_COLUMNS],
        on="fighter_name",
        how="left",
        suffixes=("", "_existing"),
    )
    if "espn_url_existing" in merged.columns:
        merged["espn_url"] = merged["espn_url_existing"].where(
            merged["espn_url_existing"].astype(str).str.strip() != "",
            merged["espn_url"],
        )
        merged = merged.drop(columns=["espn_url_existing"])
    merged["espn_url"] = merged["espn_url"].fillna("").astype(str).str.strip()
    return merged[FIGHTER_MAP_COLUMNS].copy()


def build_odds_template_frame(manifest: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fight_index, fight in enumerate(manifest["fights"]):
        fighter_a = str(fight["fighter_a"]).strip()
        fighter_b = str(fight["fighter_b"]).strip()
        scheduled_rounds = float(fight.get("scheduled_rounds", 5 if fight_index == 0 else 3))
        is_title_fight = int(fight.get("is_title_fight", 0))
        for selection in ["fighter_a", "fighter_b"]:
            rows.append(
                {
                    "event_id": manifest["event_id"],
                    "event_name": manifest["event_name"],
                    "start_time": manifest["start_time"],
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "scheduled_rounds": scheduled_rounds,
                    "is_title_fight": is_title_fight,
                    "market": "moneyline",
                    "selection": selection,
                    "book": "manual",
                    "american_odds": pd.NA,
                }
            )
    return pd.DataFrame(rows)


def build_modeled_market_template_frame(manifest: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fight_index, fight in enumerate(manifest["fights"]):
        fighter_a = str(fight["fighter_a"]).strip()
        fighter_b = str(fight["fighter_b"]).strip()
        scheduled_rounds = float(fight.get("scheduled_rounds", 5 if fight_index == 0 else 3))
        is_title_fight = int(fight.get("is_title_fight", 0))
        for market, selection, _selection_name in modeled_market_template_rows(fighter_a, fighter_b):
            rows.append(
                {
                    "event_id": manifest["event_id"],
                    "event_name": manifest["event_name"],
                    "start_time": manifest["start_time"],
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "scheduled_rounds": scheduled_rounds,
                    "is_title_fight": is_title_fight,
                    "market": market,
                    "selection": selection,
                    "book": "manual",
                    "american_odds": pd.NA,
                }
            )
    return pd.DataFrame(rows)
