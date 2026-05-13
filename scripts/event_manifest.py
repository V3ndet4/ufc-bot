from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_sources.odds_api import modeled_market_template_rows


DEFAULT_MAIN_CARD_FIGHT_COUNT = 5

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
    "news_alert_count",
    "news_radar_score",
    "news_high_confidence_alerts",
    "news_alert_confidence",
    "news_radar_label",
    "news_primary_category",
    "news_radar_summary",
    "context_notes",
]

CONTEXT_FLOAT_COLUMNS = {
    "news_alert_count",
    "news_radar_score",
    "news_high_confidence_alerts",
    "news_alert_confidence",
}

CONTEXT_TEXT_COLUMNS = {
    "news_radar_label",
    "news_primary_category",
    "news_radar_summary",
    "context_notes",
}

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

TRUE_FLAG_VALUES = {"1", "true", "yes", "y", "main", "main_card", "maincard"}
FALSE_FLAG_VALUES = {"0", "false", "no", "n", "prelim", "prelims", "early_prelim", "early_prelims"}


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


def _parse_bool_flag(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None
        return float(value) != 0.0

    text = str(value).strip().lower()
    if not text:
        return None
    normalized = text.replace("-", "_").replace(" ", "_")
    if normalized in TRUE_FLAG_VALUES:
        return True
    if normalized in FALSE_FLAG_VALUES:
        return False
    return None


def main_card_fight_count(manifest: dict[str, object]) -> int:
    for key in ["main_card_fight_count", "main_card_count"]:
        value = manifest.get(key)
        if value is None:
            continue
        try:
            count = int(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
        if count > 0:
            return count
    return DEFAULT_MAIN_CARD_FIGHT_COUNT


def is_main_card_fight(fight: dict[str, object], fight_index: int, manifest: dict[str, object]) -> bool:
    for key in ["is_main_card", "main_card"]:
        parsed = _parse_bool_flag(fight.get(key))
        if parsed is not None:
            return parsed

    for key in ["card_section", "card_segment", "bout_section", "card_type"]:
        section = str(fight.get(key, "") or "").strip().lower().replace("-", "_").replace(" ", "_")
        if "prelim" in section or "undercard" in section:
            return False
        if section.startswith("main"):
            return True

    return fight_index < main_card_fight_count(manifest)


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
        "core_board": root / "reports" / "core_board.csv",
        "core_props": root / "reports" / "core_props.csv",
        "core_parlays": root / "reports" / "core_parlays.csv",
        "parlays": root / "reports" / "parlay_board.csv",
        "operator_dashboard": root / "reports" / "operator_dashboard.html",
        "graded": root / "reports" / "graded_picks.csv",
        "learning": root / "reports" / "learning_report.csv",
        "learning_summary": root / "reports" / "learning_summary.csv",
        "learning_postmortem": root / "reports" / "learning_postmortem.csv",
        "learning_postmortem_summary": root / "reports" / "learning_postmortem_summary.csv",
        "lean_results": root / "reports" / "lean_board_results.csv",
        "lean_postmortem_summary": root / "reports" / "lean_postmortem_summary.csv",
        "filter_performance": root / "reports" / "filter_performance.csv",
        "prediction_snapshot": root / "reports" / "prediction_snapshot.csv",
        "accuracy_calibration": root / "reports" / "accuracy_calibration.csv",
        "market_accuracy": root / "reports" / "market_accuracy.csv",
        "prop_bet_market_accuracy": root / "reports" / "prop_bet_market_accuracy.csv",
        "prop_model_backtest_predictions": root / "reports" / "prop_model_backtest_predictions.csv",
        "prop_model_market_accuracy": root / "reports" / "prop_model_market_accuracy.csv",
        "prop_model_calibration": root / "reports" / "prop_model_calibration.csv",
        "prop_model_thresholds": root / "reports" / "prop_model_thresholds.csv",
        "prop_odds_archive_summary": root / "reports" / "prop_odds_archive_summary.csv",
        "odds_movement_clv": root / "reports" / "odds_movement_clv.csv",
        "tracked_clv": root / "reports" / "tracked_clv.csv",
        "segment_performance": root / "reports" / "segment_performance.csv",
        "segment_quality_gates": root / "reports" / "segment_quality_gates.csv",
        "current_prediction_quality": root / "reports" / "current_prediction_quality.csv",
        "style_matchup_diagnostics": root / "reports" / "style_matchup_diagnostics.csv",
        "accuracy_postmortem_codes": root / "reports" / "accuracy_postmortem_codes.csv",
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
        ("learning_postmortem", paths["learning_postmortem"]),
        ("learning_postmortem_summary", paths["learning_postmortem_summary"]),
        ("filter_performance", paths["filter_performance"]),
        ("prediction_snapshot", paths["prediction_snapshot"]),
        ("accuracy_calibration", paths["accuracy_calibration"]),
        ("market_accuracy", paths["market_accuracy"]),
        ("prop_bet_market_accuracy", paths["prop_bet_market_accuracy"]),
        ("prop_model_market_accuracy", paths["prop_model_market_accuracy"]),
        ("prop_model_thresholds", paths["prop_model_thresholds"]),
        ("prop_odds_archive_summary", paths["prop_odds_archive_summary"]),
        ("odds_movement_clv", paths["odds_movement_clv"]),
        ("tracked_clv", paths["tracked_clv"]),
        ("segment_performance", paths["segment_performance"]),
        ("segment_quality_gates", paths["segment_quality_gates"]),
        ("current_prediction_quality", paths["current_prediction_quality"]),
        ("style_matchup_diagnostics", paths["style_matchup_diagnostics"]),
        ("accuracy_postmortem_codes", paths["accuracy_postmortem_codes"]),
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
                "news_alert_count": 0,
                "news_radar_score": 0.0,
                "news_high_confidence_alerts": 0,
                "news_alert_confidence": 0.0,
                "news_radar_label": "",
                "news_primary_category": "",
                "news_radar_summary": "",
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
        if column in CONTEXT_TEXT_COLUMNS:
            merged[column] = merged[existing_column].fillna(merged[column]).fillna("")
        elif column in CONTEXT_FLOAT_COLUMNS:
            merged[column] = (
                pd.to_numeric(merged[existing_column], errors="coerce")
                .fillna(pd.to_numeric(merged[column], errors="coerce"))
                .fillna(0.0)
                .astype(float)
            )
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
        is_main_card = int(is_main_card_fight(fight, fight_index, manifest))
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
                    "is_main_card": is_main_card,
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
        is_main_card = int(is_main_card_fight(fight, fight_index, manifest))
        for market, selection, _selection_name in modeled_market_template_rows(fighter_a, fighter_b):
            if market in {"knockdown", "takedown"} and not is_main_card:
                continue
            rows.append(
                {
                    "event_id": manifest["event_id"],
                    "event_name": manifest["event_name"],
                    "start_time": manifest["start_time"],
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "scheduled_rounds": scheduled_rounds,
                    "is_title_fight": is_title_fight,
                    "is_main_card": is_main_card,
                    "market": market,
                    "selection": selection,
                    "book": "manual",
                    "american_odds": pd.NA,
                }
            )
    return pd.DataFrame(rows)
