from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from data_sources.external_ufc_history import (
    DATASET_URLS,
    _build_fight_level_stats,
    _build_fighter_history,
    _normalize_columns,
    _normalize_text,
)


DEFAULT_PROP_OUTCOME_HISTORY_PATH = Path("data") / "prop_outcome_history.csv"

HISTORICAL_DATASET_FILES = {
    name: f"{name}.csv"
    for name in DATASET_URLS
}

PROP_OUTCOME_LABEL_COLUMNS = [
    "takedown_count",
    "knockdown_count",
    "takedown_1plus_target",
    "takedown_2plus_target",
    "knockdown_1plus_target",
]


def default_prop_outcome_history_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_PROP_OUTCOME_HISTORY_PATH


def load_cached_external_history_datasets(cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    cache_root = Path(cache_dir)
    datasets: dict[str, pd.DataFrame] = {}
    missing: list[Path] = []
    for name, filename in HISTORICAL_DATASET_FILES.items():
        path = cache_root / filename
        if not path.exists():
            missing.append(path)
            continue
        datasets[name] = pd.read_csv(path)
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing cached UFC history files: {formatted}")
    return datasets


def build_prop_outcome_history_frame(
    *,
    fight_results: pd.DataFrame,
    fight_stats: pd.DataFrame,
    event_details: pd.DataFrame,
) -> pd.DataFrame:
    history = _build_fighter_history(fight_results, event_details)
    fight_totals = _build_fight_level_stats(fight_stats, fight_results=fight_results)
    if history.empty or fight_totals.empty:
        return pd.DataFrame()

    history = history.merge(fight_totals, on=["event", "bout", "fighter_key"], how="left")
    history = history.merge(_scheduled_rounds_frame(fight_results), on=["event", "bout"], how="left")
    history["scheduled_rounds"] = pd.to_numeric(history["scheduled_rounds"], errors="coerce").fillna(3.0)

    numeric_total_columns = [
        "minutes",
        "sig_landed_total",
        "sig_attempted_total",
        "sig_absorbed_total",
        "sig_attempted_against_total",
        "takedown_landed_total",
        "takedown_attempted_total",
        "takedowns_absorbed_total",
        "takedown_attempted_against_total",
        "control_seconds_total",
        "knockdown_total",
        "distance_landed_total",
        "clinch_landed_total",
        "ground_landed_total",
    ]
    for column in numeric_total_columns:
        if column not in history.columns:
            history[column] = 0.0
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    history["fight_knockdown_total"] = history.groupby(["event", "bout"])["knockdown_total"].transform("sum")
    history["knockdowns_absorbed_total"] = (history["fight_knockdown_total"] - history["knockdown_total"]).clip(lower=0.0)
    history["ko_win_flag"] = _finish_flag(history, result="W", patterns=("ko", "tko"))
    history["ko_loss_flag"] = _finish_flag(history, result="L", patterns=("ko", "tko"))

    profiled = (
        history.sort_values(["fighter_key", "date", "event", "bout"], na_position="last")
        .groupby("fighter_key", group_keys=False)
        .apply(_add_prior_profile_columns)
        .reset_index(drop=True)
    )
    profiled = _merge_opponent_prior_profiles(profiled)
    profiled["selection_matchup_grappling_edge"] = _matchup_grappling_edge(profiled)

    profiled["takedown_count"] = pd.to_numeric(profiled["takedown_landed_total"], errors="coerce").fillna(0.0)
    profiled["knockdown_count"] = pd.to_numeric(profiled["knockdown_total"], errors="coerce").fillna(0.0)
    profiled["takedown_1plus_target"] = (profiled["takedown_count"] >= 1).astype(int)
    profiled["takedown_2plus_target"] = (profiled["takedown_count"] >= 2).astype(int)
    profiled["knockdown_1plus_target"] = (profiled["knockdown_count"] >= 1).astype(int)

    output_columns = [
        "event",
        "bout",
        "date",
        "event_name",
        "weight_class",
        "fighter_key",
        "opponent_key",
        "result_code",
        "decision_type",
        "scheduled_rounds",
        *PROP_OUTCOME_LABEL_COLUMNS,
        "selection_ufc_fight_count",
        "opponent_ufc_fight_count",
        "selection_takedown_avg",
        "selection_takedown_accuracy_pct",
        "selection_takedown_defense_pct",
        "opponent_takedown_avg",
        "opponent_takedown_accuracy_pct",
        "opponent_takedown_defense_pct",
        "selection_recent_grappling_rate",
        "opponent_recent_grappling_rate",
        "selection_control_avg",
        "selection_recent_control_avg",
        "selection_matchup_grappling_edge",
        "selection_knockdown_avg",
        "selection_ko_win_rate",
        "opponent_ko_loss_rate",
        "selection_sig_strikes_landed_per_min",
        "opponent_sig_strikes_absorbed_per_min",
        "selection_distance_strike_share",
        "selection_clinch_strike_share",
        "selection_ground_strike_share",
    ]
    for column in output_columns:
        if column not in profiled.columns:
            profiled[column] = 0.0

    output = profiled[output_columns].copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    return output.sort_values(["date", "event", "bout", "fighter_key"], na_position="last").reset_index(drop=True)


def _scheduled_rounds_frame(fight_results: pd.DataFrame) -> pd.DataFrame:
    results = _normalize_columns(fight_results)
    results["event"] = results["EVENT"].map(_normalize_text)
    results["bout"] = results["BOUT"].map(_normalize_text)
    time_format = results["TIME FORMAT"] if "TIME FORMAT" in results.columns else pd.Series("", index=results.index)
    weight_class = results["WEIGHTCLASS"] if "WEIGHTCLASS" in results.columns else pd.Series("", index=results.index)
    results["scheduled_rounds"] = [
        _parse_scheduled_rounds(raw_time_format, raw_weight_class)
        for raw_time_format, raw_weight_class in zip(time_format, weight_class)
    ]
    return results[["event", "bout", "scheduled_rounds"]].drop_duplicates(subset=["event", "bout"], keep="first")


def _parse_scheduled_rounds(raw_time_format: object, raw_weight_class: object) -> int:
    text = _normalize_text(raw_time_format)
    match = re.search(r"(\d+)\s+Rnd", text, flags=re.IGNORECASE)
    if match:
        rounds = int(match.group(1))
        if 1 <= rounds <= 5:
            return rounds
    if "title" in _normalize_text(raw_weight_class).lower():
        return 5
    return 3


def _finish_flag(history: pd.DataFrame, *, result: str, patterns: tuple[str, ...]) -> pd.Series:
    result_code = history["result_code"].astype(str).str.upper().str.strip()
    method = history["decision_type"].astype(str).str.lower()
    pattern = "|".join(re.escape(item) for item in patterns)
    return ((result_code == result) & method.str.contains(pattern, na=False)).astype(float)


def _add_prior_profile_columns(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values(["date", "event", "bout"], na_position="last").copy()
    if "fighter_key" not in ordered.columns:
        ordered["fighter_key"] = str(group.name)
    ordered["selection_ufc_fight_count"] = range(len(ordered))

    prior_minutes = _shifted_cumsum(ordered["minutes"])
    recent_minutes = _shifted_rolling_sum(ordered["minutes"], window=3)

    ordered["selection_takedown_avg"] = _rate_per_15(_shifted_cumsum(ordered["takedown_landed_total"]), prior_minutes)
    ordered["selection_takedown_accuracy_pct"] = _percentage(
        _shifted_cumsum(ordered["takedown_landed_total"]),
        _shifted_cumsum(ordered["takedown_attempted_total"]),
    )
    ordered["selection_takedown_defense_pct"] = _defense_percentage(
        _shifted_cumsum(ordered["takedowns_absorbed_total"]),
        _shifted_cumsum(ordered["takedown_attempted_against_total"]),
        default=68.0,
    )
    ordered["selection_recent_grappling_rate"] = _rate_per_15(
        _shifted_rolling_sum(ordered["takedown_landed_total"], window=3),
        recent_minutes,
    )
    ordered["selection_control_avg"] = _rate_per_15(
        _shifted_cumsum(ordered["control_seconds_total"]) / 60.0,
        prior_minutes,
    )
    ordered["selection_recent_control_avg"] = _rate_per_15(
        _shifted_rolling_sum(ordered["control_seconds_total"], window=3) / 60.0,
        recent_minutes,
    )
    ordered["selection_knockdown_avg"] = _rate_per_15(_shifted_cumsum(ordered["knockdown_total"]), prior_minutes)
    ordered["selection_sig_strikes_landed_per_min"] = _rate_per_min(
        _shifted_cumsum(ordered["sig_landed_total"]),
        prior_minutes,
    )
    ordered["selection_sig_strikes_absorbed_per_min"] = _rate_per_min(
        _shifted_cumsum(ordered["sig_absorbed_total"]),
        prior_minutes,
    )
    prior_sig_landed = _shifted_cumsum(ordered["sig_landed_total"])
    ordered["selection_distance_strike_share"] = _share(
        _shifted_cumsum(ordered["distance_landed_total"]),
        prior_sig_landed,
        default=0.55,
    )
    ordered["selection_clinch_strike_share"] = _share(
        _shifted_cumsum(ordered["clinch_landed_total"]),
        prior_sig_landed,
        default=0.15,
    )
    ordered["selection_ground_strike_share"] = _share(
        _shifted_cumsum(ordered["ground_landed_total"]),
        prior_sig_landed,
        default=0.10,
    )
    ordered["selection_ko_win_rate"] = _rate_per_fight(_shifted_cumsum(ordered["ko_win_flag"]), ordered["selection_ufc_fight_count"])
    ordered["selection_ko_loss_rate"] = _rate_per_fight(_shifted_cumsum(ordered["ko_loss_flag"]), ordered["selection_ufc_fight_count"])
    return ordered


def _merge_opponent_prior_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    profile_columns = [
        "selection_ufc_fight_count",
        "selection_takedown_avg",
        "selection_takedown_accuracy_pct",
        "selection_takedown_defense_pct",
        "selection_recent_grappling_rate",
        "selection_control_avg",
        "selection_recent_control_avg",
        "selection_knockdown_avg",
        "selection_sig_strikes_landed_per_min",
        "selection_sig_strikes_absorbed_per_min",
        "selection_ko_loss_rate",
    ]
    opponent_profiles = frame[["event", "bout", "fighter_key", *profile_columns]].rename(
        columns={
            "fighter_key": "opponent_key",
            "selection_ufc_fight_count": "opponent_ufc_fight_count",
            "selection_takedown_avg": "opponent_takedown_avg",
            "selection_takedown_accuracy_pct": "opponent_takedown_accuracy_pct",
            "selection_takedown_defense_pct": "opponent_takedown_defense_pct",
            "selection_recent_grappling_rate": "opponent_recent_grappling_rate",
            "selection_control_avg": "opponent_control_avg",
            "selection_recent_control_avg": "opponent_recent_control_avg",
            "selection_knockdown_avg": "opponent_knockdown_avg",
            "selection_sig_strikes_landed_per_min": "opponent_sig_strikes_landed_per_min",
            "selection_sig_strikes_absorbed_per_min": "opponent_sig_strikes_absorbed_per_min",
            "selection_ko_loss_rate": "opponent_ko_loss_rate",
        }
    )
    merged = frame.merge(opponent_profiles, on=["event", "bout", "opponent_key"], how="left")
    fill_defaults = {
        "opponent_takedown_defense_pct": 68.0,
        "opponent_sig_strikes_absorbed_per_min": 0.0,
        "opponent_ko_loss_rate": 0.0,
    }
    for column in opponent_profiles.columns:
        if column in {"event", "bout", "opponent_key"}:
            continue
        default = fill_defaults.get(column, 0.0)
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default)
    return merged


def _matchup_grappling_edge(frame: pd.DataFrame) -> pd.Series:
    selection_score = (
        pd.to_numeric(frame["selection_takedown_avg"], errors="coerce").fillna(0.0)
        * ((100.0 - pd.to_numeric(frame["opponent_takedown_defense_pct"], errors="coerce").fillna(68.0)).clip(lower=5.0) / 100.0)
        + (pd.to_numeric(frame["selection_recent_grappling_rate"], errors="coerce").fillna(0.0) * 0.35)
        + (pd.to_numeric(frame["selection_takedown_accuracy_pct"], errors="coerce").fillna(0.0).clip(lower=5.0) / 100.0)
    )
    opponent_score = (
        pd.to_numeric(frame["opponent_takedown_avg"], errors="coerce").fillna(0.0)
        * ((100.0 - pd.to_numeric(frame["selection_takedown_defense_pct"], errors="coerce").fillna(68.0)).clip(lower=5.0) / 100.0)
        + (pd.to_numeric(frame["opponent_recent_grappling_rate"], errors="coerce").fillna(0.0) * 0.35)
        + (pd.to_numeric(frame["opponent_takedown_accuracy_pct"], errors="coerce").fillna(0.0).clip(lower=5.0) / 100.0)
    )
    return selection_score - opponent_score


def _shifted_cumsum(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.cumsum().shift(1, fill_value=0.0)


def _shifted_rolling_sum(series: pd.Series, *, window: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.shift(1, fill_value=0.0).rolling(window=window, min_periods=1).sum()


def _rate_per_15(total: pd.Series, minutes: pd.Series) -> pd.Series:
    return (_rate(total, minutes) * 15.0).fillna(0.0)


def _rate_per_min(total: pd.Series, minutes: pd.Series) -> pd.Series:
    return _rate(total, minutes).fillna(0.0)


def _rate(total: pd.Series, minutes: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(total, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(minutes, errors="coerce").fillna(0.0)
    return (numerator / denominator.where(denominator > 0.0)).fillna(0.0)


def _percentage(success: pd.Series, attempts: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(success, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(attempts, errors="coerce").fillna(0.0)
    return ((numerator / denominator.where(denominator > 0.0)) * 100.0).fillna(0.0)


def _defense_percentage(allowed: pd.Series, attempts_against: pd.Series, *, default: float) -> pd.Series:
    allowed_numeric = pd.to_numeric(allowed, errors="coerce").fillna(0.0)
    attempts_numeric = pd.to_numeric(attempts_against, errors="coerce").fillna(0.0)
    defense = (1.0 - (allowed_numeric / attempts_numeric.where(attempts_numeric > 0.0))) * 100.0
    return defense.fillna(default).clip(lower=0.0, upper=100.0)


def _share(part: pd.Series, total: pd.Series, *, default: float) -> pd.Series:
    part_numeric = pd.to_numeric(part, errors="coerce").fillna(0.0)
    total_numeric = pd.to_numeric(total, errors="coerce").fillna(0.0)
    return (part_numeric / total_numeric.where(total_numeric > 0.0)).fillna(default).clip(lower=0.0, upper=1.0)


def _rate_per_fight(total: pd.Series, fight_count: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(total, errors="coerce").fillna(0.0)
    denominator = pd.to_numeric(fight_count, errors="coerce").fillna(0.0)
    return (numerator / denominator.where(denominator > 0.0)).fillna(0.0).clip(lower=0.0, upper=1.0)
