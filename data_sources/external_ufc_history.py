from __future__ import annotations

from io import StringIO
from pathlib import Path
import re
import unicodedata

import pandas as pd
import requests

from data_sources.espn import (
    _age_years_from_birthdate,
    _days_since,
    _fight_duration_minutes,
    _first_round_finish_metrics,
    _loss_streak,
    _outcome_profile_metrics,
    _recent_damage_metrics,
    _recent_result_score,
)
from data_sources.ufc_stats import _parse_height_to_inches, _parse_reach_to_inches
from features.style_profile import derive_style_label


REPOSITORY_NAME = "Greco1899/scrape_ufc_stats"
RAW_BASE_URL = f"https://raw.githubusercontent.com/{REPOSITORY_NAME}/main"
REQUEST_TIMEOUT_SECONDS = 30
USER_AGENT = f"ufc-bot/1.0 (+https://github.com/{REPOSITORY_NAME})"

DATASET_URLS = {
    "fight_results": f"{RAW_BASE_URL}/ufc_fight_results.csv",
    "fight_stats": f"{RAW_BASE_URL}/ufc_fight_stats.csv",
    "event_details": f"{RAW_BASE_URL}/ufc_event_details.csv",
    "fighter_tott": f"{RAW_BASE_URL}/ufc_fighter_tott.csv",
}

HISTORY_NUMERIC_COLUMNS = [
    "recent_strike_margin_per_min",
    "recent_grappling_rate",
    "control_avg",
    "recent_control_avg",
    "recent_result_score",
    "losses_in_row",
    "first_round_finish_wins",
    "first_round_finish_rate",
    "finish_win_rate",
    "finish_loss_rate",
    "decision_rate",
    "submission_avg",
    "ko_win_rate",
    "submission_win_rate",
    "ko_loss_rate",
    "submission_loss_rate",
    "recent_finish_loss_count",
    "recent_ko_loss_count",
    "recent_finish_loss_365d",
    "recent_ko_loss_365d",
    "recent_damage_score",
    "days_since_last_fight",
    "ufc_fight_count",
    "ufc_debut_flag",
    "opponent_avg_win_rate",
    "opponent_avg_ufc_fight_count",
    "opponent_avg_recent_result_score",
    "opponent_avg_finish_win_rate",
    "opponent_quality_score",
    "recent_opponent_quality_score",
]

TECHNIQUE_NUMERIC_COLUMNS = [
    "strike_accuracy_pct",
    "strike_defense_pct",
    "takedown_accuracy_pct",
    "takedown_defense_pct",
]

STYLE_PROFILE_NUMERIC_COLUMNS = [
    "knockdown_avg",
    "head_strike_share",
    "body_strike_share",
    "leg_strike_share",
    "distance_strike_share",
    "clinch_strike_share",
    "ground_strike_share",
]

ROLLING_NUMERIC_COLUMNS = [
    "strike_margin_last_1",
    "strike_margin_last_3",
    "strike_margin_last_5",
    "grappling_rate_last_1",
    "grappling_rate_last_3",
    "grappling_rate_last_5",
    "control_avg_last_1",
    "control_avg_last_3",
    "control_avg_last_5",
    "strike_pace_last_1",
    "strike_pace_last_3",
    "strike_pace_last_5",
    "finish_win_rate_last_3",
    "finish_win_rate_last_5",
    "finish_loss_rate_last_3",
    "finish_loss_rate_last_5",
    "result_score_last_1",
    "result_score_last_3",
    "result_score_last_5",
]

RECENCY_WEIGHTED_NUMERIC_COLUMNS = [
    "recency_weighted_strike_margin",
    "recency_weighted_grappling_rate",
    "recency_weighted_control_avg",
    "recency_weighted_strike_pace",
    "recency_weighted_result_score",
    "recency_weighted_finish_win_rate",
    "recency_weighted_finish_loss_rate",
]

ROUND_TREND_NUMERIC_COLUMNS = [
    "strike_round_trend",
    "grappling_round_trend",
    "control_round_trend",
    "strike_pace_round_trend",
]

PROFILE_NUMERIC_COLUMNS = [
    "age_years",
    "height_in",
    "reach_in",
]

PROFILE_STRING_COLUMNS = [
    "stance",
]

STYLE_PROFILE_STRING_COLUMNS = [
    "history_style_label",
]

_LANDED_ATTEMPTS_PATTERN = re.compile(r"^\s*(\d+)\s+of\s+(\d+)\s*$", re.IGNORECASE)
_BOUT_SPLIT_PATTERN = re.compile(r"\s+vs\.\s+", re.IGNORECASE)


def load_external_ufc_history_datasets(
    cache_dir: str | Path,
    *,
    session: requests.Session | None = None,
) -> dict[str, pd.DataFrame]:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    client = session or requests.Session()
    client.headers.update({"User-Agent": USER_AGENT})
    return {
        name: _download_csv(client, url, cache_root / f"{name}.csv")
        for name, url in DATASET_URLS.items()
    }


def build_external_ufc_history_features(
    *,
    fight_results: pd.DataFrame,
    fight_stats: pd.DataFrame,
    event_details: pd.DataFrame,
    fighter_tott: pd.DataFrame,
) -> pd.DataFrame:
    history = _build_fighter_history(fight_results, event_details)
    fight_totals = _build_fight_level_stats(fight_stats, fight_results=fight_results)
    history = history.merge(
        fight_totals,
        on=["event", "bout", "fighter_key"],
        how="left",
    )

    for column in [
        "sig_landed_total",
        "sig_absorbed_total",
        "takedown_landed_total",
        "submission_attempt_total",
    ]:
        if column not in history.columns:
            history[column] = 0.0
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    profiles, ambiguous_keys = _build_profile_frame(fighter_tott)
    history = history.loc[~history["fighter_key"].isin(ambiguous_keys)].copy()

    metric_rows: list[dict[str, object]] = []
    for fighter_key, fighter_history in history.groupby("fighter_key", sort=False):
        ordered = fighter_history.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)
        metrics = _build_metric_row(ordered)
        metrics["fighter_key"] = fighter_key
        metric_rows.append(metrics)

    metrics_frame = pd.DataFrame(metric_rows)
    opponent_strength = _build_opponent_strength_frame(history, metrics_frame)
    features = profiles.merge(metrics_frame, on="fighter_key", how="outer").merge(
        opponent_strength,
        on="fighter_key",
        how="left",
    )

    for column in HISTORY_NUMERIC_COLUMNS:
        if column == "days_since_last_fight":
            default_value = 999.0
        elif column in {"opponent_avg_win_rate", "opponent_quality_score", "recent_opponent_quality_score"}:
            default_value = 0.5
        else:
            default_value = 0.0
        if column not in features.columns:
            features[column] = default_value
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(default_value)

    features["ufc_debut_flag"] = (features["ufc_fight_count"] <= 0).astype(float)

    for column in PROFILE_NUMERIC_COLUMNS:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    for column in PROFILE_STRING_COLUMNS:
        if column not in features.columns:
            features[column] = ""
        features[column] = features[column].fillna("").astype(str)

    for column in [
        *TECHNIQUE_NUMERIC_COLUMNS,
        *STYLE_PROFILE_NUMERIC_COLUMNS,
        *ROLLING_NUMERIC_COLUMNS,
        *RECENCY_WEIGHTED_NUMERIC_COLUMNS,
        *ROUND_TREND_NUMERIC_COLUMNS,
    ]:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)

    features["history_style_label"] = features.apply(
        lambda row: derive_style_label(
            stance=str(row.get("stance", "") or ""),
            strike_margin=float(row.get("recency_weighted_strike_margin", 0.0) or 0.0),
            grappling_rate=float(row.get("recency_weighted_grappling_rate", 0.0) or 0.0),
            control_avg=float(row.get("recency_weighted_control_avg", 0.0) or 0.0),
            ko_win_rate=float(row.get("ko_win_rate", 0.0) or 0.0),
            submission_win_rate=float(row.get("submission_win_rate", 0.0) or 0.0),
            decision_rate=float(row.get("decision_rate", 0.0) or 0.0),
            knockdown_avg=float(row.get("knockdown_avg", 0.0) or 0.0),
            distance_strike_share=float(row.get("distance_strike_share", 0.0) or 0.0),
            clinch_strike_share=float(row.get("clinch_strike_share", 0.0) or 0.0),
            ground_strike_share=float(row.get("ground_strike_share", 0.0) or 0.0),
        ),
        axis=1,
    )

    for column in STYLE_PROFILE_STRING_COLUMNS:
        if column not in features.columns:
            features[column] = ""
        features[column] = features[column].fillna("").astype(str)

    ordered_columns = [
        "fighter_key",
        *PROFILE_NUMERIC_COLUMNS,
        *PROFILE_STRING_COLUMNS,
        *HISTORY_NUMERIC_COLUMNS,
        *TECHNIQUE_NUMERIC_COLUMNS,
        *STYLE_PROFILE_NUMERIC_COLUMNS,
        *ROLLING_NUMERIC_COLUMNS,
        *RECENCY_WEIGHTED_NUMERIC_COLUMNS,
        *ROUND_TREND_NUMERIC_COLUMNS,
        *STYLE_PROFILE_STRING_COLUMNS,
    ]
    return features[ordered_columns].drop_duplicates(subset=["fighter_key"], keep="first")


def merge_external_ufc_history_into_fighter_stats(
    fighter_stats: pd.DataFrame,
    external_features: pd.DataFrame,
) -> pd.DataFrame:
    if "fighter_name" not in fighter_stats.columns:
        raise ValueError("fighter stats frame must contain fighter_name")

    enriched = fighter_stats.copy()
    enriched["fighter_key"] = enriched["fighter_name"].map(_normalize_fighter_name)
    merged = enriched.merge(external_features, on="fighter_key", how="left", suffixes=("", "__external"))

    for column in HISTORY_NUMERIC_COLUMNS:
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        current = pd.to_numeric(merged[column], errors="coerce")
        missing_mask = current.isna()
        merged.loc[missing_mask, column] = merged.loc[missing_mask, external_column]

    numeric_backfill_rules = {
        "age_years": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
        "height_in": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
        "reach_in": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
    }
    for column, rule in numeric_backfill_rules.items():
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        fill_mask = rule(merged[column]) & merged[external_column].notna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, external_column]

    numeric_backfill_rules = {
        "strike_accuracy_pct": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
        "strike_defense_pct": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
        "takedown_accuracy_pct": lambda series: pd.to_numeric(series, errors="coerce").fillna(0.0) <= 0,
    }
    for column, rule in numeric_backfill_rules.items():
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        fill_mask = rule(merged[column]) & merged[external_column].notna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, external_column]

    takedown_defense_external = "takedown_defense_pct__external"
    if takedown_defense_external in merged.columns:
        if "takedown_defense_pct" not in merged.columns:
            merged["takedown_defense_pct"] = merged[takedown_defense_external]
        else:
            current_takedown_defense = pd.to_numeric(merged["takedown_defense_pct"], errors="coerce")
            fill_mask = current_takedown_defense.isna() | (current_takedown_defense <= 0)
            if "data_notes" in merged.columns:
                proxy_mask = merged["data_notes"].fillna("").astype(str).str.contains("used as a proxy", case=False, regex=False)
                fill_mask = fill_mask | proxy_mask
            merged.loc[fill_mask & merged[takedown_defense_external].notna(), "takedown_defense_pct"] = merged.loc[
                fill_mask & merged[takedown_defense_external].notna(),
                takedown_defense_external,
            ]

    for column in STYLE_PROFILE_NUMERIC_COLUMNS:
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        current = pd.to_numeric(merged[column], errors="coerce")
        fill_mask = current.isna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, external_column]

    for column in ROLLING_NUMERIC_COLUMNS:
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        current = pd.to_numeric(merged[column], errors="coerce")
        fill_mask = current.isna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, external_column]

    for column in [*RECENCY_WEIGHTED_NUMERIC_COLUMNS, *ROUND_TREND_NUMERIC_COLUMNS]:
        external_column = f"{column}__external"
        if external_column not in merged.columns:
            continue
        if column not in merged.columns:
            merged[column] = merged[external_column]
            continue
        current = pd.to_numeric(merged[column], errors="coerce")
        fill_mask = current.isna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, external_column]

    if "stance__external" in merged.columns:
        if "stance" not in merged.columns:
            merged["stance"] = merged["stance__external"]
        else:
            fill_mask = merged["stance"].fillna("").astype(str).str.strip().eq("")
            merged.loc[fill_mask, "stance"] = merged.loc[fill_mask, "stance__external"]

    if "history_style_label__external" in merged.columns:
        if "history_style_label" not in merged.columns:
            merged["history_style_label"] = merged["history_style_label__external"]
        else:
            fill_mask = merged["history_style_label"].fillna("").astype(str).str.strip().eq("")
            merged.loc[fill_mask, "history_style_label"] = merged.loc[fill_mask, "history_style_label__external"]

    drop_columns = [
        column
        for column in merged.columns
        if column == "fighter_key" or column.endswith("__external")
    ]
    return merged.drop(columns=drop_columns)


def _download_csv(session: requests.Session, url: str, cache_path: Path) -> pd.DataFrame:
    text: str
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        text = response.text
        cache_path.write_text(text, encoding="utf-8")
    except requests.RequestException:
        if not cache_path.exists():
            raise
        text = cache_path.read_text(encoding="utf-8")
    return pd.read_csv(StringIO(text))


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [
        " ".join(str(column).replace("\xa0", " ").split()).upper()
        for column in normalized.columns
    ]
    return normalized


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split()).strip()


def _normalize_fighter_name(value: object) -> str:
    normalized = unicodedata.normalize("NFKD", _normalize_text(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(normalized.lower().replace(".", "").split())


def _split_bout(raw_bout: object) -> tuple[str, str]:
    bout = _normalize_text(raw_bout)
    parts = _BOUT_SPLIT_PATTERN.split(bout, maxsplit=1)
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _parse_outcome_pair(raw_outcome: object) -> tuple[str, str]:
    parts = [part.strip().upper() for part in str(raw_outcome or "").split("/") if part.strip()]
    if len(parts) != 2:
        return "", ""
    return parts[0], parts[1]


def _parse_landed_attempts(raw_value: object) -> tuple[int, int]:
    match = _LANDED_ATTEMPTS_PATTERN.match(_normalize_text(raw_value))
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _parse_clock_seconds(raw_value: object) -> int:
    cleaned = _normalize_text(raw_value)
    if ":" not in cleaned:
        return 0
    minutes_text, seconds_text = cleaned.split(":", 1)
    try:
        return (int(minutes_text) * 60) + int(seconds_text)
    except ValueError:
        return 0


def _parse_round_number(raw_value: object) -> int:
    match = re.search(r"(\d+)", _normalize_text(raw_value))
    if match is None:
        return 0
    return int(match.group(1))


def _build_fighter_history(fight_results: pd.DataFrame, event_details: pd.DataFrame) -> pd.DataFrame:
    results = _normalize_columns(fight_results)
    events = _normalize_columns(event_details)
    results["EVENT"] = results["EVENT"].map(_normalize_text)
    results["BOUT"] = results["BOUT"].map(_normalize_text)
    events["EVENT"] = events["EVENT"].map(_normalize_text)
    events["DATE"] = pd.to_datetime(events["DATE"], errors="coerce")
    event_dates = events[["EVENT", "DATE"]].dropna(subset=["DATE"]).drop_duplicates(subset=["EVENT"], keep="first")
    results = results.merge(event_dates, on="EVENT", how="left")

    rows: list[dict[str, object]] = []
    for row in results.to_dict(orient="records"):
        fighter_a, fighter_b = _split_bout(row.get("BOUT"))
        result_a, result_b = _parse_outcome_pair(row.get("OUTCOME"))
        if not fighter_a or not fighter_b:
            continue
        round_number = pd.to_numeric(row.get("ROUND"), errors="coerce")
        minutes = _fight_duration_minutes(round_number, row.get("TIME"))
        for fighter_name, result_code in ((fighter_a, result_a), (fighter_b, result_b)):
            opponent_name = fighter_b if fighter_name == fighter_a else fighter_a
            rows.append(
                {
                    "event": _normalize_text(row.get("EVENT")),
                    "bout": _normalize_text(row.get("BOUT")),
                    "fighter_key": _normalize_fighter_name(fighter_name),
                    "opponent_key": _normalize_fighter_name(opponent_name),
                    "date": row.get("DATE"),
                    "event_name": _normalize_text(row.get("EVENT")),
                    "weight_class": _normalize_text(row.get("WEIGHTCLASS")),
                    "result": result_code,
                    "result_code": result_code,
                    "decision_type": _normalize_text(row.get("METHOD")),
                    "round_number": round_number,
                    "minutes": float(minutes) if minutes is not None else 0.0,
                }
            )

    history = pd.DataFrame(rows)
    if history.empty:
        return history
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    return history


def _build_opponent_strength_frame(history: pd.DataFrame, metrics_frame: pd.DataFrame) -> pd.DataFrame:
    if history.empty or metrics_frame.empty or "opponent_key" not in history.columns:
        return pd.DataFrame(
            columns=[
                "fighter_key",
                "opponent_avg_win_rate",
                "opponent_avg_ufc_fight_count",
                "opponent_avg_recent_result_score",
                "opponent_avg_finish_win_rate",
                "opponent_quality_score",
                "recent_opponent_quality_score",
            ]
        )

    opponent_metrics = metrics_frame[
        [
            "fighter_key",
            "career_win_rate",
            "ufc_fight_count",
            "recent_result_score",
            "finish_win_rate",
        ]
    ].rename(
        columns={
            "fighter_key": "opponent_key",
            "career_win_rate": "opponent_win_rate",
            "ufc_fight_count": "opponent_ufc_fight_count",
            "recent_result_score": "opponent_recent_result_score",
            "finish_win_rate": "opponent_finish_win_rate",
        }
    )
    working = history[["fighter_key", "opponent_key", "date"]].merge(opponent_metrics, on="opponent_key", how="left")
    if working.empty:
        return pd.DataFrame(
            columns=[
                "fighter_key",
                "opponent_avg_win_rate",
                "opponent_avg_ufc_fight_count",
                "opponent_avg_recent_result_score",
                "opponent_avg_finish_win_rate",
                "opponent_quality_score",
                "recent_opponent_quality_score",
            ]
        )

    working["opponent_win_rate"] = pd.to_numeric(working["opponent_win_rate"], errors="coerce").fillna(0.5)
    working["opponent_ufc_fight_count"] = pd.to_numeric(
        working["opponent_ufc_fight_count"],
        errors="coerce",
    ).fillna(0.0)
    working["opponent_recent_result_score"] = pd.to_numeric(
        working["opponent_recent_result_score"],
        errors="coerce",
    ).fillna(0.0)
    working["opponent_finish_win_rate"] = pd.to_numeric(
        working["opponent_finish_win_rate"],
        errors="coerce",
    ).fillna(0.0)
    working["opponent_quality_score"] = (
        (working["opponent_win_rate"].clip(lower=0.0, upper=1.0) * 0.45)
        + ((working["opponent_ufc_fight_count"].clip(lower=0.0, upper=15.0) / 15.0) * 0.20)
        + (((working["opponent_recent_result_score"].clip(lower=-1.5, upper=1.5) + 1.5) / 3.0) * 0.20)
        + (working["opponent_finish_win_rate"].clip(lower=0.0, upper=1.0) * 0.15)
    ).round(4)

    aggregated = (
        working.groupby("fighter_key", as_index=False)
        .agg(
            opponent_avg_win_rate=("opponent_win_rate", "mean"),
            opponent_avg_ufc_fight_count=("opponent_ufc_fight_count", "mean"),
            opponent_avg_recent_result_score=("opponent_recent_result_score", "mean"),
            opponent_avg_finish_win_rate=("opponent_finish_win_rate", "mean"),
            opponent_quality_score=("opponent_quality_score", "mean"),
        )
    )
    recent = (
        working.sort_values("date", ascending=False, na_position="last")
        .groupby("fighter_key", sort=False)
        .head(3)
        .groupby("fighter_key", as_index=False)["opponent_quality_score"]
        .mean()
        .rename(columns={"opponent_quality_score": "recent_opponent_quality_score"})
    )
    return aggregated.merge(recent, on="fighter_key", how="left")


def _build_fight_level_stats(
    fight_stats: pd.DataFrame,
    *,
    fight_results: pd.DataFrame | None = None,
) -> pd.DataFrame:
    stats = _normalize_columns(fight_stats)
    stats["EVENT"] = stats["EVENT"].map(_normalize_text)
    stats["BOUT"] = stats["BOUT"].map(_normalize_text)
    stats["fighter_key"] = stats["FIGHTER"].map(_normalize_fighter_name)
    if "ROUND" in stats.columns:
        stats["round_number"] = stats["ROUND"].map(_parse_round_number)
    else:
        stats["round_number"] = 1
    stats = stats.loc[stats["round_number"] > 0].copy()

    sig_pairs = stats["SIG.STR."].map(_parse_landed_attempts)
    td_pairs = stats["TD"].map(_parse_landed_attempts)
    head_pairs = stats["HEAD"].map(_parse_landed_attempts)
    body_pairs = stats["BODY"].map(_parse_landed_attempts)
    leg_pairs = stats["LEG"].map(_parse_landed_attempts)
    distance_pairs = stats["DISTANCE"].map(_parse_landed_attempts)
    clinch_pairs = stats["CLINCH"].map(_parse_landed_attempts)
    ground_pairs = stats["GROUND"].map(_parse_landed_attempts)
    stats["sig_landed"] = sig_pairs.map(lambda pair: pair[0])
    stats["sig_attempted"] = sig_pairs.map(lambda pair: pair[1])
    stats["takedown_landed"] = td_pairs.map(lambda pair: pair[0])
    stats["takedown_attempted"] = td_pairs.map(lambda pair: pair[1])
    stats["knockdowns"] = pd.to_numeric(stats["KD"], errors="coerce").fillna(0.0)
    stats["head_landed"] = head_pairs.map(lambda pair: pair[0])
    stats["body_landed"] = body_pairs.map(lambda pair: pair[0])
    stats["leg_landed"] = leg_pairs.map(lambda pair: pair[0])
    stats["distance_landed"] = distance_pairs.map(lambda pair: pair[0])
    stats["clinch_landed"] = clinch_pairs.map(lambda pair: pair[0])
    stats["ground_landed"] = ground_pairs.map(lambda pair: pair[0])
    stats["submission_attempts"] = pd.to_numeric(stats["SUB.ATT"], errors="coerce").fillna(0.0)
    stats["control_seconds"] = stats["CTRL"].map(_parse_clock_seconds)
    stats["round_duration_minutes"] = 5.0

    if fight_results is not None and not stats.empty:
        results = _normalize_columns(fight_results)
        results["EVENT"] = results["EVENT"].map(_normalize_text)
        results["BOUT"] = results["BOUT"].map(_normalize_text)
        results["final_round_number"] = pd.to_numeric(results["ROUND"], errors="coerce").fillna(0).astype(int)
        results["final_round_minutes"] = results["TIME"].map(_parse_clock_seconds).astype(float) / 60.0
        results.loc[results["final_round_minutes"] <= 0, "final_round_minutes"] = 5.0
        round_lookup = results[
            ["EVENT", "BOUT", "final_round_number", "final_round_minutes"]
        ].drop_duplicates(subset=["EVENT", "BOUT"], keep="first")
        stats = stats.merge(
            round_lookup,
            left_on=["EVENT", "BOUT", "round_number"],
            right_on=["EVENT", "BOUT", "final_round_number"],
            how="left",
        )
        final_round_mask = stats["final_round_minutes"].notna()
        stats.loc[final_round_mask, "round_duration_minutes"] = stats.loc[final_round_mask, "final_round_minutes"]
        stats = stats.drop(columns=["final_round_number", "final_round_minutes"])

    round_grouped = (
        stats.groupby(["EVENT", "BOUT", "round_number", "fighter_key"], as_index=False)[
            [
                "sig_landed",
                "sig_attempted",
                "takedown_landed",
                "takedown_attempted",
                "submission_attempts",
                "control_seconds",
                "knockdowns",
                "round_duration_minutes",
            ]
        ]
        .sum()
        .rename(
            columns={
                "sig_landed": "sig_landed_total",
                "sig_attempted": "sig_attempted_total",
                "takedown_landed": "takedown_landed_total",
                "takedown_attempted": "takedown_attempted_total",
                "submission_attempts": "submission_attempt_total",
                "control_seconds": "control_seconds_total",
                "knockdowns": "knockdown_total",
            }
        )
    )
    if not round_grouped.empty:
        round_grouped["fight_sig_landed_total"] = round_grouped.groupby(["EVENT", "BOUT", "round_number"])["sig_landed_total"].transform("sum")
        round_grouped["fight_sig_attempted_total"] = round_grouped.groupby(["EVENT", "BOUT", "round_number"])["sig_attempted_total"].transform("sum")
        round_grouped["fight_takedown_landed_total"] = round_grouped.groupby(["EVENT", "BOUT", "round_number"])["takedown_landed_total"].transform("sum")
        round_grouped["fight_takedown_attempted_total"] = round_grouped.groupby(["EVENT", "BOUT", "round_number"])["takedown_attempted_total"].transform("sum")
        round_grouped["sig_absorbed_total"] = round_grouped["fight_sig_landed_total"] - round_grouped["sig_landed_total"]
        round_grouped["sig_attempted_against_total"] = round_grouped["fight_sig_attempted_total"] - round_grouped["sig_attempted_total"]
        round_grouped["takedowns_absorbed_total"] = round_grouped["fight_takedown_landed_total"] - round_grouped["takedown_landed_total"]
        round_grouped["takedown_attempted_against_total"] = round_grouped["fight_takedown_attempted_total"] - round_grouped["takedown_attempted_total"]

    fight_trends = _build_fight_round_trend_frame(round_grouped)

    grouped = (
        stats.groupby(["EVENT", "BOUT", "fighter_key"], as_index=False)[
            [
                "sig_landed",
                "sig_attempted",
                "takedown_landed",
                "takedown_attempted",
                "submission_attempts",
                "control_seconds",
                "knockdowns",
                "head_landed",
                "body_landed",
                "leg_landed",
                "distance_landed",
                "clinch_landed",
                "ground_landed",
            ]
        ]
        .sum()
        .rename(
            columns={
                "EVENT": "event",
                "BOUT": "bout",
                "sig_landed": "sig_landed_total",
                "sig_attempted": "sig_attempted_total",
                "takedown_landed": "takedown_landed_total",
                "takedown_attempted": "takedown_attempted_total",
                "submission_attempts": "submission_attempt_total",
                "control_seconds": "control_seconds_total",
                "knockdowns": "knockdown_total",
                "head_landed": "head_landed_total",
                "body_landed": "body_landed_total",
                "leg_landed": "leg_landed_total",
                "distance_landed": "distance_landed_total",
                "clinch_landed": "clinch_landed_total",
                "ground_landed": "ground_landed_total",
            }
        )
    )
    if grouped.empty:
        return grouped

    grouped["fight_sig_landed_total"] = grouped.groupby(["event", "bout"])["sig_landed_total"].transform("sum")
    grouped["fight_sig_attempted_total"] = grouped.groupby(["event", "bout"])["sig_attempted_total"].transform("sum")
    grouped["fight_takedown_landed_total"] = grouped.groupby(["event", "bout"])["takedown_landed_total"].transform("sum")
    grouped["fight_takedown_attempted_total"] = grouped.groupby(["event", "bout"])["takedown_attempted_total"].transform("sum")
    grouped["sig_absorbed_total"] = grouped["fight_sig_landed_total"] - grouped["sig_landed_total"]
    grouped["sig_attempted_against_total"] = grouped["fight_sig_attempted_total"] - grouped["sig_attempted_total"]
    grouped["takedowns_absorbed_total"] = grouped["fight_takedown_landed_total"] - grouped["takedown_landed_total"]
    grouped["takedown_attempted_against_total"] = grouped["fight_takedown_attempted_total"] - grouped["takedown_attempted_total"]
    grouped = grouped.drop(
        columns=[
            "fight_sig_landed_total",
            "fight_sig_attempted_total",
            "fight_takedown_landed_total",
            "fight_takedown_attempted_total",
        ]
    )
    if not fight_trends.empty:
        grouped = grouped.merge(fight_trends, on=["event", "bout", "fighter_key"], how="left")
    return grouped


def _build_fight_round_trend_frame(round_grouped: pd.DataFrame) -> pd.DataFrame:
    if round_grouped.empty:
        return pd.DataFrame(
            columns=[
                "event",
                "bout",
                "fighter_key",
                "strike_round_trend_fight",
                "grappling_round_trend_fight",
                "control_round_trend_fight",
                "strike_pace_round_trend_fight",
            ]
        )

    trend_rows: list[dict[str, object]] = []
    for (event, bout, fighter_key), fighter_rounds in round_grouped.groupby(["EVENT", "BOUT", "fighter_key"], sort=False):
        ordered = fighter_rounds.sort_values("round_number", ascending=True).reset_index(drop=True)
        early = ordered.loc[ordered["round_number"] == 1]
        late = ordered.loc[ordered["round_number"] > 1]

        strike_round_trend = float("nan")
        grappling_round_trend = float("nan")
        control_round_trend = float("nan")
        strike_pace_round_trend = float("nan")
        if not early.empty and not late.empty:
            early_minutes = float(pd.to_numeric(early["round_duration_minutes"], errors="coerce").fillna(0.0).sum())
            late_minutes = float(pd.to_numeric(late["round_duration_minutes"], errors="coerce").fillna(0.0).sum())
            if early_minutes > 0 and late_minutes > 0:
                early_sig_margin = (
                    pd.to_numeric(early["sig_landed_total"], errors="coerce").fillna(0.0).sum()
                    - pd.to_numeric(early["sig_absorbed_total"], errors="coerce").fillna(0.0).sum()
                ) / early_minutes
                late_sig_margin = (
                    pd.to_numeric(late["sig_landed_total"], errors="coerce").fillna(0.0).sum()
                    - pd.to_numeric(late["sig_absorbed_total"], errors="coerce").fillna(0.0).sum()
                ) / late_minutes
                early_grappling = (
                    pd.to_numeric(early["takedown_landed_total"], errors="coerce").fillna(0.0).sum() / early_minutes
                ) * 15
                late_grappling = (
                    pd.to_numeric(late["takedown_landed_total"], errors="coerce").fillna(0.0).sum() / late_minutes
                ) * 15
                early_control = (
                    pd.to_numeric(early["control_seconds_total"], errors="coerce").fillna(0.0).sum()
                    / 60.0
                    / early_minutes
                ) * 15
                late_control = (
                    pd.to_numeric(late["control_seconds_total"], errors="coerce").fillna(0.0).sum()
                    / 60.0
                    / late_minutes
                ) * 15
                early_strike_pace = pd.to_numeric(early["sig_attempted_total"], errors="coerce").fillna(0.0).sum() / early_minutes
                late_strike_pace = pd.to_numeric(late["sig_attempted_total"], errors="coerce").fillna(0.0).sum() / late_minutes

                strike_round_trend = round(late_sig_margin - early_sig_margin, 3)
                grappling_round_trend = round(late_grappling - early_grappling, 3)
                control_round_trend = round(late_control - early_control, 3)
                strike_pace_round_trend = round(late_strike_pace - early_strike_pace, 3)

        trend_rows.append(
            {
                "event": event,
                "bout": bout,
                "fighter_key": fighter_key,
                "strike_round_trend_fight": strike_round_trend,
                "grappling_round_trend_fight": grappling_round_trend,
                "control_round_trend_fight": control_round_trend,
                "strike_pace_round_trend_fight": strike_pace_round_trend,
            }
        )

    return pd.DataFrame(trend_rows)


def _build_profile_frame(fighter_tott: pd.DataFrame) -> tuple[pd.DataFrame, set[str]]:
    profiles = _normalize_columns(fighter_tott)
    profiles["fighter_key"] = profiles["FIGHTER"].map(_normalize_fighter_name)
    profiles["URL"] = profiles["URL"].map(_normalize_text)
    profiles = profiles.drop_duplicates(subset=["fighter_key", "URL"], keep="first")

    ambiguous_keys = set(
        profiles.groupby("fighter_key")
        .size()
        .loc[lambda counts: counts > 1]
        .index
        .tolist()
    )
    profiles = profiles.loc[~profiles["fighter_key"].isin(ambiguous_keys)].copy()

    profiles["height_in"] = profiles["HEIGHT"].map(lambda value: _parse_height_to_inches(_normalize_text(value)) or 0.0)
    profiles["reach_in"] = profiles["REACH"].map(lambda value: _parse_reach_to_inches(_normalize_text(value)) or 0.0)
    profiles["age_years"] = profiles["DOB"].map(_age_years_from_birthdate)
    profiles["stance"] = profiles["STANCE"].map(_normalize_text)

    columns = ["fighter_key", "age_years", "height_in", "reach_in", "stance"]
    return profiles[columns].drop_duplicates(subset=["fighter_key"], keep="first"), ambiguous_keys


def _build_metric_row(history: pd.DataFrame) -> dict[str, float]:
    wins = float((history["result_code"].astype(str).str.upper() == "W").sum()) if "result_code" in history.columns else 0.0
    losses = float((history["result_code"].astype(str).str.upper() == "L").sum()) if "result_code" in history.columns else 0.0
    total_results = wins + losses
    career_win_rate = round((wins / total_results), 4) if total_results > 0 else 0.5
    recent_result_score = _recent_result_score(history)
    losses_in_row = _loss_streak(history)
    ufc_fight_count = float(len(history))
    first_round_finish_wins, first_round_finish_rate = _first_round_finish_metrics(history)
    (
        finish_win_rate,
        finish_loss_rate,
        decision_rate,
        ko_win_rate,
        submission_win_rate,
        ko_loss_rate,
        submission_loss_rate,
    ) = _outcome_profile_metrics(history)
    (
        recent_finish_loss_count,
        recent_ko_loss_count,
        recent_finish_loss_365d,
        recent_ko_loss_365d,
        recent_damage_score,
    ) = _recent_damage_metrics(history)

    recent_slice = history.head(3)
    recent_minutes = float(pd.to_numeric(recent_slice["minutes"], errors="coerce").fillna(0.0).sum())
    if recent_minutes > 0:
        recent_strike_margin_per_min = round(
            (
                pd.to_numeric(recent_slice["sig_landed_total"], errors="coerce").fillna(0.0).sum()
                - pd.to_numeric(recent_slice["sig_absorbed_total"], errors="coerce").fillna(0.0).sum()
            )
            / recent_minutes,
            3,
        )
        recent_grappling_rate = round(
            pd.to_numeric(recent_slice["takedown_landed_total"], errors="coerce").fillna(0.0).sum() / recent_minutes * 15,
            3,
        )
        recent_control_avg = round(
            (
                pd.to_numeric(recent_slice["control_seconds_total"], errors="coerce").fillna(0.0).sum()
                / 60.0
                / recent_minutes
            )
            * 15,
            3,
        )
    else:
        recent_strike_margin_per_min = 0.0
        recent_grappling_rate = 0.0
        recent_control_avg = 0.0

    total_minutes = float(pd.to_numeric(history["minutes"], errors="coerce").fillna(0.0).sum())
    total_sig_landed = float(pd.to_numeric(history["sig_landed_total"], errors="coerce").fillna(0.0).sum())
    total_sig_attempted = float(pd.to_numeric(history["sig_attempted_total"], errors="coerce").fillna(0.0).sum())
    total_sig_attempted_against = float(pd.to_numeric(history["sig_attempted_against_total"], errors="coerce").fillna(0.0).sum())
    total_sig_absorbed = float(pd.to_numeric(history["sig_absorbed_total"], errors="coerce").fillna(0.0).sum())
    total_takedown_landed = float(pd.to_numeric(history["takedown_landed_total"], errors="coerce").fillna(0.0).sum())
    total_takedown_attempted = float(pd.to_numeric(history["takedown_attempted_total"], errors="coerce").fillna(0.0).sum())
    total_takedown_attempted_against = float(pd.to_numeric(history["takedown_attempted_against_total"], errors="coerce").fillna(0.0).sum())
    total_takedowns_absorbed = float(pd.to_numeric(history["takedowns_absorbed_total"], errors="coerce").fillna(0.0).sum())
    total_knockdowns = float(pd.to_numeric(history["knockdown_total"], errors="coerce").fillna(0.0).sum())
    if total_minutes > 0:
        submission_avg = round(
            pd.to_numeric(history["submission_attempt_total"], errors="coerce").fillna(0.0).sum() / total_minutes * 15,
            3,
        )
        control_avg = round(
            (
                pd.to_numeric(history["control_seconds_total"], errors="coerce").fillna(0.0).sum()
                / 60.0
                / total_minutes
            )
            * 15,
            3,
        )
        knockdown_avg = round((total_knockdowns / total_minutes) * 15, 3)
    else:
        submission_avg = 0.0
        control_avg = 0.0
        knockdown_avg = 0.0

    strike_accuracy_pct = round((total_sig_landed / total_sig_attempted) * 100, 2) if total_sig_attempted > 0 else 0.0
    strike_defense_pct = round((1 - (total_sig_absorbed / total_sig_attempted_against)) * 100, 2) if total_sig_attempted_against > 0 else 0.0
    takedown_accuracy_pct = round((total_takedown_landed / total_takedown_attempted) * 100, 2) if total_takedown_attempted > 0 else 0.0
    takedown_defense_pct = round((1 - (total_takedowns_absorbed / total_takedown_attempted_against)) * 100, 2) if total_takedown_attempted_against > 0 else 0.0

    head_share = round(pd.to_numeric(history["head_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    body_share = round(pd.to_numeric(history["body_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    leg_share = round(pd.to_numeric(history["leg_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    distance_share = round(pd.to_numeric(history["distance_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    clinch_share = round(pd.to_numeric(history["clinch_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    ground_share = round(pd.to_numeric(history["ground_landed_total"], errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    rolling_metrics = _build_rolling_metrics(history)
    recency_weighted_metrics = _build_recency_weighted_metrics(history)
    round_trend_metrics = _build_round_trend_metrics(history)
    history_style_label = derive_style_label(
        stance="",
        strike_margin=recency_weighted_metrics["recency_weighted_strike_margin"],
        grappling_rate=recency_weighted_metrics["recency_weighted_grappling_rate"],
        control_avg=recency_weighted_metrics["recency_weighted_control_avg"],
        ko_win_rate=ko_win_rate,
        submission_win_rate=submission_win_rate,
        decision_rate=decision_rate,
        knockdown_avg=knockdown_avg,
        distance_strike_share=distance_share,
        clinch_strike_share=clinch_share,
        ground_strike_share=ground_share,
    )

    last_fight_date = history["date"].max() if "date" in history.columns else None
    return {
        "recent_strike_margin_per_min": recent_strike_margin_per_min,
        "recent_grappling_rate": recent_grappling_rate,
        "control_avg": control_avg,
        "recent_control_avg": recent_control_avg,
        "recent_result_score": recent_result_score,
        "losses_in_row": float(losses_in_row),
        "first_round_finish_wins": float(first_round_finish_wins),
        "first_round_finish_rate": first_round_finish_rate,
        "finish_win_rate": finish_win_rate,
        "finish_loss_rate": finish_loss_rate,
        "decision_rate": decision_rate,
        "submission_avg": submission_avg,
        "ko_win_rate": ko_win_rate,
        "submission_win_rate": submission_win_rate,
        "ko_loss_rate": ko_loss_rate,
        "submission_loss_rate": submission_loss_rate,
        "recent_finish_loss_count": float(recent_finish_loss_count),
        "recent_ko_loss_count": float(recent_ko_loss_count),
        "recent_finish_loss_365d": float(recent_finish_loss_365d),
        "recent_ko_loss_365d": float(recent_ko_loss_365d),
        "recent_damage_score": recent_damage_score,
        "days_since_last_fight": float(_days_since(last_fight_date)),
        "ufc_fight_count": ufc_fight_count,
        "ufc_debut_flag": 0.0,
        "career_win_rate": career_win_rate,
        "strike_accuracy_pct": strike_accuracy_pct,
        "strike_defense_pct": strike_defense_pct,
        "takedown_accuracy_pct": takedown_accuracy_pct,
        "takedown_defense_pct": takedown_defense_pct,
        "knockdown_avg": knockdown_avg,
        "head_strike_share": head_share,
        "body_strike_share": body_share,
        "leg_strike_share": leg_share,
        "distance_strike_share": distance_share,
        "clinch_strike_share": clinch_share,
        "ground_strike_share": ground_share,
        "history_style_label": history_style_label,
        **recency_weighted_metrics,
        **round_trend_metrics,
        **rolling_metrics,
    }


def _build_rolling_metrics(history: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for window_size in (1, 3, 5):
        window = history.head(window_size)
        suffix = f"last_{window_size}"
        window_minutes = float(pd.to_numeric(window.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        window_sig_landed = float(pd.to_numeric(window.get("sig_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        window_sig_absorbed = float(pd.to_numeric(window.get("sig_absorbed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        window_sig_attempted = float(pd.to_numeric(window.get("sig_attempted_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        window_takedown_landed = float(pd.to_numeric(window.get("takedown_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
        window_control_seconds = float(pd.to_numeric(window.get("control_seconds_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

        if window_minutes > 0:
            metrics[f"strike_margin_{suffix}"] = round((window_sig_landed - window_sig_absorbed) / window_minutes, 3)
            metrics[f"grappling_rate_{suffix}"] = round((window_takedown_landed / window_minutes) * 15, 3)
            metrics[f"control_avg_{suffix}"] = round(((window_control_seconds / 60.0) / window_minutes) * 15, 3)
            metrics[f"strike_pace_{suffix}"] = round(window_sig_attempted / window_minutes, 3)
        else:
            metrics[f"strike_margin_{suffix}"] = 0.0
            metrics[f"grappling_rate_{suffix}"] = 0.0
            metrics[f"control_avg_{suffix}"] = 0.0
            metrics[f"strike_pace_{suffix}"] = 0.0

        metrics[f"result_score_{suffix}"] = _recent_result_score(window)
        if window_size >= 3:
            finish_win_rate, finish_loss_rate, *_ = _outcome_profile_metrics(window)
            metrics[f"finish_win_rate_{suffix}"] = finish_win_rate
            metrics[f"finish_loss_rate_{suffix}"] = finish_loss_rate

    return metrics


def _build_recency_weighted_metrics(history: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {column: 0.0 for column in RECENCY_WEIGHTED_NUMERIC_COLUMNS}

    weights = pd.Series(
        [0.5 ** (index / 2.0) for index in range(len(history))],
        index=history.index,
        dtype=float,
    )
    minutes = pd.to_numeric(history.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    strike_margin = (
        pd.to_numeric(history.get("sig_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        - pd.to_numeric(history.get("sig_absorbed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    )
    strike_margin = strike_margin.where(minutes > 0, pd.NA) / minutes.where(minutes > 0, pd.NA)
    grappling_rate = (
        pd.to_numeric(history.get("takedown_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        / minutes.where(minutes > 0, pd.NA)
    ) * 15
    control_avg = (
        pd.to_numeric(history.get("control_seconds_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        / 60.0
        / minutes.where(minutes > 0, pd.NA)
    ) * 15
    strike_pace = (
        pd.to_numeric(history.get("sig_attempted_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        / minutes.where(minutes > 0, pd.NA)
    )
    result_score = history.get("result_code", pd.Series(dtype=object)).astype(str).str.upper().map({"W": 1.0, "L": -1.0, "D": 0.0}).fillna(0.0)
    decision_text = history.get("decision_type", pd.Series(dtype=object)).astype(str).str.upper()
    is_decision = decision_text.str.contains("DEC", na=False)
    finish_win_rate = ((history.get("result_code", pd.Series(dtype=object)).astype(str).str.upper() == "W") & ~is_decision).astype(float)
    finish_loss_rate = ((history.get("result_code", pd.Series(dtype=object)).astype(str).str.upper() == "L") & ~is_decision).astype(float)

    return {
        "recency_weighted_strike_margin": round(_weighted_mean(strike_margin, weights), 3),
        "recency_weighted_grappling_rate": round(_weighted_mean(grappling_rate, weights), 3),
        "recency_weighted_control_avg": round(_weighted_mean(control_avg, weights), 3),
        "recency_weighted_strike_pace": round(_weighted_mean(strike_pace, weights), 3),
        "recency_weighted_result_score": round(_weighted_mean(result_score, weights), 3),
        "recency_weighted_finish_win_rate": round(_weighted_mean(finish_win_rate, weights), 3),
        "recency_weighted_finish_loss_rate": round(_weighted_mean(finish_loss_rate, weights), 3),
    }


def _build_round_trend_metrics(history: pd.DataFrame) -> dict[str, float]:
    if history.empty:
        return {column: 0.0 for column in ROUND_TREND_NUMERIC_COLUMNS}

    weights = pd.Series(
        [0.5 ** (index / 2.0) for index in range(len(history))],
        index=history.index,
        dtype=float,
    )
    return {
        "strike_round_trend": round(_weighted_mean(history.get("strike_round_trend_fight", pd.Series(dtype=float)), weights), 3),
        "grappling_round_trend": round(_weighted_mean(history.get("grappling_round_trend_fight", pd.Series(dtype=float)), weights), 3),
        "control_round_trend": round(_weighted_mean(history.get("control_round_trend_fight", pd.Series(dtype=float)), weights), 3),
        "strike_pace_round_trend": round(_weighted_mean(history.get("strike_pace_round_trend_fight", pd.Series(dtype=float)), weights), 3),
    }


def _weighted_mean(values: pd.Series | object, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    valid_mask = numeric_values.notna()
    if not valid_mask.any():
        return 0.0
    valid_weights = weights.loc[valid_mask]
    weight_sum = float(valid_weights.sum())
    if weight_sum <= 0:
        return 0.0
    return float((numeric_values.loc[valid_mask] * valid_weights).sum() / weight_sum)
