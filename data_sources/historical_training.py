from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_sources.espn import _first_round_finish_metrics, _loss_streak, _outcome_profile_metrics, _recent_result_score
from data_sources.external_ufc_history import (
    _build_fight_level_stats,
    _build_fighter_history,
    _build_profile_frame,
    _build_recency_weighted_metrics,
    _build_rolling_metrics,
    _build_round_trend_metrics,
    _normalize_columns,
    _normalize_fighter_name,
    _normalize_text,
    load_external_ufc_history_datasets,
)
from features.fighter_features import OPTIONAL_FIGHTER_DEFAULTS, OPTIONAL_FIGHTER_STRING_DEFAULTS, build_fight_features
from features.style_profile import derive_style_label
from models.projection import project_fight_probabilities
from normalization.odds import normalize_odds_frame

DEFAULT_ALIAS_OVERRIDE_COLUMNS = ["source_name", "canonical_name", "notes"]
DEFAULT_UNMATCHED_FIGHT_COLUMNS = [
    "event_id",
    "event_name",
    "start_time",
    "fighter_a",
    "fighter_b",
    "resolved_fighter_a",
    "resolved_fighter_b",
    "fighter_a_key",
    "fighter_b_key",
    "fighter_a_alias_applied",
    "fighter_b_alias_applied",
    "reason",
    "candidate_count",
    "nearest_event",
    "nearest_bout",
    "nearest_date",
    "nearest_date_gap_days",
]
DEFAULT_UNMATCHED_FIGHTER_COLUMNS = [
    "source_name",
    "resolved_name",
    "fighter_key",
    "alias_applied",
    "unmatched_fight_count",
    "reasons",
    "example_event_id",
    "example_event_name",
    "example_opponent",
    "nearest_event",
    "nearest_bout",
]


def load_historical_training_datasets(cache_dir: str | Path) -> dict[str, pd.DataFrame]:
    return load_external_ufc_history_datasets(cache_dir)


def load_historical_alias_overrides(path: str | Path) -> pd.DataFrame:
    alias_path = Path(path)
    if not alias_path.exists():
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)

    loaded = pd.read_csv(alias_path)
    if loaded.empty:
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)

    column_lookup = {
        _normalize_column_name(column_name): str(column_name)
        for column_name in loaded.columns
    }
    source_column = (
        column_lookup.get("sourcename")
        or column_lookup.get("aliasname")
        or column_lookup.get("fightername")
    )
    canonical_column = (
        column_lookup.get("canonicalname")
        or column_lookup.get("targetname")
        or column_lookup.get("greconame")
    )
    notes_column = column_lookup.get("notes")
    if source_column is None or canonical_column is None:
        raise ValueError(
            "Historical alias override CSV must contain source_name and canonical_name columns."
        )

    prepared = pd.DataFrame(
        {
            "source_name": loaded[source_column].fillna("").astype(str).str.strip(),
            "canonical_name": loaded[canonical_column].fillna("").astype(str).str.strip(),
            "notes": loaded[notes_column].fillna("").astype(str).str.strip()
            if notes_column is not None
            else "",
        }
    )
    prepared = prepared.loc[
        prepared["source_name"].ne("") & prepared["canonical_name"].ne("")
    ].copy()
    if prepared.empty:
        return pd.DataFrame(columns=DEFAULT_ALIAS_OVERRIDE_COLUMNS)
    return prepared.drop_duplicates(subset=["source_name"], keep="last").reset_index(drop=True)


def build_unmatched_fighter_report(unmatched_fights: pd.DataFrame) -> pd.DataFrame:
    if unmatched_fights.empty:
        return pd.DataFrame(columns=DEFAULT_UNMATCHED_FIGHTER_COLUMNS)

    records: list[dict[str, object]] = []
    for row in unmatched_fights.to_dict(orient="records"):
        for side in ("a", "b"):
            fighter_name = str(row.get(f"fighter_{side}", "") or "").strip()
            if not fighter_name:
                continue
            records.append(
                {
                    "source_name": fighter_name,
                    "resolved_name": str(row.get(f"resolved_fighter_{side}", "") or "").strip(),
                    "fighter_key": str(row.get(f"fighter_{side}_key", "") or "").strip(),
                    "alias_applied": bool(row.get(f"fighter_{side}_alias_applied", False)),
                    "reason": str(row.get("reason", "") or "").strip(),
                    "example_event_id": str(row.get("event_id", "") or "").strip(),
                    "example_event_name": str(row.get("event_name", "") or "").strip(),
                    "example_opponent": str(
                        row.get("fighter_b" if side == "a" else "fighter_a", "") or ""
                    ).strip(),
                    "nearest_event": str(row.get("nearest_event", "") or "").strip(),
                    "nearest_bout": str(row.get("nearest_bout", "") or "").strip(),
                }
            )

    if not records:
        return pd.DataFrame(columns=DEFAULT_UNMATCHED_FIGHTER_COLUMNS)

    expanded = pd.DataFrame(records)
    grouped_rows: list[dict[str, object]] = []
    group_columns = ["source_name", "resolved_name", "fighter_key", "alias_applied"]
    for group_key, fighter_rows in expanded.groupby(group_columns, dropna=False, sort=True):
        grouped_rows.append(
            {
                "source_name": group_key[0],
                "resolved_name": group_key[1],
                "fighter_key": group_key[2],
                "alias_applied": bool(group_key[3]),
                "unmatched_fight_count": int(len(fighter_rows)),
                "reasons": ", ".join(sorted(set(fighter_rows["reason"].astype(str)))),
                "example_event_id": str(fighter_rows.iloc[0]["example_event_id"]),
                "example_event_name": str(fighter_rows.iloc[0]["example_event_name"]),
                "example_opponent": str(fighter_rows.iloc[0]["example_opponent"]),
                "nearest_event": str(fighter_rows.iloc[0]["nearest_event"]),
                "nearest_bout": str(fighter_rows.iloc[0]["nearest_bout"]),
            }
        )

    return (
        pd.DataFrame(grouped_rows, columns=DEFAULT_UNMATCHED_FIGHTER_COLUMNS)
        .sort_values(["unmatched_fight_count", "source_name"], ascending=[False, True])
        .reset_index(drop=True)
    )


def write_historical_unmatched_reports(
    unmatched_fights: pd.DataFrame,
    *,
    unmatched_fights_output: str | Path | None = None,
    unmatched_fighters_output: str | Path | None = None,
) -> tuple[Path | None, Path | None]:
    fights_path: Path | None = None
    fighters_path: Path | None = None

    if unmatched_fights_output:
        fights_path = Path(unmatched_fights_output)
        fights_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched_fights.reindex(columns=DEFAULT_UNMATCHED_FIGHT_COLUMNS).to_csv(fights_path, index=False)

    if unmatched_fighters_output:
        fighters_path = Path(unmatched_fighters_output)
        fighters_path.parent.mkdir(parents=True, exist_ok=True)
        build_unmatched_fighter_report(unmatched_fights).to_csv(fighters_path, index=False)

    return fights_path, fighters_path


def build_historical_projection_dataset(
    historical_odds: pd.DataFrame,
    *,
    fight_results: pd.DataFrame,
    fight_stats: pd.DataFrame,
    event_details: pd.DataFrame,
    fighter_tott: pd.DataFrame,
    date_tolerance_days: int = 3,
    alias_overrides: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    odds = normalize_odds_frame(historical_odds)
    alias_lookup = _build_alias_lookup(alias_overrides)
    fight_index, profile_lookup = _build_greco_index(
        fight_results=fight_results,
        fight_stats=fight_stats,
        event_details=event_details,
        fighter_tott=fighter_tott,
    )
    unique_fights = odds[
        ["event_id", "event_name", "start_time", "fighter_a", "fighter_b"]
    ].drop_duplicates(ignore_index=True)

    snapshot_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []

    for fight in unique_fights.to_dict(orient="records"):
        fight_date = pd.to_datetime(fight.get("start_time"), errors="coerce")
        fighter_a = str(fight.get("fighter_a", "")).strip()
        fighter_b = str(fight.get("fighter_b", "")).strip()
        resolved_fighter_a, fighter_a_alias_applied = _resolve_fighter_alias(fighter_a, alias_lookup)
        resolved_fighter_b, fighter_b_alias_applied = _resolve_fighter_alias(fighter_b, alias_lookup)
        fighter_a_key = _normalize_fighter_name(resolved_fighter_a)
        fighter_b_key = _normalize_fighter_name(resolved_fighter_b)
        matched_fight, match_diagnostics = _match_fight_row(
            fight_index,
            fighter_a_key=fighter_a_key,
            fighter_b_key=fighter_b_key,
            fight_date=fight_date,
            date_tolerance_days=date_tolerance_days,
        )
        if matched_fight is None:
            unmatched_rows.append(
                {
                    "event_id": fight["event_id"],
                    "event_name": fight["event_name"],
                    "start_time": fight["start_time"],
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "resolved_fighter_a": resolved_fighter_a,
                    "resolved_fighter_b": resolved_fighter_b,
                    "fighter_a_key": fighter_a_key,
                    "fighter_b_key": fighter_b_key,
                    "fighter_a_alias_applied": fighter_a_alias_applied,
                    "fighter_b_alias_applied": fighter_b_alias_applied,
                    **match_diagnostics,
                }
            )
            continue

        snapshot_key_a = f"snapshot::{fight['event_id']}::fighter_a"
        snapshot_key_b = f"snapshot::{fight['event_id']}::fighter_b"
        matched_rows.append(
            {
                **fight,
                "snapshot_key_a": snapshot_key_a,
                "snapshot_key_b": snapshot_key_b,
                "resolved_fighter_a": resolved_fighter_a,
                "resolved_fighter_b": resolved_fighter_b,
                "matched_event": matched_fight["event"],
                "matched_bout": matched_fight["bout"],
                "matched_date": matched_fight["date"],
            }
        )
        snapshot_rows.extend(
            [
                _build_snapshot_row(
                    fight_index=fight_index,
                    profile_lookup=profile_lookup,
                    fighter_name=resolved_fighter_a,
                    fighter_key=fighter_a_key,
                    snapshot_key=snapshot_key_a,
                    target_fight=matched_fight,
                ),
                _build_snapshot_row(
                    fight_index=fight_index,
                    profile_lookup=profile_lookup,
                    fighter_name=resolved_fighter_b,
                    fighter_key=fighter_b_key,
                    snapshot_key=snapshot_key_b,
                    target_fight=matched_fight,
                ),
            ]
        )

    if not matched_rows:
        raise ValueError("No historical odds fights could be matched against the cached Greco UFC history.")

    matched_frame = pd.DataFrame(matched_rows)
    training_odds = odds.merge(
        matched_frame[
            [
                "event_id",
                "event_name",
                "start_time",
                "fighter_a",
                "fighter_b",
                "snapshot_key_a",
                "snapshot_key_b",
            ]
        ],
        on=["event_id", "event_name", "start_time", "fighter_a", "fighter_b"],
        how="inner",
    )
    training_odds["original_fighter_a"] = training_odds["fighter_a"]
    training_odds["original_fighter_b"] = training_odds["fighter_b"]
    training_odds["fighter_a"] = training_odds["snapshot_key_a"]
    training_odds["fighter_b"] = training_odds["snapshot_key_b"]
    training_odds = training_odds.drop(columns=["snapshot_key_a", "snapshot_key_b"])

    snapshot_stats = pd.DataFrame(snapshot_rows).drop_duplicates(subset=["fighter_name"], keep="first")
    features = build_fight_features(training_odds, snapshot_stats)
    projected = project_fight_probabilities(features)
    projected["fighter_a"] = projected["original_fighter_a"]
    projected["fighter_b"] = projected["original_fighter_b"]
    projected = projected.drop(columns=["original_fighter_a", "original_fighter_b"])
    unmatched = pd.DataFrame(unmatched_rows, columns=DEFAULT_UNMATCHED_FIGHT_COLUMNS)
    return projected, unmatched


def _build_greco_index(
    *,
    fight_results: pd.DataFrame,
    fight_stats: pd.DataFrame,
    event_details: pd.DataFrame,
    fighter_tott: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    history = _build_fighter_history(fight_results, event_details)
    fight_totals = _build_fight_level_stats(fight_stats, fight_results=fight_results)
    history = history.merge(fight_totals, on=["event", "bout", "fighter_key"], how="left")
    for column in [
        "sig_landed_total",
        "sig_attempted_total",
        "sig_absorbed_total",
        "sig_attempted_against_total",
        "takedown_landed_total",
        "takedown_attempted_total",
        "takedowns_absorbed_total",
        "takedown_attempted_against_total",
        "submission_attempt_total",
        "control_seconds_total",
        "knockdown_total",
        "head_landed_total",
        "body_landed_total",
        "leg_landed_total",
        "distance_landed_total",
        "clinch_landed_total",
        "ground_landed_total",
    ]:
        if column not in history.columns:
            history[column] = 0.0
        history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["fighter_key"]).copy()

    profiles, ambiguous_keys = _build_profile_frame(fighter_tott)
    fighter_tott_normalized = _normalize_columns(fighter_tott)
    fighter_tott_normalized["fighter_key"] = fighter_tott_normalized["FIGHTER"].map(_normalize_fighter_name)
    fighter_tott_normalized = fighter_tott_normalized.loc[
        ~fighter_tott_normalized["fighter_key"].isin(ambiguous_keys)
    ].copy()
    fighter_tott_normalized["birthdate"] = pd.to_datetime(fighter_tott_normalized["DOB"], errors="coerce")
    profile_frame = profiles.merge(
        fighter_tott_normalized[["fighter_key", "birthdate"]].drop_duplicates(subset=["fighter_key"], keep="first"),
        on="fighter_key",
        how="left",
    )
    profile_lookup = {
        str(row["fighter_key"]): row
        for row in profile_frame.to_dict(orient="records")
    }
    return history, profile_lookup


def _match_fight_row(
    fight_index: pd.DataFrame,
    *,
    fighter_a_key: str,
    fighter_b_key: str,
    fight_date: pd.Timestamp,
    date_tolerance_days: int,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    candidates = fight_index.loc[
        (fight_index["fighter_key"] == fighter_a_key)
        & (fight_index["opponent_key"] == fighter_b_key)
    ].copy()
    if candidates.empty:
        candidates = fight_index.loc[
            (fight_index["fighter_key"] == fighter_b_key)
            & (fight_index["opponent_key"] == fighter_a_key)
        ].copy()
    if candidates.empty:
        return None, {
            "reason": "fighter_pair_not_found",
            "candidate_count": 0,
            "nearest_event": "",
            "nearest_bout": "",
            "nearest_date": "",
            "nearest_date_gap_days": "",
        }
    if pd.isna(fight_date):
        candidate = candidates.sort_values("date", ascending=False, na_position="last").iloc[0]
        return candidate.to_dict(), {
            "reason": "matched_without_start_time",
            "candidate_count": int(len(candidates)),
            "nearest_event": "",
            "nearest_bout": "",
            "nearest_date": "",
            "nearest_date_gap_days": "",
        }

    normalized_fight_date = pd.Timestamp(fight_date)
    if normalized_fight_date.tzinfo is not None:
        normalized_fight_date = normalized_fight_date.tz_localize(None)
    normalized_fight_date = normalized_fight_date.normalize()
    candidates["date_gap_days"] = (candidates["date"].dt.normalize() - normalized_fight_date).abs().dt.days
    within_tolerance = candidates.loc[candidates["date_gap_days"] <= date_tolerance_days].copy()
    if within_tolerance.empty:
        nearest_candidate = candidates.sort_values(
            ["date_gap_days", "date"], ascending=[True, False], na_position="last"
        ).iloc[0]
        return None, {
            "reason": "date_out_of_tolerance",
            "candidate_count": int(len(candidates)),
            "nearest_event": str(nearest_candidate.get("event", "") or ""),
            "nearest_bout": str(nearest_candidate.get("bout", "") or ""),
            "nearest_date": _format_match_date(nearest_candidate.get("date")),
            "nearest_date_gap_days": int(nearest_candidate.get("date_gap_days", 0) or 0),
        }
    candidate = within_tolerance.sort_values(
        ["date_gap_days", "date"], ascending=[True, False], na_position="last"
    ).iloc[0]
    return candidate.to_dict(), {
        "reason": "matched",
        "candidate_count": int(len(candidates)),
        "nearest_event": "",
        "nearest_bout": "",
        "nearest_date": "",
        "nearest_date_gap_days": "",
    }


def _build_alias_lookup(alias_overrides: pd.DataFrame | None) -> dict[str, str]:
    if alias_overrides is None or alias_overrides.empty:
        return {}

    lookup: dict[str, str] = {}
    for row in alias_overrides.to_dict(orient="records"):
        source_name = str(row.get("source_name", "") or "").strip()
        canonical_name = str(row.get("canonical_name", "") or "").strip()
        if not source_name or not canonical_name:
            continue
        lookup[_normalize_fighter_name(source_name)] = canonical_name
    return lookup


def _resolve_fighter_alias(fighter_name: str, alias_lookup: dict[str, str]) -> tuple[str, bool]:
    fighter_key = _normalize_fighter_name(fighter_name)
    resolved_name = str(alias_lookup.get(fighter_key, fighter_name) or "").strip()
    alias_applied = _normalize_text(resolved_name) != _normalize_text(fighter_name)
    return resolved_name or fighter_name, alias_applied


def _normalize_column_name(value: object) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _format_match_date(value: object) -> str:
    date_value = pd.to_datetime(value, errors="coerce")
    if pd.isna(date_value):
        return ""
    return pd.Timestamp(date_value).date().isoformat()


def _build_snapshot_row(
    *,
    fight_index: pd.DataFrame,
    profile_lookup: dict[str, dict[str, object]],
    fighter_name: str,
    fighter_key: str,
    snapshot_key: str,
    target_fight: dict[str, object],
) -> dict[str, object]:
    fighter_history = fight_index.loc[fight_index["fighter_key"] == fighter_key].copy()
    target_event = _normalize_text(target_fight.get("event"))
    target_bout = _normalize_text(target_fight.get("bout"))
    target_date = pd.to_datetime(target_fight.get("date"), errors="coerce")
    if pd.notna(target_date):
        fighter_history = fighter_history.loc[fighter_history["date"] <= target_date].copy()
    prior_history = fighter_history.loc[
        ~((fighter_history["event"] == target_event) & (fighter_history["bout"] == target_bout))
    ].copy()
    prior_history = prior_history.sort_values("date", ascending=False, na_position="last").reset_index(drop=True)

    profile_row = profile_lookup.get(fighter_key, {})
    snapshot = {column: default for column, default in OPTIONAL_FIGHTER_DEFAULTS.items()}
    snapshot.update({column: default for column, default in OPTIONAL_FIGHTER_STRING_DEFAULTS.items()})

    wins = int((prior_history["result_code"].astype(str).str.upper() == "W").sum()) if not prior_history.empty else 0
    losses = int((prior_history["result_code"].astype(str).str.upper() == "L").sum()) if not prior_history.empty else 0
    draws = int((prior_history["result_code"].astype(str).str.upper() == "D").sum()) if not prior_history.empty else 0
    total_fights = wins + losses + draws
    total_minutes = float(pd.to_numeric(prior_history.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_sig_landed = float(pd.to_numeric(prior_history.get("sig_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_sig_absorbed = float(pd.to_numeric(prior_history.get("sig_absorbed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_sig_attempted = float(pd.to_numeric(prior_history.get("sig_attempted_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_sig_attempted_against = float(pd.to_numeric(prior_history.get("sig_attempted_against_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_takedown_landed = float(pd.to_numeric(prior_history.get("takedown_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_takedown_attempted = float(pd.to_numeric(prior_history.get("takedown_attempted_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_takedowns_absorbed = float(pd.to_numeric(prior_history.get("takedowns_absorbed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_takedown_attempted_against = float(pd.to_numeric(prior_history.get("takedown_attempted_against_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    total_knockdowns = float(pd.to_numeric(prior_history.get("knockdown_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    recent_slice = prior_history.head(3)
    recent_minutes = float(pd.to_numeric(recent_slice.get("minutes", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    recent_strike_margin_per_min = 0.0
    recent_grappling_rate = 0.0
    recent_control_avg = 0.0
    if recent_minutes > 0:
        recent_strike_margin_per_min = round(
            (
                pd.to_numeric(recent_slice.get("sig_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
                - pd.to_numeric(recent_slice.get("sig_absorbed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            )
            / recent_minutes,
            3,
        )
        recent_grappling_rate = round(
            pd.to_numeric(recent_slice.get("takedown_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            / recent_minutes
            * 15,
            3,
        )
        recent_control_avg = round(
            (
                pd.to_numeric(recent_slice.get("control_seconds_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
                / 60.0
                / recent_minutes
            )
            * 15,
            3,
        )

    submission_avg = 0.0
    control_avg = 0.0
    knockdown_avg = 0.0
    sig_strikes_landed_per_min = 0.0
    sig_strikes_absorbed_per_min = 0.0
    takedown_avg = 0.0
    if total_minutes > 0:
        submission_avg = round(
            pd.to_numeric(prior_history.get("submission_attempt_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            / total_minutes
            * 15,
            3,
        )
        control_avg = round(
            (
                pd.to_numeric(prior_history.get("control_seconds_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
                / 60.0
                / total_minutes
            )
            * 15,
            3,
        )
        knockdown_avg = round((total_knockdowns / total_minutes) * 15, 3)
        sig_strikes_landed_per_min = round(total_sig_landed / total_minutes, 3)
        sig_strikes_absorbed_per_min = round(total_sig_absorbed / total_minutes, 3)
        takedown_avg = round((total_takedown_landed / total_minutes) * 15, 3)

    strike_accuracy_pct = round((total_sig_landed / total_sig_attempted) * 100, 2) if total_sig_attempted > 0 else 0.0
    strike_defense_pct = round((1 - (total_sig_absorbed / total_sig_attempted_against)) * 100, 2) if total_sig_attempted_against > 0 else 0.0
    takedown_accuracy_pct = round((total_takedown_landed / total_takedown_attempted) * 100, 2) if total_takedown_attempted > 0 else 0.0
    takedown_defense_pct = round((1 - (total_takedowns_absorbed / total_takedown_attempted_against)) * 100, 2) if total_takedown_attempted_against > 0 else 0.0

    head_share = round(pd.to_numeric(prior_history.get("head_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    body_share = round(pd.to_numeric(prior_history.get("body_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    leg_share = round(pd.to_numeric(prior_history.get("leg_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    distance_share = round(pd.to_numeric(prior_history.get("distance_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    clinch_share = round(pd.to_numeric(prior_history.get("clinch_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0
    ground_share = round(pd.to_numeric(prior_history.get("ground_landed_total", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / total_sig_landed, 4) if total_sig_landed > 0 else 0.0

    recent_result_score = _recent_result_score(prior_history)
    losses_in_row = float(_loss_streak(prior_history)) if not prior_history.empty else 0.0
    first_round_finish_wins, first_round_finish_rate = _first_round_finish_metrics(prior_history)
    (
        finish_win_rate,
        finish_loss_rate,
        decision_rate,
        ko_win_rate,
        submission_win_rate,
        ko_loss_rate,
        submission_loss_rate,
    ) = _outcome_profile_metrics(prior_history)
    (
        recent_finish_loss_count,
        recent_ko_loss_count,
        recent_finish_loss_365d,
        recent_ko_loss_365d,
        recent_damage_score,
    ) = _recent_damage_metrics_as_of(prior_history, target_date)
    rolling_metrics = _build_rolling_metrics(prior_history)
    recency_weighted_metrics = _build_recency_weighted_metrics(prior_history)
    round_trend_metrics = _build_round_trend_metrics(prior_history)

    birthdate = pd.to_datetime(profile_row.get("birthdate"), errors="coerce")
    age_years = 0.0
    if pd.notna(birthdate) and pd.notna(target_date):
        age_years = round(((target_date.date() - birthdate.date()).days) / 365.25, 2)
    days_since_last_fight = 999.0
    if not prior_history.empty and pd.notna(target_date):
        last_fight_date = pd.to_datetime(prior_history["date"].max(), errors="coerce")
        if pd.notna(last_fight_date):
            days_since_last_fight = float((target_date.date() - last_fight_date.date()).days)

    stance = str(profile_row.get("stance", "") or "").strip()
    history_style_label = derive_style_label(
        stance=stance,
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

    stats_completeness = 0.55
    if total_fights >= 4:
        stats_completeness = 1.0
    elif total_fights >= 2:
        stats_completeness = 0.88
    elif total_fights == 1:
        stats_completeness = 0.72

    snapshot.update(
        {
            "fighter_name": snapshot_key,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "height_in": float(profile_row.get("height_in", 0.0) or 0.0),
            "reach_in": float(profile_row.get("reach_in", 0.0) or 0.0),
            "age_years": age_years,
            "stance": stance,
            "weight_class": str(target_fight.get("weight_class", "") or "").strip(),
            "sig_strikes_landed_per_min": sig_strikes_landed_per_min,
            "sig_strikes_absorbed_per_min": sig_strikes_absorbed_per_min,
            "takedown_avg": takedown_avg,
            "takedown_defense_pct": takedown_defense_pct,
            "strike_accuracy_pct": strike_accuracy_pct,
            "strike_defense_pct": strike_defense_pct,
            "takedown_accuracy_pct": takedown_accuracy_pct,
            "recent_strike_margin_per_min": recent_strike_margin_per_min,
            "recent_grappling_rate": recent_grappling_rate,
            "control_avg": control_avg,
            "recent_control_avg": recent_control_avg,
            "recent_result_score": recent_result_score,
            "losses_in_row": losses_in_row,
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
            "days_since_last_fight": days_since_last_fight,
            "ufc_fight_count": float(total_fights),
            "ufc_debut_flag": float(total_fights == 0),
            "stats_completeness": stats_completeness,
            "fallback_used": 0.0,
            "fighter_wins": wins,
            "fighter_losses": losses,
            "fighter_draws": draws,
            "fighter_win_rate": round((wins / total_fights), 4) if total_fights > 0 else 0.0,
            "knockdown_avg": knockdown_avg,
            "head_strike_share": head_share,
            "body_strike_share": body_share,
            "leg_strike_share": leg_share,
            "distance_strike_share": distance_share,
            "clinch_strike_share": clinch_share,
            "ground_strike_share": ground_share,
            "history_style_label": history_style_label,
        }
    )
    snapshot.update(recency_weighted_metrics)
    snapshot.update(round_trend_metrics)
    snapshot.update(rolling_metrics)
    return snapshot


def _recent_damage_metrics_as_of(history: pd.DataFrame, as_of_date: pd.Timestamp) -> tuple[int, int, int, int, float]:
    if history.empty:
        return 0, 0, 0, 0, 0.0
    decision_text = history["decision_type"].astype(str).str.upper()
    is_decision = decision_text.str.contains("DEC", na=False)
    is_submission = decision_text.str.contains("SUB", na=False)
    is_ko_tko = decision_text.str.contains("KO|TKO", na=False) & ~is_submission
    is_finish_loss = (history["result_code"] == "L") & ~is_decision
    is_ko_loss = (history["result_code"] == "L") & is_ko_tko
    recent = history.head(3)
    cutoff = (pd.Timestamp(as_of_date.date()) if pd.notna(as_of_date) else history["date"].max()) - pd.Timedelta(days=365)
    recent_window = pd.to_datetime(history["date"], errors="coerce") >= cutoff

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
