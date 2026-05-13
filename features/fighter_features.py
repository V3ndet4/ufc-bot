from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from models.ev import implied_probability, market_overround, no_vig_two_way_probabilities


REQUIRED_FIGHTER_COLUMNS = {
    "fighter_name",
    "wins",
    "losses",
    "height_in",
    "reach_in",
    "sig_strikes_landed_per_min",
    "sig_strikes_absorbed_per_min",
    "takedown_avg",
    "takedown_defense_pct",
}

OPTIONAL_FIGHTER_DEFAULTS = {
    "age_years": 0.0,
    "strike_accuracy_pct": 50.0,
    "strike_defense_pct": 50.0,
    "takedown_accuracy_pct": 50.0,
    "recent_strike_margin_per_min": 0.0,
    "recent_grappling_rate": 0.0,
    "control_avg": 0.0,
    "recent_control_avg": 0.0,
    "recent_result_score": 0.0,
    "losses_in_row": 0.0,
    "first_round_finish_wins": 0.0,
    "first_round_finish_rate": 0.0,
    "finish_win_rate": 0.0,
    "finish_loss_rate": 0.0,
    "decision_rate": 0.0,
    "submission_avg": 0.0,
    "ko_win_rate": 0.0,
    "submission_win_rate": 0.0,
    "ko_loss_rate": 0.0,
    "submission_loss_rate": 0.0,
    "recent_finish_loss_count": 0.0,
    "recent_ko_loss_count": 0.0,
    "recent_finish_loss_365d": 0.0,
    "recent_ko_loss_365d": 0.0,
    "recent_damage_score": 0.0,
    "days_since_last_fight": 999.0,
    "ufc_fight_count": 0.0,
    "ufc_debut_flag": 0.0,
    "stats_completeness": 1.0,
    "fallback_used": 0.0,
    "short_notice_flag": 0.0,
    "short_notice_acceptance_flag": 0.0,
    "short_notice_success_flag": 0.0,
    "new_gym_flag": 0.0,
    "new_contract_flag": 0.0,
    "cardio_fade_flag": 0.0,
    "injury_concern_flag": 0.0,
    "weight_cut_concern_flag": 0.0,
    "replacement_fighter_flag": 0.0,
    "travel_disadvantage_flag": 0.0,
    "camp_change_flag": 0.0,
    "news_alert_count": 0.0,
    "news_radar_score": 0.0,
    "news_high_confidence_alerts": 0.0,
    "news_alert_confidence": 0.0,
    "gym_score": 0.0,
    "gym_fighter_count": 0.0,
    "gym_total_wins": 0.0,
    "gym_total_losses": 0.0,
    "gym_total_draws": 0.0,
    "gym_win_rate": 0.0,
    "gym_elite_fighter_count": 0.0,
    "gym_changed_flag": 0.0,
    "fighter_wins": 0.0,
    "fighter_losses": 0.0,
    "fighter_draws": 0.0,
    "fighter_win_rate": 0.0,
    "fighter_elite_flag": 0.0,
    "open_implied_prob": 0.0,
    "line_movement_toward_fighter": 0.0,
    "opponent_avg_win_rate": 0.5,
    "opponent_avg_ufc_fight_count": 0.0,
    "opponent_avg_recent_result_score": 0.0,
    "opponent_avg_finish_win_rate": 0.0,
    "opponent_quality_score": 0.5,
    "recent_opponent_quality_score": 0.5,
    "knockdown_avg": 0.0,
    "head_strike_share": 0.0,
    "body_strike_share": 0.0,
    "leg_strike_share": 0.0,
    "distance_strike_share": 0.0,
    "clinch_strike_share": 0.0,
    "ground_strike_share": 0.0,
    "recency_weighted_strike_margin": 0.0,
    "recency_weighted_grappling_rate": 0.0,
    "recency_weighted_control_avg": 0.0,
    "recency_weighted_strike_pace": 0.0,
    "recency_weighted_result_score": 0.0,
    "recency_weighted_finish_win_rate": 0.0,
    "recency_weighted_finish_loss_rate": 0.0,
    "strike_round_trend": 0.0,
    "grappling_round_trend": 0.0,
    "control_round_trend": 0.0,
    "strike_pace_round_trend": 0.0,
    "strike_margin_last_1": 0.0,
    "strike_margin_last_3": 0.0,
    "strike_margin_last_5": 0.0,
    "grappling_rate_last_1": 0.0,
    "grappling_rate_last_3": 0.0,
    "grappling_rate_last_5": 0.0,
    "control_avg_last_1": 0.0,
    "control_avg_last_3": 0.0,
    "control_avg_last_5": 0.0,
    "strike_pace_last_1": 0.0,
    "strike_pace_last_3": 0.0,
    "strike_pace_last_5": 0.0,
    "finish_win_rate_last_3": 0.0,
    "finish_win_rate_last_5": 0.0,
    "finish_loss_rate_last_3": 0.0,
    "finish_loss_rate_last_5": 0.0,
    "result_score_last_1": 0.0,
    "result_score_last_3": 0.0,
    "result_score_last_5": 0.0,
}

OPTIONAL_FIGHTER_STRING_DEFAULTS = {
    "stance": "",
    "weight_class": "",
    "context_notes": "",
    "sherdog_url": "",
    "gym_name": "",
    "gym_name_normalized": "",
    "gym_page_url": "",
    "gym_tier": "",
    "gym_record": "",
    "previous_gym_name": "",
    "last_changed_at": "",
    "profile_last_refreshed_at": "",
    "last_seen_at": "",
    "history_style_label": "",
    "news_radar_label": "",
    "news_primary_category": "",
    "news_radar_summary": "",
}


def load_fighter_stats(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_FIGHTER_COLUMNS - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing fighter stat columns: {missing_list}")
    return frame


def build_fight_features(odds_frame: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
    stats = fighter_stats.copy()
    stats["fighter_name"] = stats["fighter_name"].astype(str).str.strip()
    required_numeric_columns = sorted(REQUIRED_FIGHTER_COLUMNS - {"fighter_name"})
    optional_numeric_columns = list(OPTIONAL_FIGHTER_DEFAULTS.keys())
    numeric_columns = [column for column in required_numeric_columns + optional_numeric_columns if column in stats.columns]
    for column in numeric_columns:
        stats[column] = pd.to_numeric(stats[column], errors="coerce")
    for column in required_numeric_columns:
        if column in stats.columns:
            stats[column] = stats[column].fillna(0.0)
    missing_numeric_defaults = {
        column: pd.Series(default_value, index=stats.index, dtype=float)
        for column, default_value in OPTIONAL_FIGHTER_DEFAULTS.items()
        if column not in stats.columns
    }
    if missing_numeric_defaults:
        stats = pd.concat([stats, pd.DataFrame(missing_numeric_defaults, index=stats.index)], axis=1)
    for column, default_value in OPTIONAL_FIGHTER_DEFAULTS.items():
        if column in stats.columns:
            stats[column] = stats[column].fillna(default_value)

    missing_string_defaults = {
        column: pd.Series(default_value, index=stats.index, dtype=object)
        for column, default_value in OPTIONAL_FIGHTER_STRING_DEFAULTS.items()
        if column not in stats.columns
    }
    if missing_string_defaults:
        stats = pd.concat([stats, pd.DataFrame(missing_string_defaults, index=stats.index)], axis=1)
    for column, default_value in OPTIONAL_FIGHTER_STRING_DEFAULTS.items():
        if column in stats.columns:
            stats[column] = stats[column].fillna(default_value)

    a_stats = stats.add_prefix("a_").rename(columns={"a_fighter_name": "fighter_a"})
    b_stats = stats.add_prefix("b_").rename(columns={"b_fighter_name": "fighter_b"})

    features = odds_frame.merge(a_stats, on="fighter_a", how="left").merge(b_stats, on="fighter_b", how="left")
    if features["a_wins"].isna().any() or features["b_wins"].isna().any():
        raise ValueError("Missing fighter stats for one or more selections")
    numeric_feature_columns = [
        column
        for column in features.columns
        if (column.startswith("a_") or column.startswith("b_")) and pd.api.types.is_numeric_dtype(features[column])
    ]
    for column in numeric_feature_columns:
        features[column] = features[column].fillna(0.0)

    a_total_fights = features["a_wins"] + features["a_losses"]
    b_total_fights = features["b_wins"] + features["b_losses"]
    scheduled_rounds = _derive_scheduled_rounds(features)
    is_title_fight = _coerce_numeric_column(features, "is_title_fight", 0.0)
    is_wmma_series = (
        features["a_weight_class"].astype(str).str.lower().str.contains("women", na=False)
        | features["b_weight_class"].astype(str).str.lower().str.contains("women", na=False)
    ).astype(float)
    is_heavyweight_series = _is_heavyweight_series(features["a_weight_class"], features["b_weight_class"], is_wmma_series)
    is_five_round_series = ((scheduled_rounds >= 5) | (is_title_fight >= 1)).astype(float)

    extra_columns: dict[str, pd.Series] = {}
    extra_columns["reach_diff"] = features["a_reach_in"] - features["b_reach_in"]
    extra_columns["height_diff"] = features["a_height_in"] - features["b_height_in"]
    extra_columns["age_diff"] = features["b_age_years"] - features["a_age_years"]
    extra_columns["experience_diff"] = a_total_fights - b_total_fights
    extra_columns["win_rate_diff"] = (features["a_wins"] / a_total_fights) - (features["b_wins"] / b_total_fights)
    extra_columns["strike_margin_diff"] = (
        (features["a_sig_strikes_landed_per_min"] - features["a_sig_strikes_absorbed_per_min"])
        - (features["b_sig_strikes_landed_per_min"] - features["b_sig_strikes_absorbed_per_min"])
    )
    extra_columns["strike_efficiency_diff"] = (
        (features["a_strike_accuracy_pct"] + features["a_strike_defense_pct"])
        - (features["b_strike_accuracy_pct"] + features["b_strike_defense_pct"])
    ) / 100
    a_strike_matchup_score = (
        (features["a_sig_strikes_landed_per_min"] * ((100 - features["b_strike_defense_pct"]).clip(lower=5) / 100))
        + (features["b_sig_strikes_absorbed_per_min"] * (features["a_strike_accuracy_pct"].clip(lower=5) / 100))
    )
    b_strike_matchup_score = (
        (features["b_sig_strikes_landed_per_min"] * ((100 - features["a_strike_defense_pct"]).clip(lower=5) / 100))
        + (features["a_sig_strikes_absorbed_per_min"] * (features["b_strike_accuracy_pct"].clip(lower=5) / 100))
    )
    extra_columns["matchup_striking_edge"] = a_strike_matchup_score - b_strike_matchup_score
    extra_columns["grappling_diff"] = (
        features["a_takedown_avg"] - features["b_takedown_avg"]
    ) + ((features["a_takedown_defense_pct"] - features["b_takedown_defense_pct"]) / 100)
    a_grappling_matchup_score = (
        (features["a_takedown_avg"] * ((100 - features["b_takedown_defense_pct"]).clip(lower=5) / 100))
        + (features["a_recent_grappling_rate"] * 0.35)
        + (features["a_takedown_accuracy_pct"].clip(lower=5) / 100)
    )
    b_grappling_matchup_score = (
        (features["b_takedown_avg"] * ((100 - features["a_takedown_defense_pct"]).clip(lower=5) / 100))
        + (features["b_recent_grappling_rate"] * 0.35)
        + (features["b_takedown_accuracy_pct"].clip(lower=5) / 100)
    )
    extra_columns["matchup_grappling_edge"] = a_grappling_matchup_score - b_grappling_matchup_score
    extra_columns["takedown_efficiency_diff"] = (
        (features["a_takedown_accuracy_pct"] + features["a_takedown_defense_pct"])
        - (features["b_takedown_accuracy_pct"] + features["b_takedown_defense_pct"])
    ) / 100
    extra_columns["recent_form_diff"] = features["a_recent_result_score"] - features["b_recent_result_score"]
    extra_columns["recent_strike_form_diff"] = (
        features["a_recent_strike_margin_per_min"] - features["b_recent_strike_margin_per_min"]
    )
    extra_columns["recent_grappling_form_diff"] = features["a_recent_grappling_rate"] - features["b_recent_grappling_rate"]
    extra_columns["control_diff"] = features["a_control_avg"] - features["b_control_avg"]
    extra_columns["recent_control_diff"] = features["a_recent_control_avg"] - features["b_recent_control_avg"]
    extra_columns["matchup_control_edge"] = (
        features["a_control_avg"] * ((100 - features["b_takedown_defense_pct"]).clip(lower=5) / 100)
    ) - (
        features["b_control_avg"] * ((100 - features["a_takedown_defense_pct"]).clip(lower=5) / 100)
    )
    extra_columns["grappling_pressure_diff"] = (
        (
            features["a_recent_grappling_rate"]
            + (features["a_recent_control_avg"] * 0.55)
            + (features["a_submission_avg"] * 0.35)
        )
        - (
            features["b_recent_grappling_rate"]
            + (features["b_recent_control_avg"] * 0.55)
            + (features["b_submission_avg"] * 0.35)
        )
    )
    extra_columns["first_round_finish_rate_diff"] = (
        features["a_first_round_finish_rate"] - features["b_first_round_finish_rate"]
    )
    extra_columns["finish_win_rate_diff"] = features["a_finish_win_rate"] - features["b_finish_win_rate"]
    extra_columns["ko_win_rate_diff"] = features["a_ko_win_rate"] - features["b_ko_win_rate"]
    extra_columns["submission_win_rate_diff"] = features["a_submission_win_rate"] - features["b_submission_win_rate"]
    extra_columns["ko_loss_rate_diff"] = features["b_ko_loss_rate"] - features["a_ko_loss_rate"]
    extra_columns["submission_loss_rate_diff"] = features["b_submission_loss_rate"] - features["a_submission_loss_rate"]
    extra_columns["recent_finish_damage_diff"] = features["b_recent_finish_loss_365d"] - features["a_recent_finish_loss_365d"]
    extra_columns["recent_ko_damage_diff"] = features["b_recent_ko_loss_365d"] - features["a_recent_ko_loss_365d"]
    extra_columns["recent_damage_diff"] = features["b_recent_damage_score"] - features["a_recent_damage_score"]
    extra_columns["stance_matchup_diff"] = _stance_matchup_edge(features["a_stance"], features["b_stance"])
    extra_columns["submission_avg_diff"] = features["a_submission_avg"] - features["b_submission_avg"]
    extra_columns["durability_diff"] = features["b_finish_loss_rate"] - features["a_finish_loss_rate"]
    extra_columns["decision_rate_diff"] = features["a_decision_rate"] - features["b_decision_rate"]
    extra_columns["knockdown_avg_diff"] = features["a_knockdown_avg"] - features["b_knockdown_avg"]
    extra_columns["distance_strike_share_diff"] = features["a_distance_strike_share"] - features["b_distance_strike_share"]
    extra_columns["clinch_strike_share_diff"] = features["a_clinch_strike_share"] - features["b_clinch_strike_share"]
    extra_columns["ground_strike_share_diff"] = features["a_ground_strike_share"] - features["b_ground_strike_share"]
    extra_columns["recency_weighted_strike_margin_diff"] = (
        features["a_recency_weighted_strike_margin"] - features["b_recency_weighted_strike_margin"]
    )
    extra_columns["recency_weighted_grappling_rate_diff"] = (
        features["a_recency_weighted_grappling_rate"] - features["b_recency_weighted_grappling_rate"]
    )
    extra_columns["recency_weighted_control_diff"] = (
        features["a_recency_weighted_control_avg"] - features["b_recency_weighted_control_avg"]
    )
    extra_columns["recency_weighted_strike_pace_diff"] = (
        features["a_recency_weighted_strike_pace"] - features["b_recency_weighted_strike_pace"]
    )
    extra_columns["recency_weighted_result_score_diff"] = (
        features["a_recency_weighted_result_score"] - features["b_recency_weighted_result_score"]
    )
    extra_columns["recency_weighted_finish_win_rate_diff"] = (
        features["a_recency_weighted_finish_win_rate"] - features["b_recency_weighted_finish_win_rate"]
    )
    extra_columns["recency_weighted_durability_diff"] = (
        features["b_recency_weighted_finish_loss_rate"] - features["a_recency_weighted_finish_loss_rate"]
    )
    extra_columns["strike_round_trend_diff"] = features["a_strike_round_trend"] - features["b_strike_round_trend"]
    extra_columns["grappling_round_trend_diff"] = features["a_grappling_round_trend"] - features["b_grappling_round_trend"]
    extra_columns["control_round_trend_diff"] = features["a_control_round_trend"] - features["b_control_round_trend"]
    extra_columns["strike_pace_round_trend_diff"] = (
        features["a_strike_pace_round_trend"] - features["b_strike_pace_round_trend"]
    )
    extra_columns["strike_margin_last_1_diff"] = features["a_strike_margin_last_1"] - features["b_strike_margin_last_1"]
    extra_columns["strike_margin_last_3_diff"] = features["a_strike_margin_last_3"] - features["b_strike_margin_last_3"]
    extra_columns["strike_margin_last_5_diff"] = features["a_strike_margin_last_5"] - features["b_strike_margin_last_5"]
    extra_columns["grappling_rate_last_1_diff"] = features["a_grappling_rate_last_1"] - features["b_grappling_rate_last_1"]
    extra_columns["grappling_rate_last_3_diff"] = features["a_grappling_rate_last_3"] - features["b_grappling_rate_last_3"]
    extra_columns["grappling_rate_last_5_diff"] = features["a_grappling_rate_last_5"] - features["b_grappling_rate_last_5"]
    extra_columns["control_avg_last_1_diff"] = features["a_control_avg_last_1"] - features["b_control_avg_last_1"]
    extra_columns["control_avg_last_3_diff"] = features["a_control_avg_last_3"] - features["b_control_avg_last_3"]
    extra_columns["control_avg_last_5_diff"] = features["a_control_avg_last_5"] - features["b_control_avg_last_5"]
    extra_columns["strike_pace_last_1_diff"] = features["a_strike_pace_last_1"] - features["b_strike_pace_last_1"]
    extra_columns["strike_pace_last_3_diff"] = features["a_strike_pace_last_3"] - features["b_strike_pace_last_3"]
    extra_columns["strike_pace_last_5_diff"] = features["a_strike_pace_last_5"] - features["b_strike_pace_last_5"]
    extra_columns["finish_win_rate_last_3_diff"] = features["a_finish_win_rate_last_3"] - features["b_finish_win_rate_last_3"]
    extra_columns["finish_win_rate_last_5_diff"] = features["a_finish_win_rate_last_5"] - features["b_finish_win_rate_last_5"]
    extra_columns["durability_last_3_diff"] = features["b_finish_loss_rate_last_3"] - features["a_finish_loss_rate_last_3"]
    extra_columns["durability_last_5_diff"] = features["b_finish_loss_rate_last_5"] - features["a_finish_loss_rate_last_5"]
    extra_columns["result_score_last_1_diff"] = features["a_result_score_last_1"] - features["b_result_score_last_1"]
    extra_columns["result_score_last_3_diff"] = features["a_result_score_last_3"] - features["b_result_score_last_3"]
    extra_columns["result_score_last_5_diff"] = features["a_result_score_last_5"] - features["b_result_score_last_5"]
    extra_columns["first_round_finish_rate_sum"] = features["a_first_round_finish_rate"] + features["b_first_round_finish_rate"]
    extra_columns["finish_win_rate_sum"] = features["a_finish_win_rate"] + features["b_finish_win_rate"]
    extra_columns["ko_win_rate_sum"] = features["a_ko_win_rate"] + features["b_ko_win_rate"]
    extra_columns["submission_win_rate_sum"] = features["a_submission_win_rate"] + features["b_submission_win_rate"]
    extra_columns["finish_loss_rate_sum"] = features["a_finish_loss_rate"] + features["b_finish_loss_rate"]
    extra_columns["decision_rate_sum"] = features["a_decision_rate"] + features["b_decision_rate"]
    extra_columns["combined_control_avg"] = features["a_control_avg"] + features["b_control_avg"]
    extra_columns["loss_streak_diff"] = features["b_losses_in_row"] - features["a_losses_in_row"]
    extra_columns["layoff_diff"] = _layoff_score(features["a_days_since_last_fight"]) - _layoff_score(
        features["b_days_since_last_fight"]
    )
    extra_columns["short_notice_readiness_diff"] = (
        (features["a_short_notice_flag"] * features["a_short_notice_acceptance_flag"])
        - (features["b_short_notice_flag"] * features["b_short_notice_acceptance_flag"])
    )
    extra_columns["short_notice_success_diff"] = (
        (features["a_short_notice_flag"] * features["a_short_notice_success_flag"])
        - (features["b_short_notice_flag"] * features["b_short_notice_success_flag"])
    )
    a_short_notice_capability = features[["a_short_notice_acceptance_flag", "a_short_notice_success_flag"]].max(axis=1)
    b_short_notice_capability = features[["b_short_notice_acceptance_flag", "b_short_notice_success_flag"]].max(axis=1)
    a_short_notice_instability = features["a_short_notice_flag"] * (1 - a_short_notice_capability)
    b_short_notice_instability = features["b_short_notice_flag"] * (1 - b_short_notice_capability)
    extra_columns["ufc_experience_diff"] = features["a_ufc_fight_count"] - features["b_ufc_fight_count"]
    extra_columns["ufc_debut_penalty_diff"] = features["b_ufc_debut_flag"] - features["a_ufc_debut_flag"]
    extra_columns["cardio_fade_diff"] = features["b_cardio_fade_flag"] - features["a_cardio_fade_flag"]
    extra_columns["cardio_fade_sum"] = features["a_cardio_fade_flag"] + features["b_cardio_fade_flag"]
    extra_columns["travel_advantage_diff"] = features["b_travel_disadvantage_flag"] - features["a_travel_disadvantage_flag"]
    extra_columns["gym_score_diff"] = features["a_gym_score"] - features["b_gym_score"]
    extra_columns["gym_depth_diff"] = _gym_depth_score(features["a_gym_fighter_count"]) - _gym_depth_score(features["b_gym_fighter_count"])
    extra_columns["gym_win_rate_diff"] = features["a_gym_win_rate"] - features["b_gym_win_rate"]
    extra_columns["gym_elite_diff"] = _gym_elite_score(features["a_gym_elite_fighter_count"]) - _gym_elite_score(features["b_gym_elite_fighter_count"])
    extra_columns["opponent_quality_diff"] = features["a_opponent_quality_score"] - features["b_opponent_quality_score"]
    extra_columns["recent_opponent_quality_diff"] = (
        features["a_recent_opponent_quality_score"] - features["b_recent_opponent_quality_score"]
    )
    a_schedule_strength = _schedule_strength_multiplier(
        features["a_opponent_quality_score"],
        features["a_recent_opponent_quality_score"],
    )
    b_schedule_strength = _schedule_strength_multiplier(
        features["b_opponent_quality_score"],
        features["b_recent_opponent_quality_score"],
    )
    a_recent_schedule_strength = _recent_schedule_strength_multiplier(features["a_recent_opponent_quality_score"])
    b_recent_schedule_strength = _recent_schedule_strength_multiplier(features["b_recent_opponent_quality_score"])
    a_strike_margin = features["a_sig_strikes_landed_per_min"] - features["a_sig_strikes_absorbed_per_min"]
    b_strike_margin = features["b_sig_strikes_landed_per_min"] - features["b_sig_strikes_absorbed_per_min"]
    a_grappling_profile = features["a_takedown_avg"] + (features["a_recent_grappling_rate"] * 0.35)
    b_grappling_profile = features["b_takedown_avg"] + (features["b_recent_grappling_rate"] * 0.35)
    a_control_profile = features["a_control_avg"] + (features["a_recent_control_avg"] * 0.30)
    b_control_profile = features["b_control_avg"] + (features["b_recent_control_avg"] * 0.30)
    extra_columns["schedule_strength_diff"] = a_schedule_strength - b_schedule_strength
    extra_columns["normalized_strike_margin_diff"] = (a_strike_margin * a_schedule_strength) - (
        b_strike_margin * b_schedule_strength
    )
    extra_columns["normalized_grappling_diff"] = (a_grappling_profile * a_schedule_strength) - (
        b_grappling_profile * b_schedule_strength
    )
    extra_columns["normalized_control_diff"] = (a_control_profile * a_schedule_strength) - (
        b_control_profile * b_schedule_strength
    )
    extra_columns["normalized_recent_form_diff"] = (
        features["a_recent_result_score"] * a_recent_schedule_strength
    ) - (features["b_recent_result_score"] * b_recent_schedule_strength)
    extra_columns["recent_strike_intensity_sum"] = features["a_recent_strike_margin_per_min"].abs() + features["b_recent_strike_margin_per_min"].abs()
    extra_columns["combined_grappling"] = features["a_takedown_avg"] + features["b_takedown_avg"]
    extra_columns["age_curve_diff"] = _age_curve_score(
        features["a_age_years"],
        features["a_weight_class"],
    ) - _age_curve_score(features["b_age_years"], features["b_weight_class"])
    extra_columns["data_quality"] = features[["a_stats_completeness", "b_stats_completeness"]].min(axis=1)
    extra_columns["fallback_penalty"] = features["a_fallback_used"] + features["b_fallback_used"]
    extra_columns["is_wmma"] = is_wmma_series
    extra_columns["is_heavyweight"] = is_heavyweight_series
    extra_columns["scheduled_rounds"] = scheduled_rounds
    extra_columns["is_title_fight"] = is_title_fight
    extra_columns["is_five_round_fight"] = is_five_round_series
    extra_columns["segment_label"] = _segment_label_series(
        is_wmma_series,
        is_heavyweight_series,
        is_five_round_series,
    )
    extra_columns["context_stability_diff"] = (
        (
            b_short_notice_instability
            + features["b_cardio_fade_flag"]
            + features["b_injury_concern_flag"]
            + features["b_weight_cut_concern_flag"]
            + features["b_replacement_fighter_flag"]
            + features["b_travel_disadvantage_flag"]
            + features["b_new_gym_flag"]
            + features["b_camp_change_flag"]
            + features["b_news_radar_score"]
            + (features["b_news_high_confidence_alerts"].clip(lower=0.0, upper=3.0) * 0.25)
            + features["b_news_alert_confidence"]
        )
        - (
            a_short_notice_instability
            + features["a_cardio_fade_flag"]
            + features["a_injury_concern_flag"]
            + features["a_weight_cut_concern_flag"]
            + features["a_replacement_fighter_flag"]
            + features["a_travel_disadvantage_flag"]
            + features["a_new_gym_flag"]
            + features["a_camp_change_flag"]
            + features["a_news_radar_score"]
            + (features["a_news_high_confidence_alerts"].clip(lower=0.0, upper=3.0) * 0.25)
            + features["a_news_alert_confidence"]
        )
    )
    extra_columns["news_radar_diff"] = features["b_news_radar_score"] - features["a_news_radar_score"]
    extra_columns["news_alert_confidence_diff"] = features["b_news_alert_confidence"] - features["a_news_alert_confidence"]
    extra_columns["high_confidence_news_diff"] = features["b_news_high_confidence_alerts"] - features["a_news_high_confidence_alerts"]
    if "open_american_odds" in features.columns:
        extra_columns["open_implied_prob"] = features["open_american_odds"].apply(_american_to_implied_probability)
        current_implied = features["american_odds"].apply(_american_to_implied_probability)
        extra_columns["line_movement_toward_fighter"] = features.apply(
            lambda row: 0.0
            if pd.isna(extra_columns["open_implied_prob"].loc[row.name])
            else (current_implied[row.name] - extra_columns["open_implied_prob"].loc[row.name])
            if row["selection"] == "fighter_a"
            else (extra_columns["open_implied_prob"].loc[row.name] - current_implied[row.name])
            if pd.notna(extra_columns["open_implied_prob"].loc[row.name])
            else 0.0,
            axis=1,
        )
    else:
        extra_columns["line_movement_toward_fighter"] = pd.Series(0.0, index=features.index, dtype=float)
    selection_a_mask = features["selection"] == "fighter_a"
    target_fair_probs, target_overrounds = _derive_no_vig_target_market(features)
    consensus_prob_series = (
        pd.to_numeric(features["market_consensus_prob"], errors="coerce")
        if "market_consensus_prob" in features.columns
        else pd.Series(float("nan"), index=features.index, dtype=float)
    )
    consensus_overround_series = (
        pd.to_numeric(features["market_overround"], errors="coerce")
        if "market_overround" in features.columns
        else pd.Series(float("nan"), index=features.index, dtype=float)
    )
    extra_columns["market_target_fair_prob"] = (
        pd.to_numeric(features["market_target_fair_prob"], errors="coerce")
        if "market_target_fair_prob" in features.columns
        else target_fair_probs
    ).fillna(target_fair_probs)
    extra_columns["market_target_overround"] = (
        pd.to_numeric(features["market_target_overround"], errors="coerce")
        if "market_target_overround" in features.columns
        else target_overrounds
    ).fillna(target_overrounds)
    extra_columns["market_consensus_prob"] = consensus_prob_series
    extra_columns["market_overround"] = consensus_overround_series.fillna(extra_columns["market_target_overround"])
    extra_columns["market_consensus_bookmaker_count"] = (
        pd.to_numeric(features["market_consensus_bookmaker_count"], errors="coerce")
        if "market_consensus_bookmaker_count" in features.columns
        else pd.Series(0.0, index=features.index, dtype=float)
    ).fillna(0.0)
    market_prob_series = (
        consensus_prob_series.fillna(extra_columns["market_target_fair_prob"])
        .fillna(features["american_odds"].apply(implied_probability))
    )
    extra_columns["fighter_a_current_implied_prob"] = market_prob_series.where(selection_a_mask).groupby(
        [features["event_id"], features["fighter_a"], features["fighter_b"]]
    ).transform("first")
    extra_columns["fighter_b_current_implied_prob"] = market_prob_series.where(~selection_a_mask).groupby(
        [features["event_id"], features["fighter_a"], features["fighter_b"]]
    ).transform("first")
    extra_frame = pd.DataFrame(extra_columns, index=features.index)
    overlapping = [column for column in extra_frame.columns if column in features.columns]
    if overlapping:
        features = features.drop(columns=overlapping)
    return pd.concat([features, extra_frame], axis=1)


def _american_to_implied_probability(american_odds: float) -> float:
    if pd.isna(american_odds):
        return pd.NA
    odds = int(american_odds)
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def _derive_no_vig_target_market(features: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    fair_probabilities = pd.Series(float("nan"), index=features.index, dtype=float)
    overrounds = pd.Series(float("nan"), index=features.index, dtype=float)
    group_columns = ["event_id", "fighter_a", "fighter_b"]

    for _, fight_rows in features.groupby(group_columns, dropna=False):
        fighter_a_rows = fight_rows.loc[fight_rows["selection"] == "fighter_a"]
        fighter_b_rows = fight_rows.loc[fight_rows["selection"] == "fighter_b"]
        if fighter_a_rows.empty or fighter_b_rows.empty:
            continue

        fighter_a_price = fighter_a_rows["american_odds"].iloc[0]
        fighter_b_price = fighter_b_rows["american_odds"].iloc[0]
        if pd.isna(fighter_a_price) or pd.isna(fighter_b_price):
            continue

        fair_a, fair_b = no_vig_two_way_probabilities(int(fighter_a_price), int(fighter_b_price))
        overround = market_overround(int(fighter_a_price), int(fighter_b_price))
        fair_probabilities.loc[fighter_a_rows.index] = fair_a
        fair_probabilities.loc[fighter_b_rows.index] = fair_b
        overrounds.loc[fight_rows.index] = overround

    return fair_probabilities, overrounds


def _coerce_numeric_column(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _derive_scheduled_rounds(features: pd.DataFrame) -> pd.Series:
    explicit_rounds = _coerce_numeric_column(features, "scheduled_rounds", float("nan"))
    if explicit_rounds.notna().any():
        fallback = _infer_scheduled_rounds(features)
        return explicit_rounds.fillna(fallback)
    return _infer_scheduled_rounds(features)


def _infer_scheduled_rounds(features: pd.DataFrame) -> pd.Series:
    inferred = pd.Series(3.0, index=features.index, dtype=float)
    fight_order = (
        features[["event_id", "fighter_a", "fighter_b"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    fight_order["fight_order"] = fight_order.groupby("event_id", sort=False).cumcount()
    fight_order["scheduled_rounds"] = 3.0
    fight_order.loc[fight_order["fight_order"] == 0, "scheduled_rounds"] = 5.0

    round_lookup = {
        (str(row.event_id), str(row.fighter_a), str(row.fighter_b)): float(row.scheduled_rounds)
        for row in fight_order.itertuples(index=False)
    }
    for idx, row in features[["event_id", "fighter_a", "fighter_b"]].iterrows():
        inferred.loc[idx] = round_lookup.get(
            (str(row["event_id"]), str(row["fighter_a"]), str(row["fighter_b"])),
            3.0,
        )
    return inferred


def _stance_matchup_edge(a_stance_series: pd.Series, b_stance_series: pd.Series) -> pd.Series:
    return pd.Series(
        [
            _single_stance_matchup_edge(a_stance, b_stance)
            for a_stance, b_stance in zip(a_stance_series.tolist(), b_stance_series.tolist())
        ],
        index=a_stance_series.index,
        dtype=float,
    )


def _single_stance_matchup_edge(a_stance: object, b_stance: object) -> float:
    a_value = str(a_stance).strip().lower()
    b_value = str(b_stance).strip().lower()
    if not a_value or not b_value or a_value == b_value:
        return 0.0

    edge_map = {
        ("southpaw", "orthodox"): 0.35,
        ("orthodox", "southpaw"): -0.35,
        ("switch", "orthodox"): 0.18,
        ("orthodox", "switch"): -0.18,
        ("switch", "southpaw"): 0.08,
        ("southpaw", "switch"): -0.08,
    }
    return edge_map.get((a_value, b_value), 0.0)


def _is_heavyweight_series(
    a_weight_class_series: pd.Series,
    b_weight_class_series: pd.Series,
    is_wmma_series: pd.Series,
) -> pd.Series:
    a_values = a_weight_class_series.astype(str).str.lower()
    b_values = b_weight_class_series.astype(str).str.lower()
    heavyweight_mask = (
        (a_values.str.contains("heavyweight", na=False) | b_values.str.contains("heavyweight", na=False))
        & ~a_values.str.contains("light heavy", na=False)
        & ~b_values.str.contains("light heavy", na=False)
        & (is_wmma_series < 1)
    )
    return heavyweight_mask.astype(float)


def _segment_label_series(
    is_wmma_series: pd.Series,
    is_heavyweight_series: pd.Series,
    is_five_round_series: pd.Series,
) -> pd.Series:
    labels: list[str] = []
    for is_wmma, is_heavyweight, is_five_round in zip(
        is_wmma_series.tolist(),
        is_heavyweight_series.tolist(),
        is_five_round_series.tolist(),
    ):
        active: list[str] = []
        if float(is_wmma) >= 1:
            active.append("wmma")
        if float(is_heavyweight) >= 1:
            active.append("heavyweight")
        if float(is_five_round) >= 1:
            active.append("five_round")
        labels.append("|".join(active) if active else "standard")
    return pd.Series(labels, index=is_wmma_series.index, dtype="object")


def _layoff_score(days_series: pd.Series) -> pd.Series:
    days = pd.to_numeric(days_series, errors="coerce").fillna(999)
    # Longer layoffs are treated as a prep-and-improvement upside signal.
    return (days.clip(lower=0, upper=365)) / 365


def _age_curve_score(age_series: pd.Series, weight_class_series: pd.Series) -> pd.Series:
    ages = pd.to_numeric(age_series, errors="coerce").fillna(0)
    primes = weight_class_series.astype(str).str.lower().map(_prime_age_for_weight_class).fillna(30.0)
    return 1 - ((ages - primes).abs() / 10)


def _gym_depth_score(fighter_count_series: pd.Series) -> pd.Series:
    counts = pd.to_numeric(fighter_count_series, errors="coerce").fillna(0.0)
    return counts.apply(lambda value: min(1.0, math.log1p(max(0.0, float(value))) / math.log1p(75)))


def _gym_elite_score(elite_count_series: pd.Series) -> pd.Series:
    counts = pd.to_numeric(elite_count_series, errors="coerce").fillna(0.0)
    return counts.clip(lower=0.0, upper=8.0) / 8.0


def _schedule_strength_multiplier(
    opponent_quality_series: pd.Series,
    recent_opponent_quality_series: pd.Series,
) -> pd.Series:
    opponent_quality = pd.to_numeric(opponent_quality_series, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    recent_quality = pd.to_numeric(recent_opponent_quality_series, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    blended_quality = (opponent_quality * 0.65) + (recent_quality * 0.35)
    return 0.90 + (blended_quality * 0.20)


def _recent_schedule_strength_multiplier(recent_opponent_quality_series: pd.Series) -> pd.Series:
    recent_quality = pd.to_numeric(recent_opponent_quality_series, errors="coerce").fillna(0.5).clip(lower=0.0, upper=1.0)
    return 0.92 + (recent_quality * 0.16)


def _prime_age_for_weight_class(weight_class: str) -> float:
    if "heavy" in weight_class:
        return 32.0
    if "light heavy" in weight_class or "middle" in weight_class:
        return 31.0
    if "welter" in weight_class or "light" in weight_class:
        return 30.5
    if "feather" in weight_class or "bantam" in weight_class or "fly" in weight_class:
        return 29.5
    if "straw" in weight_class:
        return 28.5
    return 30.0
