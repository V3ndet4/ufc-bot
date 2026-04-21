from __future__ import annotations

import math

import pandas as pd

from models.confidence import apply_confidence_model
from models.side import apply_side_model_adjustments


MONEYLINE_FEATURE_WEIGHTS = {
    "experience_diff": 0.008,
    "age_diff": 0.08,
    "win_rate_diff": 1.0,
    "strike_margin_diff": 0.18,
    "strike_efficiency_diff": 0.16,
    "matchup_striking_edge": 0.08,
    "grappling_diff": 0.1,
    "matchup_grappling_edge": 0.07,
    "takedown_efficiency_diff": 0.08,
    "recent_form_diff": 0.2,
    "recent_strike_form_diff": 0.16,
    "recent_grappling_form_diff": 0.08,
    "control_diff": 0.035,
    "recent_control_diff": 0.045,
    "matchup_control_edge": 0.035,
    "grappling_pressure_diff": 0.055,
    "first_round_finish_rate_diff": 0.18,
    "finish_win_rate_diff": 0.12,
    "stance_matchup_diff": 0.06,
    "recent_finish_damage_diff": 0.06,
    "recent_ko_damage_diff": 0.10,
    "recent_damage_diff": 0.14,
    "durability_diff": 0.14,
    "decision_rate_diff": 0.06,
    "knockdown_avg_diff": 0.08,
    "distance_strike_share_diff": 0.03,
    "clinch_strike_share_diff": 0.03,
    "ground_strike_share_diff": 0.04,
    "recency_weighted_strike_margin_diff": 0.10,
    "recency_weighted_grappling_rate_diff": 0.06,
    "recency_weighted_control_diff": 0.05,
    "recency_weighted_strike_pace_diff": 0.03,
    "recency_weighted_result_score_diff": 0.10,
    "recency_weighted_finish_win_rate_diff": 0.05,
    "recency_weighted_durability_diff": 0.07,
    "strike_round_trend_diff": 0.05,
    "grappling_round_trend_diff": 0.03,
    "control_round_trend_diff": 0.04,
    "strike_pace_round_trend_diff": 0.03,
    "strike_margin_last_1_diff": 0.04,
    "strike_margin_last_3_diff": 0.08,
    "strike_margin_last_5_diff": 0.05,
    "grappling_rate_last_1_diff": 0.03,
    "grappling_rate_last_3_diff": 0.05,
    "grappling_rate_last_5_diff": 0.03,
    "control_avg_last_1_diff": 0.02,
    "control_avg_last_3_diff": 0.04,
    "control_avg_last_5_diff": 0.025,
    "strike_pace_last_1_diff": 0.015,
    "strike_pace_last_3_diff": 0.03,
    "strike_pace_last_5_diff": 0.02,
    "finish_win_rate_last_3_diff": 0.06,
    "finish_win_rate_last_5_diff": 0.04,
    "durability_last_3_diff": 0.06,
    "durability_last_5_diff": 0.04,
    "result_score_last_1_diff": 0.08,
    "result_score_last_3_diff": 0.12,
    "result_score_last_5_diff": 0.08,
    "loss_streak_diff": 0.12,
    "layoff_diff": 0.12,
    "short_notice_readiness_diff": 0.08,
    "short_notice_success_diff": 0.1,
    "ufc_experience_diff": 0.01,
    "ufc_debut_penalty_diff": 0.08,
    "cardio_fade_diff": 0.2,
    "age_curve_diff": 0.24,
    "travel_advantage_diff": 0.14,
    "context_stability_diff": 0.1,
    "gym_score_diff": 0.18,
    "opponent_quality_diff": 0.14,
    "recent_opponent_quality_diff": 0.10,
    "schedule_strength_diff": 0.08,
    "normalized_strike_margin_diff": 0.06,
    "normalized_grappling_diff": 0.04,
    "normalized_control_diff": 0.03,
    "normalized_recent_form_diff": 0.08,
    "line_movement_toward_fighter": 0.25,
    "fallback_penalty": -0.18,
}

FINISH_FEATURE_WEIGHTS = {
    "first_round_finish_rate_sum": 0.85,
    "finish_win_rate_sum": 0.65,
    "finish_loss_rate_sum": 0.72,
    "decision_rate_sum": -0.92,
    "combined_control_avg": -0.025,
    "recent_strike_intensity_sum": 0.07,
    "combined_grappling": 0.03,
    "cardio_fade_sum": 0.12,
    "fallback_penalty": -0.12,
}

WMMA_MONEYLINE_OVERLAY_WEIGHTS = {
    "strike_efficiency_diff": 0.12,
    "takedown_efficiency_diff": 0.08,
    "decision_rate_diff": 0.12,
    "cardio_fade_diff": 0.10,
    "recent_damage_diff": 0.12,
    "stance_matchup_diff": 0.08,
}

HEAVYWEIGHT_MONEYLINE_OVERLAY_WEIGHTS = {
    "first_round_finish_rate_diff": 0.16,
    "finish_win_rate_diff": 0.12,
    "durability_diff": 0.18,
    "recent_ko_damage_diff": 0.18,
    "recent_damage_diff": 0.10,
    "decision_rate_diff": -0.08,
}

FIVE_ROUND_MONEYLINE_OVERLAY_WEIGHTS = {
    "cardio_fade_diff": 0.16,
    "decision_rate_diff": 0.10,
    "ufc_experience_diff": 0.01,
    "age_curve_diff": 0.10,
    "context_stability_diff": 0.14,
    "recent_damage_diff": 0.10,
    "recent_control_diff": 0.05,
    "strike_round_trend_diff": 0.05,
    "control_round_trend_diff": 0.06,
    "strike_pace_round_trend_diff": 0.04,
}

WMMA_FINISH_OVERLAY_WEIGHTS = {
    "decision_rate_sum": -0.25,
    "first_round_finish_rate_sum": -0.18,
}

HEAVYWEIGHT_FINISH_OVERLAY_WEIGHTS = {
    "first_round_finish_rate_sum": 0.22,
    "finish_win_rate_sum": 0.12,
    "decision_rate_sum": -0.18,
    "finish_loss_rate_sum": 0.08,
}

FIVE_ROUND_FINISH_OVERLAY_WEIGHTS = {
    "cardio_fade_sum": 0.15,
    "finish_loss_rate_sum": 0.10,
    "decision_rate_sum": -0.10,
}


def logistic(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _blend_toward_market(raw_prob: float, market_prob: float, confidence: float) -> tuple[float, float]:
    return _blend_toward_market_with_context(
        raw_prob,
        market_prob,
        confidence,
        consensus_count=0.0,
        overround=0.0,
    )


def _blend_toward_market_with_context(
    raw_prob: float,
    market_prob: float,
    confidence: float,
    *,
    consensus_count: float,
    overround: float,
) -> tuple[float, float]:
    disagreement = abs(raw_prob - market_prob)
    consensus_credit = _clamp((consensus_count - 1) / 6, 0.0, 1.0) * 0.15
    margin_penalty = _clamp(overround / 0.08, 0.0, 1.0) * 0.10
    blend_weight = 0.05 + (disagreement * 0.95) + ((1 - confidence) * 0.20) + consensus_credit - margin_penalty
    blend_weight = max(0.05, min(0.55, blend_weight))
    blended = (raw_prob * (1 - blend_weight)) + (market_prob * blend_weight)
    cap = 0.80 if disagreement > 0.12 or confidence < 0.90 else 0.88
    floor = 1 - cap
    return max(floor, min(cap, blended)), blend_weight


def _score_with_weights(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)
    for feature_name, weight in weights.items():
        score = score + (_numeric_series(frame, feature_name, 0.0) * weight)
    return score


def _row_scalar(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if isinstance(value, pd.Series):
        value = value.dropna().iloc[0] if not value.dropna().empty else default
    if pd.isna(value):
        return default
    return float(value)


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _clip01(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def _compute_model_confidence(scored: pd.DataFrame, baseline_raw_prob: pd.Series) -> pd.Series:
    baseline_prob = pd.to_numeric(baseline_raw_prob, errors="coerce").fillna(0.5).clip(lower=0.05, upper=0.95)
    quality_signal = _numeric_series(scored, "data_quality", 1.0).clip(lower=0.35, upper=1.0)
    fallback_penalty = _numeric_series(scored, "fallback_penalty", 0.0)
    fallback_signal = 1 - _clip01(fallback_penalty / 2.0)

    certainty_margin = (baseline_prob - 0.5).abs() * 2.0
    certainty_signal = 0.25 + (_clip01(certainty_margin / 0.35) * 0.75)

    a_ufc_fights = _numeric_series(scored, "a_ufc_fight_count", 0.0)
    b_ufc_fights = _numeric_series(scored, "b_ufc_fight_count", 0.0)
    min_ufc_sample = pd.concat([a_ufc_fights, b_ufc_fights], axis=1).min(axis=1)
    sample_signal = 0.45 + (_clip01(min_ufc_sample / 8.0) * 0.55)

    context_noise = (
        _numeric_series(scored, "a_short_notice_flag", 0.0)
        + _numeric_series(scored, "b_short_notice_flag", 0.0)
        + _numeric_series(scored, "a_cardio_fade_flag", 0.0)
        + _numeric_series(scored, "b_cardio_fade_flag", 0.0)
        + _numeric_series(scored, "a_injury_concern_flag", 0.0)
        + _numeric_series(scored, "b_injury_concern_flag", 0.0)
        + _numeric_series(scored, "a_weight_cut_concern_flag", 0.0)
        + _numeric_series(scored, "b_weight_cut_concern_flag", 0.0)
        + _numeric_series(scored, "a_replacement_fighter_flag", 0.0)
        + _numeric_series(scored, "b_replacement_fighter_flag", 0.0)
        + _numeric_series(scored, "a_travel_disadvantage_flag", 0.0)
        + _numeric_series(scored, "b_travel_disadvantage_flag", 0.0)
        + _numeric_series(scored, "a_new_gym_flag", 0.0)
        + _numeric_series(scored, "b_new_gym_flag", 0.0)
        + _numeric_series(scored, "a_camp_change_flag", 0.0)
        + _numeric_series(scored, "b_camp_change_flag", 0.0)
    )
    context_signal = 1 - _clip01(context_noise / 6.0)

    market_prob = _numeric_series(scored, "fighter_a_current_implied_prob", 0.5)
    consensus_count = _numeric_series(scored, "market_consensus_bookmaker_count", 0.0)
    overround = _numeric_series(scored, "market_overround", 0.04)
    market_alignment = 1 - _clip01((baseline_prob - market_prob).abs() / 0.18)
    market_reliability = (
        0.35
        + (_clip01((consensus_count - 1.0) / 5.0) * 0.40)
        + ((1 - _clip01(overround / 0.08)) * 0.25)
    )
    market_signal = 0.50 + ((market_alignment - 0.50) * market_reliability.clip(lower=0.0, upper=1.0))

    a_debut = _numeric_series(scored, "a_ufc_debut_flag", 0.0)
    b_debut = _numeric_series(scored, "b_ufc_debut_flag", 0.0)
    debut_penalty = pd.concat([a_debut, b_debut], axis=1).max(axis=1) * 0.08
    double_debut_penalty = ((a_debut + b_debut) >= 2).astype(float) * 0.04
    heavyweight_penalty = _numeric_series(scored, "is_heavyweight", 0.0).clip(lower=0.0, upper=1.0) * 0.03

    confidence = (
        (quality_signal * 0.30)
        + (certainty_signal * 0.28)
        + (sample_signal * 0.14)
        + (context_signal * 0.12)
        + (market_signal * 0.10)
        + (fallback_signal * 0.06)
        - debut_penalty
        - double_debut_penalty
        - heavyweight_penalty
    )
    return confidence.clip(lower=0.35, upper=0.92)


def _size_relevance_multiplier(weight_class_series: pd.Series) -> pd.Series:
    normalized = weight_class_series.astype(str).str.lower()
    multiplier = pd.Series(0.20, index=weight_class_series.index, dtype=float)
    multiplier.loc[normalized.str.contains("women", na=False)] = 0.12
    multiplier.loc[normalized.str.contains("middle", na=False)] = 0.30
    multiplier.loc[normalized.str.contains("light heavy", na=False)] = 0.40
    multiplier.loc[normalized.str.contains("heavy", na=False)] = 0.75
    return multiplier


def _masked_overlay_score(frame: pd.DataFrame, weights: dict[str, float], mask: pd.Series) -> pd.Series:
    if mask.empty:
        return pd.Series(0.0, index=frame.index, dtype=float)
    overlay = _score_with_weights(frame, weights)
    return overlay.where(mask.astype(bool), 0.0)


def _finish_share_score(scored: pd.DataFrame) -> pd.Series:
    return (
        (scored["finish_win_rate_diff"] * 1.1)
        + (scored["durability_diff"] * 0.9)
        + (scored["first_round_finish_rate_diff"] * 0.9)
        + (scored["strike_margin_diff"] * 0.15)
        + (scored["grappling_diff"] * 0.10)
        + (scored["grappling_pressure_diff"] * 0.08)
    )


def _method_split_scores(row: pd.Series, pick_side: str) -> tuple[float, float]:
    own_prefix = "a" if pick_side == "fighter_a" else "b"
    opp_prefix = "b" if pick_side == "fighter_a" else "a"
    sign = 1.0 if pick_side == "fighter_a" else -1.0

    submission_score = (
        (float(row.get(f"{own_prefix}_submission_win_rate", 0.0) or 0.0) * 2.4)
        + (float(row.get(f"{opp_prefix}_submission_loss_rate", 0.0) or 0.0) * 1.6)
        + (float(row.get(f"{own_prefix}_submission_avg", 0.0) or 0.0) * 0.55)
        + (float(row.get(f"{own_prefix}_recent_control_avg", 0.0) or 0.0) * 0.14)
        + (float(row.get(f"{own_prefix}_takedown_avg", 0.0) or 0.0) * 0.18)
        + (((100 - float(row.get(f"{opp_prefix}_takedown_defense_pct", 0.0) or 0.0)) / 100.0) * 0.9)
        + ((sign * float(row.get("grappling_diff", 0.0) or 0.0)) * 0.22)
        + ((sign * float(row.get("recent_control_diff", 0.0) or 0.0)) * 0.16)
        + ((sign * float(row.get("grappling_pressure_diff", 0.0) or 0.0)) * 0.12)
        - ((sign * float(row.get("strike_margin_diff", 0.0) or 0.0)) * 0.08)
        - (float(row.get(f"{own_prefix}_ko_win_rate", 0.0) or 0.0) * 0.8)
    )
    ko_score = (
        (float(row.get(f"{own_prefix}_ko_win_rate", 0.0) or 0.0) * 2.2)
        + (float(row.get(f"{opp_prefix}_ko_loss_rate", 0.0) or 0.0) * 1.4)
        + (max(0.0, float(row.get(f"{own_prefix}_sig_strikes_landed_per_min", 0.0) or 0.0) - 3.0) * 0.12)
        + ((sign * float(row.get("strike_margin_diff", 0.0) or 0.0)) * 0.2)
        + ((sign * float(row.get("first_round_finish_rate_diff", 0.0) or 0.0)) * 0.8)
        + ((sign * float(row.get("durability_diff", 0.0) or 0.0)) * 0.5)
        - (float(row.get(f"{own_prefix}_recent_control_avg", 0.0) or 0.0) * 0.05)
        - (float(row.get(f"{own_prefix}_submission_win_rate", 0.0) or 0.0) * 0.9)
        - (float(row.get(f"{own_prefix}_submission_avg", 0.0) or 0.0) * 0.12)
    )
    return submission_score, ko_score


def project_fight_probabilities(
    feature_frame: pd.DataFrame,
    *,
    side_model_bundle: dict[str, object] | None = None,
    confidence_model_bundle: dict[str, object] | None = None,
) -> pd.DataFrame:
    scored = feature_frame.copy()
    if scored.empty:
        scored["raw_projection_score"] = pd.Series(dtype=float)
        scored["baseline_raw_fighter_a_win_prob"] = pd.Series(dtype=float)
        scored["heuristic_model_confidence"] = pd.Series(dtype=float)
        scored["projection_score"] = pd.Series(dtype=float)
        scored["raw_projected_fighter_a_win_prob"] = pd.Series(dtype=float)
        scored["market_blend_weight"] = pd.Series(dtype=float)
        scored["base_projected_fighter_a_win_prob"] = pd.Series(dtype=float)
        scored["trained_side_fighter_a_win_prob"] = pd.Series(dtype=float)
        scored["trained_side_selection_prob"] = pd.Series(dtype=float)
        scored["side_model_blend_weight"] = pd.Series(dtype=float)
        scored["projected_fighter_a_win_prob"] = pd.Series(dtype=float)
        scored["finish_projection_score"] = pd.Series(dtype=float)
        scored["projected_finish_prob"] = pd.Series(dtype=float)
        scored["projected_decision_prob"] = pd.Series(dtype=float)
        scored["fighter_a_inside_distance_prob"] = pd.Series(dtype=float)
        scored["fighter_b_inside_distance_prob"] = pd.Series(dtype=float)
        scored["fighter_a_submission_prob"] = pd.Series(dtype=float)
        scored["fighter_b_submission_prob"] = pd.Series(dtype=float)
        scored["fighter_a_ko_tko_prob"] = pd.Series(dtype=float)
        scored["fighter_b_ko_tko_prob"] = pd.Series(dtype=float)
        scored["fighter_a_by_decision_prob"] = pd.Series(dtype=float)
        scored["fighter_b_by_decision_prob"] = pd.Series(dtype=float)
        scored["model_confidence"] = pd.Series(dtype=float)
        scored["model_projected_win_prob"] = pd.Series(dtype=float)
        return scored

    weight_class_series = scored["a_weight_class"] if "a_weight_class" in scored.columns else pd.Series("", index=scored.index)
    size_multiplier = _size_relevance_multiplier(weight_class_series)
    is_wmma_mask = pd.to_numeric(scored.get("is_wmma", pd.Series(0.0, index=scored.index)), errors="coerce").fillna(0.0) >= 1
    is_heavyweight_mask = pd.to_numeric(scored.get("is_heavyweight", pd.Series(0.0, index=scored.index)), errors="coerce").fillna(0.0) >= 1
    is_five_round_mask = pd.to_numeric(scored.get("is_five_round_fight", pd.Series(0.0, index=scored.index)), errors="coerce").fillna(0.0) >= 1

    scored["size_edge_score"] = (
        ((scored["reach_diff"] * 0.012) + (scored["height_diff"] * 0.008))
        * size_multiplier
    )
    scored["segment_projection_overlay"] = (
        _masked_overlay_score(scored, WMMA_MONEYLINE_OVERLAY_WEIGHTS, is_wmma_mask)
        + _masked_overlay_score(scored, HEAVYWEIGHT_MONEYLINE_OVERLAY_WEIGHTS, is_heavyweight_mask)
        + _masked_overlay_score(scored, FIVE_ROUND_MONEYLINE_OVERLAY_WEIGHTS, is_five_round_mask)
    )
    raw_projection_score = (
        _score_with_weights(scored, MONEYLINE_FEATURE_WEIGHTS)
        + scored["size_edge_score"]
        + scored["segment_projection_overlay"]
    )
    baseline_raw_prob = raw_projection_score.apply(logistic)
    heuristic_confidence = _compute_model_confidence(scored, baseline_raw_prob)
    scored["raw_projection_score"] = raw_projection_score
    scored["baseline_raw_fighter_a_win_prob"] = baseline_raw_prob
    scored["heuristic_model_confidence"] = heuristic_confidence
    scored["projected_fighter_a_win_prob"] = baseline_raw_prob
    confidence = apply_confidence_model(scored, confidence_model_bundle, heuristic_confidence)
    scored["projection_score"] = raw_projection_score * confidence
    scored["raw_projected_fighter_a_win_prob"] = scored["projection_score"].apply(logistic)
    scored["market_blend_weight"] = 0.0
    scored["projected_fighter_a_win_prob"] = scored["raw_projected_fighter_a_win_prob"]
    if not scored.empty and "fighter_a_current_implied_prob" in scored.columns:
        blend_values = scored.apply(
            lambda row: _blend_toward_market_with_context(
                float(row["raw_projected_fighter_a_win_prob"]),
                float(row["fighter_a_current_implied_prob"]),
                float(confidence.loc[row.name]),
                consensus_count=_row_scalar(row, "market_consensus_bookmaker_count", 0.0),
                overround=_row_scalar(row, "market_overround", 0.0),
            )
            if pd.notna(row["fighter_a_current_implied_prob"])
            else (float(row["raw_projected_fighter_a_win_prob"]), 0.0),
            axis=1,
            result_type="expand",
        )
        blend_values.columns = ["projected_fighter_a_win_prob", "market_blend_weight"]
        scored["projected_fighter_a_win_prob"] = blend_values["projected_fighter_a_win_prob"]
        scored["market_blend_weight"] = blend_values["market_blend_weight"]
    scored = apply_side_model_adjustments(scored, side_model_bundle)

    scored["segment_finish_overlay"] = (
        _masked_overlay_score(scored, WMMA_FINISH_OVERLAY_WEIGHTS, is_wmma_mask)
        + _masked_overlay_score(scored, HEAVYWEIGHT_FINISH_OVERLAY_WEIGHTS, is_heavyweight_mask)
        + _masked_overlay_score(scored, FIVE_ROUND_FINISH_OVERLAY_WEIGHTS, is_five_round_mask)
    )
    scored["finish_projection_score"] = _score_with_weights(scored, FINISH_FEATURE_WEIGHTS) + scored["segment_finish_overlay"]
    scored["finish_projection_score"] = scored["finish_projection_score"] * confidence
    scored["projected_finish_prob"] = scored["finish_projection_score"].apply(logistic).clip(lower=0.18, upper=0.82)
    scored["projected_decision_prob"] = 1 - scored["projected_finish_prob"]

    finish_share_a = _finish_share_score(scored).apply(logistic).clip(lower=0.10, upper=0.90)
    raw_a_inside = scored["projected_fighter_a_win_prob"] * finish_share_a
    raw_b_inside = (1 - scored["projected_fighter_a_win_prob"]) * (1 - finish_share_a)
    total_raw_finish = raw_a_inside + raw_b_inside
    finish_scale = pd.Series(0.0, index=scored.index, dtype=float)
    non_zero_mask = total_raw_finish > 0
    finish_scale.loc[non_zero_mask] = scored.loc[non_zero_mask, "projected_finish_prob"] / total_raw_finish.loc[non_zero_mask]

    scored["fighter_a_inside_distance_prob"] = (raw_a_inside * finish_scale).clip(lower=0.01, upper=0.97)
    scored["fighter_b_inside_distance_prob"] = (raw_b_inside * finish_scale).clip(lower=0.01, upper=0.97)
    method_split = scored.apply(
        lambda row: (
            *(_method_split_scores(row, "fighter_a")),
            *(_method_split_scores(row, "fighter_b")),
        ),
        axis=1,
        result_type="expand",
    )
    method_split.columns = [
        "fighter_a_submission_score",
        "fighter_a_ko_score",
        "fighter_b_submission_score",
        "fighter_b_ko_score",
    ]
    fighter_a_submission_share = (
        method_split["fighter_a_submission_score"].apply(logistic).clip(lower=0.08, upper=0.82)
    )
    fighter_b_submission_share = (
        method_split["fighter_b_submission_score"].apply(logistic).clip(lower=0.08, upper=0.82)
    )
    scored["fighter_a_submission_prob"] = (scored["fighter_a_inside_distance_prob"] * fighter_a_submission_share).clip(lower=0.001, upper=0.97)
    scored["fighter_b_submission_prob"] = (scored["fighter_b_inside_distance_prob"] * fighter_b_submission_share).clip(lower=0.001, upper=0.97)
    scored["fighter_a_ko_tko_prob"] = (scored["fighter_a_inside_distance_prob"] - scored["fighter_a_submission_prob"]).clip(lower=0.001, upper=0.97)
    scored["fighter_b_ko_tko_prob"] = (scored["fighter_b_inside_distance_prob"] - scored["fighter_b_submission_prob"]).clip(lower=0.001, upper=0.97)
    scored["fighter_a_by_decision_prob"] = (scored["projected_fighter_a_win_prob"] - scored["fighter_a_inside_distance_prob"]).clip(lower=0.01, upper=0.97)
    scored["fighter_b_by_decision_prob"] = ((1 - scored["projected_fighter_a_win_prob"]) - scored["fighter_b_inside_distance_prob"]).clip(lower=0.01, upper=0.97)

    scored["model_confidence"] = confidence
    scored["model_projected_win_prob"] = scored.apply(
        lambda row: row["projected_fighter_a_win_prob"] if row["selection"] == "fighter_a" else 1 - row["projected_fighter_a_win_prob"],
        axis=1,
    )
    return scored
