from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from models.ev import american_to_decimal


@dataclass(frozen=True)
class BankrollGovernorConfig:
    max_stake_pct: float = 0.05
    max_card_exposure_pct: float = 0.12
    max_fight_exposure_pct: float = 0.06
    max_market_family_exposure_pct: float = 0.08
    watchlist_multiplier: float = 0.50
    medium_fragility_multiplier: float = 0.75
    high_fragility_multiplier: float = 0.40
    prop_multiplier: float = 0.75
    disagreement_multiplier: float = 0.80
    negative_history_multiplier: float = 0.80
    min_actionable_stake: float = 0.0


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(numeric):
        return default
    return numeric


def _coerce_fraction(value: object, default: float) -> float:
    return _clamp(_coerce_float(value, default), 0.0, 1.0)


def _coerce_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _tier_rank(value: object) -> int:
    return {"A": 3, "B": 2, "C": 1}.get(_coerce_text(value).upper(), 0)


def _market_family_key(row: pd.Series | dict[str, object]) -> str:
    market_key = _coerce_text(row.get("tracked_market_key", row.get("market", "moneyline"))).lower()
    if market_key in {"moneyline", "h2h", "h2h_lay"}:
        return "moneyline"
    if market_key in {
        "fight_goes_to_decision",
        "fight_goes_distance",
        "fight_goes_the_distance",
        "fight_does_not_go_the_distance",
        "fight_doesnt_go_to_decision",
    }:
        return "decision"
    if market_key in {"inside_distance", "ko_tko", "submission", "finish"}:
        return "finish"
    if "decision" in market_key:
        return "decision"
    if "distance" in market_key or "finish" in market_key:
        return "finish"
    return "prop"


def fractional_kelly_fraction(projected_win_prob: float, american_odds: int, fraction: float = 0.25) -> float:
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1
    q = 1 - projected_win_prob
    raw_kelly = ((b * projected_win_prob) - q) / b
    return max(0.0, raw_kelly * fraction)


def suggested_stake(bankroll: float, projected_win_prob: float, american_odds: int, fraction: float = 0.25) -> float:
    return round(bankroll * fractional_kelly_fraction(projected_win_prob, american_odds, fraction), 2)


def bankroll_governor_config_from_env() -> BankrollGovernorConfig:
    return BankrollGovernorConfig(
        max_stake_pct=_coerce_fraction(os.getenv("MAX_BET_STAKE_PCT", "0.05"), 0.05),
        max_card_exposure_pct=_coerce_fraction(os.getenv("MAX_CARD_EXPOSURE_PCT", "0.12"), 0.12),
        max_fight_exposure_pct=_coerce_fraction(os.getenv("MAX_FIGHT_EXPOSURE_PCT", "0.06"), 0.06),
        max_market_family_exposure_pct=_coerce_fraction(
            os.getenv("MAX_MARKET_FAMILY_EXPOSURE_PCT", "0.08"),
            0.08,
        ),
        watchlist_multiplier=_coerce_fraction(os.getenv("WATCHLIST_STAKE_MULTIPLIER", "0.50"), 0.50),
        medium_fragility_multiplier=_coerce_fraction(
            os.getenv("MEDIUM_FRAGILITY_STAKE_MULTIPLIER", "0.75"),
            0.75,
        ),
        high_fragility_multiplier=_coerce_fraction(
            os.getenv("HIGH_FRAGILITY_STAKE_MULTIPLIER", "0.40"),
            0.40,
        ),
        prop_multiplier=_coerce_fraction(os.getenv("PROP_STAKE_MULTIPLIER", "0.75"), 0.75),
        disagreement_multiplier=_coerce_fraction(
            os.getenv("MARKET_DISAGREEMENT_STAKE_MULTIPLIER", "0.80"),
            0.80,
        ),
        negative_history_multiplier=_coerce_fraction(
            os.getenv("NEGATIVE_HISTORY_STAKE_MULTIPLIER", "0.80"),
            0.80,
        ),
        min_actionable_stake=max(0.0, _coerce_float(os.getenv("MIN_ACTIONABLE_STAKE", "0.0"), 0.0)),
    )


def _governor_multiplier(row: pd.Series, config: BankrollGovernorConfig) -> tuple[float, list[str]]:
    reasons: list[str] = []
    action = _coerce_text(row.get("recommended_action", ""))
    tier = _coerce_text(row.get("recommended_tier", "")).upper()
    hard_gate_reason = _coerce_text(row.get("hard_gate_reason", ""))

    if hard_gate_reason:
        return 0.0, ["hard_gate"]
    if action == "Pass" or tier == "C":
        return 0.0, ["model_pass"]

    multiplier = 1.0
    if action == "Watchlist":
        multiplier *= config.watchlist_multiplier
        reasons.append("watchlist_half_stake")

    fragility_bucket = _coerce_text(row.get("fragility_bucket", "low"), "low").lower()
    if fragility_bucket == "medium":
        multiplier *= config.medium_fragility_multiplier
        reasons.append("fragility_medium_trim")
    elif fragility_bucket == "high":
        multiplier *= config.high_fragility_multiplier
        reasons.append("fragility_high_trim")

    tracked_market_key = _coerce_text(
        row.get("tracked_market_key", row.get("market", "moneyline")),
        "moneyline",
    ).lower()
    if tracked_market_key != "moneyline":
        multiplier *= config.prop_multiplier
        reasons.append("prop_volatility_trim")

    if _coerce_float(row.get("market_blend_weight", 0.0), 0.0) >= 0.40:
        multiplier *= config.disagreement_multiplier
        reasons.append("market_disagreement_trim")

    historical_grade = _coerce_text(row.get("historical_overlay_grade", "low_sample"), "low_sample")
    historical_sample_size = int(_coerce_float(row.get("historical_sample_size", 0.0), 0.0))
    if historical_sample_size >= 4 and "negative" in historical_grade:
        multiplier *= config.negative_history_multiplier
        reasons.append("historical_negative_trim")

    return round(multiplier, 4), reasons


def apply_bankroll_governor(
    frame: pd.DataFrame,
    *,
    bankroll: float,
    config: BankrollGovernorConfig | None = None,
) -> pd.DataFrame:
    governed = frame.copy()
    if governed.empty:
        for column, default_value in {
            "raw_suggested_stake": pd.Series(dtype=float),
            "raw_chosen_expression_stake": pd.Series(dtype=float),
            "stake_governor_multiplier": pd.Series(dtype=float),
            "stake_cap_per_bet": pd.Series(dtype=float),
            "stake_cap_per_fight": pd.Series(dtype=float),
            "stake_cap_per_card": pd.Series(dtype=float),
            "market_family_key": pd.Series(dtype=str),
            "stake_governor_reason": pd.Series(dtype=str),
        }.items():
            if column not in governed.columns:
                governed[column] = default_value
        return governed

    settings = config or BankrollGovernorConfig()
    bankroll_value = max(0.0, float(bankroll))
    per_bet_cap = round(bankroll_value * settings.max_stake_pct, 2)
    per_fight_cap = round(bankroll_value * settings.max_fight_exposure_pct, 2)
    per_card_cap = round(bankroll_value * settings.max_card_exposure_pct, 2)

    raw_suggested = pd.to_numeric(
        governed.get("suggested_stake", pd.Series(0.0, index=governed.index)),
        errors="coerce",
    ).fillna(0.0)
    raw_chosen = pd.to_numeric(
        governed.get(
            "chosen_expression_stake",
            governed.get("suggested_stake", pd.Series(0.0, index=governed.index)),
        ),
        errors="coerce",
    ).fillna(0.0)
    governed["raw_suggested_stake"] = raw_suggested.round(2)
    governed["raw_chosen_expression_stake"] = raw_chosen.round(2)
    governed["stake_governor_multiplier"] = 0.0
    governed["stake_cap_per_bet"] = per_bet_cap
    governed["stake_cap_per_fight"] = per_fight_cap
    governed["stake_cap_per_card"] = per_card_cap
    governed["stake_governor_reason"] = ""
    governed["chosen_expression_stake"] = 0.0

    priority = governed.copy()
    priority["_event_key"] = priority.get("event_id", pd.Series("", index=priority.index)).fillna("").astype(str).str.strip()
    priority.loc[priority["_event_key"] == "", "_event_key"] = (
        priority.get("event_name", pd.Series("", index=priority.index)).fillna("").astype(str).str.strip()
    )
    priority["_fight_key"] = priority.get("fight_key", pd.Series("", index=priority.index)).fillna("").astype(str).str.strip()
    fallback_fight_keys = (
        priority.get("fighter_a", pd.Series("", index=priority.index)).fillna("").astype(str).str.strip()
        + "||"
        + priority.get("fighter_b", pd.Series("", index=priority.index)).fillna("").astype(str).str.strip()
    )
    priority.loc[priority["_fight_key"] == "", "_fight_key"] = fallback_fight_keys.loc[priority["_fight_key"] == ""]
    priority["_tier_rank"] = priority.get("recommended_tier", pd.Series("", index=priority.index)).apply(_tier_rank)
    priority["_score_rank"] = pd.to_numeric(
        priority.get("bet_quality_score", pd.Series(0.0, index=priority.index)),
        errors="coerce",
    ).fillna(0.0)
    priority["_edge_rank"] = pd.to_numeric(
        priority.get("effective_edge", priority.get("chosen_expression_edge", pd.Series(0.0, index=priority.index))),
        errors="coerce",
    ).fillna(0.0)
    priority["_raw_stake_rank"] = governed["raw_chosen_expression_stake"]
    priority = priority.sort_values(
        by=["_event_key", "_tier_rank", "_score_rank", "_edge_rank", "_raw_stake_rank"],
        ascending=[True, False, False, False, False],
    )

    card_usage: dict[str, float] = {}
    fight_usage: dict[tuple[str, str], float] = {}
    family_usage: dict[tuple[str, str], float] = {}
    final_stakes: dict[object, float] = {}
    final_reasons: dict[object, str] = {}
    final_multipliers: dict[object, float] = {}
    final_families: dict[object, str] = {}

    for idx, row in priority.iterrows():
        multiplier, reasons = _governor_multiplier(row, settings)
        raw_stake = float(governed.at[idx, "raw_chosen_expression_stake"])
        desired_stake = raw_stake * multiplier
        if desired_stake > per_bet_cap + 1e-9:
            reasons.append("per_bet_cap")
        desired_stake = min(desired_stake, per_bet_cap)

        event_key = _coerce_text(row.get("_event_key", ""), "unknown_event")
        fight_key = _coerce_text(row.get("_fight_key", ""), "unknown_fight")
        market_family_key = _market_family_key(row)
        card_remaining = max(0.0, per_card_cap - card_usage.get(event_key, 0.0))
        fight_remaining = max(0.0, per_fight_cap - fight_usage.get((event_key, fight_key), 0.0))
        family_remaining = max(0.0, round(bankroll_value * settings.max_market_family_exposure_pct, 2) - family_usage.get((event_key, market_family_key), 0.0))

        if desired_stake > fight_remaining + 1e-9:
            reasons.append("fight_exposure_cap")
        if desired_stake > card_remaining + 1e-9:
            reasons.append("card_exposure_cap")
        if desired_stake > family_remaining + 1e-9:
            reasons.append("market_family_exposure_cap")

        final_stake = min(desired_stake, fight_remaining, card_remaining, family_remaining)
        if 0.0 < final_stake < settings.min_actionable_stake:
            final_stake = 0.0
            reasons.append("below_min_actionable_stake")

        final_stake = round(max(0.0, final_stake), 2)
        if final_stake > 0.0:
            card_usage[event_key] = round(card_usage.get(event_key, 0.0) + final_stake, 2)
            fight_usage[(event_key, fight_key)] = round(fight_usage.get((event_key, fight_key), 0.0) + final_stake, 2)
            family_usage[(event_key, market_family_key)] = round(family_usage.get((event_key, market_family_key), 0.0) + final_stake, 2)

        final_stakes[idx] = final_stake
        final_multipliers[idx] = multiplier
        final_reasons[idx] = ", ".join(dict.fromkeys(reasons))
        final_families[idx] = market_family_key

    governed["chosen_expression_stake"] = governed.index.to_series().map(final_stakes).fillna(0.0).astype(float)
    governed["stake_governor_multiplier"] = governed.index.to_series().map(final_multipliers).fillna(0.0).astype(float)
    governed["stake_governor_reason"] = governed.index.to_series().map(final_reasons).fillna("").astype(str)
    governed["market_family_key"] = governed.index.to_series().map(final_families).fillna("prop").astype(str)
    return governed
