from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ev import american_to_decimal, implied_probability, probability_to_american
from scripts.build_fight_week_report import _colorize


DEFAULT_MIN_LEGS = 2
DEFAULT_MAX_LEGS = 5
DEFAULT_MIN_EDGE = 0.04
DEFAULT_MIN_MODEL_PROB = 0.42
DEFAULT_MIN_CONFIDENCE = 0.64
DEFAULT_MIN_DATA_QUALITY = 0.85
DEFAULT_MAX_FAVORITE_AMERICAN = -300
DEFAULT_EXTENDED_MAX_FAVORITE_AMERICAN = -400
DEFAULT_MAX_DOG_AMERICAN = 250
DEFAULT_MAX_MARKET_BLEND_WEIGHT = 0.55
DEFAULT_MAX_CANDIDATES = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build best-value 2-to-5 leg UFC parlays from the value report.")
    parser.add_argument("--input", required=True, help="Value-report CSV path.")
    parser.add_argument("--output", required=True, help="Parlay-board CSV output path.")
    parser.add_argument("--min-legs", type=int, default=DEFAULT_MIN_LEGS, help="Minimum parlay size.")
    parser.add_argument("--max-legs", type=int, default=DEFAULT_MAX_LEGS, help="Maximum parlay size.")
    parser.add_argument("--min-edge", type=float, default=DEFAULT_MIN_EDGE, help="Minimum single-leg edge.")
    parser.add_argument(
        "--min-model-prob",
        type=float,
        default=DEFAULT_MIN_MODEL_PROB,
        help="Minimum single-leg projected probability.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="Minimum single-leg model confidence.",
    )
    parser.add_argument(
        "--min-data-quality",
        type=float,
        default=DEFAULT_MIN_DATA_QUALITY,
        help="Minimum single-leg data quality.",
    )
    parser.add_argument(
        "--max-favorite-american",
        type=int,
        default=DEFAULT_MAX_FAVORITE_AMERICAN,
        help="Reject favorites shorter than this price.",
    )
    parser.add_argument(
        "--extended-max-favorite-american",
        type=int,
        default=DEFAULT_EXTENDED_MAX_FAVORITE_AMERICAN,
        help="Allow premium heavy-favorite legs down to this price when the profile is strong enough.",
    )
    parser.add_argument(
        "--max-dog-american",
        type=int,
        default=DEFAULT_MAX_DOG_AMERICAN,
        help="Reject underdogs longer than this price.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=DEFAULT_MAX_CANDIDATES,
        help="Maximum filtered legs to consider before generating combinations.",
    )
    parser.add_argument(
        "--max-market-blend-weight",
        type=float,
        default=DEFAULT_MAX_MARKET_BLEND_WEIGHT,
        help="Maximum allowed model-vs-market blend weight for a parlay leg.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _numeric_series(frame: pd.DataFrame, names: list[str], default: float) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _selection_oriented_numeric_series(frame: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    series = _numeric_series(frame, names, default)
    selection = frame.get("selection", pd.Series("fighter_a", index=frame.index)).fillna("fighter_a").astype(str)
    signs = selection.eq("fighter_a").map({True: 1.0, False: -1.0})
    return series * signs


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_decimal(value: float) -> str:
    return f"{value:.2f}"


def _parse_value(value: object) -> float:
    text = _safe_text(value)
    if not text:
        return 0.0
    normalized = text.replace("%", "")
    try:
        numeric = float(normalized)
    except ValueError:
        return 0.0
    if "%" in text or abs(numeric) > 1.0:
        return numeric / 100.0
    return numeric


def _highlight_title(title: str, color: str) -> str:
    return _colorize(title, color)


def _highlight_edge(value: object) -> str:
    numeric = _parse_value(value)
    if numeric >= 0.10:
        color = "green"
    elif numeric >= 0.05:
        color = "cyan"
    elif numeric >= 0:
        color = "yellow"
    else:
        color = "red"
    return _colorize(_safe_text(value), color)


def _highlight_confidence(value: object) -> str:
    numeric = _parse_value(value)
    if numeric >= 0.80:
        color = "green"
    elif numeric >= 0.68:
        color = "cyan"
    elif numeric >= 0.58:
        color = "yellow"
    else:
        color = "red"
    return _colorize(_safe_text(value), color)


def _highlight_stake_profile(value: object) -> str:
    text = _safe_text(value, "unknown stake profile")
    normalized = text.lower()
    if "all a-tier" in normalized:
        color = "green"
    elif "core" in normalized:
        color = "cyan"
    else:
        color = "yellow"
    return _colorize(text, color)


def _highlight_risk_summary(value: object) -> str:
    text = _safe_text(value, "none")
    normalized = text.lower()
    if "medium-risk" in normalized:
        color = "yellow"
    elif "premium heavy favorites" in normalized and "no premium heavy favorites" not in normalized:
        color = "red"
    else:
        color = "green"
    return _colorize(text, color)


def _highlight_parlay_name(value: object) -> str:
    return _colorize(_safe_text(value), "cyan")


def _highlight_leg_count(value: object) -> str:
    return _colorize(f"{int(value)}-leg", "cyan")


def _fragility_penalty(bucket: str) -> float:
    normalized = str(bucket or "").strip().lower()
    if normalized == "high":
        return 18.0
    if normalized == "medium":
        return 7.0
    return 0.0


def _tier_bonus(tier: str) -> float:
    return {"A": 6.0, "B": 2.5}.get(str(tier or "").strip().upper(), 0.0)


def _infer_expression_market(value: object) -> str:
    text = _safe_text(value).lower()
    if not text:
        return "moneyline"
    if text in {"prop", "props", "alternative_market", "runner_up_expression"}:
        return "prop"
    if text in {"moneyline", "side_market"}:
        return "moneyline"
    prop_keywords = [
        "inside distance",
        "decision",
        "ko",
        "tko",
        "sub",
        "submission",
        "round",
        "goes to",
        "doesn't go",
        "does not go",
        "over ",
        "under ",
    ]
    if any(keyword in text for keyword in prop_keywords):
        return "prop"
    return "moneyline"


def _format_market_mix(counts: dict[str, int]) -> str:
    parts: list[str] = []
    moneylines = counts.get("moneyline", 0)
    props = counts.get("prop", 0)
    if moneylines:
        parts.append(f"{moneylines} moneyline")
    if props:
        parts.append(f"{props} prop")
    return " / ".join(parts) if parts else "unknown"


def _expand_expression_candidates(working: pd.DataFrame) -> pd.DataFrame:
    required = {
        "runner_up_expression",
        "runner_up_odds",
        "runner_up_prob",
        "runner_up_implied_prob",
        "runner_up_edge",
    }
    if not required.issubset(set(working.columns)):
        return working

    base = working.copy()
    runner_rows = base.copy()
    runner_expression = runner_rows["runner_up_expression"].fillna("").astype(str).str.strip()
    chosen_expression = runner_rows.get("chosen_value_expression", pd.Series("", index=runner_rows.index)).fillna("").astype(str).str.strip()
    runner_odds = pd.to_numeric(runner_rows["runner_up_odds"], errors="coerce")
    runner_prob = pd.to_numeric(runner_rows["runner_up_prob"], errors="coerce")
    runner_implied = pd.to_numeric(runner_rows["runner_up_implied_prob"], errors="coerce")
    runner_edge = pd.to_numeric(runner_rows["runner_up_edge"], errors="coerce")
    runner_mask = (
        runner_expression.ne("")
        & runner_expression.ne(chosen_expression)
        & runner_odds.notna()
        & runner_prob.gt(0.0)
        & runner_prob.lt(1.0)
        & runner_implied.gt(0.0)
        & runner_implied.lt(1.0)
        & runner_edge.notna()
    )
    runner_rows = runner_rows.loc[runner_mask].copy()
    if runner_rows.empty:
        return base

    runner_rows["chosen_value_expression"] = runner_rows["runner_up_expression"]
    runner_rows["chosen_expression_odds"] = runner_rows["runner_up_odds"]
    runner_rows["chosen_expression_prob"] = runner_rows["runner_up_prob"]
    runner_rows["chosen_expression_implied_prob"] = runner_rows["runner_up_implied_prob"]
    runner_rows["chosen_expression_edge"] = runner_rows["runner_up_edge"]
    runner_rows["effective_american_odds"] = runner_rows["runner_up_odds"]
    runner_rows["effective_projected_prob"] = runner_rows["runner_up_prob"]
    runner_rows["effective_implied_prob"] = runner_rows["runner_up_implied_prob"]
    runner_rows["effective_edge"] = runner_rows["runner_up_edge"]
    runner_rows["expression_pick_source"] = "runner_up_expression"
    runner_rows["tracked_market_key"] = runner_rows["runner_up_expression"].map(_infer_expression_market)

    expanded = pd.concat([base, runner_rows], ignore_index=True, sort=False)
    dedupe_columns = [
        column
        for column in ["fight_key", "chosen_value_expression", "effective_american_odds"]
        if column in expanded.columns
    ]
    return expanded.drop_duplicates(
        subset=dedupe_columns,
        keep="first",
    )


def _leg_expected_value(probability: float, american_odds: float) -> float:
    try:
        return (float(probability) * float(american_to_decimal(int(float(american_odds))))) - 1.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _premium_heavy_favorite_mask(
    working: pd.DataFrame,
    *,
    min_edge: float,
    min_confidence: float,
    min_data_quality: float,
) -> pd.Series:
    support_count = _numeric_series(working, ["support_count"], 0.0)
    quality_score = _numeric_series(working, ["bet_quality_score"], 0.0)
    age_edge = _selection_oriented_numeric_series(working, ["age_diff"], 0.0)
    reach_edge = _selection_oriented_numeric_series(working, ["reach_diff"], 0.0)
    grappling_edge = _selection_oriented_numeric_series(
        working,
        ["matchup_grappling_edge", "grappling_diff", "normalized_grappling_diff"],
        0.0,
    )
    gym_edge = _selection_oriented_numeric_series(working, ["gym_score_diff", "gym_depth_diff"], 0.0)
    control_edge = _selection_oriented_numeric_series(working, ["matchup_control_edge", "recent_control_diff"], 0.0)
    striking_edge = _selection_oriented_numeric_series(
        working,
        ["matchup_striking_edge", "strike_margin_diff", "normalized_strike_margin_diff"],
        0.0,
    )
    support_signals = working.get("support_signals", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    risk_flags = working.get("risk_flags", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()

    advantage_count = (
        age_edge.ge(2.5).astype(int)
        + reach_edge.ge(2.0).astype(int)
        + grappling_edge.ge(0.8).astype(int)
        + gym_edge.ge(0.05).astype(int)
        + control_edge.ge(0.75).astype(int)
        + striking_edge.ge(0.75).astype(int)
        + support_signals.str.contains("gym edge|gym depth|younger|reach edge|grappling edge|wrestling|control edge").astype(int)
    )

    blocker_mask = risk_flags.str.contains(
        "injury_concern|weight_cut_concern|late_replacement|travel_disadvantage|recent_ko_damage|recent_finish_damage|camp_change|recent_gym_switch"
    )

    return (
        working.get("recommended_tier", pd.Series("", index=working.index)).astype(str).eq("A")
        & working.get("recommended_action", pd.Series("", index=working.index)).astype(str).eq("Bettable now")
        & (working["effective_edge_numeric"] >= max(min_edge, 0.05))
        & (working["model_confidence_numeric"] >= max(min_confidence, 0.72))
        & (working["data_quality_numeric"] >= max(min_data_quality, 0.90))
        & (quality_score >= 82.0)
        & (support_count >= 4.0)
        & (advantage_count >= 3)
        & ~blocker_mask
    )


def build_parlay_board(
    report: pd.DataFrame,
    *,
    min_legs: int = DEFAULT_MIN_LEGS,
    max_legs: int = DEFAULT_MAX_LEGS,
    min_edge: float = DEFAULT_MIN_EDGE,
    min_model_prob: float = DEFAULT_MIN_MODEL_PROB,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    min_data_quality: float = DEFAULT_MIN_DATA_QUALITY,
    max_favorite_american: int = DEFAULT_MAX_FAVORITE_AMERICAN,
    extended_max_favorite_american: int = DEFAULT_EXTENDED_MAX_FAVORITE_AMERICAN,
    max_dog_american: int = DEFAULT_MAX_DOG_AMERICAN,
    max_market_blend_weight: float = DEFAULT_MAX_MARKET_BLEND_WEIGHT,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> pd.DataFrame:
    columns = [
        "event_name",
        "leg_count",
        "parlay_rank",
        "parlay_name",
        "legs",
        "market_mix",
        "american_odds",
        "decimal_odds",
        "model_prob",
        "implied_prob",
        "edge",
        "expected_value",
        "parlay_confidence",
        "average_leg_edge",
        "average_leg_confidence",
        "average_leg_score",
        "risk_summary",
        "stake_profile",
    ]
    if report.empty:
        return pd.DataFrame(columns=columns)

    working = report.copy()
    working["fight"] = working["fighter_a"].astype(str) + " vs " + working["fighter_b"].astype(str)
    working["fight_key"] = working.get("fight_key", working["fight"]).fillna(working["fight"]).astype(str)
    working = _expand_expression_candidates(working)
    working["effective_edge_numeric"] = _numeric_series(working, ["effective_edge", "chosen_expression_edge", "edge"], 0.0)
    working["effective_expected_value_numeric"] = _numeric_series(
        working,
        ["effective_expected_value", "chosen_expression_expected_value"],
        0.0,
    )
    working["model_confidence_numeric"] = _numeric_series(working, ["model_confidence"], 0.0)
    working["data_quality_numeric"] = _numeric_series(working, ["data_quality"], 0.0)
    working["american_odds_numeric"] = _numeric_series(
        working,
        ["effective_american_odds", "chosen_expression_odds", "american_odds"],
        float("nan"),
    )
    working["model_prob_numeric"] = _numeric_series(
        working,
        ["effective_projected_prob", "chosen_expression_prob", "model_projected_win_prob"],
        0.0,
    )
    working["implied_prob_numeric"] = _numeric_series(
        working,
        ["effective_implied_prob", "chosen_expression_implied_prob", "implied_prob"],
        0.0,
    )
    working["leg_expected_value_numeric"] = [
        _leg_expected_value(probability, american_odds)
        for probability, american_odds in zip(working["model_prob_numeric"], working["american_odds_numeric"])
    ]
    working["leg_quality_score"] = _numeric_series(working, ["bet_quality_score"], 0.0)
    working["leg_stake_numeric"] = _numeric_series(
        working,
        ["effective_suggested_stake", "chosen_expression_stake", "suggested_stake"],
        0.0,
    )
    working["market_family"] = working.apply(
        lambda row: _infer_expression_market(
            row.get("chosen_value_expression", row.get("selection_name", row.get("market", "")))
        )
        if _safe_text(row.get("expression_pick_source", "")) in {"alternative_market", "runner_up_expression"}
        else _safe_text(row.get("tracked_market_key", row.get("market", "moneyline")), "moneyline"),
        axis=1,
    )
    working["market_family"] = working["market_family"].map(lambda value: "prop" if _infer_expression_market(value) == "prop" else "moneyline")

    candidate_mask = (
        working.get("recommended_tier", pd.Series("", index=working.index)).astype(str).isin(["A", "B"])
        & working.get("recommended_action", pd.Series("", index=working.index)).astype(str).isin(["Bettable now", "Watchlist"])
        & (working["effective_edge_numeric"] >= min_edge)
        & (working["model_prob_numeric"] >= min_model_prob)
        & (working["model_confidence_numeric"] >= min_confidence)
        & (working["data_quality_numeric"] >= min_data_quality)
        & working["american_odds_numeric"].notna()
        & (working["model_prob_numeric"] > 0.0)
        & (working["model_prob_numeric"] < 1.0)
        & (working["implied_prob_numeric"] > 0.0)
        & (working["implied_prob_numeric"] < 1.0)
        & ((working["leg_stake_numeric"] > 0.0) | (working["leg_expected_value_numeric"] > 0.0))
    )
    candidate_mask &= ~working.get("fragility_bucket", pd.Series("", index=working.index)).astype(str).str.lower().eq("high")
    candidate_mask &= _numeric_series(working, ["selection_fallback_used"], 0.0).le(0.0)
    candidate_mask &= _numeric_series(working, ["market_blend_weight"], 0.0).le(max_market_blend_weight)
    premium_heavy_favorite_mask = _premium_heavy_favorite_mask(
        working,
        min_edge=min_edge,
        min_confidence=min_confidence,
        min_data_quality=min_data_quality,
    )
    standard_favorite_mask = working["american_odds_numeric"].ge(max_favorite_american)
    premium_favorite_mask = (
        working["american_odds_numeric"].lt(max_favorite_american)
        & working["american_odds_numeric"].ge(extended_max_favorite_american)
        & premium_heavy_favorite_mask
    )
    candidate_mask &= (standard_favorite_mask | premium_favorite_mask)
    candidate_mask &= working["american_odds_numeric"].le(max_dog_american)

    candidates = working.loc[candidate_mask].copy()
    if candidates.empty:
        return pd.DataFrame(columns=columns)

    selective_clv = (
        pd.to_numeric(candidates["selective_clv_prob"], errors="coerce").fillna(0.0)
        if "selective_clv_prob" in candidates.columns
        else pd.Series(0.0, index=candidates.index, dtype=float)
    )
    tier_bonus = (
        candidates["recommended_tier"].astype(str).map(_tier_bonus).fillna(0.0)
        if "recommended_tier" in candidates.columns
        else pd.Series(0.0, index=candidates.index, dtype=float)
    )
    fragility_penalty = (
        candidates["fragility_bucket"].astype(str).map(_fragility_penalty).fillna(0.0)
        if "fragility_bucket" in candidates.columns
        else pd.Series(0.0, index=candidates.index, dtype=float)
    )
    prop_bonus = candidates["market_family"].astype(str).eq("prop").astype(float) * 2.0
    candidates["candidate_score"] = (
        (candidates["effective_edge_numeric"] * 100 * 22)
        + (candidates["leg_expected_value_numeric"] * 100 * 10)
        + (candidates["model_prob_numeric"] * 100 * 0.45)
        + (candidates["model_confidence_numeric"] * 24)
        + (candidates["leg_quality_score"] * 0.35)
        + (selective_clv * 7)
        + tier_bonus
        + prop_bonus
        - fragility_penalty
    )
    candidates = (
        candidates.sort_values(
            by=["candidate_score", "effective_edge_numeric", "model_confidence_numeric"],
            ascending=[False, False, False],
        )
        .drop_duplicates(subset=["fight_key"], keep="first")
        .head(max_candidates)
        .reset_index(drop=True)
    )
    if len(candidates) < min_legs:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    event_name = _safe_text(candidates.iloc[0].get("event_name", ""))
    for leg_count in range(min_legs, max_legs + 1):
        if len(candidates) < leg_count:
            continue
        parlay_rows: list[dict[str, object]] = []
        for combo in itertools.combinations(candidates.to_dict("records"), leg_count):
            decimal_odds = 1.0
            model_prob = 1.0
            implied_prob = 1.0
            average_leg_edge = 0.0
            average_leg_confidence = 0.0
            average_leg_score = 0.0
            average_leg_model_prob = 0.0
            medium_risk_legs = 0
            premium_heavy_favorites = 0
            tier_counts = {"A": 0, "B": 0}
            market_counts = {"moneyline": 0, "prop": 0}
            leg_parts: list[str] = []
            for leg in combo:
                american_price = int(float(leg["american_odds_numeric"]))
                decimal_price = float(american_to_decimal(american_price))
                decimal_odds *= decimal_price
                model_prob *= float(leg["model_prob_numeric"])
                implied_prob *= float(leg["implied_prob_numeric"])
                average_leg_edge += float(leg["effective_edge_numeric"])
                average_leg_confidence += float(leg["model_confidence_numeric"])
                average_leg_score += float(leg["candidate_score"])
                average_leg_model_prob += float(leg["model_prob_numeric"])
                if str(leg.get("fragility_bucket", "")).strip().lower() == "medium":
                    medium_risk_legs += 1
                if american_price < max_favorite_american:
                    premium_heavy_favorites += 1
                tier = _safe_text(leg.get("recommended_tier", ""))
                if tier in tier_counts:
                    tier_counts[tier] += 1
                market_family = _safe_text(leg.get("market_family", "moneyline"), "moneyline")
                market_counts["prop" if market_family == "prop" else "moneyline"] += 1
                leg_parts.append(f"{_safe_text(leg.get('chosen_value_expression', leg.get('selection_name', '')))} ({american_price:+d})")
            average_leg_edge /= leg_count
            average_leg_confidence /= leg_count
            average_leg_score /= leg_count
            average_leg_model_prob /= leg_count
            combined_edge = model_prob - implied_prob
            expected_value = (model_prob * decimal_odds) - 1
            parlay_confidence = max(0.35, min(0.95, average_leg_confidence - (0.025 * max(0, leg_count - 3)) - (0.03 * medium_risk_legs)))
            parlay_score = (
                (combined_edge * 100 * 28)
                + (expected_value * 14)
                + (model_prob * 100 * 16)
                + (average_leg_model_prob * 100 * 0.8)
                + (parlay_confidence * 16)
                + (average_leg_score * 0.28)
                - (medium_risk_legs * 4)
                - (max(0, leg_count - 3) * 2)
            )
            try:
                american_odds = probability_to_american(1 / decimal_odds)
            except ValueError:
                american_odds = pd.NA
            stake_profile = "best add-value parlay"
            if tier_counts["A"] == leg_count:
                stake_profile = "all A-tier value legs"
            elif tier_counts["A"] >= max(1, leg_count - 1):
                stake_profile = "A-tier core with one support leg"
            risk_summary = []
            if medium_risk_legs:
                risk_summary.append(f"{medium_risk_legs} medium-risk leg{'s' if medium_risk_legs != 1 else ''}")
            if premium_heavy_favorites:
                risk_summary.append(f"{premium_heavy_favorites} premium heavy favorite{'s' if premium_heavy_favorites != 1 else ''}")
            else:
                risk_summary.append("no premium heavy favorites")
            if market_counts["moneyline"] and market_counts["prop"]:
                risk_summary.append("moneyline/prop mix")
            parlay_rows.append(
                {
                    "event_name": event_name,
                    "leg_count": leg_count,
                    "parlay_name": f"Top {leg_count}-Leg Value Parlay",
                    "legs": " | ".join(leg_parts),
                    "market_mix": _format_market_mix(market_counts),
                    "american_odds": american_odds if pd.notna(american_odds) else "",
                    "decimal_odds": round(decimal_odds, 2),
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied_prob, 4),
                    "edge": round(combined_edge, 4),
                    "expected_value": round(expected_value, 4),
                    "parlay_confidence": round(parlay_confidence, 4),
                    "average_leg_edge": round(average_leg_edge, 4),
                    "average_leg_confidence": round(average_leg_confidence, 4),
                    "average_leg_score": round(average_leg_score, 2),
                    "risk_summary": ", ".join(risk_summary),
                    "stake_profile": stake_profile,
                    "_score": round(parlay_score, 4),
                }
            )
        if not parlay_rows:
            continue
        ranked = pd.DataFrame(parlay_rows).sort_values(
            by=["_score", "model_prob", "edge", "expected_value", "average_leg_score"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
        ranked["parlay_rank"] = ranked.index + 1
        rows.extend(ranked.head(1).to_dict("records"))

    if not rows:
        return pd.DataFrame(columns=columns)
    parlays = pd.DataFrame(rows).sort_values(by=["leg_count", "parlay_rank"]).reset_index(drop=True)
    parlays["american_odds"] = parlays["american_odds"].apply(lambda value: f"{int(value):+d}" if str(value).strip() else "")
    parlays["decimal_odds"] = parlays["decimal_odds"].apply(lambda value: _format_decimal(float(value)))
    parlays["model_prob"] = parlays["model_prob"].apply(lambda value: _format_percent(float(value)))
    parlays["implied_prob"] = parlays["implied_prob"].apply(lambda value: _format_percent(float(value)))
    parlays["edge"] = parlays["edge"].apply(lambda value: _format_percent(float(value)))
    parlays["expected_value"] = parlays["expected_value"].apply(lambda value: _format_percent(float(value)))
    parlays["parlay_confidence"] = parlays["parlay_confidence"].apply(lambda value: f"{float(value):.2f}")
    parlays["average_leg_edge"] = parlays["average_leg_edge"].apply(lambda value: _format_percent(float(value)))
    parlays["average_leg_confidence"] = parlays["average_leg_confidence"].apply(lambda value: f"{float(value):.2f}")
    return parlays[columns]


def print_parlay_summary(parlays: pd.DataFrame) -> None:
    if parlays.empty:
        print(_highlight_title("Parlay board: no qualifying 2-5 leg combinations.", "yellow"))
        print()
        return
    print(_highlight_title(f"Parlay board: {len(parlays)} best-value combinations", "yellow"))
    print()
    for _, row in parlays.iterrows():
        print(
            f"- {_highlight_parlay_name(row['parlay_name'])} | {row['american_odds']} / {row['decimal_odds']} | "
            f"edge {_highlight_edge(row['edge'])} | EV {_highlight_edge(row['expected_value'])}"
        )
        print(
            f"  Model {_colorize(row['model_prob'], 'green')} | implied {_colorize(row['implied_prob'], 'gray')} | "
            f"confidence {_highlight_confidence(row['parlay_confidence'])} | {_highlight_stake_profile(row['stake_profile'])}"
        )
        print(f"  {_colorize('Legs:', 'cyan')} {row['legs']}")
        print(f"  {_colorize('Risks:', 'yellow')} {_highlight_risk_summary(row['risk_summary'])}")
    print()


def print_compact_parlay_summary(parlays: pd.DataFrame) -> None:
    print(format_compact_parlay_summary(parlays), end="")


def format_compact_parlay_summary(parlays: pd.DataFrame) -> str:
    lines: list[str] = []
    if parlays.empty:
        lines.append(_highlight_title("Parlay board: no qualifying 2-5 leg combinations.", "yellow"))
        lines.append("")
        return "\n".join(lines)
    lines.append(_highlight_title(f"Parlay board: {len(parlays)} top combinations", "yellow"))
    lines.append("")
    for _, row in parlays.iterrows():
        lines.append(
            f"{_highlight_leg_count(row['leg_count'])} | {row['american_odds']} | "
            f"edge {_highlight_edge(row['edge'])} | EV {_highlight_edge(row['expected_value'])}"
        )
        lines.append(f"  {_colorize('Legs:', 'cyan')} {row['legs']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = pd.read_csv(args.input) if Path(args.input).exists() else pd.DataFrame()
    parlays = build_parlay_board(
        report,
        min_legs=args.min_legs,
        max_legs=args.max_legs,
        min_edge=args.min_edge,
        min_model_prob=args.min_model_prob,
        min_confidence=args.min_confidence,
        min_data_quality=args.min_data_quality,
        max_favorite_american=args.max_favorite_american,
        extended_max_favorite_american=args.extended_max_favorite_american,
        max_dog_american=args.max_dog_american,
        max_market_blend_weight=args.max_market_blend_weight,
        max_candidates=args.max_candidates,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parlays.to_csv(output_path, index=False)
    if not args.quiet:
        print_parlay_summary(parlays)
        print(f"Saved parlay board to {output_path}")


if __name__ == "__main__":
    main()
