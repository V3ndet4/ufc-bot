from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from backtests.grading import attach_tracked_expression_columns
from data_sources.storage import load_tracked_picks


HISTORICAL_FULL_SAMPLE = 8
HISTORICAL_LIGHT_SAMPLE = 4

HISTORICAL_KEY_COLUMNS = [
    "tracked_market_bucket",
    "segment_label",
    "price_bucket",
    "edge_bucket",
    "confidence_bucket",
    "line_movement_bucket",
]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _confidence_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.75:
        return "0.75_plus"
    if numeric >= 0.65:
        return "0.65_to_0.74"
    if numeric >= 0.55:
        return "0.55_to_0.64"
    return "below_0.55"


def _edge_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.10:
        return "0.10_plus"
    if numeric >= 0.05:
        return "0.05_to_0.099"
    if numeric >= 0.03:
        return "0.03_to_0.049"
    return "below_0.03"


def _favorite_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric < 0:
        return "favorite"
    if numeric > 0:
        return "underdog"
    return "pickem"


def _line_movement_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.02:
        return "toward_pick"
    if numeric <= -0.02:
        return "against_pick"
    return "flat"


def enrich_feedback_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    if "tracked_market_key" not in enriched.columns:
        enriched = attach_tracked_expression_columns(enriched)

    enriched["tracked_market_bucket"] = (
        enriched.get("tracked_market_key", enriched.get("market", pd.Series("", index=enriched.index)))
        .fillna("unknown")
        .astype(str)
    )
    enriched["confidence_bucket"] = enriched.get(
        "model_confidence",
        pd.Series(pd.NA, index=enriched.index),
    ).apply(_confidence_bucket)
    enriched["edge_bucket"] = enriched.get(
        "chosen_expression_edge",
        enriched.get("edge", pd.Series(pd.NA, index=enriched.index)),
    ).apply(_edge_bucket)
    enriched["price_bucket"] = enriched.get(
        "chosen_expression_odds",
        enriched.get("american_odds", pd.Series(pd.NA, index=enriched.index)),
    ).apply(_favorite_bucket)
    enriched["line_movement_bucket"] = enriched.get(
        "line_movement_toward_fighter",
        pd.Series(pd.NA, index=enriched.index),
    ).apply(_line_movement_bucket)
    if "segment_label" not in enriched.columns:
        enriched["segment_label"] = "standard"
    enriched["segment_label"] = enriched["segment_label"].fillna("standard").astype(str)
    return enriched


def _historical_overlay_grade(
    sample_size: int,
    roi_pct: float,
    avg_clv_delta: float | pd.NA,
) -> tuple[str, str, float]:
    clv_missing = pd.isna(avg_clv_delta)
    full_weight = sample_size >= HISTORICAL_FULL_SAMPLE
    sample_label = f"{sample_size} comps"

    if sample_size < HISTORICAL_LIGHT_SAMPLE:
        return ("low_sample", f"{sample_label}, not enough graded comps", 0.0)

    if roi_pct >= 5.0 and (clv_missing or float(avg_clv_delta) >= 0.0):
        score = 6.0 if full_weight else 2.0
        return ("strong_positive" if full_weight else "light_positive", f"{sample_label}, positive ROI and CLV", score)

    if roi_pct >= 0.0 and (clv_missing or float(avg_clv_delta) >= 0.0):
        score = 2.0 if full_weight else 1.0
        return ("mild_positive" if full_weight else "light_positive", f"{sample_label}, stable positive history", score)

    if roi_pct < 0.0 and (not clv_missing and float(avg_clv_delta) < 0.0):
        score = -8.0 if full_weight else -3.0
        return ("strong_negative" if full_weight else "light_negative", f"{sample_label}, negative ROI and CLV", score)

    return ("mixed", f"{sample_label}, mixed history", 0.0)


def build_historical_overlay_lookup(tracked_frame: pd.DataFrame) -> dict[tuple[str, ...], dict[str, Any]]:
    if tracked_frame.empty:
        return {}

    working = enrich_feedback_buckets(tracked_frame)
    if "grade_status" not in working.columns:
        working["grade_status"] = "pending"
    if "closing_american_odds" not in working.columns:
        working["closing_american_odds"] = pd.NA
    working["grade_status"] = working["grade_status"].fillna("pending").astype(str)
    working["closing_american_odds"] = _coerce_numeric(working["closing_american_odds"])
    working = working.loc[
        (working["grade_status"].str.lower() != "pending")
        & working["closing_american_odds"].notna()
    ].copy()
    if working.empty:
        return {}

    working["stake_at_pick"] = _coerce_numeric(
        working.get("chosen_expression_stake", working.get("suggested_stake", pd.Series(0.0, index=working.index)))
    ).fillna(0.0)
    working["profit"] = _coerce_numeric(working.get("profit", pd.Series(0.0, index=working.index))).fillna(0.0)
    working["clv_delta"] = _coerce_numeric(working.get("clv_delta", pd.Series(pd.NA, index=working.index)))
    working["actual_result"] = working.get("actual_result", pd.Series("", index=working.index)).fillna("").astype(str)

    overlays: dict[tuple[str, ...], dict[str, Any]] = {}
    for key, group in working.groupby(HISTORICAL_KEY_COLUMNS, dropna=False):
        sample_size = int(len(group))
        wins = int((group["actual_result"] == "win").sum())
        win_rate = wins / sample_size if sample_size > 0 else 0.0
        total_stake = float(group["stake_at_pick"].sum())
        total_profit = float(group["profit"].sum())
        roi_pct = round((total_profit / total_stake) * 100, 2) if total_stake > 0 else 0.0
        avg_clv_delta = group["clv_delta"].dropna().mean()
        grade, reason, score_adjustment = _historical_overlay_grade(sample_size, roi_pct, avg_clv_delta)
        overlays[tuple(str(value) for value in key)] = {
            "historical_sample_size": sample_size,
            "historical_roi_pct": roi_pct,
            "historical_avg_clv_delta": round(0.0 if pd.isna(avg_clv_delta) else float(avg_clv_delta), 2),
            "historical_win_rate": round(win_rate, 4),
            "historical_overlay_grade": grade,
            "historical_overlay_reason": reason,
            "historical_overlay_score_adjustment": score_adjustment,
        }
    return overlays


def apply_historical_overlays(
    frame: pd.DataFrame,
    *,
    db_path: str | Path | None = None,
    tracked_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    enriched = enrich_feedback_buckets(frame)
    history = tracked_frame
    if history is None and db_path:
        db_file = Path(db_path)
        if db_file.exists():
            history = load_tracked_picks(db_file)
    if history is None or history.empty:
        for column, default_value in {
            "historical_sample_size": 0,
            "historical_roi_pct": pd.NA,
            "historical_avg_clv_delta": pd.NA,
            "historical_win_rate": pd.NA,
            "historical_overlay_grade": "low_sample",
            "historical_overlay_reason": "No graded history yet",
            "historical_overlay_score_adjustment": 0.0,
        }.items():
            enriched[column] = default_value
        return enriched

    lookup = build_historical_overlay_lookup(history)
    defaults = {
        "historical_sample_size": 0,
        "historical_roi_pct": pd.NA,
        "historical_avg_clv_delta": pd.NA,
        "historical_win_rate": pd.NA,
        "historical_overlay_grade": "low_sample",
        "historical_overlay_reason": "No matching graded comps",
        "historical_overlay_score_adjustment": 0.0,
    }
    overlay_rows: list[dict[str, Any]] = []
    for row in enriched.to_dict("records"):
        key = tuple(str(row.get(column, "unknown")) for column in HISTORICAL_KEY_COLUMNS)
        overlay_rows.append({**defaults, **lookup.get(key, {})})
    overlay_frame = pd.DataFrame(overlay_rows, index=enriched.index)
    return pd.concat([enriched, overlay_frame], axis=1)


def calculate_fragility_metrics(
    *,
    short_notice_flag: float = 0.0,
    short_notice_acceptance_flag: float = 0.0,
    short_notice_success_flag: float = 0.0,
    days_since_last_fight: float = 999.0,
    ufc_fight_count: float = 0.0,
    ufc_debut_flag: float = 0.0,
    injury_concern_flag: float = 0.0,
    weight_cut_concern_flag: float = 0.0,
    replacement_fighter_flag: float = 0.0,
    travel_disadvantage_flag: float = 0.0,
    camp_change_flag: float = 0.0,
    gym_changed_flag: float = 0.0,
    fallback_used: float = 0.0,
    data_quality: float = 1.0,
    market_blend_weight: float = 0.0,
    consensus_count: float = 0.0,
    consensus_price_edge: float = 0.0,
) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []

    short_notice_capability = max(float(short_notice_acceptance_flag or 0.0), float(short_notice_success_flag or 0.0))
    if float(short_notice_flag or 0.0) >= 1.0 and short_notice_capability < 1.0:
        score += 1
        reasons.append("unproven short notice")
    if float(injury_concern_flag or 0.0) >= 1.0:
        score += 2
        reasons.append("injury concern")
    if float(weight_cut_concern_flag or 0.0) >= 1.0:
        score += 2
        reasons.append("weight cut")
    if float(replacement_fighter_flag or 0.0) >= 1.0:
        score += 2
        reasons.append("late replacement")
    if float(days_since_last_fight or 999.0) > 420:
        score += 1
        reasons.append("long layoff")
    if float(days_since_last_fight or 999.0) < 45 and float(ufc_fight_count or 0.0) > 0.0:
        score += 1
        reasons.append("quick turnaround")
    if float(camp_change_flag or 0.0) >= 1.0 or float(gym_changed_flag or 0.0) >= 1.0:
        score += 1
        reasons.append("camp change")
    if float(fallback_used or 0.0) >= 1.0 or float(data_quality or 0.0) < 0.85:
        score += 1
        reasons.append("data uncertainty")
    if float(market_blend_weight or 0.0) >= 0.40 or (
        float(consensus_count or 0.0) >= 3.0 and float(consensus_price_edge or 0.0) <= -0.015
    ):
        score += 1
        reasons.append("market disagreement")
    if float(ufc_debut_flag or 0.0) >= 1.0 or float(ufc_fight_count or 0.0) < 3.0:
        score += 1
        reasons.append("thin UFC sample")
    if float(travel_disadvantage_flag or 0.0) >= 1.0:
        score += 1
        reasons.append("travel disadvantage")

    bucket = "high" if score >= 4 else "medium" if score >= 2 else "low"
    return {
        "fragility_score": int(score),
        "fragility_bucket": bucket,
        "fragility_reasons": ", ".join(dict.fromkeys(reasons)),
    }
