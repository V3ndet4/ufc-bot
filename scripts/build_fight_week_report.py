from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import (
    DEFAULT_BOOKMAKER,
    OddsApiError,
    extract_alternative_market_keys,
    fetch_the_odds_api_event_markets,
    fetch_the_odds_api_event_odds,
    load_api_key,
    load_odds_csv,
)
from data_sources.bestfightodds import fetch_html as fetch_bestfightodds_html
from features.fighter_features import build_fight_features, load_fighter_stats
from models.confidence import default_confidence_model_path, load_confidence_model
from models.decision_support import apply_historical_overlays, calculate_fragility_metrics
from models.ev import implied_probability
from models.projection import project_fight_probabilities
from models.side import default_side_model_path, load_side_model
from normalization.odds import normalize_odds_frame
from scripts.event_manifest import MODEL_CONTEXT_FLAG_COLUMNS, OPERATOR_CONTEXT_FLAG_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fight-level market vs model comparison report for an upcoming UFC card."
    )
    parser.add_argument("--odds", required=True, help="Path to the odds CSV input.")
    parser.add_argument("--fighter-stats", required=True, help="Path to the fighter stats CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path for the fight-week report.")
    parser.add_argument(
        "--skipped-output",
        help="Optional output CSV path for fights skipped from the main report, with reasons.",
    )
    parser.add_argument(
        "--bestfightodds-event-url",
        action="append",
        default=[],
        help="Optional BestFightOdds event URL used to enrich alternative markets like goes to decision.",
    )
    parser.add_argument(
        "--odds-api-bookmaker",
        default=DEFAULT_BOOKMAKER,
        help="Bookmaker key used for The Odds API alternative-market fallback.",
    )
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "ufc_betting.db"),
        help="SQLite database path used for historical decision overlays.",
    )
    parser.add_argument(
        "--side-model",
        help="Optional pickle path for the calibrated side model. Defaults to models/side_model.pkl when present.",
    )
    parser.add_argument(
        "--confidence-model",
        help="Optional pickle path for the calibrated confidence model. Defaults to models/confidence_model.pkl when present.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _fighter_flag_summary(row: pd.Series, fighter_prefix: str, flag_columns: list[str]) -> str:
    active_flags = [
        flag_name
        for flag_name in flag_columns
        if float(row.get(f"{fighter_prefix}_{flag_name}", 0.0) or 0.0) > 0
    ]
    return ", ".join(active_flags)


def _pick_side(row: pd.Series) -> tuple[str, str, float, float]:
    if float(row["fighter_a_edge_vs_current"]) >= float(row["fighter_b_edge_vs_current"]):
        return (
            row["fighter_a"],
            "fighter_a",
            float(row["projected_fighter_a_win_prob"]),
            float(row["fighter_a_edge_vs_current"]),
        )
    return (
        row["fighter_b"],
        "fighter_b",
        float(row["fighter_b_model_win_prob"]),
        float(row["fighter_b_edge_vs_current"]),
    )


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _tier_rank(tier: object) -> int:
    mapping = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    return mapping.get(str(tier or "").strip().upper(), 0)


def _row_value(row: pd.Series | object, *names: str, default: object = pd.NA) -> object:
    for name in names:
        if isinstance(row, Mapping) and name in row:
            value = row.get(name, default)
            if not pd.isna(value):
                return value
        elif hasattr(row, name):
            value = getattr(row, name)
            if not pd.isna(value):
                return value
        elif isinstance(row, pd.Series) and name in row.index:
            value = row.get(name, default)
            if not pd.isna(value):
                return value
    return default


def _oriented_metric(row: pd.Series | object, column: str, pick_side: str, default: float = 0.0) -> float:
    alias_map = {
        "fighter_a_reach_advantage_in": ("fighter_a_reach_advantage_in", "reach_diff"),
        "fighter_a_height_advantage_in": ("fighter_a_height_advantage_in", "height_diff"),
        "fighter_a_fallback_used": ("fighter_a_fallback_used", "a_fallback_used"),
        "fighter_b_fallback_used": ("fighter_b_fallback_used", "b_fallback_used"),
        "fighter_a_short_notice_flag": ("fighter_a_short_notice_flag", "a_short_notice_flag"),
        "fighter_b_short_notice_flag": ("fighter_b_short_notice_flag", "b_short_notice_flag"),
        "fighter_a_cardio_fade_flag": ("fighter_a_cardio_fade_flag", "a_cardio_fade_flag"),
        "fighter_b_cardio_fade_flag": ("fighter_b_cardio_fade_flag", "b_cardio_fade_flag"),
    }
    value = _safe_float(_row_value(row, *(alias_map.get(column, (column,))), default=default), default)
    return value if pick_side == "fighter_a" else -value


def _best_pick_context(row: pd.Series | object) -> tuple[str, str, float, float]:
    fighter_a_edge = _safe_float(_row_value(row, "fighter_a_edge_vs_current", default=0.0))
    fighter_b_edge = _safe_float(_row_value(row, "fighter_b_edge_vs_current", default=0.0))
    if fighter_a_edge >= fighter_b_edge:
        fighter_a_name = _row_value(row, "fighter_a", default="")
        fighter_a_prob = _safe_float(_row_value(row, "fighter_a_model_win_prob", "projected_fighter_a_win_prob", default=0.0))
        return str(fighter_a_name), "fighter_a", fighter_a_prob, fighter_a_edge
    fighter_b_name = _row_value(row, "fighter_b", default="")
    fighter_b_prob = _safe_float(_row_value(row, "fighter_b_model_win_prob", default=0.0))
    return str(fighter_b_name), "fighter_b", fighter_b_prob, fighter_b_edge


def _selected_value(row: pd.Series | object, pick_side: str, fighter_a_name: str, fighter_b_name: str) -> dict[str, object]:
    selected_name = fighter_a_name if pick_side == "fighter_a" else fighter_b_name
    selection_recent_grappling_rate = _safe_float(
        _row_value(
            row,
            "fighter_a_recent_grappling_rate" if pick_side == "fighter_a" else "fighter_b_recent_grappling_rate",
            "a_recent_grappling_rate" if pick_side == "fighter_a" else "b_recent_grappling_rate",
            default=0.0,
        ),
        0.0,
    )
    selection_control_avg = _safe_float(
        _row_value(
            row,
            "fighter_a_control_avg" if pick_side == "fighter_a" else "fighter_b_control_avg",
            "a_control_avg" if pick_side == "fighter_a" else "b_control_avg",
            default=0.0,
        ),
        0.0,
    )
    selection_recent_control_avg = _safe_float(
        _row_value(
            row,
            "fighter_a_recent_control_avg" if pick_side == "fighter_a" else "fighter_b_recent_control_avg",
            "a_recent_control_avg" if pick_side == "fighter_a" else "b_recent_control_avg",
            default=0.0,
        ),
        0.0,
    )
    selection_submission_avg = _safe_float(
        _row_value(
            row,
            "fighter_a_submission_avg" if pick_side == "fighter_a" else "fighter_b_submission_avg",
            "a_submission_avg" if pick_side == "fighter_a" else "b_submission_avg",
            default=0.0,
        ),
        0.0,
    )
    return {
        "selection": pick_side,
        "selection_name": selected_name,
        "selection_gym_name": str(
            _row_value(
                row,
                "fighter_a_gym_name" if pick_side == "fighter_a" else "fighter_b_gym_name",
                "a_gym_name" if pick_side == "fighter_a" else "b_gym_name",
                default="",
            )
            or ""
        ).strip(),
        "selection_gym_tier": str(
            _row_value(
                row,
                "fighter_a_gym_tier" if pick_side == "fighter_a" else "fighter_b_gym_tier",
                "a_gym_tier" if pick_side == "fighter_a" else "b_gym_tier",
                default="",
            )
            or ""
        ).strip().upper(),
        "selection_gym_record": str(
            _row_value(
                row,
                "fighter_a_gym_record" if pick_side == "fighter_a" else "fighter_b_gym_record",
                "a_gym_record" if pick_side == "fighter_a" else "b_gym_record",
                default="",
            )
            or ""
        ).strip(),
        "selection_gym_score": _safe_float(
            _row_value(
                row,
                "fighter_a_gym_score" if pick_side == "fighter_a" else "fighter_b_gym_score",
                "a_gym_score" if pick_side == "fighter_a" else "b_gym_score",
                default=0.0,
            ),
            0.0,
        ),
        "selection_recent_grappling_rate": selection_recent_grappling_rate,
        "selection_control_avg": selection_control_avg,
        "selection_recent_control_avg": selection_recent_control_avg,
        "selection_grappling_pressure_score": round(
            selection_recent_grappling_rate + (selection_recent_control_avg * 0.55) + (selection_submission_avg * 0.35),
            3,
        ),
        "selection_previous_gym_name": str(
            _row_value(
                row,
                "fighter_a_previous_gym_name" if pick_side == "fighter_a" else "fighter_b_previous_gym_name",
                "a_previous_gym_name" if pick_side == "fighter_a" else "b_previous_gym_name",
                default="",
            )
            or ""
        ).strip(),
        "selection_fallback_used": _safe_float(
            _row_value(
                row,
                "fighter_a_fallback_used" if pick_side == "fighter_a" else "fighter_b_fallback_used",
                "a_fallback_used" if pick_side == "fighter_a" else "b_fallback_used",
                default=0.0,
            ),
            0.0,
        ),
        "selection_short_notice_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_short_notice_flag" if pick_side == "fighter_a" else "fighter_b_short_notice_flag",
                "a_short_notice_flag" if pick_side == "fighter_a" else "b_short_notice_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_short_notice_acceptance_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_short_notice_acceptance_flag" if pick_side == "fighter_a" else "fighter_b_short_notice_acceptance_flag",
                "a_short_notice_acceptance_flag" if pick_side == "fighter_a" else "b_short_notice_acceptance_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_short_notice_success_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_short_notice_success_flag" if pick_side == "fighter_a" else "fighter_b_short_notice_success_flag",
                "a_short_notice_success_flag" if pick_side == "fighter_a" else "b_short_notice_success_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_injury_concern_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_injury_concern_flag" if pick_side == "fighter_a" else "fighter_b_injury_concern_flag",
                "a_injury_concern_flag" if pick_side == "fighter_a" else "b_injury_concern_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_weight_cut_concern_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_weight_cut_concern_flag" if pick_side == "fighter_a" else "fighter_b_weight_cut_concern_flag",
                "a_weight_cut_concern_flag" if pick_side == "fighter_a" else "b_weight_cut_concern_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_replacement_fighter_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_replacement_fighter_flag" if pick_side == "fighter_a" else "fighter_b_replacement_fighter_flag",
                "a_replacement_fighter_flag" if pick_side == "fighter_a" else "b_replacement_fighter_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_travel_disadvantage_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_travel_disadvantage_flag" if pick_side == "fighter_a" else "fighter_b_travel_disadvantage_flag",
                "a_travel_disadvantage_flag" if pick_side == "fighter_a" else "b_travel_disadvantage_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_camp_change_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_camp_change_flag" if pick_side == "fighter_a" else "fighter_b_camp_change_flag",
                "a_camp_change_flag" if pick_side == "fighter_a" else "b_camp_change_flag",
                default=0.0,
            ),
            0.0,
        ),
        "selection_gym_changed_flag": _safe_float(
            _row_value(
                row,
                "fighter_a_gym_changed_flag" if pick_side == "fighter_a" else "fighter_b_gym_changed_flag",
                "a_gym_changed_flag" if pick_side == "fighter_a" else "b_gym_changed_flag",
                default=0.0,
            ),
            0.0,
        ),
    }


def _moneyline_expression_context(row: pd.Series | object) -> dict[str, object]:
    pick_name, pick_side, _, _ = _best_pick_context(row)
    return {
        "side_expression": pick_name,
        "pick_side": pick_side,
        "side_market_american_odds": _row_value(row, "side_market_american_odds", default=pd.NA),
        "side_market_projected_prob": _row_value(row, "side_market_projected_prob", default=pd.NA),
        "side_market_implied_prob": _row_value(row, "side_market_implied_prob", default=pd.NA),
        "side_market_edge": _row_value(row, "side_market_edge", default=pd.NA),
    }


def _selected_line_movement(row: pd.Series | object, pick_side: str) -> float:
    open_probability = _safe_float(
        _row_value(
            row,
            "fighter_a_open_implied_prob" if pick_side == "fighter_a" else "fighter_b_open_implied_prob",
            default=pd.NA,
        ),
        default=float("nan"),
    )
    current_probability = _safe_float(
        _row_value(
            row,
            "fighter_a_current_implied_prob" if pick_side == "fighter_a" else "fighter_b_current_implied_prob",
            default=pd.NA,
        ),
        default=float("nan"),
    )
    if pd.isna(open_probability) or pd.isna(current_probability):
        return 0.0
    return float(current_probability - open_probability)


def _ansi_enabled() -> bool:
    stream = getattr(sys, "stdout", None)
    if stream is None or not hasattr(stream, "isatty"):
        return False
    if os.getenv("NO_COLOR"):
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _colorize(text: str, color: str) -> str:
    if not _ansi_enabled():
        return text
    palette = {
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "red": "\x1b[31m",
        "cyan": "\x1b[36m",
        "gray": "\x1b[90m",
    }
    prefix = palette.get(color, "")
    suffix = "\x1b[0m" if prefix else ""
    return f"{prefix}{text}{suffix}"


def _gym_tier_color(tier: object) -> str:
    return {
        "S": "cyan",
        "A": "green",
        "B": "yellow",
        "C": "gray",
        "D": "red",
    }.get(str(tier or "").strip().upper(), "gray")


def _format_gym_tier_label(tier: object) -> str:
    normalized = str(tier or "").strip().upper()
    if not normalized:
        return ""
    return _colorize(f"{normalized}-tier", _gym_tier_color(normalized))


def _signal_tier(value: float, thresholds: tuple[float, float, float, float]) -> str:
    for tier, cutoff in zip(("S", "A", "B", "C"), thresholds):
        if value >= cutoff:
            return tier
    return ""


def _tiered_signal_label(label: str, value: float, thresholds: tuple[float, float, float, float]) -> str:
    tier = _signal_tier(value, thresholds)
    if not tier:
        return ""
    return f"{_format_gym_tier_label(tier)} {label}"


def _history_summary(row: pd.Series | object) -> str:
    sample_size = int(_safe_float(_row_value(row, "historical_sample_size", default=0.0), 0.0))
    grade = _safe_text(_row_value(row, "historical_overlay_grade", default="low_sample"), "low_sample")
    if sample_size <= 0:
        return _colorize("low sample | no graded history", "gray")
    label = (
        "HIST+"
        if "positive" in grade
        else "HIST-"
        if "negative" in grade
        else "HIST?"
        if grade == "mixed"
        else "HIST~"
    )
    color = "green" if "positive" in grade else "red" if "negative" in grade else "yellow" if grade == "mixed" else "gray"
    roi_pct = _row_value(row, "historical_roi_pct", default=pd.NA)
    avg_clv_delta = _row_value(row, "historical_avg_clv_delta", default=pd.NA)
    win_rate = _row_value(row, "historical_win_rate", default=pd.NA)
    parts = [
        _colorize(f"[{label}]", color),
        f"{sample_size} comps",
        f"ROI {_format_percent((float(roi_pct) / 100) if pd.notna(roi_pct) else pd.NA)}",
        f"CLV {float(avg_clv_delta):+.2f}" if pd.notna(avg_clv_delta) else "CLV n/a",
        f"WR {_format_percent(win_rate)}" if pd.notna(win_rate) else "WR n/a",
    ]
    return " | ".join(parts)


def _fragility_summary(row: pd.Series | object) -> str:
    bucket = _safe_text(_row_value(row, "fragility_bucket", default="low"), "low").lower()
    score = int(_safe_float(_row_value(row, "fragility_score", default=0.0), 0.0))
    reasons = _safe_text(_row_value(row, "fragility_reasons", default=""))
    label = f"[RISK:{bucket.upper()}]"
    color = "red" if bucket == "high" else "yellow" if bucket == "medium" else "green"
    summary = f"{_colorize(label, color)} {score}"
    if reasons:
        summary += f" | {reasons}"
    return summary


def _decision_summary(row: pd.Series | object) -> str:
    fallback_expression = _safe_text(
        _row_value(row, "preferred_market_expression", "selection_name", default=_best_pick_context(row)[0]),
        _best_pick_context(row)[0],
    )
    chosen_expression = _safe_text(_row_value(row, "chosen_value_expression", default=fallback_expression), fallback_expression)
    runner_up_expression = _safe_text(_row_value(row, "runner_up_expression", default=""))
    source = _safe_text(_row_value(row, "expression_pick_source", default="side_market"), "side_market")
    gap = _row_value(row, "expression_edge_gap", default=pd.NA)
    prefix = _colorize("[PROP]", "cyan") if source == "alternative_market" else _colorize("[SIDE]", "green")
    parts = [prefix, chosen_expression]
    if runner_up_expression:
        gap_text = f" | gap {_format_percent(gap)}" if pd.notna(gap) else ""
        parts.append(f"runner-up {runner_up_expression}{gap_text}")
    return " | ".join(parts)

def _driver_labels(row: pd.Series | object, pick_side: str) -> list[str]:
    labels: list[str] = []
    reach_advantage = _oriented_metric(row, "fighter_a_reach_advantage_in", pick_side)
    height_advantage = _oriented_metric(row, "fighter_a_height_advantage_in", pick_side)
    strike_margin = _oriented_metric(row, "strike_margin_diff", pick_side)
    matchup_striking_edge = _oriented_metric(row, "matchup_striking_edge", pick_side)
    grappling_edge = _oriented_metric(row, "grappling_diff", pick_side)
    matchup_grappling_edge = _oriented_metric(row, "matchup_grappling_edge", pick_side)
    control_edge = _oriented_metric(row, "control_diff", pick_side)
    recent_control_edge = _oriented_metric(row, "recent_control_diff", pick_side)
    matchup_control_edge = _oriented_metric(row, "matchup_control_edge", pick_side)
    grappling_pressure_edge = _oriented_metric(row, "grappling_pressure_diff", pick_side)
    schedule_strength_edge = _oriented_metric(row, "schedule_strength_diff", pick_side)
    normalized_striking_edge = _oriented_metric(row, "normalized_strike_margin_diff", pick_side)
    normalized_grappling_edge = _oriented_metric(row, "normalized_grappling_diff", pick_side)
    normalized_control_edge = _oriented_metric(row, "normalized_control_diff", pick_side)
    first_round_finish_edge = _oriented_metric(row, "first_round_finish_rate_diff", pick_side)
    durability_edge = _oriented_metric(row, "durability_diff", pick_side)
    decision_edge = _oriented_metric(row, "decision_rate_diff", pick_side)
    market_blend_weight = _safe_float(getattr(row, "market_blend_weight") if hasattr(row, "market_blend_weight") else row.get("market_blend_weight"))
    finish_prob = _safe_float(getattr(row, "projected_finish_prob") if hasattr(row, "projected_finish_prob") else row.get("projected_finish_prob"), 0.5)
    decision_prob = _safe_float(getattr(row, "projected_decision_prob") if hasattr(row, "projected_decision_prob") else row.get("projected_decision_prob"), 0.5)
    gym_score_edge = _oriented_metric(row, "gym_score_diff", pick_side)
    selected_tier = str(
        _row_value(
            row,
            "fighter_a_gym_tier" if pick_side == "fighter_a" else "fighter_b_gym_tier",
            "a_gym_tier" if pick_side == "fighter_a" else "b_gym_tier",
            default="",
        )
        or ""
    ).strip().upper()
    tier_label = _format_gym_tier_label(selected_tier)
    tier_prefix = f"{tier_label} " if tier_label else ""
    grappling_label = _tiered_signal_label("grappling edge", grappling_edge, (1.8, 1.45, 1.0, 0.6))
    recent_control_label = _tiered_signal_label("control edge", recent_control_edge, (5.0, 3.5, 2.25, 1.0))
    positional_control_label = _tiered_signal_label("positional control", control_edge, (4.0, 2.8, 1.8, 1.2))
    grappling_pressure_label = _tiered_signal_label("grappling pressure", grappling_pressure_edge, (4.5, 3.0, 1.75, 0.8))
    matchup_striking_label = _tiered_signal_label("matchup striking", matchup_striking_edge, (2.4, 1.8, 1.2, 0.7))
    matchup_grappling_label = _tiered_signal_label("matchup wrestling", matchup_grappling_edge, (2.0, 1.5, 1.0, 0.55))
    matchup_control_label = _tiered_signal_label("matchup control", matchup_control_edge, (2.2, 1.6, 1.0, 0.5))
    schedule_strength_label = _tiered_signal_label("schedule strength", schedule_strength_edge, (0.10, 0.07, 0.04, 0.02))
    normalized_striking_label = _tiered_signal_label(
        "opponent-adjusted striking",
        normalized_striking_edge,
        (2.2, 1.5, 0.9, 0.45),
    )
    normalized_grappling_label = _tiered_signal_label(
        "opponent-adjusted wrestling",
        normalized_grappling_edge,
        (1.6, 1.1, 0.7, 0.35),
    )
    normalized_control_label = _tiered_signal_label(
        "opponent-adjusted control",
        normalized_control_edge,
        (3.0, 2.1, 1.2, 0.6),
    )

    if strike_margin >= 1.0:
        labels.append(f"striking edge ({strike_margin:+.2f}/min)")
    if matchup_striking_label:
        labels.append(matchup_striking_label)
    if grappling_label:
        labels.append(grappling_label)
    if matchup_grappling_label:
        labels.append(matchup_grappling_label)
    if recent_control_label:
        labels.append(recent_control_label)
    elif positional_control_label:
        labels.append(positional_control_label)
    if matchup_control_label:
        labels.append(matchup_control_label)
    if grappling_pressure_label:
        labels.append(grappling_pressure_label)
    if schedule_strength_label:
        labels.append(schedule_strength_label)
    if normalized_striking_label:
        labels.append(normalized_striking_label)
    if normalized_grappling_label:
        labels.append(normalized_grappling_label)
    if normalized_control_label:
        labels.append(normalized_control_label)
    if reach_advantage >= 2.0:
        labels.append(f"reach edge (+{reach_advantage:.0f}\")")
    if height_advantage >= 2.0:
        labels.append(f"height edge (+{height_advantage:.0f}\")")
    if first_round_finish_edge >= 0.15:
        labels.append(f"early finish threat ({first_round_finish_edge:+.1%})")
    if durability_edge >= 0.12:
        labels.append(f"durability edge ({durability_edge:+.1%})")
    if decision_edge >= 0.10 and decision_prob >= 0.54:
        labels.append(f"decision equity ({decision_prob:.1%})")
    if gym_score_edge >= 0.10:
        labels.append(f"{tier_prefix}camp edge")
    elif gym_score_edge >= 0.05:
        labels.append(f"{tier_prefix}camp quality")
    if finish_prob >= 0.60:
        labels.append(f"high finish environment ({finish_prob:.1%})")
    if market_blend_weight <= 0.20:
        labels.append("model vs market aligned")
    return labels


def _risk_labels(row: pd.Series | object, pick_side: str) -> list[str]:
    labels: list[str] = []
    market_blend_weight = _safe_float(_row_value(row, "market_blend_weight", default=0.0))
    model_confidence = _safe_float(_row_value(row, "model_confidence", default=0.5), 0.5)
    data_quality = _safe_float(_row_value(row, "data_quality", default=1.0), 1.0)
    current_best_edge = _safe_float(_row_value(row, "fighter_a_edge_vs_current", default=0.0))
    current_best_edge = max(
        current_best_edge,
        _safe_float(_row_value(row, "fighter_b_edge_vs_current", default=0.0)),
    )
    fallback_used = _safe_float(
        _row_value(
            row,
            "fighter_a_fallback_used" if pick_side == "fighter_a" else "fighter_b_fallback_used",
            "a_fallback_used" if pick_side == "fighter_a" else "b_fallback_used",
            default=0.0,
        ),
        0.0,
    )
    short_notice = _safe_float(
        _row_value(
            row,
            "fighter_a_short_notice_flag" if pick_side == "fighter_a" else "fighter_b_short_notice_flag",
            "a_short_notice_flag" if pick_side == "fighter_a" else "b_short_notice_flag",
            default=0.0,
        ),
        0.0,
    )
    cardio_flag = _safe_float(
        _row_value(
            row,
            "fighter_a_cardio_fade_flag" if pick_side == "fighter_a" else "fighter_b_cardio_fade_flag",
            "a_cardio_fade_flag" if pick_side == "fighter_a" else "b_cardio_fade_flag",
            default=0.0,
        ),
        0.0,
    )
    camp_change_flag = _safe_float(
        _row_value(
            row,
            "fighter_a_camp_change_flag" if pick_side == "fighter_a" else "fighter_b_camp_change_flag",
            "a_camp_change_flag" if pick_side == "fighter_a" else "b_camp_change_flag",
            default=0.0,
        ),
        0.0,
    )
    gym_changed_flag = _safe_float(
        _row_value(
            row,
            "fighter_a_gym_changed_flag" if pick_side == "fighter_a" else "fighter_b_gym_changed_flag",
            "a_gym_changed_flag" if pick_side == "fighter_a" else "b_gym_changed_flag",
            default=0.0,
        ),
        0.0,
    )
    gym_name = str(
        _row_value(
            row,
            "fighter_a_gym_name" if pick_side == "fighter_a" else "fighter_b_gym_name",
            "a_gym_name" if pick_side == "fighter_a" else "b_gym_name",
            default="",
        )
        or ""
    ).strip()
    previous_gym_name = str(
        _row_value(
            row,
            "fighter_a_previous_gym_name" if pick_side == "fighter_a" else "fighter_b_previous_gym_name",
            "a_previous_gym_name" if pick_side == "fighter_a" else "b_previous_gym_name",
            default="",
        )
        or ""
    ).strip()

    if current_best_edge < 0.03:
        labels.append("thin or negative edge")
    if market_blend_weight >= 0.40:
        labels.append("market disagreement")
    if model_confidence < 0.70 or data_quality < 0.85:
        labels.append("weaker model confidence")
    if fallback_used >= 1:
        labels.append("fallback stats used")
    if short_notice >= 1:
        labels.append("short-notice context")
    if cardio_flag >= 1:
        labels.append("cardio concern")
    if gym_changed_flag >= 1 and previous_gym_name:
        switch_target = gym_name if gym_name else "current camp"
        labels.append(f"recent gym switch ({previous_gym_name} -> {switch_target})")
    elif camp_change_flag >= 1:
        labels.append("camp change flag")
    return labels


def _fighter_camp_note(row: pd.Series | object, fighter_prefix: str) -> str:
    short_prefix = "a" if fighter_prefix == "fighter_a" else "b" if fighter_prefix == "fighter_b" else fighter_prefix
    fighter_name = str(_row_value(row, fighter_prefix, default="") or "").strip()
    gym_name = str(
        _row_value(
            row,
            f"{fighter_prefix}_gym_name",
            f"{short_prefix}_gym_name",
            default="",
        )
        or ""
    ).strip()
    gym_tier = str(
        _row_value(
            row,
            f"{fighter_prefix}_gym_tier",
            f"{short_prefix}_gym_tier",
            default="",
        )
        or ""
    ).strip().upper()
    gym_record = str(
        _row_value(
            row,
            f"{fighter_prefix}_gym_record",
            f"{short_prefix}_gym_record",
            default="",
        )
        or ""
    ).strip()
    gym_changed_flag = _safe_float(
        _row_value(
            row,
            f"{fighter_prefix}_gym_changed_flag",
            f"{short_prefix}_gym_changed_flag",
            default=0.0,
        ),
        0.0,
    )
    previous_gym_name = str(
        _row_value(
            row,
            f"{fighter_prefix}_previous_gym_name",
            f"{short_prefix}_previous_gym_name",
            default="",
        )
        or ""
    ).strip()
    camp_change_flag = _safe_float(
        _row_value(
            row,
            f"{fighter_prefix}_camp_change_flag",
            f"{short_prefix}_camp_change_flag",
            default=0.0,
        ),
        0.0,
    )
    new_gym_flag = _safe_float(
        _row_value(
            row,
            f"{fighter_prefix}_new_gym_flag",
            f"{short_prefix}_new_gym_flag",
            default=0.0,
        ),
        0.0,
    )

    if not any([gym_name, gym_tier, gym_record, gym_changed_flag, camp_change_flag, new_gym_flag]):
        return ""

    summary = fighter_name
    if gym_name:
        meta: list[str] = []
        if gym_tier:
            meta.append(_format_gym_tier_label(gym_tier))
        if gym_record:
            meta.append(gym_record)
        summary += f" {gym_name}"
        if meta:
            summary += f" ({', '.join(meta)})"
    else:
        summary += " camp watch"

    if gym_changed_flag >= 1 and previous_gym_name:
        summary += f" | switched from {previous_gym_name}"
    elif camp_change_flag >= 1 or new_gym_flag >= 1:
        summary += " | camp-change flag"
    return summary


def _actionable_expression(row: pd.Series | object) -> tuple[str, str]:
    pick_name, _, _, best_edge = _best_pick_context(row)
    preferred_expression = str(_row_value(row, "preferred_market_expression", default=""))
    value_expression_winner = str(_row_value(row, "value_expression_winner", default=""))
    preferred_market_odds = _row_value(row, "preferred_market_american_odds", default=pd.NA)

    if best_edge < 0.01:
        return ("Pass", "No positive model edge at current prices.")
    if best_edge < 0.03:
        return ("Watch", "Small edge only; wait for a better number or more confirmation.")

    if pd.notna(preferred_market_odds) and value_expression_winner == "alternative_market":
        return ("Bet", f"{preferred_expression} is the best priced expression right now.")

    if value_expression_winner == "side_market":
        return ("Bet", f"{pick_name} moneyline is priced better than the alternative look.")

    if pd.isna(preferred_market_odds) and preferred_expression != f"{pick_name} moneyline":
        return ("Bet", f"{pick_name} moneyline now; keep a lean toward {preferred_expression} if a price appears.")

    return ("Bet", f"{pick_name} moneyline is the cleanest currently priced play.")


def _derive_market_read(row: pd.Series) -> tuple[str, str, str, float]:
    pick_name, pick_side, pick_prob, pick_edge = _pick_side(row)
    pick_odds = int(row["american_odds"]) if pick_side == "fighter_a" else int(row["fighter_b_current_american_odds"])
    finish_prob = _safe_float(row.get("projected_finish_prob"), 0.50)
    decision_prob = _safe_float(row.get("projected_decision_prob"), 0.50)
    pick_inside_prob = _safe_float(
        row.get("fighter_a_inside_distance_prob" if pick_side == "fighter_a" else "fighter_b_inside_distance_prob"),
        0.0,
    )
    pick_by_decision_prob = _safe_float(
        row.get("fighter_a_by_decision_prob" if pick_side == "fighter_a" else "fighter_b_by_decision_prob"),
        0.0,
    )
    big_favorite = abs(pick_odds) >= 180 if pick_odds < 0 else False
    wmma = "women" in str(row["a_weight_class"]).lower() or "women" in str(row["b_weight_class"]).lower()

    style_tags: list[str] = []
    if decision_prob >= 0.56 or wmma:
        style_tags.append("decision_lean")
    if finish_prob >= 0.56:
        style_tags.append("finish_lean")
    if pick_inside_prob >= 0.28:
        style_tags.append("favorite_finish_lean" if pick_prob >= 0.55 else "underdog_finish_lean")
    if (_safe_float(row["a_takedown_avg"]) + _safe_float(row["b_takedown_avg"])) >= 4.0:
        style_tags.append("grappling_heavy")
    if pick_edge >= 0.12:
        style_tags.append("clear_side_value")

    if (decision_prob >= 0.62 or wmma) and big_favorite:
        return (
            "Fight goes to decision",
            ", ".join(style_tags),
            f"{pick_name} looks favored and the separate finish/decision model still leans long-fight, so the decision prop is cleaner than a heavy side price.",
            decision_prob,
        )
    if pick_by_decision_prob >= 0.30 and decision_prob >= 0.55:
        drivers = _driver_labels(row, pick_side)
        return (
            f"{pick_name} by decision",
            ", ".join(style_tags),
            f"Decision equity is strongest through {pick_name}; key drivers: {', '.join(drivers[:3]) if drivers else 'low-chaos fight shape'}.",
            pick_by_decision_prob,
        )
    if pick_inside_prob >= 0.33 and finish_prob >= 0.56 and big_favorite:
        drivers = _driver_labels(row, pick_side)
        return (
            f"{pick_name} inside distance",
            ", ".join(style_tags),
            f"The stoppage angle for {pick_name} is backed by {', '.join(drivers[:3]) if drivers else 'strong finish equity'}.",
            pick_inside_prob,
        )
    if finish_prob >= 0.60:
        drivers = _driver_labels(row, pick_side)
        return (
            "Fight doesn't go to decision",
            ", ".join(style_tags),
            f"Finish pressure is real here: {', '.join(drivers[:3]) if drivers else f'modeled finish rate {finish_prob:.1%}'}.",
            finish_prob,
        )
    if pick_inside_prob >= 0.27 and pick_prob >= 0.62:
        drivers = _driver_labels(row, pick_side)
        return (
            f"{pick_name} inside distance",
            ", ".join(style_tags),
            f"{pick_name} carries enough finish equity to consider the prop, driven by {', '.join(drivers[:3]) if drivers else 'its side profile'}.",
            pick_inside_prob,
        )
    drivers = _driver_labels(row, pick_side)
    return (
        f"{pick_name} moneyline",
        ", ".join(style_tags),
        f"The moneyline edge is the cleanest path, driven by {', '.join(drivers[:3]) if drivers else 'the side profile more than prop-specific outcomes'}.",
        0.45,
    )


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().replace("’", "'").split())


def _normalize_market_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normalize_text(value)).strip("_")


def _extract_odds_tokens(cells: list[str]) -> list[int]:
    tokens: list[int] = []
    for cell in cells:
        tokens.extend(int(token) for token in re.findall(r"(?<!\d)([+-]\d{2,4})(?!\d)", cell))
    return [token for token in tokens if abs(token) <= 2000]


def _consensus_price_from_tokens(tokens: list[int]) -> int | None:
    if not tokens:
        return None
    implied_probs = []
    for odds in tokens:
        if odds > 0:
            implied_probs.append(100 / (odds + 100))
        else:
            implied_probs.append(abs(odds) / (abs(odds) + 100))
    implied_probs.sort()
    mid = len(implied_probs) // 2
    if len(implied_probs) % 2:
        consensus_prob = implied_probs[mid]
    else:
        consensus_prob = (implied_probs[mid - 1] + implied_probs[mid]) / 2
    if consensus_prob >= 0.5:
        return int(round(-(consensus_prob / (1 - consensus_prob)) * 100))
    return int(round(((1 - consensus_prob) / consensus_prob) * 100))


def _is_fighter_name_row(label: str) -> bool:
    normalized = _normalize_text(label)
    if not normalized:
        return False
    blocked_prefixes = (
        "over ",
        "under ",
        "fight ",
        "any other result",
        "not ",
    )
    if normalized.startswith(blocked_prefixes):
        return False
    blocked_contains = (" wins ", " round ", "draw", "decision", "distance", "submission", "tko/ko")
    if any(token in normalized for token in blocked_contains):
        return False
    return True


def _extract_bestfightodds_alt_markets(html: str) -> dict[tuple[str, str], dict[str, int]]:
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if len(tables) < 2:
        return {}

    label_rows = tables[0].find_all("tr")
    odds_rows = tables[1].find_all("tr")
    market_map: dict[tuple[str, str], dict[str, int]] = {}
    current_fight: tuple[str, str] | None = None
    pending_fighter: str | None = None

    for label_row, odds_row in zip(label_rows[1:], odds_rows[1:]):
        label_cells = [" ".join(cell.stripped_strings) for cell in label_row.find_all(["th", "td"])]
        odds_cells = [" ".join(cell.stripped_strings) for cell in odds_row.find_all(["th", "td"])]
        if not label_cells:
            continue
        label = label_cells[0].strip()
        if not label:
            continue

        # Fighter name rows appear as two consecutive rows before the market labels.
        if odds_cells and _normalize_text(label) == _normalize_text(odds_cells[0]) and _is_fighter_name_row(label):
            if pending_fighter is None:
                pending_fighter = label
            else:
                current_fight = (pending_fighter, label)
                market_map.setdefault(current_fight, {})
                pending_fighter = None
            continue

        if current_fight is None:
            continue

        normalized_label = _normalize_text(label)
        tokens = _extract_odds_tokens(odds_cells[1:])
        consensus_price = _consensus_price_from_tokens(tokens)
        if consensus_price is None:
            continue
        fighter_a, fighter_b = current_fight
        key: str | None = None
        if normalized_label == "fight goes to decision":
            key = "fight_goes_to_decision_odds"
        elif normalized_label == "fight doesn't go to decision":
            key = "fight_doesnt_go_to_decision_odds"
        elif normalized_label == _normalize_text(f"{fighter_a} wins inside distance"):
            key = "fighter_a_inside_distance_odds"
        elif normalized_label == _normalize_text(f"{fighter_b} wins inside distance"):
            key = "fighter_b_inside_distance_odds"
        elif normalized_label == _normalize_text(f"{fighter_a} wins by decision"):
            key = "fighter_a_by_decision_odds"
        elif normalized_label == _normalize_text(f"{fighter_b} wins by decision"):
            key = "fighter_b_by_decision_odds"
        if key:
            market_map.setdefault(current_fight, {})[key] = consensus_price

    return market_map


def _label_to_report_column(label: str, fighter_a: str, fighter_b: str, market_key: str = "") -> str | None:
    normalized_label = _normalize_text(label)
    normalized_market_key = _normalize_market_key(market_key)
    fighter_a_label = _normalize_text(fighter_a)
    fighter_b_label = _normalize_text(fighter_b)

    if normalized_label in {"yes", "to_go_the_distance", "goes_the_distance"}:
        if any(token in normalized_market_key for token in ["goes_distance", "goes_the_distance", "to_go_the_distance", "decision"]):
            return "fight_goes_to_decision_odds"
    if normalized_label in {"no", "not_to_go_the_distance", "does_not_go_the_distance"}:
        if any(token in normalized_market_key for token in ["goes_distance", "goes_the_distance", "to_go_the_distance", "decision"]):
            return "fight_doesnt_go_to_decision_odds"

    if normalized_label in {"fight goes to decision", "fight goes the distance", "to go the distance"}:
        return "fight_goes_to_decision_odds"
    if normalized_label in {"fight doesn t go to decision", "fight doesnt go to decision", "fight does not go the distance", "not to go the distance"}:
        return "fight_doesnt_go_to_decision_odds"

    if fighter_a_label in normalized_label and "by decision" in normalized_label:
        return "fighter_a_by_decision_odds"
    if fighter_b_label in normalized_label and "by decision" in normalized_label:
        return "fighter_b_by_decision_odds"
    if fighter_a_label in normalized_label and "inside distance" in normalized_label:
        return "fighter_a_inside_distance_odds"
    if fighter_b_label in normalized_label and "inside distance" in normalized_label:
        return "fighter_b_inside_distance_odds"
    return None


def _extract_oddsapi_alt_markets(payload: dict[str, object], bookmaker_key: str, fighter_a: str, fighter_b: str) -> dict[str, int]:
    prices: dict[str, int] = {}
    for bookmaker in payload.get("bookmakers", []):
        if str(bookmaker.get("key", "")).strip() != bookmaker_key:
            continue
        for market in bookmaker.get("markets", []):
            market_key = str(market.get("key", "")).strip()
            for outcome in market.get("outcomes", []):
                name = str(outcome.get("name", "")).strip()
                description = str(outcome.get("description", "")).strip()
                combined_label = " ".join(part for part in [description, name] if part)
                column = _label_to_report_column(combined_label or name, fighter_a, fighter_b, market_key)
                if not column:
                    continue
                price = outcome.get("price")
                if price is None:
                    continue
                prices[column] = int(price)
    return prices


def enrich_with_bestfightodds_alternative_markets(report: pd.DataFrame, event_urls: list[str]) -> pd.DataFrame:
    if not event_urls:
        return report

    enriched = report.copy()
    enriched["fight_goes_to_decision_odds"] = pd.NA
    enriched["fight_doesnt_go_to_decision_odds"] = pd.NA
    enriched["fighter_a_inside_distance_odds"] = pd.NA
    enriched["fighter_b_inside_distance_odds"] = pd.NA
    enriched["fighter_a_by_decision_odds"] = pd.NA
    enriched["fighter_b_by_decision_odds"] = pd.NA
    enriched["_fight_key"] = enriched.apply(
        lambda row: frozenset({_normalize_text(row["fighter_a"]), _normalize_text(row["fighter_b"])}),
        axis=1,
    )

    for event_url in event_urls:
        try:
            html = fetch_bestfightodds_html(event_url)
        except Exception as exc:
            print(f"Warning: skipping BestFightOdds alt-market fetch for {event_url}: {exc}", file=sys.stderr)
            continue
        market_map = _extract_bestfightodds_alt_markets(html)
        for (fighter_a, fighter_b), prices in market_map.items():
            fight_key = frozenset({_normalize_text(fighter_a), _normalize_text(fighter_b)})
            mask = enriched["_fight_key"] == fight_key
            if not mask.any():
                continue
            for column, value in prices.items():
                enriched.loc[mask, column] = value

    enriched["preferred_market_american_odds"] = enriched.apply(
        lambda row: (
            row["fight_goes_to_decision_odds"]
            if row["preferred_market_expression"] == "Fight goes to decision"
            else row["fight_doesnt_go_to_decision_odds"]
            if row["preferred_market_expression"] == "Fight doesn't go to decision"
            else row["fighter_a_inside_distance_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} inside distance"
            else row["fighter_b_inside_distance_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} inside distance"
            else row["fighter_a_by_decision_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} by decision"
            else row["fighter_b_by_decision_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} by decision"
            else row["side_market_american_odds"]
            if str(row["preferred_market_expression"]).endswith("moneyline")
            else pd.NA
        ),
        axis=1,
    )
    enriched["side_market_american_odds"] = enriched.apply(
        lambda row: row["fighter_a_current_american_odds"]
        if row["preferred_market_expression"].startswith(row["fighter_a"])
        else row["fighter_b_current_american_odds"]
        if row["preferred_market_expression"].startswith(row["fighter_b"])
        else row["fighter_a_current_american_odds"]
        if row["fighter_a_edge_vs_current"] >= row["fighter_b_edge_vs_current"]
        else row["fighter_b_current_american_odds"],
        axis=1,
    )
    enriched["market_comparison_summary"] = enriched.apply(
        lambda row: (
            f"Side {int(row['side_market_american_odds']):+d} vs alt {int(row['preferred_market_american_odds']):+d}"
            if pd.notna(row["preferred_market_american_odds"])
            else "No alternative market price found"
        ),
        axis=1,
    )
    expression_values = enriched.apply(_expression_decision, axis=1, result_type="expand")
    expression_values.columns = ["value_expression_winner", "value_expression_reason", "side_market_implied_prob", "alternative_market_implied_prob"]
    for column in expression_values.columns:
        enriched[column] = expression_values[column]
    enriched = enriched.drop(columns=["_fight_key"])
    return enriched


def enrich_with_oddsapi_alternative_markets(
    report: pd.DataFrame,
    *,
    bookmaker_key: str = DEFAULT_BOOKMAKER,
) -> pd.DataFrame:
    if report.empty or "odds_api_event_id" not in report.columns:
        return report

    event_ids = sorted({str(value).strip() for value in report["odds_api_event_id"].dropna() if str(value).strip()})
    if not event_ids:
        return report

    try:
        api_key = load_api_key(os.getenv("ODDS_API_KEY"))
    except OddsApiError:
        return report

    enriched = report.copy()
    for event_id in event_ids:
        try:
            markets_payload = fetch_the_odds_api_event_markets(api_key=api_key, event_id=event_id)
            alt_market_keys = extract_alternative_market_keys(markets_payload, bookmaker_key)
            candidate_market_keys = [
                key
                for key in alt_market_keys
                if any(token in _normalize_market_key(key) for token in ["decision", "distance"])
            ]
            if not candidate_market_keys:
                continue
            odds_payload = fetch_the_odds_api_event_odds(
                api_key=api_key,
                event_id=event_id,
                bookmakers=bookmaker_key,
                markets=",".join(candidate_market_keys),
            )
        except Exception:
            continue

        mask = enriched["odds_api_event_id"].astype(str) == event_id
        for idx, row in enriched.loc[mask].iterrows():
            prices = _extract_oddsapi_alt_markets(odds_payload, bookmaker_key, str(row["fighter_a"]), str(row["fighter_b"]))
            if not prices:
                continue
            for column, value in prices.items():
                enriched.at[idx, column] = value

    enriched["preferred_market_american_odds"] = enriched.apply(
        lambda row: (
            row["fight_goes_to_decision_odds"]
            if row["preferred_market_expression"] == "Fight goes to decision"
            else row["fight_doesnt_go_to_decision_odds"]
            if row["preferred_market_expression"] == "Fight doesn't go to decision"
            else row["fighter_a_inside_distance_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} inside distance"
            else row["fighter_b_inside_distance_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} inside distance"
            else row["fighter_a_by_decision_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} by decision"
            else row["fighter_b_by_decision_odds"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} by decision"
            else row["side_market_american_odds"]
            if str(row["preferred_market_expression"]).endswith("moneyline")
            else pd.NA
        ),
        axis=1,
    )
    enriched["market_comparison_summary"] = enriched.apply(
        lambda row: (
            f"Side {int(row['side_market_american_odds']):+d} vs alt {int(row['preferred_market_american_odds']):+d}"
            if pd.notna(row["preferred_market_american_odds"])
            else row["market_comparison_summary"]
        ),
        axis=1,
    )
    expression_values = enriched.apply(_expression_decision, axis=1, result_type="expand")
    expression_values.columns = ["value_expression_winner", "value_expression_reason", "side_market_implied_prob", "alternative_market_implied_prob"]
    for column in expression_values.columns:
        enriched[column] = expression_values[column]
    return enriched


def build_fight_week_report(
    odds_path: str | Path,
    fighter_stats_path: str | Path,
    *,
    bestfightodds_event_urls: list[str] | None = None,
    odds_api_bookmaker: str = DEFAULT_BOOKMAKER,
    db_path: str | Path | None = None,
    side_model_bundle: dict[str, object] | None = None,
    confidence_model_bundle: dict[str, object] | None = None,
) -> pd.DataFrame:
    odds = normalize_odds_frame(load_odds_csv(odds_path))
    for column in ["open_american_odds", "current_best_range_low", "current_best_range_high"]:
        if column not in odds.columns:
            odds[column] = pd.NA
    fighter_stats = load_fighter_stats(fighter_stats_path)
    features = build_fight_features(odds, fighter_stats)
    projections = project_fight_probabilities(
        features,
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )
    if projections.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "start_time",
                "fighter_a",
                "fighter_b",
                "fighter_a_model_win_prob",
                "fighter_b_model_win_prob",
                "projected_finish_prob",
                "projected_decision_prob",
                "preferred_market_expression",
                "market_style_tags",
                "market_substitution_reason",
                "value_expression_winner",
                "value_expression_reason",
            ]
        )

    fight_rows = projections.loc[projections["selection"] == "fighter_a"].copy()
    if "odds_api_event_id" in fight_rows.columns:
        pass
    fighter_b_rows = (
        projections.loc[
            projections["selection"] == "fighter_b",
            [
                "event_id",
                "fighter_a",
                "fighter_b",
                "american_odds",
                "open_american_odds",
                "current_best_range_low",
                "current_best_range_high",
                "model_projected_win_prob",
            ],
        ]
        .rename(
            columns={
                "american_odds": "fighter_b_current_american_odds",
                "open_american_odds": "fighter_b_open_american_odds",
                "current_best_range_low": "fighter_b_range_low",
                "current_best_range_high": "fighter_b_range_high",
                "model_projected_win_prob": "fighter_b_model_win_prob",
            }
        )
        .copy()
    )

    fight_rows = fight_rows.merge(fighter_b_rows, on=["event_id", "fighter_a", "fighter_b"], how="left")
    fight_rows = fight_rows.loc[
        fight_rows["american_odds"].notna() & fight_rows["fighter_b_current_american_odds"].notna()
    ].copy()
    if fight_rows.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "start_time",
                "fighter_a",
                "fighter_b",
                "fighter_a_model_win_prob",
                "fighter_b_model_win_prob",
                "projected_finish_prob",
                "projected_decision_prob",
                "preferred_market_expression",
                "market_style_tags",
                "market_substitution_reason",
                "value_expression_winner",
                "value_expression_reason",
            ]
        )
    fight_rows["fighter_a_current_implied_prob"] = fight_rows["american_odds"].apply(implied_probability)
    fight_rows["fighter_b_current_implied_prob"] = fight_rows["fighter_b_current_american_odds"].apply(implied_probability)
    fight_rows["fighter_a_open_implied_prob"] = fight_rows["open_american_odds"].apply(
        lambda odds: implied_probability(int(odds)) if pd.notna(odds) else pd.NA
    )
    fight_rows["fighter_b_open_implied_prob"] = fight_rows["fighter_b_open_american_odds"].apply(
        lambda odds: implied_probability(int(odds)) if pd.notna(odds) else pd.NA
    )
    fight_rows["fighter_a_edge_vs_current"] = (
        fight_rows["projected_fighter_a_win_prob"] - fight_rows["fighter_a_current_implied_prob"]
    )
    fight_rows["fighter_b_edge_vs_current"] = (
        fight_rows["fighter_b_model_win_prob"] - fight_rows["fighter_b_current_implied_prob"]
    )
    market_reads = fight_rows.apply(_derive_market_read, axis=1, result_type="expand")
    market_reads.columns = [
        "preferred_market_expression",
        "market_style_tags",
        "market_substitution_reason",
        "market_style_confidence",
    ]
    fight_rows = pd.concat([fight_rows, market_reads], axis=1)
    for column, default_value in {
        "a_gym_name": "",
        "b_gym_name": "",
        "a_gym_tier": "",
        "b_gym_tier": "",
        "a_gym_record": "",
        "b_gym_record": "",
        "a_previous_gym_name": "",
        "b_previous_gym_name": "",
        "a_gym_score": 0.0,
        "b_gym_score": 0.0,
        "a_gym_changed_flag": 0.0,
        "b_gym_changed_flag": 0.0,
    }.items():
        if column not in fight_rows.columns:
            fight_rows[column] = default_value

    columns = [
        "event_name",
        "start_time",
        "fighter_a",
        "fighter_b",
        "american_odds",
        "fighter_b_current_american_odds",
        "open_american_odds",
        "fighter_b_open_american_odds",
        "current_best_range_low",
        "current_best_range_high",
        "fighter_b_range_low",
        "fighter_b_range_high",
        "a_height_in",
        "b_height_in",
        "a_reach_in",
        "b_reach_in",
        "a_age_years",
        "b_age_years",
        "a_stance",
        "b_stance",
        "a_sig_strikes_landed_per_min",
        "b_sig_strikes_landed_per_min",
        "a_sig_strikes_absorbed_per_min",
        "b_sig_strikes_absorbed_per_min",
        "a_takedown_avg",
        "b_takedown_avg",
        "a_takedown_defense_pct",
        "b_takedown_defense_pct",
        "a_recent_grappling_rate",
        "b_recent_grappling_rate",
        "a_control_avg",
        "b_control_avg",
        "a_recent_control_avg",
        "b_recent_control_avg",
        "a_ko_win_rate",
        "b_ko_win_rate",
        "a_submission_win_rate",
        "b_submission_win_rate",
        "a_days_since_last_fight",
        "b_days_since_last_fight",
        "a_ufc_fight_count",
        "b_ufc_fight_count",
        "a_ufc_debut_flag",
        "b_ufc_debut_flag",
        "a_fighter_wins",
        "b_fighter_wins",
        "a_fighter_losses",
        "b_fighter_losses",
        "a_fighter_draws",
        "b_fighter_draws",
        "height_diff",
        "reach_diff",
        "age_diff",
        "experience_diff",
        "projected_fighter_a_win_prob",
        "raw_projected_fighter_a_win_prob",
        "market_blend_weight",
        "fighter_b_model_win_prob",
        "projected_finish_prob",
        "projected_decision_prob",
        "fighter_a_inside_distance_prob",
        "fighter_b_inside_distance_prob",
        "fighter_a_submission_prob",
        "fighter_b_submission_prob",
        "fighter_a_ko_tko_prob",
        "fighter_b_ko_tko_prob",
        "fighter_a_by_decision_prob",
        "fighter_b_by_decision_prob",
        "model_confidence",
        "data_quality",
        "fighter_a_current_implied_prob",
        "fighter_b_current_implied_prob",
        "fighter_a_open_implied_prob",
        "fighter_b_open_implied_prob",
        "fighter_a_edge_vs_current",
        "fighter_b_edge_vs_current",
        "strike_margin_diff",
        "matchup_striking_edge",
        "grappling_diff",
        "matchup_grappling_edge",
        "control_diff",
        "recent_control_diff",
        "matchup_control_edge",
        "grappling_pressure_diff",
        "opponent_quality_diff",
        "recent_opponent_quality_diff",
        "schedule_strength_diff",
        "normalized_strike_margin_diff",
        "normalized_grappling_diff",
        "normalized_control_diff",
        "normalized_recent_form_diff",
        "combined_control_avg",
        "base_projected_fighter_a_win_prob",
        "trained_side_fighter_a_win_prob",
        "side_model_blend_weight",
        "first_round_finish_rate_diff",
        "durability_diff",
        "decision_rate_diff",
        "layoff_diff",
        "ufc_experience_diff",
        "ufc_debut_penalty_diff",
        "a_stats_completeness",
        "b_stats_completeness",
        "a_fallback_used",
        "b_fallback_used",
        "a_short_notice_flag",
        "b_short_notice_flag",
        "a_short_notice_acceptance_flag",
        "b_short_notice_acceptance_flag",
        "a_short_notice_success_flag",
        "b_short_notice_success_flag",
        "a_new_gym_flag",
        "b_new_gym_flag",
        "a_new_contract_flag",
        "b_new_contract_flag",
        "a_cardio_fade_flag",
        "b_cardio_fade_flag",
        "a_injury_concern_flag",
        "b_injury_concern_flag",
        "a_weight_cut_concern_flag",
        "b_weight_cut_concern_flag",
        "a_replacement_fighter_flag",
        "b_replacement_fighter_flag",
        "a_travel_disadvantage_flag",
        "b_travel_disadvantage_flag",
        "a_camp_change_flag",
        "b_camp_change_flag",
        "a_first_round_finish_rate",
        "b_first_round_finish_rate",
        "a_finish_win_rate",
        "b_finish_win_rate",
        "a_finish_loss_rate",
        "b_finish_loss_rate",
        "a_decision_rate",
        "b_decision_rate",
        "a_history_style_label",
        "b_history_style_label",
        "a_context_notes",
        "b_context_notes",
        "a_gym_name",
        "b_gym_name",
        "a_gym_tier",
        "b_gym_tier",
        "a_gym_record",
        "b_gym_record",
        "a_gym_score",
        "b_gym_score",
        "a_gym_changed_flag",
        "b_gym_changed_flag",
        "a_previous_gym_name",
        "b_previous_gym_name",
        "preferred_market_expression",
        "market_style_tags",
        "market_substitution_reason",
    ]
    if "odds_api_event_id" in fight_rows.columns:
        columns.append("odds_api_event_id")
    report = fight_rows[columns].rename(
        columns={
            "american_odds": "fighter_a_current_american_odds",
            "open_american_odds": "fighter_a_open_american_odds",
            "current_best_range_low": "fighter_a_range_low",
            "current_best_range_high": "fighter_a_range_high",
            "a_height_in": "fighter_a_height_in",
            "b_height_in": "fighter_b_height_in",
            "a_reach_in": "fighter_a_reach_in",
            "b_reach_in": "fighter_b_reach_in",
            "a_age_years": "fighter_a_age_years",
            "b_age_years": "fighter_b_age_years",
            "a_stance": "fighter_a_stance",
            "b_stance": "fighter_b_stance",
            "a_sig_strikes_landed_per_min": "fighter_a_sig_strikes_landed_per_min",
            "b_sig_strikes_landed_per_min": "fighter_b_sig_strikes_landed_per_min",
            "a_sig_strikes_absorbed_per_min": "fighter_a_sig_strikes_absorbed_per_min",
            "b_sig_strikes_absorbed_per_min": "fighter_b_sig_strikes_absorbed_per_min",
            "a_takedown_avg": "fighter_a_takedown_avg",
            "b_takedown_avg": "fighter_b_takedown_avg",
            "a_takedown_defense_pct": "fighter_a_takedown_defense_pct",
            "b_takedown_defense_pct": "fighter_b_takedown_defense_pct",
            "a_recent_grappling_rate": "fighter_a_recent_grappling_rate",
            "b_recent_grappling_rate": "fighter_b_recent_grappling_rate",
            "a_control_avg": "fighter_a_control_avg",
            "b_control_avg": "fighter_b_control_avg",
            "a_recent_control_avg": "fighter_a_recent_control_avg",
            "b_recent_control_avg": "fighter_b_recent_control_avg",
            "a_ko_win_rate": "fighter_a_ko_win_rate",
            "b_ko_win_rate": "fighter_b_ko_win_rate",
            "a_submission_win_rate": "fighter_a_submission_win_rate",
            "b_submission_win_rate": "fighter_b_submission_win_rate",
            "a_days_since_last_fight": "fighter_a_days_since_last_fight",
            "b_days_since_last_fight": "fighter_b_days_since_last_fight",
            "a_ufc_fight_count": "fighter_a_ufc_fight_count",
            "b_ufc_fight_count": "fighter_b_ufc_fight_count",
            "a_ufc_debut_flag": "fighter_a_ufc_debut_flag",
            "b_ufc_debut_flag": "fighter_b_ufc_debut_flag",
            "a_fighter_wins": "fighter_a_record_wins",
            "b_fighter_wins": "fighter_b_record_wins",
            "a_fighter_losses": "fighter_a_record_losses",
            "b_fighter_losses": "fighter_b_record_losses",
            "a_fighter_draws": "fighter_a_record_draws",
            "b_fighter_draws": "fighter_b_record_draws",
            "height_diff": "fighter_a_height_advantage_in",
            "reach_diff": "fighter_a_reach_advantage_in",
            "projected_fighter_a_win_prob": "fighter_a_model_win_prob",
            "raw_projected_fighter_a_win_prob": "fighter_a_raw_model_win_prob",
            "a_stats_completeness": "fighter_a_stats_completeness",
            "b_stats_completeness": "fighter_b_stats_completeness",
            "a_fallback_used": "fighter_a_fallback_used",
            "b_fallback_used": "fighter_b_fallback_used",
            "a_short_notice_flag": "fighter_a_short_notice_flag",
            "b_short_notice_flag": "fighter_b_short_notice_flag",
            "a_short_notice_acceptance_flag": "fighter_a_short_notice_acceptance_flag",
            "b_short_notice_acceptance_flag": "fighter_b_short_notice_acceptance_flag",
            "a_short_notice_success_flag": "fighter_a_short_notice_success_flag",
            "b_short_notice_success_flag": "fighter_b_short_notice_success_flag",
            "a_new_gym_flag": "fighter_a_new_gym_flag",
            "b_new_gym_flag": "fighter_b_new_gym_flag",
            "a_new_contract_flag": "fighter_a_new_contract_flag",
            "b_new_contract_flag": "fighter_b_new_contract_flag",
            "a_cardio_fade_flag": "fighter_a_cardio_fade_flag",
            "b_cardio_fade_flag": "fighter_b_cardio_fade_flag",
            "a_injury_concern_flag": "fighter_a_injury_concern_flag",
            "b_injury_concern_flag": "fighter_b_injury_concern_flag",
            "a_weight_cut_concern_flag": "fighter_a_weight_cut_concern_flag",
            "b_weight_cut_concern_flag": "fighter_b_weight_cut_concern_flag",
            "a_replacement_fighter_flag": "fighter_a_replacement_fighter_flag",
            "b_replacement_fighter_flag": "fighter_b_replacement_fighter_flag",
            "a_travel_disadvantage_flag": "fighter_a_travel_disadvantage_flag",
            "b_travel_disadvantage_flag": "fighter_b_travel_disadvantage_flag",
            "a_camp_change_flag": "fighter_a_camp_change_flag",
            "b_camp_change_flag": "fighter_b_camp_change_flag",
            "a_first_round_finish_rate": "fighter_a_first_round_finish_rate",
            "b_first_round_finish_rate": "fighter_b_first_round_finish_rate",
            "a_finish_win_rate": "fighter_a_finish_win_rate",
            "b_finish_win_rate": "fighter_b_finish_win_rate",
            "a_finish_loss_rate": "fighter_a_finish_loss_rate",
            "b_finish_loss_rate": "fighter_b_finish_loss_rate",
            "a_decision_rate": "fighter_a_decision_rate",
            "b_decision_rate": "fighter_b_decision_rate",
            "a_history_style_label": "fighter_a_history_style_label",
            "b_history_style_label": "fighter_b_history_style_label",
            "a_context_notes": "fighter_a_context_notes",
            "b_context_notes": "fighter_b_context_notes",
            "a_gym_name": "fighter_a_gym_name",
            "b_gym_name": "fighter_b_gym_name",
            "a_gym_tier": "fighter_a_gym_tier",
            "b_gym_tier": "fighter_b_gym_tier",
            "a_gym_record": "fighter_a_gym_record",
            "b_gym_record": "fighter_b_gym_record",
            "a_gym_score": "fighter_a_gym_score",
            "b_gym_score": "fighter_b_gym_score",
            "a_gym_changed_flag": "fighter_a_gym_changed_flag",
            "b_gym_changed_flag": "fighter_b_gym_changed_flag",
            "a_previous_gym_name": "fighter_a_previous_gym_name",
            "b_previous_gym_name": "fighter_b_previous_gym_name",
        }
    )
    report["fight_goes_to_decision_odds"] = pd.NA
    report["fight_doesnt_go_to_decision_odds"] = pd.NA
    report["fighter_a_inside_distance_odds"] = pd.NA
    report["fighter_b_inside_distance_odds"] = pd.NA
    report["fighter_a_by_decision_odds"] = pd.NA
    report["fighter_b_by_decision_odds"] = pd.NA
    report["preferred_market_american_odds"] = pd.NA
    report["fight_goes_to_decision_model_prob"] = report["projected_decision_prob"]
    report["fight_doesnt_go_to_decision_model_prob"] = report["projected_finish_prob"]
    report["preferred_market_projected_prob"] = report.apply(
        lambda row: (
            row["fight_goes_to_decision_model_prob"]
            if row["preferred_market_expression"] == "Fight goes to decision"
            else row["fight_doesnt_go_to_decision_model_prob"]
            if row["preferred_market_expression"] == "Fight doesn't go to decision"
            else row["fighter_a_inside_distance_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} inside distance"
            else row["fighter_b_inside_distance_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} inside distance"
            else row["fighter_a_by_decision_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} by decision"
            else row["fighter_b_by_decision_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} by decision"
            else row["fighter_a_model_win_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_a']} moneyline"
            else row["fighter_b_model_win_prob"]
            if row["preferred_market_expression"] == f"{row['fighter_b']} moneyline"
            else pd.NA
        ),
        axis=1,
    )
    report["side_market_american_odds"] = report.apply(
        lambda row: row["fighter_a_current_american_odds"]
        if row["preferred_market_expression"].startswith(row["fighter_a"])
        else row["fighter_b_current_american_odds"]
        if row["preferred_market_expression"].startswith(row["fighter_b"])
        else row["fighter_a_current_american_odds"]
        if row["fighter_a_edge_vs_current"] >= row["fighter_b_edge_vs_current"]
        else row["fighter_b_current_american_odds"],
        axis=1,
    )
    report["preferred_market_american_odds"] = report.apply(
        lambda row: row["side_market_american_odds"]
        if str(row["preferred_market_expression"]).endswith("moneyline")
        else row["preferred_market_american_odds"],
        axis=1,
    )
    report["side_market_projected_prob"] = report.apply(
        lambda row: row["fighter_a_model_win_prob"]
        if row["preferred_market_expression"].startswith(row["fighter_a"])
        else row["fighter_b_model_win_prob"]
        if row["preferred_market_expression"].startswith(row["fighter_b"])
        else row["fighter_a_model_win_prob"]
        if row["fighter_a_edge_vs_current"] >= row["fighter_b_edge_vs_current"]
        else row["fighter_b_model_win_prob"],
        axis=1,
    )
    report["market_comparison_summary"] = "No alternative market price found"
    longshot_watch = report.apply(_longshot_prop_watch, axis=1, result_type="expand")
    longshot_watch.columns = [
        "speculative_prop_expression",
        "speculative_prop_model_prob",
        "speculative_prop_fair_american_odds",
        "speculative_prop_fair_decimal_odds",
        "speculative_prop_reason",
    ]
    report = pd.concat([report, longshot_watch], axis=1)
    expression_values = report.apply(_expression_decision, axis=1, result_type="expand")
    expression_values.columns = [
        "value_expression_winner",
        "value_expression_reason",
        "side_market_implied_prob",
        "alternative_market_implied_prob",
    ]
    for column in expression_values.columns:
        report[column] = expression_values[column]
    report = enrich_with_bestfightodds_alternative_markets(report, bestfightodds_event_urls or [])
    if report["preferred_market_american_odds"].isna().any():
        report = enrich_with_oddsapi_alternative_markets(report, bookmaker_key=odds_api_bookmaker)
    report["preferred_market_implied_prob"] = report["preferred_market_american_odds"].apply(_american_to_implied_probability)
    report["preferred_market_edge"] = report["preferred_market_projected_prob"] - report["preferred_market_implied_prob"]
    report["side_market_edge"] = report["side_market_projected_prob"] - report["side_market_american_odds"].apply(_american_to_implied_probability)
    for column in [
        "fighter_a_current_american_odds",
        "fighter_b_current_american_odds",
        "fighter_a_open_american_odds",
        "fighter_b_open_american_odds",
        "preferred_market_american_odds",
        "side_market_american_odds",
    ]:
        report[column.replace("_american_odds", "_decimal_odds")] = report[column].apply(_american_to_decimal)
    for fighter_prefix in ["fighter_a", "fighter_b"]:
        report[f"{fighter_prefix}_model_context_flags"] = report.apply(
            lambda row: _fighter_flag_summary(row, fighter_prefix, MODEL_CONTEXT_FLAG_COLUMNS),
            axis=1,
        )
        report[f"{fighter_prefix}_operator_context_flags"] = report.apply(
            lambda row: _fighter_flag_summary(row, fighter_prefix, OPERATOR_CONTEXT_FLAG_COLUMNS),
            axis=1,
        )
    report = _attach_decision_diagnostics(report, db_path=db_path)
    return report.sort_values(by=["start_time", "fighter_a"]).reset_index(drop=True)


def _format_percent(value: float | pd.NA) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _american_to_decimal(odds: object) -> float | pd.NA:
    if pd.isna(odds):
        return pd.NA
    value = int(float(odds))
    if value > 0:
        return round((value / 100) + 1, 2)
    return round((100 / abs(value)) + 1, 2)


def _format_decimal(odds: object) -> str:
    decimal_odds = _american_to_decimal(odds)
    if pd.isna(decimal_odds):
        return "n/a"
    return f"{float(decimal_odds):.2f}"


def _american_to_implied_probability(odds: object) -> float | pd.NA:
    if pd.isna(odds):
        return pd.NA
    value = int(odds)
    if value > 0:
        return 100 / (value + 100)
    return abs(value) / (abs(value) + 100)


def _probability_to_american_odds(probability: object) -> int | pd.NA:
    if pd.isna(probability):
        return pd.NA
    value = float(probability)
    if value <= 0 or value >= 1:
        return pd.NA
    if value >= 0.5:
        return int(round(-(value / (1 - value)) * 100))
    return int(round(((1 - value) / value) * 100))


def _longshot_prop_watch(row: pd.Series) -> tuple[object, object, object, object, object]:
    fighter_a_edge = _safe_float(row.get("fighter_a_edge_vs_current"), 0.0)
    fighter_b_edge = _safe_float(row.get("fighter_b_edge_vs_current"), 0.0)
    if fighter_a_edge >= fighter_b_edge:
        pick_name = str(row["fighter_a"])
        pick_side = "fighter_a"
        pick_prob = _safe_float(row.get("fighter_a_model_win_prob"), 0.0)
    else:
        pick_name = str(row["fighter_b"])
        pick_side = "fighter_b"
        pick_prob = _safe_float(row.get("fighter_b_model_win_prob"), 0.0)
    pick_inside_prob = _safe_float(
        row.get("fighter_a_inside_distance_prob" if pick_side == "fighter_a" else "fighter_b_inside_distance_prob"),
        0.0,
    )
    pick_submission_prob = _safe_float(
        row.get("fighter_a_submission_prob" if pick_side == "fighter_a" else "fighter_b_submission_prob"),
        0.0,
    )
    pick_ko_prob = _safe_float(
        row.get("fighter_a_ko_tko_prob" if pick_side == "fighter_a" else "fighter_b_ko_tko_prob"),
        0.0,
    )
    pick_decision_prob = _safe_float(
        row.get("fighter_a_by_decision_prob" if pick_side == "fighter_a" else "fighter_b_by_decision_prob"),
        0.0,
    )
    sign = 1.0 if pick_side == "fighter_a" else -1.0
    grappling_edge = sign * _safe_float(row.get("grappling_diff"), 0.0)
    strike_edge = sign * _safe_float(row.get("strike_margin_diff"), 0.0)
    finish_edge = sign * _safe_float(row.get("first_round_finish_rate_diff"), 0.0)
    durability_edge = sign * _safe_float(row.get("durability_diff"), 0.0)
    decision_edge = sign * _safe_float(row.get("decision_rate_diff"), 0.0)
    selection_takedowns = _safe_float(row.get("fighter_a_takedown_avg" if pick_side == "fighter_a" else "fighter_b_takedown_avg"), 0.0)
    opponent_td_def = _safe_float(row.get("fighter_b_takedown_defense_pct" if pick_side == "fighter_a" else "fighter_a_takedown_defense_pct"), 0.0)

    inside_style = "inside distance lean"
    if grappling_edge >= 0.45 and selection_takedowns >= 1.2 and opponent_td_def <= 72:
        inside_style = "submission lean"
    elif strike_edge >= 0.75 or finish_edge >= 0.15 or durability_edge >= 0.12:
        inside_style = "KO/TKO lean"

    if pick_inside_prob >= 0.16:
        if inside_style == "submission lean" and pick_submission_prob >= 0.12:
            fair_american = _probability_to_american_odds(pick_submission_prob)
            fair_decimal = _american_to_decimal(fair_american)
            if pd.notna(fair_american) and int(fair_american) >= 250:
                return (
                    f"{pick_name} submission",
                    round(pick_submission_prob, 4),
                    fair_american,
                    fair_decimal,
                    f"Aggressive shot only. Model gives {pick_name} {pick_submission_prob:.1%} submission equity in a grappling-friendly matchup.",
                )
        if inside_style == "KO/TKO lean" and pick_ko_prob >= 0.14:
            fair_american = _probability_to_american_odds(pick_ko_prob)
            fair_decimal = _american_to_decimal(fair_american)
            if pd.notna(fair_american) and int(fair_american) >= 225:
                return (
                    f"{pick_name} KO/TKO",
                    round(pick_ko_prob, 4),
                    fair_american,
                    fair_decimal,
                    f"Aggressive shot only. Model gives {pick_name} {pick_ko_prob:.1%} KO/TKO equity with a strong striking finish lean.",
                )
        fair_american = _probability_to_american_odds(pick_inside_prob)
        fair_decimal = _american_to_decimal(fair_american)
        if pd.notna(fair_american) and int(fair_american) >= 175:
            return (
                f"{pick_name} inside distance",
                round(pick_inside_prob, 4),
                fair_american,
                fair_decimal,
                f"Aggressive shot only. Model gives {pick_name} {pick_inside_prob:.1%} inside-distance equity with a {inside_style}.",
            )

    if pick_decision_prob >= 0.18 and row.get("projected_decision_prob", 0.0) >= 0.52 and decision_edge >= 0.05:
        fair_american = _probability_to_american_odds(pick_decision_prob)
        fair_decimal = _american_to_decimal(fair_american)
        if pd.notna(fair_american) and int(fair_american) >= 200:
            return (
                f"{pick_name} by decision",
                round(pick_decision_prob, 4),
                fair_american,
                fair_decimal,
                f"Aggressive shot only. Model gives {pick_name} {pick_decision_prob:.1%} decision equity in a lower-chaos fight shape.",
            )

    return (pd.NA, pd.NA, pd.NA, pd.NA, pd.NA)


def _expression_decision(row: pd.Series) -> tuple[str, str, float | pd.NA, float | pd.NA]:
    side_odds = row["side_market_american_odds"]
    alt_odds = row["preferred_market_american_odds"]
    side_implied = _american_to_implied_probability(side_odds)
    alt_implied = _american_to_implied_probability(alt_odds)
    side_projected = _safe_float(row.get("side_market_projected_prob"), 0.0)
    alt_projected = _safe_float(row.get("preferred_market_projected_prob"), 0.0)
    style_confidence = _safe_float(row.get("market_style_confidence", 0.45), 0.45)
    preferred_expression = str(row.get("preferred_market_expression", ""))

    if pd.isna(alt_implied):
        return (
            "side_only",
            "No alternative price found, so the side remains the only priced expression.",
            side_implied,
            alt_implied,
        )

    if pd.isna(side_implied):
        return (
            "alternative_market",
            "Only the alternative market was priced, so it is the default expression.",
            side_implied,
            alt_implied,
        )

    premium = float(alt_implied) - float(side_implied)
    side_edge = side_projected - float(side_implied)
    alt_edge = alt_projected - float(alt_implied)
    confidence_credit = max(0.0, style_confidence - 0.5) * 0.18

    if alt_edge >= side_edge + 0.02:
        return (
            "alternative_market",
            "The separate finish/decision model gives the alternative market the stronger edge.",
            side_implied,
            alt_implied,
        )

    if side_edge >= alt_edge + 0.02:
        return (
            "side_market",
            "The side still has the better priced edge after comparing the separate outcome models.",
            side_implied,
            alt_implied,
        )

    if preferred_expression == "Fight goes to decision" or preferred_expression.endswith("by decision"):
        if premium <= confidence_credit:
            return (
                "alternative_market",
                "The decision-style market cost is justified by the model's low-chaos read for this fight.",
                side_implied,
                alt_implied,
            )

    if preferred_expression == "Fight doesn't go to decision" or preferred_expression.endswith("inside distance"):
        if premium <= confidence_credit:
            return (
                "alternative_market",
                "The finish-style market cost is justified by the model's finish confidence for this fight.",
                side_implied,
                alt_implied,
            )

    if alt_implied + 0.03 < side_implied:
        return (
            "alternative_market",
            "The alternative market is meaningfully cheaper than the side for the same fight read.",
            side_implied,
            alt_implied,
        )

    if side_implied + 0.03 < alt_implied:
        return (
            "side_market",
            "The side is cheaper than the alternative market, so the prop is not buying enough extra safety.",
            side_implied,
            alt_implied,
        )

    return (
        "comparable",
        "The side and alternative market are priced similarly; choose based on how strongly you trust the fight shape.",
        side_implied,
        alt_implied,
    )


def _attach_decision_diagnostics(report: pd.DataFrame, *, db_path: str | Path | None) -> pd.DataFrame:
    if report.empty:
        return report.copy()

    enriched = report.copy()
    decision_rows: list[dict[str, object]] = []
    for row in enriched.to_dict("records"):
        context = _moneyline_expression_context(row)
        side_expression = str(context["side_expression"])
        pick_side = str(context["pick_side"])
        preferred_expression = str(row.get("preferred_market_expression", "") or "")
        preferred_is_priced = pd.notna(row.get("preferred_market_american_odds"))
        use_alternative = str(row.get("value_expression_winner", "") or "") == "alternative_market" and preferred_is_priced

        chosen_expression = preferred_expression if use_alternative else side_expression
        chosen_odds = row.get("preferred_market_american_odds") if use_alternative else context["side_market_american_odds"]
        chosen_prob = row.get("preferred_market_projected_prob") if use_alternative else context["side_market_projected_prob"]
        chosen_implied = row.get("preferred_market_implied_prob") if use_alternative else context["side_market_implied_prob"]
        chosen_edge = row.get("preferred_market_edge") if use_alternative else context["side_market_edge"]

        runner_up_expression = ""
        runner_up_odds: object = pd.NA
        runner_up_prob: object = pd.NA
        runner_up_implied: object = pd.NA
        runner_up_edge: object = pd.NA
        if use_alternative:
            runner_up_expression = side_expression
            runner_up_odds = context["side_market_american_odds"]
            runner_up_prob = context["side_market_projected_prob"]
            runner_up_implied = context["side_market_implied_prob"]
            runner_up_edge = context["side_market_edge"]
        elif preferred_expression and preferred_expression != side_expression:
            runner_up_expression = preferred_expression
            runner_up_odds = row.get("preferred_market_american_odds")
            runner_up_prob = row.get("preferred_market_projected_prob")
            runner_up_implied = row.get("preferred_market_implied_prob")
            runner_up_edge = row.get("preferred_market_edge")

        selection_context = _selected_value(row, pick_side, str(row.get("fighter_a", "")), str(row.get("fighter_b", "")))
        fragility = calculate_fragility_metrics(
            short_notice_flag=selection_context["selection_short_notice_flag"],
            short_notice_acceptance_flag=selection_context["selection_short_notice_acceptance_flag"],
            short_notice_success_flag=selection_context["selection_short_notice_success_flag"],
            days_since_last_fight=_safe_float(
                _row_value(
                    row,
                    "fighter_a_days_since_last_fight" if pick_side == "fighter_a" else "fighter_b_days_since_last_fight",
                    "a_days_since_last_fight" if pick_side == "fighter_a" else "b_days_since_last_fight",
                    default=999.0,
                ),
                999.0,
            ),
            ufc_fight_count=_safe_float(
                _row_value(
                    row,
                    "fighter_a_ufc_fight_count" if pick_side == "fighter_a" else "fighter_b_ufc_fight_count",
                    "a_ufc_fight_count" if pick_side == "fighter_a" else "b_ufc_fight_count",
                    default=0.0,
                ),
                0.0,
            ),
            ufc_debut_flag=_safe_float(
                _row_value(
                    row,
                    "fighter_a_ufc_debut_flag" if pick_side == "fighter_a" else "fighter_b_ufc_debut_flag",
                    "a_ufc_debut_flag" if pick_side == "fighter_a" else "b_ufc_debut_flag",
                    default=0.0,
                ),
                0.0,
            ),
            injury_concern_flag=selection_context["selection_injury_concern_flag"],
            weight_cut_concern_flag=selection_context["selection_weight_cut_concern_flag"],
            replacement_fighter_flag=selection_context["selection_replacement_fighter_flag"],
            travel_disadvantage_flag=selection_context["selection_travel_disadvantage_flag"],
            camp_change_flag=selection_context["selection_camp_change_flag"],
            gym_changed_flag=selection_context["selection_gym_changed_flag"],
            fallback_used=selection_context["selection_fallback_used"],
            data_quality=_safe_float(row.get("data_quality", 1.0), 1.0),
            market_blend_weight=_safe_float(row.get("market_blend_weight", 0.0), 0.0),
            consensus_count=_safe_float(row.get("market_consensus_bookmaker_count", 0.0), 0.0),
            consensus_price_edge=_safe_float(row.get("price_edge_vs_consensus", 0.0), 0.0),
        )
        hard_gate_reason = ""
        tracked_market_key = "moneyline"
        if "inside distance" in chosen_expression.lower():
            tracked_market_key = "inside_distance"
        elif "by decision" in chosen_expression.lower():
            tracked_market_key = "by_decision"
        elif "fight goes to decision" == chosen_expression.lower():
            tracked_market_key = "fight_goes_to_decision"
        elif "fight doesn't go to decision" == chosen_expression.lower():
            tracked_market_key = "fight_doesnt_go_to_decision"
        if tracked_market_key in {"inside_distance", "by_decision"} and (
            float(selection_context["selection_fallback_used"]) >= 1.0 or _safe_float(row.get("data_quality", 1.0), 1.0) < 0.85
        ):
            hard_gate_reason = "prop_data_quality_gate"

        decision_rows.append(
            {
                "selection": pick_side,
                "selection_name": selection_context["selection_name"],
                "line_movement_toward_fighter": _selected_line_movement(row, pick_side),
                "chosen_value_expression": chosen_expression,
                "expression_pick_source": "alternative_market" if use_alternative else "side_market",
                "chosen_expression_odds": chosen_odds,
                "chosen_expression_prob": chosen_prob,
                "chosen_expression_implied_prob": chosen_implied,
                "chosen_expression_edge": chosen_edge,
                "runner_up_expression": runner_up_expression,
                "runner_up_odds": runner_up_odds,
                "runner_up_prob": runner_up_prob,
                "runner_up_implied_prob": runner_up_implied,
                "runner_up_edge": runner_up_edge,
                "expression_edge_gap": (
                    float(chosen_edge) - float(runner_up_edge)
                    if pd.notna(chosen_edge) and pd.notna(runner_up_edge)
                    else pd.NA
                ),
                "hard_gate_reason": hard_gate_reason,
                **selection_context,
                **fragility,
            }
        )

    enriched = pd.concat([enriched, pd.DataFrame(decision_rows, index=enriched.index)], axis=1)
    enriched = apply_historical_overlays(enriched, db_path=db_path)
    return enriched


def build_skipped_fights_report(raw_odds: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"event_id", "event_name", "start_time", "fighter_a", "fighter_b", "selection", "american_odds"}
    if not required_columns.issubset(raw_odds.columns):
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "start_time",
                "fighter_a",
                "fighter_b",
                "skip_reason",
                "priced_rows",
            ]
        )

    working = raw_odds.copy()
    for column in ["event_id", "event_name", "start_time", "fighter_a", "fighter_b", "selection"]:
        working[column] = working[column].astype(str).str.strip()
    working["american_odds_numeric"] = pd.to_numeric(working["american_odds"], errors="coerce")

    skipped_rows: list[dict[str, object]] = []
    group_columns = ["event_id", "event_name", "start_time", "fighter_a", "fighter_b"]
    for keys, fight_rows in working.groupby(group_columns, dropna=False):
        fighter_a_priced = bool(
            fight_rows.loc[fight_rows["selection"] == "fighter_a", "american_odds_numeric"].notna().any()
        )
        fighter_b_priced = bool(
            fight_rows.loc[fight_rows["selection"] == "fighter_b", "american_odds_numeric"].notna().any()
        )
        if fighter_a_priced and fighter_b_priced:
            continue
        if not fighter_a_priced and not fighter_b_priced:
            skip_reason = "no_priced_odds"
        elif not fighter_a_priced:
            skip_reason = "missing_fighter_a_price"
        else:
            skip_reason = "missing_fighter_b_price"
        skipped_rows.append(
            {
                "event_id": keys[0],
                "event_name": keys[1],
                "start_time": keys[2],
                "fighter_a": keys[3],
                "fighter_b": keys[4],
                "skip_reason": skip_reason,
                "priced_rows": int(fight_rows["american_odds_numeric"].notna().sum()),
            }
        )

    return pd.DataFrame(skipped_rows).sort_values(by=["start_time", "fighter_a"]).reset_index(drop=True) if skipped_rows else pd.DataFrame(
        columns=["event_id", "event_name", "start_time", "fighter_a", "fighter_b", "skip_reason", "priced_rows"]
    )


def print_report_summary(report: pd.DataFrame, skipped_report: pd.DataFrame | None = None) -> None:
    if report.empty:
        print("No active odds rows available to build a fight-week report.")
        if skipped_report is not None and not skipped_report.empty:
            print(f"Skipped fights: {len(skipped_report)}")
        print()
        return
    skipped_count = 0 if skipped_report is None else len(skipped_report)
    print(f"Fight-week report summary: {len(report)} fights | {skipped_count} skipped")
    print()
    for row in report.itertuples(index=False):
        best_side, pick_side, best_prob, best_edge = _best_pick_context(row)
        action_label, action_reason = _actionable_expression(row)
        drivers = _driver_labels(row, pick_side)
        risks = _risk_labels(row, pick_side)
        camp_notes = [note for note in [_fighter_camp_note(row, "fighter_a"), _fighter_camp_note(row, "fighter_b")] if note]
        chosen_odds = getattr(row, "chosen_expression_odds", pd.NA)
        if pd.isna(chosen_odds):
            chosen_odds = getattr(row, "preferred_market_american_odds", pd.NA)
        if pd.isna(chosen_odds):
            chosen_odds = getattr(row, "fighter_a_current_american_odds", pd.NA) if pick_side == "fighter_a" else getattr(row, "fighter_b_current_american_odds", pd.NA)
        chosen_prob_value = getattr(row, "chosen_expression_prob", pd.NA)
        if pd.isna(chosen_prob_value):
            chosen_prob_value = best_prob
        chosen_implied_value = getattr(row, "chosen_expression_implied_prob", pd.NA)
        if pd.isna(chosen_implied_value) and pd.notna(chosen_odds):
            chosen_implied_value = implied_probability(int(float(chosen_odds)))
        chosen_edge_value = getattr(row, "chosen_expression_edge", pd.NA)
        if pd.isna(chosen_edge_value) and pd.notna(chosen_prob_value) and pd.notna(chosen_implied_value):
            chosen_edge_value = float(chosen_prob_value) - float(chosen_implied_value)
        print(
            f"- {row.fighter_a} vs {row.fighter_b} | "
            f"model {_format_percent(row.fighter_a_model_win_prob)} / {_format_percent(row.fighter_b_model_win_prob)} | "
            f"best edge {best_side} {_format_percent(best_edge)} | "
            f"{action_label}"
        )
        print(f"  Action: {action_reason}")
        print(f"  Decision: {_decision_summary(row)}")
        print(
            f"  Numbers: line {_format_decimal(chosen_odds)} | model {_format_percent(chosen_prob_value)} | "
            f"implied {_format_percent(chosen_implied_value)} | edge {_format_percent(chosen_edge_value)}"
        )
        print(f"  History: {_history_summary(row)}")
        print(f"  Fragility: {_fragility_summary(row)}")
        print(f"  Drivers: {', '.join(drivers[:4]) if drivers else 'No single feature cleared the driver threshold.'}")
        if camp_notes:
            print(f"  Camp: {' | '.join(camp_notes)}")
        if risks:
            print(f"  Risks: {', '.join(risks[:4])}")
        print(
            f"  Prices: {row.fighter_a} {_format_decimal(row.fighter_a_current_american_odds)} | "
            f"{row.fighter_b} {_format_decimal(row.fighter_b_current_american_odds)}"
        )
    if skipped_report is not None and not skipped_report.empty:
        print()
        print("Skipped fights")
        for row in skipped_report.itertuples(index=False):
            print(f"- {row.fighter_a} vs {row.fighter_b} | {row.skip_reason}")
    print()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    load_dotenv(ROOT / ".env")
    side_model_path = Path(args.side_model) if args.side_model else default_side_model_path(ROOT)
    side_model_bundle = load_side_model(side_model_path) if side_model_path.exists() else None
    confidence_model_path = Path(args.confidence_model) if args.confidence_model else default_confidence_model_path(ROOT)
    confidence_model_bundle = load_confidence_model(confidence_model_path) if confidence_model_path.exists() else None
    raw_odds = load_odds_csv(args.odds)
    skipped_report = build_skipped_fights_report(raw_odds)
    report = build_fight_week_report(
        args.odds,
        args.fighter_stats,
        bestfightodds_event_urls=args.bestfightodds_event_url,
        odds_api_bookmaker=args.odds_api_bookmaker,
        db_path=args.db,
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    if args.skipped_output:
        skipped_output_path = Path(args.skipped_output)
        skipped_output_path.parent.mkdir(parents=True, exist_ok=True)
        skipped_report.to_csv(skipped_output_path, index=False)
    if not args.quiet:
        print_report_summary(report, skipped_report)
        print(f"\nSaved fight-week report to {output_path}")
        if args.skipped_output:
            print(f"Saved skipped fights report to {args.skipped_output}")


if __name__ == "__main__":
    main()
