from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ev import american_to_decimal, implied_probability
from scripts.build_fight_week_report import (
    _colorize,
    _driver_labels,
    _format_decimal,
    _format_gym_tier_label,
    _risk_labels,
)


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
LEAN_STRENGTH_ORDER = ["Strong Lean", "Lean", "Slight Lean", "Coin Flip"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a side-lean board from the fight-week report.")
    parser.add_argument("--input", required=True, help="Fight-week report CSV path.")
    parser.add_argument("--output", required=True, help="Lean-board CSV output path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def _strip_ansi(value: object) -> str:
    return ANSI_ESCAPE_RE.sub("", _safe_text(value))


def _safe_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    return int(round(float(value)))


def _plain_tier_label(tier: object) -> str:
    normalized = _safe_text(tier).upper()
    return f"{normalized}-tier" if normalized else "unranked"


def _colored_tier_label(tier: object) -> str:
    if not _safe_text(tier):
        return "unranked"
    label = _format_gym_tier_label(tier)
    return label if label else "unranked"


def _camp_label(fighter: str, gym_name: str, gym_tier: str) -> str:
    label = gym_name or "unknown camp"
    tier = _plain_tier_label(gym_tier)
    return f"{fighter} {label} ({tier})"


def _selected_value(row: dict[str, object], pick_side: str, fighter_a_key: str, fighter_b_key: str, default: object = pd.NA) -> object:
    key = fighter_a_key if pick_side == "fighter_a" else fighter_b_key
    return row.get(key, default)


def _oriented_metric(row: dict[str, object], column: str, pick_side: str, default: float = 0.0) -> float:
    value = _safe_float(row.get(column, default), default)
    return value if pick_side == "fighter_a" else -value


def _fair_american_odds(probability: float) -> int:
    probability = min(max(float(probability), 0.01), 0.99)
    if probability >= 0.5:
        return int(round(-(probability / (1 - probability)) * 100))
    return int(round(((1 - probability) / probability) * 100))


def _lean_strength(edge: float, confidence: float, fragility_bucket: str) -> str:
    if edge >= 0.08 and confidence >= 0.72 and fragility_bucket == "low":
        return "Strong Lean"
    if edge >= 0.05 and confidence >= 0.64:
        return "Lean"
    if edge >= 0.02:
        return "Slight Lean"
    return "Coin Flip"


def _lean_action(edge: float, line_movement: float, fragility_bucket: str, confidence: float) -> str:
    if edge < 0.02:
        return "Pass"
    if confidence < 0.55 or fragility_bucket == "high":
        return "Lean only"
    if edge >= 0.03 and line_movement > 0.01:
        return "Wait for a better number"
    if edge >= 0.05 and confidence >= 0.58 and fragility_bucket != "high":
        return "Bet now"
    return "Lean only"


def _format_record(wins: int, losses: int, draws: int) -> str:
    if draws > 0:
        return f"{wins}-{losses}-{draws}"
    return f"{wins}-{losses}"


def _style_label(
    stance: str,
    strike_margin: float,
    grappling_rate: float,
    control_avg: float,
    ko_win_rate: float,
    submission_win_rate: float,
    decision_rate: float,
) -> str:
    stance_label = stance.title() if stance else "Unknown stance"
    if control_avg >= 4.0 or grappling_rate >= 2.2:
        base = "Control grappler"
    elif submission_win_rate >= 0.35 and grappling_rate >= 1.2:
        base = "Submission grappler"
    elif ko_win_rate >= 0.45 and strike_margin >= 1.0:
        base = "Power striker"
    elif strike_margin >= 1.0:
        base = "Volume striker"
    elif decision_rate >= 0.55 and abs(strike_margin) < 0.8 and grappling_rate < 1.2:
        base = "Point fighter"
    else:
        base = "All-rounder"
    return f"{base} | {stance_label}"


def _resolved_style_label(
    row: dict[str, object],
    pick_side: str,
    *,
    fighter_a_key: str,
    fighter_b_key: str,
    stance: str,
    strike_margin: float,
    grappling_rate: float,
    control_avg: float,
    ko_win_rate: float,
    submission_win_rate: float,
    decision_rate: float,
) -> str:
    history_style = _safe_text(_selected_value(row, pick_side, fighter_a_key, fighter_b_key, default=""))
    if history_style:
        return history_style
    return _style_label(
        stance,
        strike_margin,
        grappling_rate,
        control_avg,
        ko_win_rate,
        submission_win_rate,
        decision_rate,
    )


def _style_color(style_label: str) -> str:
    normalized = style_label.lower()
    if "grappler" in normalized:
        return "green"
    if "striker" in normalized:
        return "red"
    if "point fighter" in normalized:
        return "yellow"
    return "cyan"


def _highlight_style(style_label: str) -> str:
    return _colorize(style_label, _style_color(style_label))


def _highlight_strength(value: str) -> str:
    mapping = {
        "Strong Lean": "cyan",
        "Lean": "green",
        "Slight Lean": "yellow",
        "Coin Flip": "gray",
    }
    return _colorize(value, mapping.get(value, "gray"))


def _highlight_action(value: str) -> str:
    mapping = {
        "Bet now": "green",
        "Wait for a better number": "yellow",
        "Lean only": "cyan",
        "Pass": "gray",
    }
    return _colorize(value, mapping.get(value, "gray"))


def _highlight_edge(edge: float) -> str:
    if edge >= 0.08:
        color = "green"
    elif edge >= 0.03:
        color = "cyan"
    elif edge >= 0:
        color = "yellow"
    else:
        color = "red"
    return _colorize(f"{edge:+.1%}", color)


def _highlight_risk(bucket: str) -> str:
    normalized = _safe_text(bucket, "low").lower()
    mapping = {"low": "green", "medium": "yellow", "high": "red"}
    return _colorize(normalized.upper(), mapping.get(normalized, "gray"))


def _highlight_confidence(confidence: float) -> str:
    if confidence >= 0.8:
        color = "green"
    elif confidence >= 0.68:
        color = "cyan"
    elif confidence >= 0.58:
        color = "yellow"
    else:
        color = "red"
    return _colorize(f"{confidence:.2f}", color)


def _highlight_title(title: str, color: str = "cyan") -> str:
    return _colorize(title, color)


def _highlight_fight_name(fight: str) -> str:
    return _colorize(fight, "cyan")


def _highlight_metric(label: str, value: float, strong_cutoff: float, light_cutoff: float) -> str:
    color = "gray"
    if value >= strong_cutoff:
        color = "green"
    elif value >= light_cutoff:
        color = "cyan"
    elif value <= -strong_cutoff:
        color = "red"
    elif value <= -light_cutoff:
        color = "yellow"
    return _colorize(label, color)


def _age_note(lean_side: str, pick_age: float, opponent_age: float) -> str:
    if pick_age <= 0 or opponent_age <= 0:
        return "Age: limited data"
    difference = abs(pick_age - opponent_age)
    if difference < 0.5:
        return f"Age: near even ({pick_age:.1f} vs {opponent_age:.1f})"
    direction = "younger" if pick_age < opponent_age else "older"
    return f"Age: {lean_side} is {direction} by {difference:.1f}y ({pick_age:.1f} vs {opponent_age:.1f})"


def _layoff_note(lean_side: str, opponent_side: str, pick_days: float, opponent_days: float) -> str:
    if pick_days >= 999 or opponent_days >= 999:
        return "Layoff: limited data"
    difference = abs(pick_days - opponent_days)
    if difference < 21:
        return f"Layoff: similar prep windows ({_safe_int(pick_days)}d vs {_safe_int(opponent_days)}d)"
    if pick_days < opponent_days:
        return (
            f"Layoff: {lean_side} is more active by {difference:.0f}d "
            f"({_safe_int(pick_days)}d vs {_safe_int(opponent_days)}d)"
        )
    return (
        f"Layoff: {opponent_side} is more active by {difference:.0f}d "
        f"({_safe_int(pick_days)}d vs {_safe_int(opponent_days)}d)"
    )


def _experience_note(lean_side: str, pick_ufc_fights: float, opponent_ufc_fights: float) -> str:
    if pick_ufc_fights <= 0 and opponent_ufc_fights <= 0:
        return "UFC sample: limited data"
    difference = pick_ufc_fights - opponent_ufc_fights
    if abs(difference) < 1:
        return f"UFC sample: similar ({_safe_int(pick_ufc_fights)} vs {_safe_int(opponent_ufc_fights)})"
    sign = "+" if difference > 0 else "-"
    return (
        f"UFC sample: {lean_side} {sign}{abs(difference):.0f} UFC fights "
        f"({_safe_int(pick_ufc_fights)} vs {_safe_int(opponent_ufc_fights)})"
    )


def _debut_note(lean_side: str, opponent_side: str, pick_debut_flag: bool, opponent_debut_flag: bool) -> str:
    if pick_debut_flag and opponent_debut_flag:
        return "Debut: both fighters making UFC debut"
    if pick_debut_flag:
        return f"Debut: {lean_side} is making a UFC debut"
    if opponent_debut_flag:
        return f"Debut: {opponent_side} is making a UFC debut"
    return "Debut: neither side making UFC debut"


def _watch_for(
    lean_side: str,
    opponent_side: str,
    striking_diff: float,
    matchup_striking_edge: float,
    grappling_diff: float,
    matchup_grappling_edge: float,
    control_diff: float,
    reach_advantage: float,
    height_advantage: float,
    pick_days: float,
    opponent_days: float,
    pick_debut_flag: bool,
    opponent_debut_flag: bool,
    risk_labels: list[str],
) -> str:
    notes: list[str] = []

    grappling_pressure = max(grappling_diff, matchup_grappling_edge, control_diff / 1.8)
    striking_pressure = max(striking_diff / 2.0, matchup_striking_edge, reach_advantage / 2.0, height_advantage / 2.0)

    if grappling_pressure >= striking_pressure and grappling_pressure >= 0.7:
        notes.append(f"Watch whether {lean_side} can force wrestling exchanges and bank control time")
        notes.append(f"If {opponent_side} keeps it standing early, the edge tightens fast")
    elif striking_pressure >= 0.7:
        notes.append(f"Watch whether {lean_side} wins range and lands the cleaner shots")
        notes.append(f"If {opponent_side} closes distance and turns it into a clinch fight, the read gets closer")
    else:
        notes.append("Watch the first round pace closely because the phase edge is not overwhelming")

    if opponent_debut_flag:
        notes.append(f"{opponent_side} is making a UFC debut, so composure and pace matter")
    elif pick_debut_flag:
        notes.append(f"{lean_side} is making a UFC debut, which adds volatility despite the model side")
    elif pick_days < 999 and opponent_days < 999 and abs(pick_days - opponent_days) >= 120:
        active_side = lean_side if pick_days < opponent_days else opponent_side
        notes.append(f"Activity leans toward {active_side} off the shorter layoff")

    if risk_labels:
        notes.append(f"Main caution: {risk_labels[0]}")

    return ". ".join(note.rstrip(".") for note in notes[:3]) + "."


def _empty_board() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "report_order",
            "event_name",
            "fight",
            "lean_side",
            "opponent_side",
            "lean_strength",
            "lean_action",
            "lean_prob",
            "fair_american_odds",
            "fair_decimal_odds",
            "current_american_odds",
            "current_decimal_odds",
            "edge",
            "market_tax",
            "line_movement_toward_pick",
            "model_confidence",
            "fragility_bucket",
            "pick_gym_name",
            "pick_gym_tier",
            "opponent_gym_name",
            "opponent_gym_tier",
            "camp_summary",
            "striking_diff",
            "matchup_striking_edge",
            "grappling_diff",
            "matchup_grappling_edge",
            "control_diff",
            "matchup_control_edge",
            "reach_advantage",
            "height_advantage",
            "pick_age_years",
            "opponent_age_years",
            "pick_days_since_last_fight",
            "opponent_days_since_last_fight",
            "pick_ufc_fight_count",
            "opponent_ufc_fight_count",
            "pick_ufc_debut_flag",
            "opponent_ufc_debut_flag",
            "pick_record",
            "opponent_record",
            "pick_stance",
            "opponent_stance",
            "pick_style",
            "opponent_style",
            "context_summary",
            "top_reasons",
            "risk_flags",
            "watch_for",
        ]
    )


def build_lean_board(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return _empty_board()

    rows: list[dict[str, object]] = []
    for report_order, row in enumerate(report.to_dict("records"), start=1):
        fighter_a_prob = _safe_float(row.get("fighter_a_model_win_prob"), 0.5)
        fighter_b_prob = _safe_float(row.get("fighter_b_model_win_prob"), 0.5)
        pick_side = "fighter_a" if fighter_a_prob >= fighter_b_prob else "fighter_b"
        lean_side = _safe_text(row["fighter_a"] if pick_side == "fighter_a" else row["fighter_b"])
        opponent_side = _safe_text(row["fighter_b"] if pick_side == "fighter_a" else row["fighter_a"])
        lean_prob = fighter_a_prob if pick_side == "fighter_a" else fighter_b_prob

        current_odds = int(
            row["fighter_a_current_american_odds"] if pick_side == "fighter_a" else row["fighter_b_current_american_odds"]
        )
        implied = implied_probability(current_odds)
        edge = lean_prob - implied
        current_other_odds = int(
            row["fighter_b_current_american_odds"] if pick_side == "fighter_a" else row["fighter_a_current_american_odds"]
        )
        market_tax = implied_probability(current_odds) + implied_probability(current_other_odds) - 1
        open_odds = row.get(
            "fighter_a_open_american_odds" if pick_side == "fighter_a" else "fighter_b_open_american_odds",
            pd.NA,
        )
        line_movement = implied - implied_probability(int(open_odds)) if pd.notna(open_odds) else 0.0
        fragility_bucket = _safe_text(row.get("fragility_bucket", "low"), "low").lower()
        confidence = _safe_float(row.get("model_confidence"), 0.5)

        reasons = [_strip_ansi(label) for label in _driver_labels(pd.Series(row), pick_side)]
        risk_labels = [_strip_ansi(label) for label in _risk_labels(pd.Series(row), pick_side)]

        striking_diff = _oriented_metric(row, "strike_margin_diff", pick_side)
        matchup_striking_edge = _oriented_metric(row, "matchup_striking_edge", pick_side)
        grappling_diff = _oriented_metric(row, "grappling_diff", pick_side)
        matchup_grappling_edge = _oriented_metric(row, "matchup_grappling_edge", pick_side)
        control_diff = _oriented_metric(row, "control_diff", pick_side)
        matchup_control_edge = _oriented_metric(row, "matchup_control_edge", pick_side)
        reach_advantage = _oriented_metric(row, "fighter_a_reach_advantage_in", pick_side)
        height_advantage = _oriented_metric(row, "fighter_a_height_advantage_in", pick_side)

        pick_gym_name = _safe_text(
            _selected_value(row, pick_side, "fighter_a_gym_name", "fighter_b_gym_name", default="")
        )
        pick_gym_tier = _safe_text(
            _selected_value(row, pick_side, "fighter_a_gym_tier", "fighter_b_gym_tier", default="")
        ).upper()
        opponent_gym_name = _safe_text(
            _selected_value(row, pick_side, "fighter_b_gym_name", "fighter_a_gym_name", default="")
        )
        opponent_gym_tier = _safe_text(
            _selected_value(row, pick_side, "fighter_b_gym_tier", "fighter_a_gym_tier", default="")
        ).upper()

        pick_age = _safe_float(_selected_value(row, pick_side, "fighter_a_age_years", "fighter_b_age_years", default=0.0))
        opponent_age = _safe_float(
            _selected_value(row, pick_side, "fighter_b_age_years", "fighter_a_age_years", default=0.0)
        )
        pick_days = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_a_days_since_last_fight",
                "fighter_b_days_since_last_fight",
                default=999.0,
            ),
            999.0,
        )
        opponent_days = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_b_days_since_last_fight",
                "fighter_a_days_since_last_fight",
                default=999.0,
            ),
            999.0,
        )
        pick_ufc_fights = _safe_float(
            _selected_value(row, pick_side, "fighter_a_ufc_fight_count", "fighter_b_ufc_fight_count", default=0.0)
        )
        opponent_ufc_fights = _safe_float(
            _selected_value(row, pick_side, "fighter_b_ufc_fight_count", "fighter_a_ufc_fight_count", default=0.0)
        )
        pick_debut_flag = _safe_float(
            _selected_value(row, pick_side, "fighter_a_ufc_debut_flag", "fighter_b_ufc_debut_flag", default=0.0)
        ) >= 1
        opponent_debut_flag = _safe_float(
            _selected_value(row, pick_side, "fighter_b_ufc_debut_flag", "fighter_a_ufc_debut_flag", default=0.0)
        ) >= 1
        pick_sig_landed = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_a_sig_strikes_landed_per_min",
                "fighter_b_sig_strikes_landed_per_min",
                default=0.0,
            )
        )
        pick_sig_absorbed = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_a_sig_strikes_absorbed_per_min",
                "fighter_b_sig_strikes_absorbed_per_min",
                default=0.0,
            )
        )
        opponent_sig_landed = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_b_sig_strikes_landed_per_min",
                "fighter_a_sig_strikes_landed_per_min",
                default=0.0,
            )
        )
        opponent_sig_absorbed = _safe_float(
            _selected_value(
                row,
                pick_side,
                "fighter_b_sig_strikes_absorbed_per_min",
                "fighter_a_sig_strikes_absorbed_per_min",
                default=0.0,
            )
        )
        pick_strike_margin = pick_sig_landed - pick_sig_absorbed
        opponent_strike_margin = opponent_sig_landed - opponent_sig_absorbed
        pick_record = _format_record(
            _safe_int(_selected_value(row, pick_side, "fighter_a_record_wins", "fighter_b_record_wins", default=0)),
            _safe_int(_selected_value(row, pick_side, "fighter_a_record_losses", "fighter_b_record_losses", default=0)),
            _safe_int(_selected_value(row, pick_side, "fighter_a_record_draws", "fighter_b_record_draws", default=0)),
        )
        opponent_record = _format_record(
            _safe_int(_selected_value(row, pick_side, "fighter_b_record_wins", "fighter_a_record_wins", default=0)),
            _safe_int(_selected_value(row, pick_side, "fighter_b_record_losses", "fighter_a_record_losses", default=0)),
            _safe_int(_selected_value(row, pick_side, "fighter_b_record_draws", "fighter_a_record_draws", default=0)),
        )
        pick_stance = _safe_text(_selected_value(row, pick_side, "fighter_a_stance", "fighter_b_stance", default=""))
        opponent_stance = _safe_text(_selected_value(row, pick_side, "fighter_b_stance", "fighter_a_stance", default=""))
        pick_style = _resolved_style_label(
            row,
            pick_side,
            fighter_a_key="fighter_a_history_style_label",
            fighter_b_key="fighter_b_history_style_label",
            stance=pick_stance,
            strike_margin=pick_strike_margin,
            grappling_rate=_safe_float(_selected_value(row, pick_side, "fighter_a_recent_grappling_rate", "fighter_b_recent_grappling_rate", default=0.0)),
            control_avg=_safe_float(_selected_value(row, pick_side, "fighter_a_control_avg", "fighter_b_control_avg", default=0.0)),
            ko_win_rate=_safe_float(_selected_value(row, pick_side, "fighter_a_ko_win_rate", "fighter_b_ko_win_rate", default=0.0)),
            submission_win_rate=_safe_float(_selected_value(row, pick_side, "fighter_a_submission_win_rate", "fighter_b_submission_win_rate", default=0.0)),
            decision_rate=_safe_float(_selected_value(row, pick_side, "fighter_a_decision_rate", "fighter_b_decision_rate", default=0.0)),
        )
        opponent_style = _resolved_style_label(
            row,
            pick_side,
            fighter_a_key="fighter_b_history_style_label",
            fighter_b_key="fighter_a_history_style_label",
            stance=opponent_stance,
            strike_margin=opponent_strike_margin,
            grappling_rate=_safe_float(_selected_value(row, pick_side, "fighter_b_recent_grappling_rate", "fighter_a_recent_grappling_rate", default=0.0)),
            control_avg=_safe_float(_selected_value(row, pick_side, "fighter_b_control_avg", "fighter_a_control_avg", default=0.0)),
            ko_win_rate=_safe_float(_selected_value(row, pick_side, "fighter_b_ko_win_rate", "fighter_a_ko_win_rate", default=0.0)),
            submission_win_rate=_safe_float(_selected_value(row, pick_side, "fighter_b_submission_win_rate", "fighter_a_submission_win_rate", default=0.0)),
            decision_rate=_safe_float(_selected_value(row, pick_side, "fighter_b_decision_rate", "fighter_a_decision_rate", default=0.0)),
        )

        context_parts = [
            _age_note(lean_side, pick_age, opponent_age),
            _layoff_note(lean_side, opponent_side, pick_days, opponent_days),
            _experience_note(lean_side, pick_ufc_fights, opponent_ufc_fights),
            _debut_note(lean_side, opponent_side, pick_debut_flag, opponent_debut_flag),
        ]
        context_summary = " | ".join(part for part in context_parts if part)

        rows.append(
            {
                "report_order": report_order,
                "event_name": row.get("event_name", ""),
                "fight": f"{row['fighter_a']} vs {row['fighter_b']}",
                "lean_side": lean_side,
                "opponent_side": opponent_side,
                "lean_strength": _lean_strength(edge, confidence, fragility_bucket),
                "lean_action": _lean_action(edge, line_movement, fragility_bucket, confidence),
                "lean_prob": round(lean_prob, 4),
                "fair_american_odds": _fair_american_odds(lean_prob),
                "fair_decimal_odds": round(float(american_to_decimal(_fair_american_odds(lean_prob))), 2),
                "current_american_odds": current_odds,
                "current_decimal_odds": round(float(american_to_decimal(current_odds)), 2),
                "edge": round(edge, 4),
                "market_tax": round(market_tax, 4),
                "line_movement_toward_pick": round(line_movement, 4),
                "model_confidence": round(confidence, 3),
                "fragility_bucket": fragility_bucket,
                "pick_gym_name": pick_gym_name,
                "pick_gym_tier": pick_gym_tier,
                "opponent_gym_name": opponent_gym_name,
                "opponent_gym_tier": opponent_gym_tier,
                "camp_summary": (
                    f"{_camp_label(lean_side, pick_gym_name, pick_gym_tier)} vs "
                    f"{_camp_label(opponent_side, opponent_gym_name, opponent_gym_tier)}"
                ),
                "striking_diff": round(striking_diff, 3),
                "matchup_striking_edge": round(matchup_striking_edge, 3),
                "grappling_diff": round(grappling_diff, 3),
                "matchup_grappling_edge": round(matchup_grappling_edge, 3),
                "control_diff": round(control_diff, 3),
                "matchup_control_edge": round(matchup_control_edge, 3),
                "reach_advantage": round(reach_advantage, 3),
                "height_advantage": round(height_advantage, 3),
                "pick_age_years": round(pick_age, 2),
                "opponent_age_years": round(opponent_age, 2),
                "pick_days_since_last_fight": _safe_int(pick_days, 999),
                "opponent_days_since_last_fight": _safe_int(opponent_days, 999),
                "pick_ufc_fight_count": _safe_int(pick_ufc_fights),
                "opponent_ufc_fight_count": _safe_int(opponent_ufc_fights),
                "pick_ufc_debut_flag": int(pick_debut_flag),
                "opponent_ufc_debut_flag": int(opponent_debut_flag),
                "pick_record": pick_record,
                "opponent_record": opponent_record,
                "pick_stance": pick_stance,
                "opponent_stance": opponent_stance,
                "pick_style": pick_style,
                "opponent_style": opponent_style,
                "context_summary": context_summary,
                "top_reasons": ", ".join(reasons[:3]) if reasons else "No clear driver edge",
                "risk_flags": ", ".join(risk_labels[:3]) if risk_labels else "none",
                "watch_for": _watch_for(
                    lean_side=lean_side,
                    opponent_side=opponent_side,
                    striking_diff=striking_diff,
                    matchup_striking_edge=matchup_striking_edge,
                    grappling_diff=grappling_diff,
                    matchup_grappling_edge=matchup_grappling_edge,
                    control_diff=control_diff,
                    reach_advantage=reach_advantage,
                    height_advantage=height_advantage,
                    pick_days=pick_days,
                    opponent_days=opponent_days,
                    pick_debut_flag=pick_debut_flag,
                    opponent_debut_flag=opponent_debut_flag,
                    risk_labels=risk_labels,
                ),
            }
        )

    board = pd.DataFrame(rows)
    board["lean_strength"] = pd.Categorical(
        board["lean_strength"],
        categories=LEAN_STRENGTH_ORDER,
        ordered=True,
    )
    return board.sort_values(
        by=["lean_strength", "edge", "model_confidence"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _full_card_rows(board: pd.DataFrame) -> pd.DataFrame:
    if "report_order" not in board.columns:
        return board.copy()
    return board.sort_values(by=["report_order", "fight"], ascending=[True, True]).reset_index(drop=True)


def _best_choice_rows(board: pd.DataFrame, *, max_rows: int | None = None) -> pd.DataFrame:
    if board.empty:
        return board.copy()
    eligible = board.loc[board["lean_action"] != "Pass"].copy()
    if eligible.empty:
        return eligible
    if max_rows is not None:
        eligible = eligible.head(max_rows)
    return eligible.reset_index(drop=True)


def format_full_card_breakdown(board: pd.DataFrame) -> str:
    lines: list[str] = []
    if board.empty:
        lines.append(_highlight_title("Full card read: no fights available.", "cyan"))
        lines.append("")
        return "\n".join(lines)

    ordered = _full_card_rows(board)
    lines.append(_highlight_title(f"Full card read: {len(ordered)} fights", "cyan"))
    lines.append("")
    for _, row in ordered.iterrows():
        pick_gym_name = _safe_text(row.get("pick_gym_name"), "unknown camp")
        opponent_gym_name = _safe_text(row.get("opponent_gym_name"), "unknown camp")
        lines.append(_highlight_fight_name(str(row["fight"])))
        lines.append(
            f"  Lean {_colorize(row['lean_side'], 'green')} | {_highlight_strength(str(row['lean_strength']))} | {_highlight_action(str(row['lean_action']))} | "
            f"model {float(row['lean_prob']):.1%} | fair {_format_decimal(row['fair_american_odds'])} | "
            f"current {_format_decimal(row['current_american_odds'])} | edge {_highlight_edge(float(row['edge']))}"
        )
        lines.append(
            f"  Camps {_colorize(row['lean_side'], 'green')} {pick_gym_name} ({_colored_tier_label(row['pick_gym_tier'])}, record {row['pick_record']}) "
            f"vs {_colorize(row['opponent_side'], 'yellow')} {opponent_gym_name} ({_colored_tier_label(row['opponent_gym_tier'])}, record {row['opponent_record']})"
        )
        lines.append(
            f"  Styles {_colorize(row['lean_side'], 'green')} {_highlight_style(str(row['pick_style']))} "
            f"vs {_colorize(row['opponent_side'], 'yellow')} {_highlight_style(str(row['opponent_style']))}"
        )
        lines.append(
            f"  Matchup {_highlight_metric('striking', float(row['striking_diff']), 1.5, 0.5)} {float(row['striking_diff']):+.2f}/min, "
            f"{_highlight_metric('striking matchup', float(row['matchup_striking_edge']), 1.2, 0.4)} {float(row['matchup_striking_edge']):+.2f}, "
            f"{_highlight_metric('grappling', float(row['grappling_diff']), 1.2, 0.4)} {float(row['grappling_diff']):+.2f}, "
            f"{_highlight_metric('control', float(row['control_diff']), 1.8, 0.7)} {float(row['control_diff']):+.2f}"
        )
        lines.append(
            f"  Context {row['context_summary']} | confidence {_highlight_confidence(float(row['model_confidence']))} | "
            f"fragility {_highlight_risk(str(row['fragility_bucket']))}"
        )
        lines.append(f"  {_colorize('Look for', 'cyan')} {row['watch_for']}")
        if _safe_text(row.get("risk_flags", "none"), "none") != "none":
            lines.append(f"  Risks {_colorize(str(row['risk_flags']), 'yellow')}")
        lines.append("")
    return "\n".join(lines)


def format_best_leans_summary(board: pd.DataFrame, *, max_rows: int | None = None) -> str:
    lines: list[str] = []
    if board.empty:
        lines.append(_highlight_title("Lean board: no fights available.", "green"))
        lines.append("")
        return "\n".join(lines)

    eligible = _best_choice_rows(board, max_rows=max_rows)
    if eligible.empty:
        lines.append(_highlight_title("Lean board: no actionable leans right now.", "green"))
        lines.append("")
        return "\n".join(lines)

    lines.append(_highlight_title(f"Lean board: {len(eligible)} best choices", "green"))
    lines.append("")
    for _, row in eligible.iterrows():
        pick_gym_name = _safe_text(row.get("pick_gym_name"), "unknown camp")
        opponent_gym_name = _safe_text(row.get("opponent_gym_name"), "unknown camp")
        lines.append(
            f"{_highlight_fight_name(str(row['fight']))} | {_colorize(row['lean_side'], 'green')} | {_highlight_strength(str(row['lean_strength']))} | {_highlight_action(str(row['lean_action']))} | "
            f"edge {_highlight_edge(float(row['edge']))} | fair {_format_decimal(row['fair_american_odds'])} | "
            f"current {_format_decimal(row['current_american_odds'])}"
        )
        lines.append(
            f"  Camps {pick_gym_name} ({_colored_tier_label(row['pick_gym_tier'])}, {row['pick_record']}) "
            f"vs {opponent_gym_name} ({_colored_tier_label(row['opponent_gym_tier'])}, {row['opponent_record']})"
        )
        lines.append(
            f"  Styles {_highlight_style(str(row['pick_style']))} vs {_highlight_style(str(row['opponent_style']))}"
        )
        lines.append(f"  {_colorize('Drivers', 'cyan')} {_colorize(str(row['top_reasons']), 'cyan')}")
        lines.append(f"  {_colorize('Watch', 'cyan')} {row['watch_for']}")
        if _safe_text(row.get("risk_flags", "none"), "none") != "none":
            lines.append(f"  {_colorize('Risks', 'yellow')} {_colorize(str(row['risk_flags']), 'yellow')}")
    lines.append("")
    return "\n".join(lines)


def print_lean_board_summary(board: pd.DataFrame) -> None:
    print(format_full_card_breakdown(board), end="")
    print(format_best_leans_summary(board), end="")


def print_compact_lean_board_summary(board: pd.DataFrame, *, max_rows: int = 5) -> None:
    print(format_compact_lean_board_summary(board, max_rows=max_rows), end="")


def format_compact_lean_board_summary(board: pd.DataFrame, *, max_rows: int = 5) -> str:
    return format_best_leans_summary(board, max_rows=max_rows)


def main() -> None:
    args = parse_args()
    report = pd.read_csv(args.input)
    board = build_lean_board(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_path, index=False)
    if not args.quiet:
        print_lean_board_summary(board)
        print(f"Saved lean board to {output_path}")


if __name__ == "__main__":
    main()
