from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency for local env loading
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bankroll.sizing import (
    apply_bankroll_governor,
    bankroll_governor_config_from_env,
    suggested_stake,
)
from data_sources.odds_api import load_odds_csv
from data_sources.storage import save_tracked_picks
from features.fighter_features import build_fight_features, load_fighter_stats
from models.confidence import default_confidence_model_path, load_confidence_model
from models.decision_support import apply_historical_overlays, calculate_fragility_metrics
from models.ev import expected_value, implied_probability
from models.projection import project_fight_probabilities
from models.selective import (
    default_selective_model_path,
    load_selective_model,
    predict_selective_clv_prob,
)
from models.side import default_side_model_path, load_side_model
from models.threshold_policy import (
    default_threshold_policy_path,
    load_threshold_policy,
    resolve_scan_thresholds,
)
from normalization.odds import normalize_odds_frame
from scripts.event_manifest import MODEL_CONTEXT_FLAG_COLUMNS, OPERATOR_CONTEXT_FLAG_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan UFC markets for positive EV selections.")
    parser.add_argument("--input", required=True, help="Path to the odds CSV input.")
    parser.add_argument("--fighter-stats", help="Optional fighter stats CSV used for model projections.")
    parser.add_argument("--output", required=True, help="Path to the output CSV report.")
    parser.add_argument(
        "--fight-report",
        help="Optional fight-week report CSV used to carry preferred market expression into the scan output.",
    )
    parser.add_argument(
        "--shortlist-output",
        help="Optional output path for a tighter shortlist report. Defaults next to --output.",
    )
    parser.add_argument(
        "--board-output",
        help="Optional output path for a compact final betting board. Defaults next to --output.",
    )
    parser.add_argument(
        "--passes-output",
        help="Optional output path for processed plays that were filtered out, with pass reasons.",
    )
    parser.add_argument(
        "--selective-model",
        help="Optional pickle path for the selective CLV model. Defaults to models/selective_clv_model.pkl when present.",
    )
    parser.add_argument(
        "--side-model",
        help="Optional pickle path for the calibrated side model. Defaults to models/side_model.pkl when present.",
    )
    parser.add_argument(
        "--confidence-model",
        help="Optional pickle path for the calibrated confidence model. Defaults to models/confidence_model.pkl when present.",
    )
    parser.add_argument(
        "--threshold-policy",
        help="Optional JSON path for an optimized threshold policy. Defaults to models/threshold_policy.json when present.",
    )
    parser.add_argument("--db", help="Optional SQLite path used to persist tracked picks from this run.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _selection_value(row: object, fighter_a_column: str, fighter_b_column: str) -> float:
    return row[fighter_a_column] if row["selection"] == "fighter_a" else row[fighter_b_column]


def _oriented_feature(row: object, feature_name: str) -> float:
    value = float(row.get(feature_name, 0.0) or 0.0)
    return value if row["selection"] == "fighter_a" else -value


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _format_reasons(signals: list[str], risks: list[str]) -> str:
    if signals and risks:
        return f"{'; '.join(signals)} | Risks: {', '.join(risks)}"
    if signals:
        return "; ".join(signals)
    if risks:
        return f"Risks: {', '.join(risks)}"
    return "No strong supporting signals"


def _default_shortlist_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_shortlist{output_path.suffix}")


def _default_board_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_board{output_path.suffix}")


def _default_passes_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_passes{output_path.suffix}")


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _append_label_value(existing: object, value: str) -> str:
    items = [item.strip() for item in str(existing or "").split(",") if item.strip()]
    if value and value not in items:
        items.append(value)
    return ", ".join(items)


def _tier_rank(tier: object) -> int:
    mapping = {"S": 5, "A": 4, "B": 3, "C": 2, "D": 1}
    return mapping.get(str(tier or "").strip().upper(), 0)


def _selection_camp_summary(row: pd.Series, colorize_tier: bool = False) -> str:
    gym_name = str(row.get("selection_gym_name", "") or "").strip()
    gym_tier = str(row.get("selection_gym_tier", "") or "").strip().upper()
    gym_record = str(row.get("selection_gym_record", "") or "").strip()
    previous_gym_name = str(row.get("selection_previous_gym_name", "") or "").strip()
    gym_changed = float(row.get("selection_gym_changed_flag", 0.0) or 0.0) >= 1
    camp_change = float(row.get("selection_camp_change_flag", 0.0) or 0.0) >= 1

    if not any([gym_name, gym_tier, gym_record, previous_gym_name, gym_changed, camp_change]):
        return ""

    parts: list[str] = []
    if gym_name:
        meta: list[str] = []
        if gym_tier:
            meta.append(_format_gym_tier_label(gym_tier) if colorize_tier else f"{gym_tier}-tier")
        if gym_record:
            meta.append(gym_record)
        gym_label = gym_name
        if meta:
            gym_label += f" ({', '.join(meta)})"
        parts.append(gym_label)
    else:
        parts.append("gym unknown")

    if gym_changed and previous_gym_name:
        parts.append(f"switched from {previous_gym_name}")
    elif camp_change:
        parts.append("camp-change flag")
    return " | ".join(parts)


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _split_items(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


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


def _plain_tier_label(tier: object) -> str:
    normalized = str(tier or "").strip().upper()
    if not normalized:
        return ""
    return f"{normalized}-tier"


def _signal_tier(value: float, thresholds: tuple[float, float, float, float]) -> str:
    for tier, cutoff in zip(("S", "A", "B", "C"), thresholds):
        if value >= cutoff:
            return tier
    return ""


def _tiered_signal_label(label: str, value: float, thresholds: tuple[float, float, float, float]) -> str:
    tier = _signal_tier(value, thresholds)
    if not tier:
        return ""
    return f"{_plain_tier_label(tier)} {label}"


def _colorize_tier_tokens(text: object) -> str:
    value = _safe_text(text)
    if not value:
        return ""
    return re.sub(
        r"\b([SABCD])-tier\b",
        lambda match: _format_gym_tier_label(match.group(1)),
        value,
    )


def _history_summary(row: pd.Series | object) -> str:
    sample_size = int(_safe_float(row.get("historical_sample_size", 0.0), 0.0))
    grade = _safe_text(row.get("historical_overlay_grade", "low_sample"), "low_sample")
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
    roi_pct = row.get("historical_roi_pct", pd.NA)
    avg_clv_delta = row.get("historical_avg_clv_delta", pd.NA)
    win_rate = row.get("historical_win_rate", pd.NA)
    parts = [
        _colorize(f"[{label}]", color),
        f"{sample_size} comps",
        f"ROI {_format_percent((float(roi_pct) / 100) if pd.notna(roi_pct) else pd.NA)}",
        f"CLV {float(avg_clv_delta):+.2f}" if pd.notna(avg_clv_delta) else "CLV n/a",
        f"WR {_format_percent(win_rate)}" if pd.notna(win_rate) else "WR n/a",
    ]
    return " | ".join(parts)


def _fragility_summary(row: pd.Series | object) -> str:
    bucket = _safe_text(row.get("fragility_bucket", "low"), "low").lower()
    score = int(_safe_float(row.get("fragility_score", 0.0), 0.0))
    reasons = _safe_text(row.get("fragility_reasons", ""))
    label = f"[RISK:{bucket.upper()}]"
    color = "red" if bucket == "high" else "yellow" if bucket == "medium" else "green"
    summary = f"{_colorize(label, color)} {score}"
    if reasons:
        summary += f" | {reasons}"
    return summary


def _decision_summary(row: pd.Series | object) -> str:
    chosen_expression = _safe_text(row.get("chosen_value_expression", row.get("selection_name", "")))
    runner_up_expression = _safe_text(row.get("runner_up_expression", ""))
    source = _safe_text(row.get("expression_pick_source", "side_market"), "side_market")
    gap = row.get("expression_edge_gap", pd.NA)
    prefix = _colorize("[PROP]", "cyan") if source == "alternative_market" else _colorize("[SIDE]", "green")
    parts = [prefix, chosen_expression]
    if runner_up_expression:
        gap_text = f" | gap {_format_percent(gap)}" if pd.notna(gap) else ""
        parts.append(f"runner-up {runner_up_expression}{gap_text}")
    return " | ".join(parts)


def _apply_selective_model(
    frame: pd.DataFrame,
    model_path: Path | None,
) -> pd.DataFrame:
    enriched = frame.copy()
    for column, default_value in {
        "bet_quality_score": 0.0,
        "recommended_tier": "",
        "recommended_action": "",
        "support_signals": "",
        "risk_flags": "",
        "why_it_rates_well": "",
        "support_count": 0,
        "risk_flag_count": 0,
    }.items():
        if column not in enriched.columns:
            enriched[column] = default_value
    enriched["selective_clv_prob"] = pd.NA
    if model_path is None or not model_path.exists() or enriched.empty:
        return enriched

    bundle = load_selective_model(model_path)
    probabilities = predict_selective_clv_prob(enriched, bundle)
    enriched["selective_clv_prob"] = probabilities.round(4)

    for idx, probability in probabilities.items():
        score = float(enriched.at[idx, "bet_quality_score"]) if "bet_quality_score" in enriched.columns else 0.0
        tier = str(enriched.at[idx, "recommended_tier"]) if "recommended_tier" in enriched.columns else ""
        action = str(enriched.at[idx, "recommended_action"]) if "recommended_action" in enriched.columns else ""

        if probability >= 0.65:
            score += 8
            enriched.at[idx, "support_signals"] = _append_label_value(enriched.at[idx, "support_signals"], "strong CLV profile")
        elif probability >= 0.55:
            score += 4
            enriched.at[idx, "support_signals"] = _append_label_value(enriched.at[idx, "support_signals"], "positive CLV profile")
        elif probability < 0.40:
            score -= 10
            enriched.at[idx, "risk_flags"] = _append_label_value(enriched.at[idx, "risk_flags"], "poor_clv_profile")
        elif probability < 0.48:
            score -= 5
            enriched.at[idx, "risk_flags"] = _append_label_value(enriched.at[idx, "risk_flags"], "weak_clv_profile")

        if probability < 0.40:
            tier = "C"
            action = "Pass"
        elif tier == "A" and probability < 0.58:
            tier = "B"
            action = "Watchlist"
        elif tier == "B" and probability < 0.50:
            tier = "C"
            action = "Pass"

        score = round(_clamp(score, 0.0, 100.0), 2)
        enriched.at[idx, "bet_quality_score"] = score
        if tier:
            enriched.at[idx, "recommended_tier"] = tier
        if action:
            enriched.at[idx, "recommended_action"] = action
        enriched.at[idx, "why_it_rates_well"] = _format_reasons(
            [item.strip() for item in str(enriched.at[idx, "support_signals"] or "").split(",") if item.strip()][:6],
            [item.strip() for item in str(enriched.at[idx, "risk_flags"] or "").split(",") if item.strip()],
        )

    if "support_signals" in enriched.columns:
        enriched["support_count"] = enriched["support_signals"].astype(str).apply(
            lambda value: len([item for item in value.split(", ") if item])
        )
    if "risk_flags" in enriched.columns:
        enriched["risk_flag_count"] = enriched["risk_flags"].astype(str).apply(
            lambda value: len([item for item in value.split(", ") if item])
        )
    return enriched


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


def _format_summary_line(row: pd.Series, probability_column: str) -> str:
    expression = row.get("chosen_value_expression", row["selection_name"])
    matchup = f"{row['fighter_a']} vs {row['fighter_b']}"
    odds_value = row.get("effective_american_odds", row.get("chosen_expression_odds", row["american_odds"]))
    odds_display = _format_decimal(odds_value)
    parts = [
        matchup,
        f"{expression} ({odds_display})",
        f"prob {_format_percent(float(row.get('effective_projected_prob', row[probability_column])))}",
        f"edge {_format_percent(float(row.get('effective_edge', row['edge'])))}",
        f"score {float(row['bet_quality_score']):.1f}",
        f"stake ${float(row.get('effective_suggested_stake', row['suggested_stake'])):.2f}",
    ]
    return " | ".join(parts)


def _flag_summary(row: pd.Series, fighter_prefix: str, flag_columns: list[str]) -> str:
    active_flags = [
        flag_name
        for flag_name in flag_columns
        if float(row.get(f"{fighter_prefix}_{flag_name}", 0.0) or 0.0) > 0
    ]
    return ", ".join(active_flags)


def _print_console_summary(report: pd.DataFrame, shortlist: pd.DataFrame, probability_column: str) -> None:
    tier_counts = report["recommended_tier"].astype(str).value_counts().to_dict() if "recommended_tier" in report.columns else {}
    print(
        "Value scan summary: "
        f"{len(report)} plays | "
        f"A {tier_counts.get('A', 0)} | "
        f"B {tier_counts.get('B', 0)} | "
        f"C {tier_counts.get('C', 0)}"
    )
    print()

    if not shortlist.empty:
        print("Shortlist")
        for tier in ["A", "B"]:
            tier_rows = shortlist.loc[shortlist["recommended_tier"].astype(str) == tier].head(5)
            if tier_rows.empty:
                continue
            tier_label = _colorize(f"{tier}-tier", "green" if tier == "A" else "yellow")
            print(tier_label)
            for _, row in tier_rows.iterrows():
                matchup = f"{row['fighter_a']} vs {row['fighter_b']}"
                print(
                    f"- {matchup} | score {float(row['bet_quality_score']):.1f} | "
                    f"stake ${float(row.get('effective_suggested_stake', row['suggested_stake'])):.2f}"
                )
                print(f"  Decision: {_decision_summary(row)}")
                print(
                    f"  Numbers: line {_format_decimal(row.get('effective_american_odds', row['american_odds']))} | "
                    f"model {_format_percent(float(row.get('effective_projected_prob', row[probability_column])))} | "
                    f"implied {_format_percent(float(row.get('effective_implied_prob', row['implied_prob'])))} | "
                    f"edge {_format_percent(float(row.get('effective_edge', row['edge'])))} | "
                    f"EV {float(row.get('effective_expected_value', row['expected_value'])):+.3f}"
                )
                print(f"  History: {_history_summary(row)}")
                print(f"  Fragility: {_fragility_summary(row)}")
                stake_rule = _safe_text(row.get("stake_governor_reason", ""))
                if stake_rule:
                    print(f"  Stake rule: {stake_rule}")
                if row.get("expression_pick_source") == "alternative_market" and row.get("value_expression_reason"):
                    print(f"  Why this expression: {row['value_expression_reason']}")
                if row.get("support_signals"):
                    print(f"  Supports: {_colorize_tier_tokens(row['support_signals'])}")
                camp_summary = _selection_camp_summary(row, colorize_tier=True)
                if camp_summary:
                    print(f"  Camp: {camp_summary}")
                if row.get("risk_flags"):
                    print(f"  Risks: {row['risk_flags']}")
            print()

    c_rows = report.loc[report["recommended_tier"].astype(str) == "C"].head(3) if "recommended_tier" in report.columns else pd.DataFrame()
    if not c_rows.empty:
        print("Passes")
        for _, row in c_rows.iterrows():
            matchup = f"{row['fighter_a']} vs {row['fighter_b']}"
            print(
                f"- {matchup} | score {float(row['bet_quality_score']):.1f} | "
                f"edge {_format_percent(float(row.get('effective_edge', row['edge'])))}"
            )
            print(f"  Decision: {_decision_summary(row)}")
            print(f"  History: {_history_summary(row)}")
            print(f"  Fragility: {_fragility_summary(row)}")
            stake_rule = _safe_text(row.get("stake_governor_reason", ""))
            if stake_rule:
                print(f"  Stake rule: {stake_rule}")
            camp_summary = _selection_camp_summary(row, colorize_tier=True)
            if camp_summary:
                print(f"  Camp: {camp_summary}")
            hard_gate_reason = _safe_text(row.get("hard_gate_reason", ""))
            if hard_gate_reason:
                print(f"  Gate: {hard_gate_reason}")
            if row.get("risk_flags"):
                print(f"  Risks: {row['risk_flags']}")
        print()


def _build_betting_board(report: pd.DataFrame) -> pd.DataFrame:
    board = report.copy()
    if "recommended_tier" in board.columns:
        board = board.loc[board["recommended_tier"].isin(["A", "B"])].copy()
    if board.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "fight",
                "bet",
                "line",
                "american_line",
                "model_prob",
                "implied_prob",
                "edge",
                "confidence",
                "tier",
                "action",
                "stake",
                "stake_notes",
                "support_signals",
                "risk_flags",
                "model_context_flags",
                "operator_context_flags",
                "speculative_prop",
                "speculative_prop_fair_line",
                "speculative_prop_reason",
                "context_notes",
            ]
        )

    board["fight"] = board["fighter_a"] + " vs " + board["fighter_b"]
    board["bet"] = board["chosen_value_expression"]
    board["line"] = board["effective_american_odds"].apply(_format_decimal)
    board["american_line"] = board["effective_american_odds"].apply(lambda value: f"{int(float(value)):+d}")
    board["model_prob"] = board["effective_projected_prob"].apply(_format_percent)
    board["implied_prob"] = board["effective_implied_prob"].apply(_format_percent)
    board["edge"] = board["effective_edge"].apply(_format_percent)
    board["confidence"] = board.get("model_confidence", pd.Series(0.0, index=board.index)).apply(lambda value: f"{float(value):.2f}")
    board["tier"] = board.get("recommended_tier", "")
    board["action"] = board.get("recommended_action", "")
    board["stake"] = board["effective_suggested_stake"].apply(lambda value: round(float(value), 2))
    board["stake_notes"] = board.get("stake_governor_reason", "")
    board["runner_up_bet"] = board.get("runner_up_expression", "")
    board["decision_gap"] = board.get("expression_edge_gap", pd.Series(pd.NA, index=board.index)).apply(
        lambda value: _format_percent(value) if pd.notna(value) else ""
    )
    board["history"] = board.apply(_history_summary, axis=1)
    board["fragility"] = board.apply(_fragility_summary, axis=1)
    board["model_context_flags"] = board.apply(
        lambda row: _flag_summary(row, "a", MODEL_CONTEXT_FLAG_COLUMNS)
        if row["selection"] == "fighter_a"
        else _flag_summary(row, "b", MODEL_CONTEXT_FLAG_COLUMNS),
        axis=1,
    )
    board["operator_context_flags"] = board.apply(
        lambda row: _flag_summary(row, "a", OPERATOR_CONTEXT_FLAG_COLUMNS)
        if row["selection"] == "fighter_a"
        else _flag_summary(row, "b", OPERATOR_CONTEXT_FLAG_COLUMNS),
        axis=1,
    )
    board["speculative_prop"] = board.get("speculative_prop_expression", "")
    board["speculative_prop_fair_line"] = board.apply(
        lambda row: (
            f"{int(float(row['speculative_prop_fair_american_odds'])):+d} / {float(row['speculative_prop_fair_decimal_odds']):.2f}"
            if pd.notna(row.get("speculative_prop_fair_american_odds")) and pd.notna(row.get("speculative_prop_fair_decimal_odds"))
            else ""
        ),
        axis=1,
    )
    board["speculative_prop_reason"] = board.get("speculative_prop_reason", "")
    board["context_notes"] = board.apply(
        lambda row: " | ".join(
            note for note in [
                str(row.get("a_context_notes", "")).strip() if row["selection"] == "fighter_a" else str(row.get("b_context_notes", "")).strip(),
                _selection_camp_summary(row),
                str(row.get("value_expression_reason", "")).strip() if row.get("expression_pick_source") == "alternative_market" else "",
            ] if note
        ),
        axis=1,
    )
    columns = [
        "event_name",
        "fight",
        "bet",
        "line",
        "american_line",
        "model_prob",
        "implied_prob",
        "edge",
        "confidence",
        "tier",
        "action",
        "stake",
        "stake_notes",
        "runner_up_bet",
        "decision_gap",
        "history",
        "fragility",
        "support_signals",
        "risk_flags",
        "model_context_flags",
        "operator_context_flags",
        "speculative_prop",
        "speculative_prop_fair_line",
        "speculative_prop_reason",
        "context_notes",
    ]
    sort_columns = ["tier", "bet_quality_score", "effective_edge"] if "bet_quality_score" in board.columns else ["tier", "effective_edge"]
    ascending = [True, False, False] if len(sort_columns) == 3 else [True, False]
    return board.sort_values(by=sort_columns, ascending=ascending)[columns].reset_index(drop=True)


def _build_pass_reasons_report(
    normalized: pd.DataFrame,
    report: pd.DataFrame,
    *,
    min_edge: float,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
) -> pd.DataFrame:
    included_keys = set()
    if not report.empty:
        included_keys = set(
            zip(
                report["event_id"].astype(str),
                report["fighter_a"].astype(str),
                report["fighter_b"].astype(str),
                report["selection"].astype(str),
            )
        )

    passes = normalized.copy()
    passes["_row_key"] = list(
        zip(
            passes["event_id"].astype(str),
            passes["fighter_a"].astype(str),
            passes["fighter_b"].astype(str),
            passes["selection"].astype(str),
        )
    )
    passes = passes.loc[~passes["_row_key"].isin(included_keys)].copy()
    if passes.empty:
        return pd.DataFrame(
            columns=[
                "event_name",
                "fight",
                "selection_name",
                "line",
                "american_line",
                "edge",
                "confidence",
                "data_quality",
                "pass_reason",
                "risk_flags",
                "model_context_flags",
                "operator_context_flags",
                "context_notes",
            ]
        )

    def describe_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if float(row.get("effective_edge", row.get("edge", 0.0))) < min_edge:
            reasons.append("edge_below_threshold")
        if "model_confidence" in row.index and float(row.get("model_confidence", 0.0) or 0.0) < min_model_confidence:
            reasons.append("confidence_below_threshold")
        if "data_quality" in row.index and float(row.get("data_quality", 0.0) or 0.0) < min_stats_completeness:
            reasons.append("stats_completeness_below_threshold")
        if exclude_fallback_rows and float(row.get("fallback_penalty", 0.0) or 0.0) > 0:
            reasons.append("fallback_data_excluded")
        if "recommended_tier" in row.index and str(row.get("recommended_tier", "")) == "C":
            reasons.append("model_marked_pass")
        if "market_blend_weight" in row.index and float(row.get("market_blend_weight", 0.0) or 0.0) >= 0.40:
            reasons.append("market_disagreement")
        hard_gate_reason = _safe_text(row.get("hard_gate_reason", ""))
        if hard_gate_reason:
            reasons.append(hard_gate_reason)
        governor_reason = _safe_text(row.get("stake_governor_reason", ""))
        if _safe_float(row.get("chosen_expression_stake", 0.0), 0.0) <= 0.0 and governor_reason:
            reasons.append(governor_reason)
        return ", ".join(dict.fromkeys(reasons)) if reasons else "filtered_out"

    passes["fight"] = passes["fighter_a"] + " vs " + passes["fighter_b"]
    passes["line"] = passes["effective_american_odds"].apply(_format_decimal)
    passes["american_line"] = passes["effective_american_odds"].apply(lambda value: f"{int(float(value)):+d}")
    passes["edge"] = passes["effective_edge"].apply(_format_percent)
    passes["confidence"] = passes.get("model_confidence", pd.Series(0.0, index=passes.index)).apply(lambda value: f"{float(value):.2f}")
    passes["data_quality"] = passes.get("data_quality", pd.Series(0.0, index=passes.index)).apply(lambda value: f"{float(value):.2f}")
    passes["pass_reason"] = passes.apply(describe_reason, axis=1)
    passes["runner_up_bet"] = passes.get("runner_up_expression", "")
    passes["decision_gap"] = passes.get("expression_edge_gap", pd.Series(pd.NA, index=passes.index)).apply(
        lambda value: _format_percent(value) if pd.notna(value) else ""
    )
    passes["history"] = passes.apply(_history_summary, axis=1)
    passes["fragility"] = passes.apply(_fragility_summary, axis=1)
    passes["model_context_flags"] = passes.apply(
        lambda row: _flag_summary(row, "a", MODEL_CONTEXT_FLAG_COLUMNS)
        if row["selection"] == "fighter_a"
        else _flag_summary(row, "b", MODEL_CONTEXT_FLAG_COLUMNS),
        axis=1,
    )
    passes["operator_context_flags"] = passes.apply(
        lambda row: _flag_summary(row, "a", OPERATOR_CONTEXT_FLAG_COLUMNS)
        if row["selection"] == "fighter_a"
        else _flag_summary(row, "b", OPERATOR_CONTEXT_FLAG_COLUMNS),
        axis=1,
    )
    passes["context_notes"] = passes.apply(
        lambda row: " | ".join(
            note
            for note in [
                str(row.get("a_context_notes", "")).strip()
                if row["selection"] == "fighter_a"
                else str(row.get("b_context_notes", "")).strip(),
                _selection_camp_summary(row),
                str(row.get("value_expression_reason", "")).strip()
                if row.get("expression_pick_source") == "alternative_market"
                else "",
            ]
            if note
        ),
        axis=1,
    )
    columns = [
        "event_name",
        "fight",
        "selection_name",
        "line",
        "american_line",
        "edge",
        "confidence",
        "data_quality",
        "pass_reason",
        "runner_up_bet",
        "decision_gap",
        "history",
        "fragility",
        "risk_flags",
        "model_context_flags",
        "operator_context_flags",
        "context_notes",
    ]
    return passes.sort_values(by=["event_name", "fight", "selection_name"])[columns].reset_index(drop=True)


def _apply_expression_overrides(normalized: pd.DataFrame, fight_report_path: str | None, probability_column: str) -> pd.DataFrame:
    if fight_report_path:
        fight_report = pd.read_csv(fight_report_path)
        merge_columns = [
            "fighter_a",
            "fighter_b",
            "selection_name",
            "preferred_market_expression",
            "preferred_market_american_odds",
            "preferred_market_projected_prob",
            "preferred_market_edge",
            "preferred_market_implied_prob",
            "value_expression_winner",
            "value_expression_reason",
            "speculative_prop_expression",
            "speculative_prop_model_prob",
            "speculative_prop_fair_american_odds",
            "speculative_prop_fair_decimal_odds",
            "speculative_prop_reason",
        ]
        available_columns = [column for column in merge_columns if column in fight_report.columns]
        if len(available_columns) >= 3:
            report_subset = fight_report[available_columns].drop_duplicates(subset=["fighter_a", "fighter_b"]).rename(
                columns={"selection_name": "report_selection_name"}
            )
            normalized = normalized.merge(
                report_subset,
                on=["fighter_a", "fighter_b"],
                how="left",
            )
        else:
            normalized["report_selection_name"] = ""
    else:
        normalized["report_selection_name"] = ""

    for column, default_value in {
        "preferred_market_expression": pd.NA,
        "preferred_market_american_odds": pd.NA,
        "preferred_market_projected_prob": pd.NA,
        "preferred_market_edge": pd.NA,
        "preferred_market_implied_prob": pd.NA,
        "value_expression_winner": "",
        "value_expression_reason": "",
    }.items():
        if column not in normalized.columns:
            normalized[column] = default_value

    def use_alternative_expression(row: pd.Series) -> bool:
        if _safe_text(row.get("value_expression_winner", "")) != "alternative_market":
            return False
        if pd.isna(row.get("preferred_market_expression")) or pd.isna(row.get("preferred_market_american_odds")):
            return False
        expression = _safe_text(row.get("preferred_market_expression", ""))
        if not expression:
            return False
        selection_name = _safe_text(row.get("selection_name", ""))
        report_selection_name = _safe_text(row.get("report_selection_name", ""))
        if expression in {"Fight goes to decision", "Fight doesn't go to decision"}:
            if report_selection_name:
                return selection_name == report_selection_name
            return _safe_float(row.get(probability_column, 0.0), 0.0) >= 0.5
        return expression.startswith(selection_name)

    normalized["_use_alternative_expression"] = normalized.apply(use_alternative_expression, axis=1)
    normalized["chosen_value_expression"] = normalized.apply(
        lambda row: row["preferred_market_expression"] if row["_use_alternative_expression"] else row["selection_name"],
        axis=1,
    )
    normalized["expression_pick_source"] = normalized["_use_alternative_expression"].map(
        {True: "alternative_market", False: "side_market"}
    )
    normalized["chosen_expression_odds"] = normalized.apply(
        lambda row: row.get("preferred_market_american_odds") if row["_use_alternative_expression"] else row["american_odds"],
        axis=1,
    )
    normalized["chosen_expression_prob"] = normalized.apply(
        lambda row: float(row.get("preferred_market_projected_prob"))
        if row["_use_alternative_expression"] and pd.notna(row.get("preferred_market_projected_prob"))
        else float(row[probability_column]),
        axis=1,
    )
    normalized["chosen_expression_implied_prob"] = normalized.apply(
        lambda row: float(row.get("preferred_market_implied_prob"))
        if row["_use_alternative_expression"] and pd.notna(row.get("preferred_market_implied_prob"))
        else float(row["implied_prob"]),
        axis=1,
    )
    normalized["chosen_expression_edge"] = normalized["chosen_expression_prob"] - normalized["chosen_expression_implied_prob"]
    normalized["runner_up_expression"] = ""
    normalized["runner_up_odds"] = pd.NA
    normalized["runner_up_prob"] = pd.NA
    normalized["runner_up_implied_prob"] = pd.NA
    normalized["runner_up_edge"] = pd.NA

    def alternative_relevant_to_selection(row: pd.Series) -> bool:
        expression = _safe_text(row.get("preferred_market_expression", ""))
        if not expression or pd.isna(row.get("preferred_market_american_odds")):
            return False
        selection_name = _safe_text(row.get("selection_name", ""))
        report_selection_name = _safe_text(row.get("report_selection_name", ""))
        if expression in {"Fight goes to decision", "Fight doesn't go to decision"}:
            if report_selection_name:
                return selection_name == report_selection_name
            return _safe_float(row.get(probability_column, 0.0), 0.0) >= 0.5
        return expression.startswith(selection_name)

    alternative_relevant = normalized.apply(alternative_relevant_to_selection, axis=1)
    use_alternative = normalized["_use_alternative_expression"]
    normalized.loc[use_alternative, "runner_up_expression"] = normalized.loc[use_alternative, "selection_name"]
    normalized.loc[use_alternative, "runner_up_odds"] = normalized.loc[use_alternative, "american_odds"]
    normalized.loc[use_alternative, "runner_up_prob"] = normalized.loc[use_alternative, probability_column]
    normalized.loc[use_alternative, "runner_up_implied_prob"] = normalized.loc[use_alternative, "implied_prob"]
    normalized.loc[use_alternative, "runner_up_edge"] = normalized.loc[use_alternative, "edge"]

    runner_up_mask = ~use_alternative & alternative_relevant
    normalized.loc[runner_up_mask, "runner_up_expression"] = normalized.loc[runner_up_mask, "preferred_market_expression"]
    normalized.loc[runner_up_mask, "runner_up_odds"] = normalized.loc[runner_up_mask, "preferred_market_american_odds"]
    normalized.loc[runner_up_mask, "runner_up_prob"] = normalized.loc[runner_up_mask, "preferred_market_projected_prob"]
    normalized.loc[runner_up_mask, "runner_up_implied_prob"] = normalized.loc[runner_up_mask, "preferred_market_implied_prob"]
    normalized.loc[runner_up_mask, "runner_up_edge"] = normalized.loc[runner_up_mask, "preferred_market_edge"]
    normalized["expression_edge_gap"] = normalized.apply(
        lambda row: float(row["chosen_expression_edge"]) - float(row["runner_up_edge"])
        if pd.notna(row.get("runner_up_edge"))
        else pd.NA,
        axis=1,
    )
    if "speculative_prop_expression" in normalized.columns:
        normalized["speculative_prop_expression"] = normalized.apply(
            lambda row: row["speculative_prop_expression"]
            if pd.notna(row.get("speculative_prop_expression"))
            and str(row["speculative_prop_expression"]).startswith(str(row["selection_name"]))
            else pd.NA,
            axis=1,
        )
        for column in [
            "speculative_prop_model_prob",
            "speculative_prop_fair_american_odds",
            "speculative_prop_fair_decimal_odds",
            "speculative_prop_reason",
        ]:
            normalized[column] = normalized.apply(
                lambda row: row.get(column) if pd.notna(row.get("speculative_prop_expression")) else pd.NA,
                axis=1,
            )
    normalized = normalized.drop(columns=["_use_alternative_expression"], errors="ignore")
    return normalized


def _attach_diagnostic_context(normalized: pd.DataFrame, *, db_path: str | Path | None) -> pd.DataFrame:
    enriched = normalized.copy()
    if "line_movement_toward_fighter" not in enriched.columns:
        enriched["line_movement_toward_fighter"] = enriched.apply(
            lambda row: (
                implied_probability(int(row["american_odds"])) - implied_probability(int(row["open_american_odds"]))
                if pd.notna(row.get("open_american_odds")) and pd.notna(row.get("american_odds"))
                else 0.0
            ),
            axis=1,
        )
    enriched = apply_historical_overlays(enriched, db_path=db_path)

    fragility_rows: list[dict[str, object]] = []
    for row in enriched.to_dict("records"):
        fragility = calculate_fragility_metrics(
            short_notice_flag=_safe_float(row.get("selection_short_notice_flag", 0.0), 0.0),
            short_notice_acceptance_flag=_safe_float(row.get("selection_short_notice_acceptance_flag", 0.0), 0.0),
            short_notice_success_flag=_safe_float(row.get("selection_short_notice_success_flag", 0.0), 0.0),
            days_since_last_fight=_safe_float(row.get("selection_days_since_last_fight", 999.0), 999.0),
            ufc_fight_count=_safe_float(row.get("selection_ufc_fight_count", 0.0), 0.0),
            ufc_debut_flag=_safe_float(row.get("selection_ufc_debut_flag", 0.0), 0.0),
            injury_concern_flag=_safe_float(row.get("selection_injury_concern_flag", 0.0), 0.0),
            weight_cut_concern_flag=_safe_float(row.get("selection_weight_cut_concern_flag", 0.0), 0.0),
            replacement_fighter_flag=_safe_float(row.get("selection_replacement_fighter_flag", 0.0), 0.0),
            travel_disadvantage_flag=_safe_float(row.get("selection_travel_disadvantage_flag", 0.0), 0.0),
            camp_change_flag=_safe_float(row.get("selection_camp_change_flag", 0.0), 0.0),
            gym_changed_flag=_safe_float(row.get("selection_gym_changed_flag", 0.0), 0.0),
            fallback_used=_safe_float(row.get("selection_fallback_used", 0.0), 0.0),
            data_quality=min(
                _safe_float(row.get("data_quality", 1.0), 1.0),
                _safe_float(row.get("selection_stats_completeness", row.get("data_quality", 1.0)), 1.0),
            ),
            market_blend_weight=_safe_float(row.get("market_blend_weight", 0.0), 0.0),
            consensus_count=_safe_float(row.get("market_consensus_bookmaker_count", 0.0), 0.0),
            consensus_price_edge=_safe_float(row.get("price_edge_vs_consensus", 0.0), 0.0),
        )
        tracked_market_key = _safe_text(row.get("tracked_market_key", "moneyline"), "moneyline")
        hard_gate_reason = ""
        data_quality = min(
            _safe_float(row.get("data_quality", 1.0), 1.0),
            _safe_float(row.get("selection_stats_completeness", row.get("data_quality", 1.0)), 1.0),
        )
        if tracked_market_key in {"inside_distance", "by_decision"} and (
            _safe_float(row.get("selection_fallback_used", 0.0), 0.0) >= 1.0 or data_quality < 0.85
        ):
            hard_gate_reason = "prop_data_quality_gate"
        elif (
            tracked_market_key in {"inside_distance", "by_decision"}
            and int(_safe_float(row.get("historical_sample_size", 0.0), 0.0)) >= 8
            and "negative" in _safe_text(row.get("historical_overlay_grade", ""))
        ):
            hard_gate_reason = "negative_prop_history_gate"
        elif (
            _safe_text(fragility.get("fragility_bucket", "low"), "low") == "high"
            and _safe_float(row.get("chosen_expression_edge", row.get("edge", 0.0)), 0.0) < 0.06
        ):
            hard_gate_reason = "high_fragility_thin_edge"
        fragility_rows.append({"hard_gate_reason": hard_gate_reason, **fragility})

    return pd.concat([enriched, pd.DataFrame(fragility_rows, index=enriched.index)], axis=1)


def _apply_diagnostic_overrides(normalized: pd.DataFrame) -> pd.DataFrame:
    if "bet_quality_score" not in normalized.columns or normalized.empty:
        return normalized

    adjusted = normalized.copy()
    updated_scores: list[float] = []
    updated_tiers: list[str] = []
    updated_actions: list[str] = []
    updated_supports: list[str] = []
    updated_risks: list[str] = []
    updated_rationales: list[str] = []
    support_counts: list[int] = []
    risk_counts: list[int] = []

    for row in adjusted.to_dict("records"):
        support_items = _split_items(row.get("support_signals", ""))
        risk_items = _split_items(row.get("risk_flags", ""))
        score = _safe_float(row.get("bet_quality_score", 0.0), 0.0)
        tier = _safe_text(row.get("recommended_tier", ""))
        action = _safe_text(row.get("recommended_action", ""))
        historical_grade = _safe_text(row.get("historical_overlay_grade", "low_sample"), "low_sample")
        historical_sample_size = int(_safe_float(row.get("historical_sample_size", 0.0), 0.0))
        historical_roi_pct = row.get("historical_roi_pct", pd.NA)
        fragility_bucket = _safe_text(row.get("fragility_bucket", "low"), "low").lower()
        hard_gate_reason = _safe_text(row.get("hard_gate_reason", ""))
        expression_gap = row.get("expression_edge_gap", pd.NA)
        tracked_market_key = _safe_text(row.get("tracked_market_key", "moneyline"), "moneyline")
        effective_edge = _safe_float(row.get("effective_edge", row.get("chosen_expression_edge", row.get("edge", 0.0))), 0.0)

        score += _safe_float(row.get("historical_overlay_score_adjustment", 0.0), 0.0)
        if fragility_bucket == "medium":
            score -= 4.0
        elif fragility_bucket == "high":
            score -= 10.0

        if historical_sample_size >= 4 and "positive" in historical_grade:
            roi_text = f"{float(historical_roi_pct):+.1f}% ROI" if pd.notna(historical_roi_pct) else "positive history"
            support_items.append(f"historical {historical_sample_size}-comp edge ({roi_text})")
        elif historical_sample_size >= 4 and "negative" in historical_grade:
            risk_items.append("historical_negative")
        elif historical_sample_size >= 4 and historical_grade == "mixed":
            risk_items.append("historical_mixed")

        if fragility_bucket == "medium":
            risk_items.append("fragility_medium")
        elif fragility_bucket == "high":
            risk_items.append("fragility_high")
        else:
            support_items.append("low fragility")

        if pd.notna(expression_gap) and float(expression_gap) >= 0.03:
            support_items.append("clear expression gap")
        elif (
            pd.notna(expression_gap)
            and float(expression_gap) <= 0.01
            and _safe_text(row.get("expression_pick_source", "")) == "alternative_market"
        ):
            risk_items.append("thin_expression_gap")

        if hard_gate_reason:
            risk_items.append(hard_gate_reason)

        if hard_gate_reason in {"prop_data_quality_gate", "negative_prop_history_gate", "high_fragility_thin_edge"}:
            tier = "C"
            action = "Pass"
        elif fragility_bucket == "high" and tier == "A":
            tier = "B"
            action = "Watchlist"
        elif fragility_bucket == "high" and tier == "B" and effective_edge < 0.08:
            tier = "C"
            action = "Pass"
        elif historical_sample_size >= 8 and "negative" in historical_grade and tier == "A":
            tier = "B"
            action = "Watchlist"

        score = round(_clamp(score, 0.0, 100.0), 2)
        if tier == "A" and score < 82:
            tier = "B"
            action = "Watchlist"
        if tier in {"A", "B"} and score < 66:
            tier = "C"
            action = "Pass"
        if tracked_market_key in {"inside_distance", "by_decision"} and tier == "A" and fragility_bucket != "low":
            tier = "B"
            action = "Watchlist"

        support_items = list(dict.fromkeys(support_items))
        risk_items = list(dict.fromkeys(risk_items))
        updated_scores.append(score)
        updated_tiers.append(tier)
        updated_actions.append(action)
        updated_supports.append(", ".join(support_items[:8]))
        updated_risks.append(", ".join(risk_items))
        updated_rationales.append(_format_reasons(support_items[:6], risk_items))
        support_counts.append(len(support_items))
        risk_counts.append(len(risk_items))

    adjusted["bet_quality_score"] = updated_scores
    adjusted["recommended_tier"] = updated_tiers
    adjusted["recommended_action"] = updated_actions
    adjusted["support_signals"] = updated_supports
    adjusted["risk_flags"] = updated_risks
    adjusted["why_it_rates_well"] = updated_rationales
    adjusted["support_count"] = support_counts
    adjusted["risk_flag_count"] = risk_counts
    return adjusted


def _apply_bankroll_overrides(normalized: pd.DataFrame) -> pd.DataFrame:
    if normalized.empty or "chosen_expression_stake" not in normalized.columns:
        return normalized

    adjusted = normalized.copy()
    updated_scores: list[float] = []
    updated_tiers: list[str] = []
    updated_actions: list[str] = []
    updated_supports: list[str] = []
    updated_risks: list[str] = []
    updated_rationales: list[str] = []
    support_counts: list[int] = []
    risk_counts: list[int] = []

    for row in adjusted.to_dict("records"):
        support_items = _split_items(row.get("support_signals", ""))
        risk_items = _split_items(row.get("risk_flags", ""))
        score = _safe_float(row.get("bet_quality_score", 0.0), 0.0)
        tier = _safe_text(row.get("recommended_tier", ""))
        action = _safe_text(row.get("recommended_action", ""))
        raw_stake = _safe_float(
            row.get("raw_chosen_expression_stake", row.get("chosen_expression_stake", 0.0)),
            0.0,
        )
        final_stake = _safe_float(row.get("chosen_expression_stake", 0.0), 0.0)
        governor_reasons = _split_items(row.get("stake_governor_reason", ""))

        if raw_stake > 0.0 and final_stake < raw_stake:
            risk_items.append("stake_trimmed")
        for reason in governor_reasons:
            risk_items.append(reason)

        if raw_stake > 0.0 and final_stake <= 0.0 and action in {"Bettable now", "Watchlist"}:
            tier = "C"
            action = "Pass"
            score = min(score, 65.0)

        support_items = list(dict.fromkeys(support_items))
        risk_items = list(dict.fromkeys(risk_items))
        updated_scores.append(round(_clamp(score, 0.0, 100.0), 2))
        updated_tiers.append(tier)
        updated_actions.append(action)
        updated_supports.append(", ".join(support_items[:8]))
        updated_risks.append(", ".join(risk_items))
        updated_rationales.append(_format_reasons(support_items[:6], risk_items))
        support_counts.append(len(support_items))
        risk_counts.append(len(risk_items))

    adjusted["bet_quality_score"] = updated_scores
    adjusted["recommended_tier"] = updated_tiers
    adjusted["recommended_action"] = updated_actions
    adjusted["support_signals"] = updated_supports
    adjusted["risk_flags"] = updated_risks
    adjusted["why_it_rates_well"] = updated_rationales
    adjusted["support_count"] = support_counts
    adjusted["risk_flag_count"] = risk_counts
    return adjusted


def _filter_and_rank_report(
    normalized: pd.DataFrame,
    *,
    min_edge: float,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
) -> pd.DataFrame:
    report = normalized.loc[normalized["effective_edge"] >= min_edge].copy()
    if "model_confidence" in report.columns:
        report = report.loc[report["model_confidence"] >= min_model_confidence].copy()
    if "data_quality" in report.columns:
        report = report.loc[report["data_quality"] >= min_stats_completeness].copy()
    if exclude_fallback_rows and "fallback_penalty" in report.columns:
        report = report.loc[report["fallback_penalty"] == 0].copy()
    if "recommended_action" in report.columns:
        report = report.loc[report["recommended_action"].isin(["Bettable now", "Watchlist"])].copy()
    sort_columns = ["edge", "expected_value"]
    if "bet_quality_score" in report.columns:
        if "expression_pick_source" in report.columns:
            report["expression_bonus"] = report["expression_pick_source"].map(
                {"alternative_market": 2.5, "side_market": 0.0}
            ).fillna(0.0)
            report["expression_rank_score"] = (report["bet_quality_score"].astype(float) + report["expression_bonus"].astype(float)).round(2)
        else:
            report["expression_rank_score"] = report["bet_quality_score"].astype(float)
        sort_columns = ["recommended_tier", "bet_quality_score", "effective_edge", "effective_expected_value"]
        if "expression_rank_score" in report.columns:
            sort_columns = ["recommended_tier", "expression_rank_score", "bet_quality_score", "effective_edge", "effective_expected_value"]
        report["recommended_tier"] = pd.Categorical(report["recommended_tier"], categories=["A", "B", "C"], ordered=True)
    if "bet_quality_score" in report.columns:
        ascending = [True] + [False] * (len(sort_columns) - 1)
        return report.sort_values(by=sort_columns, ascending=ascending)
    return report.sort_values(by=sort_columns, ascending=False)


def _write_scan_outputs(
    report: pd.DataFrame,
    normalized: pd.DataFrame,
    *,
    output_path: Path,
    shortlist_path: Path,
    board_path: Path,
    passes_path: Path,
    min_edge: float,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    shortlist = report.copy()
    if "recommended_tier" in shortlist.columns:
        shortlist = shortlist.loc[shortlist["recommended_tier"].isin(["A", "B"])].copy()
    shortlist.to_csv(shortlist_path, index=False)
    board = _build_betting_board(report)
    board.to_csv(board_path, index=False)
    passes = _build_pass_reasons_report(
        normalized,
        report,
        min_edge=min_edge,
        min_model_confidence=min_model_confidence,
        min_stats_completeness=min_stats_completeness,
        exclude_fallback_rows=exclude_fallback_rows,
    )
    passes.to_csv(passes_path, index=False)
    return shortlist, board


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    selective_model_path = Path(args.selective_model) if args.selective_model else default_selective_model_path(ROOT)
    side_model_path = Path(args.side_model) if args.side_model else default_side_model_path(ROOT)
    side_model_bundle = load_side_model(side_model_path) if side_model_path.exists() else None
    confidence_model_path = Path(args.confidence_model) if args.confidence_model else default_confidence_model_path(ROOT)
    confidence_model_bundle = load_confidence_model(confidence_model_path) if confidence_model_path.exists() else None

    min_edge = float(os.getenv("MIN_EDGE", "0.03"))
    bankroll = float(os.getenv("BANKROLL", "1000"))
    fractional_kelly = float(os.getenv("FRACTIONAL_KELLY", "0.25"))
    bankroll_governor = bankroll_governor_config_from_env()
    min_model_confidence = float(os.getenv("MIN_MODEL_CONFIDENCE", "0.60"))
    min_stats_completeness = float(os.getenv("MIN_STATS_COMPLETENESS", "0.80"))
    exclude_fallback_rows = os.getenv("EXCLUDE_FALLBACK_ROWS", "true").lower() == "true"
    threshold_policy_path = Path(args.threshold_policy) if args.threshold_policy else default_threshold_policy_path(ROOT)
    threshold_policy = load_threshold_policy(threshold_policy_path if threshold_policy_path.exists() else None)
    threshold_settings = resolve_scan_thresholds(
        min_edge=min_edge,
        min_model_confidence=min_model_confidence,
        min_stats_completeness=min_stats_completeness,
        exclude_fallback_rows=exclude_fallback_rows,
        policy=threshold_policy,
    )
    min_edge = float(threshold_settings["min_edge"])
    min_model_confidence = float(threshold_settings["min_model_confidence"])
    min_stats_completeness = float(threshold_settings["min_stats_completeness"])
    exclude_fallback_rows = bool(threshold_settings["exclude_fallback_rows"])

    raw = load_odds_csv(args.input)
    normalized = normalize_odds_frame(raw)
    if args.fighter_stats:
        fighter_stats = load_fighter_stats(args.fighter_stats)
        features = build_fight_features(normalized, fighter_stats)
        normalized = project_fight_probabilities(
            features,
            side_model_bundle=side_model_bundle,
            confidence_model_bundle=confidence_model_bundle,
        )
        probability_column = "model_projected_win_prob"
    else:
        probability_column = "projected_win_prob"
    normalized["implied_prob"] = normalized["american_odds"].apply(implied_probability)
    normalized["edge"] = normalized[probability_column] - normalized["implied_prob"]
    normalized["market_consensus_prob"] = pd.to_numeric(
        normalized.get("market_consensus_prob", pd.Series(pd.NA, index=normalized.index)),
        errors="coerce",
    )
    normalized["market_consensus_bookmaker_count"] = pd.to_numeric(
        normalized.get("market_consensus_bookmaker_count", pd.Series(0.0, index=normalized.index)),
        errors="coerce",
    ).fillna(0.0)
    normalized["market_overround"] = pd.to_numeric(
        normalized.get("market_overround", pd.Series(0.0, index=normalized.index)),
        errors="coerce",
    ).fillna(0.0)
    normalized["price_edge_vs_consensus"] = (
        normalized["market_consensus_prob"] - normalized["implied_prob"]
    ).fillna(0.0)
    normalized["expected_value"] = normalized.apply(
        lambda row: expected_value(row[probability_column], int(row["american_odds"])),
        axis=1,
    )
    normalized["suggested_stake"] = normalized.apply(
        lambda row: suggested_stake(
            bankroll=bankroll,
            projected_win_prob=row[probability_column],
            american_odds=int(row["american_odds"]),
            fraction=fractional_kelly,
        ),
        axis=1,
    )
    if probability_column == "model_projected_win_prob":
        normalized["scheduled_rounds"] = pd.to_numeric(
            normalized.get("scheduled_rounds", pd.Series(3.0, index=normalized.index)),
            errors="coerce",
        ).fillna(3.0)
        normalized["is_wmma"] = pd.to_numeric(
            normalized.get("is_wmma", pd.Series(0.0, index=normalized.index)),
            errors="coerce",
        ).fillna(0.0)
        normalized["is_heavyweight"] = pd.to_numeric(
            normalized.get("is_heavyweight", pd.Series(0.0, index=normalized.index)),
            errors="coerce",
        ).fillna(0.0)
        normalized["is_five_round_fight"] = pd.to_numeric(
            normalized.get("is_five_round_fight", pd.Series(0.0, index=normalized.index)),
            errors="coerce",
        ).fillna(0.0)
        normalized["segment_label"] = normalized.get(
            "segment_label",
            pd.Series("standard", index=normalized.index),
        ).fillna("standard")
        normalized["selection_days_since_last_fight"] = normalized.apply(
            lambda row: _selection_value(row, "a_days_since_last_fight", "b_days_since_last_fight"),
            axis=1,
        )
        normalized["selection_ufc_fight_count"] = normalized.apply(
            lambda row: _selection_value(row, "a_ufc_fight_count", "b_ufc_fight_count"),
            axis=1,
        )
        normalized["selection_ufc_debut_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_ufc_debut_flag", "b_ufc_debut_flag"),
            axis=1,
        )
        normalized["selection_stats_completeness"] = normalized.apply(
            lambda row: _selection_value(row, "a_stats_completeness", "b_stats_completeness"),
            axis=1,
        )
        normalized["selection_fallback_used"] = normalized.apply(
            lambda row: _selection_value(row, "a_fallback_used", "b_fallback_used"),
            axis=1,
        )
        normalized["selection_short_notice_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_short_notice_flag", "b_short_notice_flag"),
            axis=1,
        )
        normalized["selection_short_notice_acceptance_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_short_notice_acceptance_flag", "b_short_notice_acceptance_flag"),
            axis=1,
        )
        normalized["selection_short_notice_success_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_short_notice_success_flag", "b_short_notice_success_flag"),
            axis=1,
        )
        normalized["selection_first_round_finish_rate"] = normalized.apply(
            lambda row: _selection_value(row, "a_first_round_finish_rate", "b_first_round_finish_rate"),
            axis=1,
        )
        normalized["selection_finish_loss_rate"] = normalized.apply(
            lambda row: _selection_value(row, "a_finish_loss_rate", "b_finish_loss_rate"),
            axis=1,
        )
        normalized["selection_recent_finish_damage"] = normalized.apply(
            lambda row: _selection_value(row, "a_recent_finish_loss_365d", "b_recent_finish_loss_365d"),
            axis=1,
        )
        normalized["selection_recent_ko_damage"] = normalized.apply(
            lambda row: _selection_value(row, "a_recent_ko_loss_365d", "b_recent_ko_loss_365d"),
            axis=1,
        )
        normalized["selection_recent_damage_score"] = normalized.apply(
            lambda row: _selection_value(row, "a_recent_damage_score", "b_recent_damage_score"),
            axis=1,
        )
        normalized["selection_recent_grappling_rate"] = normalized.apply(
            lambda row: _selection_value(row, "a_recent_grappling_rate", "b_recent_grappling_rate"),
            axis=1,
        )
        normalized["selection_control_avg"] = normalized.apply(
            lambda row: _selection_value(row, "a_control_avg", "b_control_avg"),
            axis=1,
        )
        normalized["selection_recent_control_avg"] = normalized.apply(
            lambda row: _selection_value(row, "a_recent_control_avg", "b_recent_control_avg"),
            axis=1,
        )
        normalized["selection_grappling_pressure_score"] = normalized.apply(
            lambda row: round(
                float(_selection_value(row, "a_recent_grappling_rate", "b_recent_grappling_rate") or 0.0)
                + (float(_selection_value(row, "a_recent_control_avg", "b_recent_control_avg") or 0.0) * 0.55)
                + (float(_selection_value(row, "a_submission_avg", "b_submission_avg") or 0.0) * 0.35),
                3,
            ),
            axis=1,
        )
        normalized["selection_stance_matchup_edge"] = normalized.apply(
            lambda row: _oriented_feature(row, "stance_matchup_diff"),
            axis=1,
        )
        normalized["selection_matchup_striking_edge"] = normalized.apply(
            lambda row: _oriented_feature(row, "matchup_striking_edge"),
            axis=1,
        )
        normalized["selection_matchup_grappling_edge"] = normalized.apply(
            lambda row: _oriented_feature(row, "matchup_grappling_edge"),
            axis=1,
        )
        normalized["selection_matchup_control_edge"] = normalized.apply(
            lambda row: _oriented_feature(row, "matchup_control_edge"),
            axis=1,
        )
        normalized["selection_gym_name"] = normalized.apply(
            lambda row: row.get("a_gym_name", "") if row["selection"] == "fighter_a" else row.get("b_gym_name", ""),
            axis=1,
        )
        normalized["selection_gym_record"] = normalized.apply(
            lambda row: row.get("a_gym_record", "") if row["selection"] == "fighter_a" else row.get("b_gym_record", ""),
            axis=1,
        )
        normalized["selection_gym_tier"] = normalized.apply(
            lambda row: row.get("a_gym_tier", "") if row["selection"] == "fighter_a" else row.get("b_gym_tier", ""),
            axis=1,
        )
        normalized["selection_gym_score"] = normalized.apply(
            lambda row: _selection_value(row, "a_gym_score", "b_gym_score"),
            axis=1,
        )
        normalized["selection_gym_changed_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_gym_changed_flag", "b_gym_changed_flag"),
            axis=1,
        )
        normalized["selection_previous_gym_name"] = normalized.apply(
            lambda row: row.get("a_previous_gym_name", "") if row["selection"] == "fighter_a" else row.get("b_previous_gym_name", ""),
            axis=1,
        )
        normalized["selection_injury_concern_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_injury_concern_flag", "b_injury_concern_flag"),
            axis=1,
        )
        normalized["selection_weight_cut_concern_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_weight_cut_concern_flag", "b_weight_cut_concern_flag"),
            axis=1,
        )
        normalized["selection_replacement_fighter_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_replacement_fighter_flag", "b_replacement_fighter_flag"),
            axis=1,
        )
        normalized["selection_travel_disadvantage_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_travel_disadvantage_flag", "b_travel_disadvantage_flag"),
            axis=1,
        )
        normalized["selection_camp_change_flag"] = normalized.apply(
            lambda row: _selection_value(row, "a_camp_change_flag", "b_camp_change_flag"),
            axis=1,
        )
        normalized["selection_context_instability"] = normalized.apply(
            lambda row: (
                _selection_value(row, "a_short_notice_flag", "b_short_notice_flag")
                * (
                    1
                    - max(
                        _selection_value(row, "a_short_notice_acceptance_flag", "b_short_notice_acceptance_flag"),
                        _selection_value(row, "a_short_notice_success_flag", "b_short_notice_success_flag"),
                    )
                )
            )
            + _selection_value(row, "a_new_gym_flag", "b_new_gym_flag")
            + _selection_value(row, "a_new_contract_flag", "b_new_contract_flag")
            + _selection_value(row, "a_cardio_fade_flag", "b_cardio_fade_flag")
            + _selection_value(row, "a_injury_concern_flag", "b_injury_concern_flag")
            + _selection_value(row, "a_weight_cut_concern_flag", "b_weight_cut_concern_flag")
            + _selection_value(row, "a_replacement_fighter_flag", "b_replacement_fighter_flag")
            + _selection_value(row, "a_travel_disadvantage_flag", "b_travel_disadvantage_flag")
            + _selection_value(row, "a_camp_change_flag", "b_camp_change_flag"),
            axis=1,
        )
        normalized["support_count"] = 0
        normalized["risk_flag_count"] = 0
        bet_quality_scores: list[float] = []
        recommended_tiers: list[str] = []
        recommended_actions: list[str] = []
        support_labels: list[str] = []
        risk_flags: list[str] = []
        rationale: list[str] = []

        for row in normalized.to_dict("records"):
            signals: list[str] = []
            risks: list[str] = []
            support_count = 0

            edge = float(row["edge"])
            projected_prob = float(row[probability_column])
            confidence = float(row.get("model_confidence", 0.5) or 0.5)
            data_quality = float(row.get("data_quality", 0.5) or 0.5)
            selection_quality = float(row.get("selection_stats_completeness", data_quality) or data_quality)
            fallback_penalty = float(row.get("fallback_penalty", 0.0) or 0.0)
            selection_fallback = float(row.get("selection_fallback_used", 0.0) or 0.0)
            selection_days = float(row.get("selection_days_since_last_fight", 999.0) or 999.0)
            selection_ufc_fights = float(row.get("selection_ufc_fight_count", 0.0) or 0.0)
            selection_debut = float(row.get("selection_ufc_debut_flag", 0.0) or 0.0)
            selection_instability = float(row.get("selection_context_instability", 0.0) or 0.0)
            selection_injury = float(row.get("selection_injury_concern_flag", 0.0) or 0.0)
            selection_weight_cut = float(row.get("selection_weight_cut_concern_flag", 0.0) or 0.0)
            selection_replacement = float(row.get("selection_replacement_fighter_flag", 0.0) or 0.0)
            selection_travel = float(row.get("selection_travel_disadvantage_flag", 0.0) or 0.0)
            selection_camp_change = float(row.get("selection_camp_change_flag", 0.0) or 0.0)
            selection_gym_tier = str(row.get("selection_gym_tier", "") or "").strip().upper()
            selection_first_round_finish_rate = float(row.get("selection_first_round_finish_rate", 0.0) or 0.0)
            market_blend_weight = float(row.get("market_blend_weight", 0.0) or 0.0)
            consensus_price_edge = float(row.get("price_edge_vs_consensus", 0.0) or 0.0)
            consensus_count = float(row.get("market_consensus_bookmaker_count", 0.0) or 0.0)
            market_overround = float(row.get("market_overround", 0.0) or 0.0)
            is_wmma = float(row.get("is_wmma", 0.0) or 0.0)
            odds = int(row["american_odds"])
            is_opposite_model_side = projected_prob < 0.5
            is_positive_odds = odds > 0

            if edge >= 0.08:
                signals.append(f"strong edge {edge:.1%}")
                support_count += 1
            elif edge >= 0.05:
                signals.append(f"solid edge {edge:.1%}")
                support_count += 1
            else:
                risks.append("thin_edge")

            if projected_prob >= 0.60:
                signals.append(f"win probability {projected_prob:.1%}")
                support_count += 1
            elif projected_prob < 0.52:
                risks.append("low_win_probability")

            if confidence >= 0.75:
                signals.append(f"confidence {confidence:.2f}")
                support_count += 1
            elif confidence < 0.65:
                risks.append("low_confidence")

            if selection_quality >= 0.9 and data_quality >= 0.9:
                signals.append("clean stat coverage")
                support_count += 1
            elif data_quality < 0.85:
                risks.append("incomplete_stats")

            grappling_edge = max(
                _oriented_feature(row, "grappling_diff"),
                _oriented_feature(row, "recent_grappling_form_diff"),
            )
            matchup_striking_edge = _oriented_feature(row, "matchup_striking_edge")
            matchup_grappling_edge = _oriented_feature(row, "matchup_grappling_edge")
            matchup_control_edge = _oriented_feature(row, "matchup_control_edge")
            control_edge = _oriented_feature(row, "recent_control_diff")
            grappling_pressure_edge = _oriented_feature(row, "grappling_pressure_diff")
            schedule_strength_edge = _oriented_feature(row, "schedule_strength_diff")
            opponent_quality_edge = _oriented_feature(row, "opponent_quality_diff")
            normalized_striking_edge = _oriented_feature(row, "normalized_strike_margin_diff")
            normalized_grappling_edge = _oriented_feature(row, "normalized_grappling_diff")
            normalized_control_edge = _oriented_feature(row, "normalized_control_diff")
            grappling_signal = _tiered_signal_label("grappling edge", grappling_edge, (1.8, 1.45, 1.0, 0.45))
            control_signal = _tiered_signal_label("control edge", control_edge, (5.0, 3.5, 2.25, 1.0))
            grappling_pressure_signal = _tiered_signal_label(
                "grappling pressure",
                grappling_pressure_edge,
                (4.5, 3.0, 1.75, 0.8),
            )
            matchup_striking_signal = _tiered_signal_label("matchup striking", matchup_striking_edge, (2.4, 1.8, 1.2, 0.7))
            matchup_grappling_signal = _tiered_signal_label("matchup wrestling", matchup_grappling_edge, (2.0, 1.5, 1.0, 0.55))
            matchup_control_signal = _tiered_signal_label("matchup control", matchup_control_edge, (2.2, 1.6, 1.0, 0.5))
            schedule_strength_signal = _tiered_signal_label("schedule strength", schedule_strength_edge, (0.10, 0.07, 0.04, 0.02))
            normalized_striking_signal = _tiered_signal_label(
                "opponent-adjusted striking",
                normalized_striking_edge,
                (2.2, 1.5, 0.9, 0.45),
            )
            normalized_grappling_signal = _tiered_signal_label(
                "opponent-adjusted wrestling",
                normalized_grappling_edge,
                (1.6, 1.1, 0.7, 0.35),
            )
            normalized_control_signal = _tiered_signal_label(
                "opponent-adjusted control",
                normalized_control_edge,
                (3.0, 2.1, 1.2, 0.6),
            )

            if _oriented_feature(row, "win_rate_diff") >= 0.08:
                signals.append("better win-rate profile")
                support_count += 1
            if _oriented_feature(row, "reach_diff") >= 2.0:
                signals.append("reach advantage")
                support_count += 1
            if _oriented_feature(row, "height_diff") >= 2.0:
                signals.append("height advantage")
                support_count += 1
            if _oriented_feature(row, "strike_margin_diff") >= 0.75 or _oriented_feature(row, "recent_strike_form_diff") >= 0.45:
                signals.append("striking edge")
                support_count += 1
            if matchup_striking_signal:
                signals.append(matchup_striking_signal)
                support_count += 1
            if grappling_signal:
                signals.append(grappling_signal)
                support_count += 1
            if matchup_grappling_signal:
                signals.append(matchup_grappling_signal)
                support_count += 1
            if control_signal:
                signals.append(control_signal)
                support_count += 1
            if matchup_control_signal:
                signals.append(matchup_control_signal)
                support_count += 1
            if grappling_pressure_signal:
                signals.append(grappling_pressure_signal)
                support_count += 1
            if schedule_strength_signal:
                signals.append(schedule_strength_signal)
                support_count += 1
            elif opponent_quality_edge >= 0.06:
                signals.append("tested against better opposition")
                support_count += 1
            if normalized_striking_signal:
                signals.append(normalized_striking_signal)
                support_count += 1
            if normalized_grappling_signal:
                signals.append(normalized_grappling_signal)
                support_count += 1
            if normalized_control_signal:
                signals.append(normalized_control_signal)
                support_count += 1
            if _oriented_feature(row, "strike_efficiency_diff") >= 0.12:
                signals.append("accuracy edge")
                support_count += 1
            if _oriented_feature(row, "takedown_efficiency_diff") >= 0.12:
                signals.append("wrestling efficiency edge")
                support_count += 1
            if _oriented_feature(row, "first_round_finish_rate_diff") >= 0.15:
                signals.append("early finish threat")
                support_count += 1
            if _oriented_feature(row, "durability_diff") >= 0.12:
                signals.append("durability edge")
                support_count += 1
            if _oriented_feature(row, "age_diff") >= 2.5:
                signals.append("younger fighter")
                support_count += 1
            if _oriented_feature(row, "experience_diff") >= 3 or _oriented_feature(row, "ufc_experience_diff") >= 3:
                signals.append("experience edge")
                support_count += 1
            if _oriented_feature(row, "recent_form_diff") >= 0.5:
                signals.append("better recent form")
                support_count += 1
            if _oriented_feature(row, "layoff_diff") >= 0.15 and selection_days <= 300:
                signals.append("activity advantage")
                support_count += 1
            if _oriented_feature(row, "travel_advantage_diff") >= 1.0:
                signals.append("travel edge")
                support_count += 1
            if _oriented_feature(row, "gym_score_diff") >= 0.10:
                signals.append(f"{selection_gym_tier + '-tier ' if selection_gym_tier else ''}gym edge")
                support_count += 1
            elif _oriented_feature(row, "gym_score_diff") >= 0.05:
                signals.append(f"{selection_gym_tier + '-tier ' if selection_gym_tier else ''}camp quality")
                support_count += 1
            elif _tier_rank(selection_gym_tier) >= 4:
                signals.append(f"{selection_gym_tier}-tier camp")
                support_count += 1
            if _oriented_feature(row, "gym_depth_diff") >= 0.10:
                signals.append("better camp depth")
                support_count += 1
            if _oriented_feature(row, "stance_matchup_diff") >= 0.15:
                signals.append("stance matchup edge")
                support_count += 1
            if _oriented_feature(row, "recent_damage_diff") >= 0.30:
                signals.append("opponent recent damage risk")
                support_count += 1
            if _oriented_feature(row, "line_movement_toward_fighter") >= 0.02:
                signals.append("line moved toward pick")
                support_count += 1
            if consensus_count >= 3 and consensus_price_edge >= 0.015:
                signals.append("better than market consensus")
                support_count += 1
            if is_wmma >= 1 and odds < 0 and projected_prob >= 0.58:
                signals.append("wmma favorite profile")
                support_count += 1
            if float(row.get("is_heavyweight", 0.0) or 0.0) >= 1 and _oriented_feature(row, "durability_diff") >= 0.12:
                signals.append("heavyweight durability edge")
                support_count += 1
            if float(row.get("is_five_round_fight", 0.0) or 0.0) >= 1 and _oriented_feature(row, "cardio_fade_diff") >= 0.5:
                signals.append("five-round cardio edge")
                support_count += 1

            if selection_days > 420:
                risks.append("long_layoff")
            if selection_days < 45 and selection_ufc_fights > 0:
                risks.append("quick_turnaround")
            if selection_debut >= 1:
                risks.append("ufc_debut")
            if selection_ufc_fights < 3:
                risks.append("small_ufc_sample")
            if selection_fallback >= 1 or fallback_penalty > 0:
                risks.append("fallback_data")
            if selection_instability > 0:
                risks.append("fight_week_instability")
            if selection_injury >= 1:
                risks.append("injury_concern")
            if selection_weight_cut >= 1:
                risks.append("weight_cut_concern")
            if selection_replacement >= 1:
                risks.append("late_replacement")
            if selection_travel >= 1:
                risks.append("travel_disadvantage")
            if selection_camp_change >= 1:
                risks.append("camp_change")
            if float(row.get("selection_gym_changed_flag", 0.0) or 0.0) >= 1:
                risks.append("recent_gym_switch")
            if odds <= -300:
                risks.append("pricey_favorite")
            if is_opposite_model_side and is_positive_odds:
                risks.append("dog_flip")
            if market_blend_weight >= 0.40:
                risks.append("market_disagreement")
            if consensus_count >= 3 and consensus_price_edge <= -0.015:
                risks.append("worse_than_market_consensus")
            if market_overround >= 0.06:
                risks.append("high_vig_market")
            if is_wmma >= 1 and is_positive_odds and projected_prob < 0.53:
                risks.append("wmma_dog_thin_edge")
            if float(row.get("selection_recent_ko_damage", 0.0) or 0.0) >= 1:
                risks.append("recent_ko_damage")
            elif float(row.get("selection_recent_finish_damage", 0.0) or 0.0) >= 2:
                risks.append("recent_finish_damage")

            score = 0.0
            score += _clamp(edge / 0.12, 0.0, 1.5) * 35
            score += _clamp((projected_prob - 0.50) / 0.25, 0.0, 1.0) * 20
            score += confidence * 15
            score += data_quality * 10
            score += min(support_count, 6) * 4
            score += _clamp(_oriented_feature(row, "recent_form_diff") / 1.5, -0.5, 0.5) * 6
            score += _clamp(_oriented_feature(row, "strike_margin_diff") / 3.0, -0.5, 0.5) * 6
            score += _clamp(_oriented_feature(row, "grappling_diff") / 2.0, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "recent_control_diff") / 3.0, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "grappling_pressure_diff") / 2.5, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "matchup_striking_edge") / 2.0, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "matchup_grappling_edge") / 1.8, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "matchup_control_edge") / 1.5, -0.5, 0.5) * 2
            score += _clamp(_oriented_feature(row, "normalized_strike_margin_diff") / 3.0, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "normalized_grappling_diff") / 2.0, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "normalized_control_diff") / 2.5, -0.5, 0.5) * 2
            score += _clamp(_oriented_feature(row, "normalized_recent_form_diff") / 1.5, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "opponent_quality_diff") / 0.20, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "schedule_strength_diff") / 0.12, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "age_diff") / 6.0, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "strike_efficiency_diff") / 0.25, -0.5, 0.5) * 4
            score += _clamp(_oriented_feature(row, "travel_advantage_diff"), -1.0, 1.0) * 3
            score += _clamp(_oriented_feature(row, "gym_score_diff") / 0.20, -0.5, 0.5) * 6
            score += _clamp(_oriented_feature(row, "gym_depth_diff") / 0.25, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "stance_matchup_diff") / 0.4, -0.5, 0.5) * 3
            score += _clamp(_oriented_feature(row, "recent_damage_diff") / 1.5, -0.5, 0.5) * 5
            score += _clamp(consensus_price_edge / 0.05, -0.5, 0.5) * 8
            score -= max(0.0, fallback_penalty) * 10
            score -= 8 if selection_days > 420 else 0
            score -= 5 if selection_debut >= 1 else 0
            score -= 4 if selection_ufc_fights < 3 else 0
            score -= 4 if confidence < 0.65 else 0
            score -= 4 if data_quality < 0.85 else 0
            score -= 3 if odds <= -300 else 0
            score -= 9 if (is_opposite_model_side and is_positive_odds) else 0
            score -= max(0.0, market_blend_weight - 0.30) * 18
            score -= 3 if market_overround >= 0.06 else 0
            score -= 4 if consensus_count >= 3 and consensus_price_edge <= -0.015 else 0
            score = round(_clamp(score, 0.0, 100.0), 2)

            opposite_side_gate = not (is_opposite_model_side and is_positive_odds)
            if (
                score >= 82
                and edge >= 0.08
                and projected_prob >= 0.60
                and confidence >= 0.72
                and data_quality >= 0.90
                and "fallback_data" not in risks
                and "market_disagreement" not in risks
                and opposite_side_gate
            ):
                tier = "A"
                action = "Bettable now"
            elif (
                score >= 66
                and edge >= (0.10 if is_opposite_model_side and is_positive_odds else 0.05)
                and confidence >= (0.75 if is_opposite_model_side and is_positive_odds else 0.60)
                and data_quality >= (0.90 if is_opposite_model_side and is_positive_odds else 0.85)
                and "fallback_data" not in risks
            ):
                tier = "B"
                action = "Watchlist"
            else:
                tier = "C"
                action = "Pass"

            bet_quality_scores.append(score)
            recommended_tiers.append(tier)
            recommended_actions.append(action)
            support_labels.append(", ".join(signals[:6]))
            risk_flags.append(", ".join(risks))
            rationale.append(_format_reasons(signals[:6], risks))

        normalized["bet_quality_score"] = bet_quality_scores
        normalized["recommended_tier"] = recommended_tiers
        normalized["recommended_action"] = recommended_actions
        normalized["support_signals"] = support_labels
        normalized["risk_flags"] = risk_flags
        normalized["why_it_rates_well"] = rationale
        normalized["support_count"] = [len([item for item in value.split(", ") if item]) for value in support_labels]
        normalized["risk_flag_count"] = [len([item for item in value.split(", ") if item]) for value in risk_flags]

    normalized = _apply_expression_overrides(normalized, args.fight_report, probability_column)
    normalized = _attach_diagnostic_context(normalized, db_path=args.db)
    normalized = _apply_selective_model(normalized, selective_model_path)
    normalized["chosen_expression_expected_value"] = normalized.apply(
        lambda row: expected_value(float(row["chosen_expression_prob"]), int(row["chosen_expression_odds"])),
        axis=1,
    )
    normalized["chosen_expression_stake"] = normalized.apply(
        lambda row: suggested_stake(
            bankroll=bankroll,
            projected_win_prob=float(row["chosen_expression_prob"]),
            american_odds=int(row["chosen_expression_odds"]),
            fraction=fractional_kelly,
        ),
        axis=1,
    )
    normalized = apply_bankroll_governor(
        normalized,
        bankroll=bankroll,
        config=bankroll_governor,
    )
    normalized["effective_projected_prob"] = normalized["chosen_expression_prob"]
    normalized["effective_implied_prob"] = normalized["chosen_expression_implied_prob"]
    normalized["effective_edge"] = normalized["chosen_expression_edge"]
    normalized["effective_expected_value"] = normalized["chosen_expression_expected_value"]
    normalized["effective_suggested_stake"] = normalized["chosen_expression_stake"]
    normalized["effective_american_odds"] = normalized["chosen_expression_odds"]
    normalized = _apply_diagnostic_overrides(normalized)
    normalized = _apply_bankroll_overrides(normalized)

    report = _filter_and_rank_report(
        normalized,
        min_edge=min_edge,
        min_model_confidence=min_model_confidence,
        min_stats_completeness=min_stats_completeness,
        exclude_fallback_rows=exclude_fallback_rows,
    )

    output_path = Path(args.output)
    shortlist_path = Path(args.shortlist_output) if args.shortlist_output else _default_shortlist_path(output_path)
    board_path = Path(args.board_output) if args.board_output else _default_board_path(output_path)
    passes_path = Path(args.passes_output) if args.passes_output else _default_passes_path(output_path)
    shortlist, _board = _write_scan_outputs(
        report,
        normalized,
        output_path=output_path,
        shortlist_path=shortlist_path,
        board_path=board_path,
        passes_path=passes_path,
        min_edge=min_edge,
        min_model_confidence=min_model_confidence,
        min_stats_completeness=min_stats_completeness,
        exclude_fallback_rows=exclude_fallback_rows,
    )
    if args.db and not report.empty:
        saved = save_tracked_picks(report, args.db)
        if not args.quiet:
            print(f"Tracked {saved} picks in {args.db}")

    if not args.quiet and bool(threshold_settings.get("policy_applied")):
        print(f"Threshold policy: {threshold_settings.get('policy_summary', '')}")

    if report.empty:
        if not args.quiet:
            print("No qualifying bets found for the configured edge threshold.")
        return

    if not args.quiet:
        if "bet_quality_score" in report.columns and "recommended_tier" in report.columns:
            _print_console_summary(report, shortlist, probability_column)
        else:
            columns = [
                "event_name",
                "market",
                "selection_name",
                "book",
                "american_odds",
                probability_column,
                "implied_prob",
                "edge",
                "expected_value",
                "suggested_stake",
            ]
            print(report[columns].to_string(index=False))
        print(f"\nSaved report to {output_path}")
        print(f"Saved shortlist to {shortlist_path}")
        print(f"Saved betting board to {board_path}")
        print(f"Saved pass reasons to {passes_path}")


if __name__ == "__main__":
    main()
