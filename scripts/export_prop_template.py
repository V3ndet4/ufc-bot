from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_manual_props import _american_to_decimal, _probability_to_american_odds
from scripts.event_manifest import DEFAULT_MAIN_CARD_FIGHT_COUNT, is_main_card_fight


PROP_TEMPLATE_SPECS = [
    ("moneyline", "fighter_a_model_win_prob", "fighter"),
    ("moneyline", "fighter_b_model_win_prob", "fighter"),
    ("inside_distance", "fighter_a_inside_distance_prob", "fighter"),
    ("inside_distance", "fighter_b_inside_distance_prob", "fighter"),
    ("submission", "fighter_a_submission_prob", "fighter"),
    ("submission", "fighter_b_submission_prob", "fighter"),
    ("ko_tko", "fighter_a_ko_tko_prob", "fighter"),
    ("ko_tko", "fighter_b_ko_tko_prob", "fighter"),
    ("fight_ends_by_submission", "fight_ends_by_submission_model_prob", "fight"),
    ("fight_ends_by_ko_tko", "fight_ends_by_ko_tko_model_prob", "fight"),
    ("knockdown", "fighter_a_knockdown_prop_prob", "fighter"),
    ("knockdown", "fighter_b_knockdown_prop_prob", "fighter"),
    ("takedown", "fighter_a_takedown_prop_prob", "fighter"),
    ("takedown", "fighter_b_takedown_prop_prob", "fighter"),
    ("by_decision", "fighter_a_by_decision_prob", "fighter"),
    ("by_decision", "fighter_b_by_decision_prob", "fighter"),
    ("fight_goes_to_decision", "fight_goes_to_decision_model_prob", "fight"),
    ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision_model_prob", "fight"),
]

PROP_TEMPLATE_COLUMNS = [
    "fighter_a",
    "fighter_b",
    "is_main_card",
    "selection_name",
    "prop_type",
    "prop_label",
    "american_odds",
    "sportsbook",
    "model_prob",
    "fair_american_odds",
    "fair_decimal_odds",
    "notes",
]

PROP_LEAN_MIN_PROB = {
    "inside_distance": 0.24,
    "submission": 0.16,
    "ko_tko": 0.16,
    "fight_ends_by_submission": 0.22,
    "fight_ends_by_ko_tko": 0.28,
    "knockdown": 0.18,
    "takedown": 0.35,
    "by_decision": 0.26,
    "fight_goes_to_decision": 0.55,
    "fight_doesnt_go_to_decision": 0.55,
}

PROP_LEAN_BONUS = {
    "fight_doesnt_go_to_decision": 0.06,
    "fight_goes_to_decision": 0.05,
    "inside_distance": 0.04,
    "submission": 0.05,
    "ko_tko": 0.05,
    "fight_ends_by_submission": 0.05,
    "fight_ends_by_ko_tko": 0.05,
    "knockdown": 0.06,
    "takedown": 0.06,
    "by_decision": 0.02,
}

PROP_LEAN_TYPE_LIMITS = {
    "fight_doesnt_go_to_decision": 4,
    "fight_goes_to_decision": 3,
    "inside_distance": 2,
    "submission": 2,
    "ko_tko": 2,
    "fight_ends_by_submission": 2,
    "fight_ends_by_ko_tko": 2,
    "knockdown": 3,
    "takedown": 3,
    "by_decision": 2,
}

FORCED_CONSOLE_PROP_TYPES = ["knockdown", "takedown"]
MAIN_CARD_ONLY_PROP_TYPES = {"knockdown", "takedown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a manual prop entry template from a fight-week report.")
    parser.add_argument("--fight-report", required=True, help="Fight-week report CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--min-model-prob",
        type=float,
        default=0.08,
        help="Minimum model probability required to include a prop row.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export every supported prop row instead of a curated subset.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _include_curated_prop(report_row: dict[str, object], prop_type: str, model_prob: float, fair_american: object) -> bool:
    decision_prob = float(report_row.get("projected_decision_prob", 0.0) or 0.0)
    finish_prob = float(report_row.get("projected_finish_prob", 0.0) or 0.0)
    if prop_type == "moneyline":
        return model_prob >= 0.22
    if prop_type == "inside_distance":
        return model_prob >= 0.18 and pd.notna(fair_american) and int(fair_american) >= 125
    if prop_type == "submission":
        return model_prob >= 0.10 and pd.notna(fair_american) and int(fair_american) >= 220
    if prop_type == "ko_tko":
        return model_prob >= 0.12 and pd.notna(fair_american) and int(fair_american) >= 180
    if prop_type == "fight_ends_by_submission":
        return model_prob >= 0.16 and pd.notna(fair_american) and int(fair_american) >= 180
    if prop_type == "fight_ends_by_ko_tko":
        return model_prob >= 0.20 and pd.notna(fair_american) and int(fair_american) >= 130
    if prop_type == "knockdown":
        return model_prob >= 0.16 and pd.notna(fair_american) and int(fair_american) >= 125
    if prop_type == "takedown":
        return model_prob >= 0.32 and pd.notna(fair_american)
    if prop_type == "by_decision":
        return model_prob >= 0.14 and decision_prob >= 0.42 and pd.notna(fair_american) and int(fair_american) >= 180
    if prop_type == "fight_goes_to_decision":
        return decision_prob >= 0.22 and pd.notna(fair_american) and int(fair_american) >= 120
    if prop_type == "fight_doesnt_go_to_decision":
        return finish_prob >= 0.22 and pd.notna(fair_american) and int(fair_american) >= 120
    return False


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _truthy_flag(value: object, default: bool = True) -> bool:
    if value is None or pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "main", "main_card"}:
        return True
    if text in {"0", "false", "no", "n", "prelim", "prelims"}:
        return False
    return default


def _main_card_keys_from_event_manifest(fight_report: pd.DataFrame) -> set[str]:
    event_names = (
        set(fight_report["event_name"].dropna().astype(str).str.strip())
        if "event_name" in fight_report.columns
        else set()
    )
    event_ids = (
        set(fight_report["event_id"].dropna().astype(str).str.strip())
        if "event_id" in fight_report.columns
        else set()
    )
    if not event_names and not event_ids:
        return set()

    keys: set[str] = set()
    for manifest_path in sorted((ROOT / "events").glob("*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        manifest_event_id = str(manifest.get("event_id", "")).strip()
        manifest_event_name = str(manifest.get("event_name", "")).strip()
        if manifest_event_id not in event_ids and manifest_event_name not in event_names:
            continue

        for fight_index, fight in enumerate(manifest.get("fights", [])):
            if not isinstance(fight, dict) or not is_main_card_fight(fight, fight_index, manifest):
                continue
            fighter_a = str(fight.get("fighter_a", "")).strip()
            fighter_b = str(fight.get("fighter_b", "")).strip()
            if fighter_a and fighter_b:
                keys.add(f"{fighter_a}||{fighter_b}")
                keys.add(f"{fighter_b}||{fighter_a}")
    return keys


def _ensure_main_card_column(fight_report: pd.DataFrame) -> pd.DataFrame:
    prepared = fight_report.copy()
    if "is_main_card" in prepared.columns:
        prepared["is_main_card"] = prepared["is_main_card"].apply(lambda value: int(_truthy_flag(value, default=False)))
        return prepared

    if {"fighter_a", "fighter_b"}.issubset(prepared.columns):
        fight_keys = prepared["fighter_a"].astype(str) + "||" + prepared["fighter_b"].astype(str)
        main_card_keys = _main_card_keys_from_event_manifest(prepared)
        if not main_card_keys:
            main_card_keys = set(fight_keys.drop_duplicates().head(DEFAULT_MAIN_CARD_FIGHT_COUNT))
        prepared["is_main_card"] = fight_keys.isin(main_card_keys).astype(int)
    else:
        prepared["is_main_card"] = 1
    return prepared


def _is_main_card_report_row(report_row: dict[str, object]) -> bool:
    return _truthy_flag(report_row.get("is_main_card"), default=True)


def _knockdown_prop_probability(row: pd.Series, side: str) -> float:
    own_prefix = f"{side}_"
    opponent_side = "fighter_b" if side == "fighter_a" else "fighter_a"
    opponent_prefix = f"{opponent_side}_"
    knockdown_avg = _safe_float(row.get(f"{own_prefix}knockdown_avg"), 0.0)
    if knockdown_avg <= 0.0:
        return 0.0

    scheduled_rounds = _safe_float(row.get("scheduled_rounds"), 3.0)
    exposure = max(0.75, min(1.45, scheduled_rounds / 3.0))
    own_ko_rate = _safe_float(row.get(f"{own_prefix}ko_win_rate"), 0.0)
    opponent_ko_loss_rate = _safe_float(row.get(f"{opponent_prefix}ko_loss_rate"), 0.0)
    own_strike_rate = _safe_float(row.get(f"{own_prefix}sig_strikes_landed_per_min"), 0.0)
    opponent_absorbed_rate = _safe_float(row.get(f"{opponent_prefix}sig_strikes_absorbed_per_min"), 0.0)
    distance_share = _safe_float(row.get(f"{own_prefix}distance_strike_share"), 0.55)

    pressure_multiplier = (
        1.0
        + (own_ko_rate * 0.22)
        + (opponent_ko_loss_rate * 0.18)
        + (max(0.0, own_strike_rate - 4.0) * 0.035)
        + (max(0.0, opponent_absorbed_rate - 4.0) * 0.025)
        + (max(0.0, distance_share - 0.55) * 0.20)
    )
    knockdown_expectation = knockdown_avg * exposure * max(0.75, min(1.45, pressure_multiplier))
    probability = 1 - pow(2.718281828459045, -knockdown_expectation)
    return round(max(0.0, min(0.72, probability)), 4)


def _takedown_prop_probability(row: pd.Series, side: str) -> float:
    own_prefix = f"{side}_"
    opponent_side = "fighter_b" if side == "fighter_a" else "fighter_a"
    opponent_prefix = f"{opponent_side}_"
    takedown_avg = _safe_float(row.get(f"{own_prefix}takedown_avg"), 0.0)
    if takedown_avg <= 0.0:
        return 0.0

    scheduled_rounds = _safe_float(row.get("scheduled_rounds"), 3.0)
    exposure = max(0.75, min(1.45, scheduled_rounds / 3.0))
    opponent_td_def = _safe_float(row.get(f"{opponent_prefix}takedown_defense_pct"), 68.0)
    recent_grappling = _safe_float(row.get(f"{own_prefix}recent_grappling_rate"), 0.0)
    control_avg = _safe_float(row.get(f"{own_prefix}control_avg"), 0.0)
    recent_control = _safe_float(row.get(f"{own_prefix}recent_control_avg"), 0.0)
    grappling_edge = _safe_float(row.get("matchup_grappling_edge"), 0.0)
    if side == "fighter_b":
        grappling_edge *= -1

    defense_multiplier = 1.0 + max(-0.28, min(0.36, (68.0 - opponent_td_def) / 100.0))
    pressure_multiplier = (
        1.0
        + (max(0.0, recent_grappling - 1.0) * 0.045)
        + (max(0.0, control_avg - 1.0) * 0.035)
        + (max(0.0, recent_control - 1.0) * 0.040)
        + (max(-0.18, min(0.28, grappling_edge * 0.08)))
    )
    takedown_expectation = takedown_avg * exposure * max(0.65, min(1.60, defense_multiplier * pressure_multiplier))
    probability = 1 - pow(2.718281828459045, -takedown_expectation)
    return round(max(0.0, min(0.88, probability)), 4)


def _prepare_prop_report(fight_report: pd.DataFrame) -> pd.DataFrame:
    prepared = _ensure_main_card_column(fight_report)
    for side in ["fighter_a", "fighter_b"]:
        knockdown_column = f"{side}_knockdown_prop_prob"
        if knockdown_column not in prepared.columns:
            prepared[knockdown_column] = prepared.apply(lambda row: _knockdown_prop_probability(row, side), axis=1)
        takedown_column = f"{side}_takedown_prop_prob"
        if takedown_column not in prepared.columns:
            prepared[takedown_column] = prepared.apply(lambda row: _takedown_prop_probability(row, side), axis=1)
    if "fight_ends_by_submission_model_prob" not in prepared.columns:
        prepared["fight_ends_by_submission_model_prob"] = (
            _numeric_report_column(prepared, "fighter_a_submission_prob")
            + _numeric_report_column(prepared, "fighter_b_submission_prob")
        ).clip(lower=0.0, upper=0.97)
    if "fight_ends_by_ko_tko_model_prob" not in prepared.columns:
        prepared["fight_ends_by_ko_tko_model_prob"] = (
            _numeric_report_column(prepared, "fighter_a_ko_tko_prob")
            + _numeric_report_column(prepared, "fighter_b_ko_tko_prob")
        ).clip(lower=0.0, upper=0.97)
    return prepared


def _numeric_report_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def export_prop_template(
    fight_report: pd.DataFrame,
    *,
    min_model_prob: float = 0.08,
    full: bool = False,
) -> pd.DataFrame:
    fight_report = _prepare_prop_report(fight_report)
    rows: list[dict[str, object]] = []
    for report_row in fight_report.to_dict("records"):
        fighter_a = str(report_row["fighter_a"]).strip()
        fighter_b = str(report_row["fighter_b"]).strip()
        is_main_card = _is_main_card_report_row(report_row)
        for prop_type, model_column, scope in PROP_TEMPLATE_SPECS:
            if prop_type in MAIN_CARD_ONLY_PROP_TYPES and not is_main_card:
                continue
            model_prob = report_row.get(model_column)
            if pd.isna(model_prob) or float(model_prob) < min_model_prob:
                continue

            fair_american = _probability_to_american_odds(model_prob)
            fair_decimal = _american_to_decimal(fair_american)
            if not full and not _include_curated_prop(report_row, prop_type, float(model_prob), fair_american):
                continue
            if scope == "fight":
                selection_name = ""
                fight_prop_labels = {
                    "fight_goes_to_decision": "Fight goes to decision",
                    "fight_doesnt_go_to_decision": "Fight doesn't go the distance",
                    "fight_ends_by_submission": "Fight ends by submission",
                    "fight_ends_by_ko_tko": "Fight ends by KO/TKO",
                }
                prop_label = fight_prop_labels[prop_type]
            else:
                selection_name = fighter_a if "_a_" in model_column else fighter_b
                suffix_map = {
                    "moneyline": "moneyline",
                    "inside_distance": "inside distance",
                    "submission": "submission",
                    "ko_tko": "KO/TKO",
                    "knockdown": "knockdown",
                    "takedown": "takedown",
                    "by_decision": "by decision",
                }
                prop_label = f"{selection_name} {suffix_map[prop_type]}"

            rows.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "is_main_card": int(is_main_card),
                    "selection_name": selection_name,
                    "prop_type": prop_type,
                    "prop_label": prop_label,
                    "american_odds": "",
                    "sportsbook": "",
                    "model_prob": round(float(model_prob), 4),
                    "fair_american_odds": fair_american,
                    "fair_decimal_odds": fair_decimal,
                    "notes": "",
                }
            )

    if not rows:
        return pd.DataFrame(columns=PROP_TEMPLATE_COLUMNS)
    return (
        pd.DataFrame(rows, columns=PROP_TEMPLATE_COLUMNS)
        .sort_values(by=["fighter_a", "fighter_b", "prop_type", "selection_name"])
        .reset_index(drop=True)
    )


def _format_percent(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def _format_american(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    odds = int(float(value))
    return f"+{odds}" if odds > 0 else str(odds)


def _prop_lean_note(prop_type: str, model_prob: float) -> str:
    if prop_type == "fight_doesnt_go_to_decision":
        return "finish lean"
    if prop_type == "fight_goes_to_decision":
        return "decision lean"
    if prop_type == "inside_distance":
        return "inside-distance lean"
    if prop_type == "submission":
        return "submission lean"
    if prop_type == "ko_tko":
        return "KO/TKO lean"
    if prop_type == "fight_ends_by_submission":
        return "fight submission lean"
    if prop_type == "fight_ends_by_ko_tko":
        return "fight KO/TKO lean"
    if prop_type == "knockdown":
        return "knockdown lean"
    if prop_type == "takedown":
        return "takedown lean"
    if prop_type == "by_decision":
        return "fighter decision lean"
    return "model lean"


def build_prop_leans(
    fight_report: pd.DataFrame,
    *,
    limit: int = 8,
    min_model_prob: float = 0.08,
) -> pd.DataFrame:
    template = export_prop_template(fight_report, min_model_prob=min_model_prob, full=True)
    if template.empty:
        return pd.DataFrame(columns=[*template.columns, "lean_score", "lean_note"])

    props = template.loc[template["prop_type"].ne("moneyline")].copy()
    if props.empty:
        return props
    props["model_prob_numeric"] = pd.to_numeric(props["model_prob"], errors="coerce").fillna(0.0)
    props["min_required_prob"] = props["prop_type"].map(PROP_LEAN_MIN_PROB).fillna(0.50)
    props = props.loc[props["model_prob_numeric"] >= props["min_required_prob"]].copy()
    if props.empty:
        return props

    props["lean_score"] = props["model_prob_numeric"] + props["prop_type"].map(PROP_LEAN_BONUS).fillna(0.0)
    props["lean_note"] = props.apply(
        lambda row: _prop_lean_note(str(row["prop_type"]), float(row["model_prob_numeric"])),
        axis=1,
    )
    props = props.sort_values(
        by=["lean_score", "model_prob_numeric", "fair_american_odds"],
        ascending=[False, False, True],
    )

    def _select_rows(frame: pd.DataFrame, limit_value: int) -> list[int]:
        selected: list[int] = []
        fight_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        for row in frame.itertuples():
            fight_key = f"{row.fighter_a}||{row.fighter_b}"
            prop_type = str(row.prop_type)
            if fight_counts.get(fight_key, 0) >= 2:
                continue
            if type_counts.get(prop_type, 0) >= PROP_LEAN_TYPE_LIMITS.get(prop_type, limit_value):
                continue
            selected.append(row.Index)
            fight_counts[fight_key] = fight_counts.get(fight_key, 0) + 1
            type_counts[prop_type] = type_counts.get(prop_type, 0) + 1
            if len(selected) >= limit_value:
                break
        return selected

    def _fight_counts_for_indexes(indexes: list[int]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in props.loc[indexes].itertuples():
            fight_key = f"{row.fighter_a}||{row.fighter_b}"
            counts[fight_key] = counts.get(fight_key, 0) + 1
        return counts

    keep_indexes: list[int] = []
    if limit > 0:
        keep_indexes = _select_rows(props, limit)
        for forced_prop_type in FORCED_CONSOLE_PROP_TYPES:
            selected_props = props.loc[keep_indexes] if keep_indexes else pd.DataFrame()
            has_forced_type = (
                not selected_props.empty
                and selected_props["prop_type"].astype(str).eq(forced_prop_type).any()
            )
            available_forced_rows = props.loc[props["prop_type"].astype(str).eq(forced_prop_type)]
            if has_forced_type or available_forced_rows.empty:
                continue
            fight_counts = _fight_counts_for_indexes(keep_indexes)
            chosen_forced_index = None
            for row in available_forced_rows.itertuples():
                fight_key = f"{row.fighter_a}||{row.fighter_b}"
                if fight_counts.get(fight_key, 0) < 3:
                    chosen_forced_index = row.Index
                    break
            if chosen_forced_index is not None:
                if len(keep_indexes) >= limit:
                    removable = props.loc[keep_indexes]
                    non_forced_rows = removable.loc[
                        ~removable["prop_type"].astype(str).isin(FORCED_CONSOLE_PROP_TYPES)
                    ]
                    if not non_forced_rows.empty:
                        remove_index = non_forced_rows.sort_values(
                            by=["lean_score", "model_prob_numeric"],
                            ascending=[True, True],
                        ).index[0]
                        keep_indexes.remove(remove_index)
                if len(keep_indexes) < limit:
                    keep_indexes.append(chosen_forced_index)
    return props.loc[keep_indexes].reset_index(drop=True)


def format_prop_leans_summary(
    fight_report: pd.DataFrame,
    *,
    limit: int = 8,
    min_model_prob: float = 0.08,
) -> str:
    prop_leans = build_prop_leans(fight_report, limit=limit, min_model_prob=min_model_prob)
    if prop_leans.empty:
        return "Prop leans: no qualifying model-only prop leans.\n"

    lines = [f"Prop leans: top {len(prop_leans)} model-only fair lines"]
    for row in prop_leans.itertuples(index=False):
        lines.append(
            f"- {row.fighter_a} vs {row.fighter_b} | {row.prop_label} | "
            f"model {_format_percent(row.model_prob)} | fair {_format_american(row.fair_american_odds)} | {row.lean_note}"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    fight_report = pd.read_csv(args.fight_report)
    template = export_prop_template(
        fight_report,
        min_model_prob=args.min_model_prob,
        full=args.full,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    if not args.quiet:
        print(f"Exported {len(template)} prop rows")
        print(f"Saved prop template to {output_path}")


if __name__ == "__main__":
    main()
