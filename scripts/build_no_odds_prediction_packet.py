from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.fighter_features import build_fight_features, load_fighter_stats
from models.confidence import default_confidence_model_path, load_confidence_model
from models.ev import probability_to_american
from models.projection import project_fight_probabilities
from models.side import default_side_model_path, load_side_model
from normalization.odds import normalize_odds_frame
from scripts.event_manifest import derived_paths, load_manifest
from scripts.run_core_scan import (
    _camp_summary,
    _context_summary,
    _derived_risk_flags,
    _news_summary,
    _oriented,
    _safe_float,
    _safe_text,
    _watch_for,
)


NO_ODDS_COLUMNS = [
    "event_name",
    "fight",
    "lean_strength",
    "lean_side",
    "opponent",
    "model_prob",
    "fair_line",
    "confidence",
    "data_quality",
    "method_lean",
    "pick_best_method",
    "projected_finish_prob",
    "projected_decision_prob",
    "pick_gym_name",
    "pick_gym_tier",
    "pick_gym_record",
    "opponent_gym_name",
    "opponent_gym_tier",
    "opponent_gym_record",
    "camp_summary",
    "top_reasons",
    "risk_flags",
    "context_summary",
    "news_summary",
    "watch_for",
    "fighter_a",
    "fighter_b",
    "scheduled_rounds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-odds UFC leans from cached stats, gyms, and context.")
    parser.add_argument("--manifest", required=True, help="Event manifest JSON.")
    parser.add_argument("--fighter-stats", help="Cached fighter stats CSV. Defaults from --manifest.")
    parser.add_argument("--output", help="Output CSV. Defaults to cards/<slug>/reports/no_odds_prediction_packet.csv.")
    parser.add_argument("--side-model", help="Optional calibrated side-model pickle path.")
    parser.add_argument("--confidence-model", help="Optional calibrated confidence-model pickle path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary.")
    return parser.parse_args()


def _pseudo_even_money_odds(manifest: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fight_index, fight in enumerate(manifest["fights"]):
        fighter_a = str(fight["fighter_a"]).strip()
        fighter_b = str(fight["fighter_b"]).strip()
        for selection in ("fighter_a", "fighter_b"):
            rows.append(
                {
                    "event_id": manifest["event_id"],
                    "event_name": manifest["event_name"],
                    "start_time": manifest["start_time"],
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "scheduled_rounds": float(fight.get("scheduled_rounds", 5 if fight_index == 0 else 3)),
                    "is_title_fight": int(fight.get("is_title_fight", 0)),
                    "market": "moneyline",
                    "selection": selection,
                    "book": "no_odds",
                    "american_odds": 100,
                }
            )
    return normalize_odds_frame(pd.DataFrame(rows))


def _lean_strength(probability: float, confidence: float) -> str:
    if probability >= 0.62 and confidence >= 0.72:
        return "Strong Lean"
    if probability >= 0.57 and confidence >= 0.64:
        return "Lean"
    if probability >= 0.53:
        return "Slight Lean"
    return "Coin Flip"


def _method_lean(row: pd.Series) -> str:
    finish = _safe_float(row.get("projected_finish_prob"), 0.0)
    decision = _safe_float(row.get("projected_decision_prob"), 0.0)
    if finish >= 0.57:
        return "Finish lean"
    if decision >= 0.57:
        return "Decision lean"
    return "Mixed method"


def _pick_best_method(row: pd.Series, pick_side: str) -> str:
    prefix = "fighter_a" if pick_side == "fighter_a" else "fighter_b"
    choices = [
        ("KO/TKO", _safe_float(row.get(f"{prefix}_ko_tko_prob"), 0.0)),
        ("Submission", _safe_float(row.get(f"{prefix}_submission_prob"), 0.0)),
        ("Decision", _safe_float(row.get(f"{prefix}_by_decision_prob"), 0.0)),
    ]
    label, probability = max(choices, key=lambda item: item[1])
    return f"{label} ({probability:.1%})"


def _support_reasons(row: pd.Series) -> str:
    reasons: list[tuple[float, str]] = []
    strike_margin = _oriented(row, "strike_margin_diff")
    striking = _oriented(row, "matchup_striking_edge")
    grappling = _oriented(row, "matchup_grappling_edge")
    control = _oriented(row, "matchup_control_edge")
    reach = _oriented(row, "reach_diff")
    schedule = _oriented(row, "schedule_strength_diff")
    pick_age = _safe_float(row.get(f"{_side_prefix(str(row.get('selection')))}_age_years"), 0.0)
    opponent_age = _safe_float(row.get(f"{_opponent_prefix(str(row.get('selection')))}_age_years"), 0.0)
    age_edge = opponent_age - pick_age if pick_age > 0 and opponent_age > 0 else 0.0

    if strike_margin >= 0.5:
        reasons.append((strike_margin, f"striking margin +{strike_margin:.2f}/min"))
    if striking >= 0.75:
        reasons.append((striking, f"matchup striking +{striking:.2f}"))
    if grappling >= 0.20:
        reasons.append((grappling, f"grappling +{grappling:.2f}"))
    if control >= 0.25:
        reasons.append((control, f"control +{control:.2f}"))
    if reach >= 2.0:
        reasons.append((reach / 2.0, f"reach +{reach:.0f} in"))
    if age_edge >= 4.0:
        reasons.append((age_edge / 4.0, f"younger by {age_edge:.1f}y"))
    if schedule >= 0.08:
        reasons.append((schedule * 10.0, f"schedule strength +{schedule:.2f}"))

    if not reasons:
        return "Composite model lean; no single support signal cleared threshold"
    return ", ".join(reason for _, reason in sorted(reasons, reverse=True)[:3])


def _technical_cautions(row: pd.Series) -> str:
    cautions: list[tuple[float, str]] = []
    strike_margin = _oriented(row, "strike_margin_diff")
    striking = _oriented(row, "matchup_striking_edge")
    grappling = _oriented(row, "matchup_grappling_edge")
    control = _oriented(row, "matchup_control_edge")
    reach = _oriented(row, "reach_diff")
    schedule = _oriented(row, "schedule_strength_diff")
    pick_age = _safe_float(row.get(f"{_side_prefix(str(row.get('selection')))}_age_years"), 0.0)
    opponent_age = _safe_float(row.get(f"{_opponent_prefix(str(row.get('selection')))}_age_years"), 0.0)
    age_edge = opponent_age - pick_age if pick_age > 0 and opponent_age > 0 else 0.0

    if strike_margin <= -0.5:
        cautions.append((abs(strike_margin), f"opponent striking margin {strike_margin:.2f}/min"))
    if striking <= -0.75:
        cautions.append((abs(striking), f"opponent matchup striking {striking:.2f}"))
    if grappling <= -0.20:
        cautions.append((abs(grappling), f"opponent grappling {grappling:.2f}"))
    if control <= -0.25:
        cautions.append((abs(control), f"opponent control {control:.2f}"))
    if reach <= -2.0:
        cautions.append((abs(reach) / 2.0, f"opponent reach {abs(reach):.0f} in"))
    if age_edge <= -4.0:
        cautions.append((abs(age_edge) / 4.0, f"older by {abs(age_edge):.1f}y"))
    if schedule <= -0.08:
        cautions.append((abs(schedule) * 10.0, f"opponent schedule strength {schedule:.2f}"))

    if not cautions:
        return ""
    return ", ".join(reason for _, reason in sorted(cautions, reverse=True)[:3])


def _merge_risks(*parts: str) -> str:
    risks: list[str] = []
    for part in parts:
        text = _safe_text(part)
        if not text or text == "none":
            continue
        risks.extend(item.strip() for item in text.split(",") if item.strip())
    return ", ".join(dict.fromkeys(risks)) if risks else "none"


def _side_prefix(selection: str) -> str:
    return "a" if selection == "fighter_a" else "b"


def _opponent_prefix(selection: str) -> str:
    return "b" if selection == "fighter_a" else "a"


def _side_text(row: pd.Series, prefix: str, suffix: str) -> str:
    return _safe_text(row.get(f"{prefix}_{suffix}", ""))


def _prepare_no_market_features(odds: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
    features = build_fight_features(odds, fighter_stats)
    for column in [
        "fighter_a_current_implied_prob",
        "fighter_b_current_implied_prob",
        "market_target_fair_prob",
        "market_consensus_prob",
    ]:
        if column in features.columns:
            features[column] = pd.NA
    return features


def build_no_odds_prediction_packet(
    manifest: dict[str, object],
    fighter_stats: pd.DataFrame,
    *,
    side_model_bundle: dict[str, object] | None = None,
    confidence_model_bundle: dict[str, object] | None = None,
) -> pd.DataFrame:
    odds = _pseudo_even_money_odds(manifest)
    features = _prepare_no_market_features(odds, fighter_stats)
    projected = project_fight_probabilities(
        features,
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )

    fight_rows = projected.loc[projected["selection"].eq("fighter_a")].copy()
    packet_rows: list[dict[str, object]] = []
    for _, row in fight_rows.iterrows():
        fighter_a_prob = _safe_float(row.get("projected_fighter_a_win_prob"), 0.5)
        pick_side = "fighter_a" if fighter_a_prob >= 0.5 else "fighter_b"
        pick_prefix = _side_prefix(pick_side)
        opponent_prefix = _opponent_prefix(pick_side)
        pick = str(row["fighter_a"] if pick_side == "fighter_a" else row["fighter_b"])
        opponent = str(row["fighter_b"] if pick_side == "fighter_a" else row["fighter_a"])
        pick_prob = fighter_a_prob if pick_side == "fighter_a" else 1 - fighter_a_prob
        selected_row = row.copy()
        selected_row["selection"] = pick_side
        selected_row["selection_name"] = pick
        selected_row["pick"] = pick
        selected_row["opponent"] = opponent
        selected_row["fight"] = f"{row['fighter_a']} vs {row['fighter_b']}"
        selected_row["model_projected_win_prob"] = pick_prob
        selected_row["model_prob"] = pick_prob
        selected_row["raw_model_prob"] = pick_prob
        selected_row["market_prob"] = pick_prob
        selected_row["confidence"] = _safe_float(row.get("model_confidence"), 0.0)
        selected_row["data_quality"] = min(
            _safe_float(row.get("a_stats_completeness"), 0.0),
            _safe_float(row.get("b_stats_completeness"), 0.0),
        )
        selected_row["pick_gym_name"] = _side_text(row, pick_prefix, "gym_name")
        selected_row["pick_gym_tier"] = _side_text(row, pick_prefix, "gym_tier").upper()
        selected_row["pick_gym_record"] = _side_text(row, pick_prefix, "gym_record")
        selected_row["opponent_gym_name"] = _side_text(row, opponent_prefix, "gym_name")
        selected_row["opponent_gym_tier"] = _side_text(row, opponent_prefix, "gym_tier").upper()
        selected_row["opponent_gym_record"] = _side_text(row, opponent_prefix, "gym_record")
        selected_row["top_reasons"] = _support_reasons(selected_row)
        selected_row["risk_flags"] = _merge_risks(_derived_risk_flags(selected_row), _technical_cautions(selected_row))
        selected_row["context_summary"] = _context_summary(selected_row)
        selected_row["news_summary"] = _news_summary(selected_row)
        selected_row["watch_for"] = _watch_for(selected_row)

        packet_rows.append(
            {
                "event_name": row["event_name"],
                "fight": selected_row["fight"],
                "lean_strength": _lean_strength(pick_prob, _safe_float(row.get("model_confidence"), 0.0)),
                "lean_side": pick,
                "opponent": opponent,
                "model_prob": round(pick_prob, 4),
                "fair_line": probability_to_american(max(0.001, min(0.999, pick_prob))),
                "confidence": round(_safe_float(row.get("model_confidence"), 0.0), 4),
                "data_quality": round(_safe_float(selected_row.get("data_quality"), 0.0), 4),
                "method_lean": _method_lean(row),
                "pick_best_method": _pick_best_method(row, pick_side),
                "projected_finish_prob": round(_safe_float(row.get("projected_finish_prob"), 0.0), 4),
                "projected_decision_prob": round(_safe_float(row.get("projected_decision_prob"), 0.0), 4),
                "pick_gym_name": selected_row["pick_gym_name"],
                "pick_gym_tier": selected_row["pick_gym_tier"],
                "pick_gym_record": selected_row["pick_gym_record"],
                "opponent_gym_name": selected_row["opponent_gym_name"],
                "opponent_gym_tier": selected_row["opponent_gym_tier"],
                "opponent_gym_record": selected_row["opponent_gym_record"],
                "camp_summary": _camp_summary(selected_row),
                "top_reasons": selected_row["top_reasons"],
                "risk_flags": selected_row["risk_flags"],
                "context_summary": selected_row["context_summary"],
                "news_summary": selected_row["news_summary"],
                "watch_for": selected_row["watch_for"],
                "fighter_a": row["fighter_a"],
                "fighter_b": row["fighter_b"],
                "scheduled_rounds": _safe_float(row.get("scheduled_rounds"), 3.0),
            }
        )
    return pd.DataFrame(packet_rows, columns=NO_ODDS_COLUMNS)


def _print_summary(packet: pd.DataFrame, output_path: Path) -> None:
    print(f"No-odds prediction packet: {len(packet)} fights")
    for row in packet.itertuples(index=False):
        print(
            f"{row.lean_strength}: {row.lean_side} over {row.opponent} "
            f"({row.model_prob:.1%}, fair {row.fair_line}, conf {row.confidence:.2f})"
        )
        print(f"  Gyms: {row.camp_summary}")
        print(f"  Drivers: {row.top_reasons}")
        print(f"  Risks: {row.risk_flags}")
    print(f"\nSaved no-odds prediction packet to {output_path}")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)
    stats_path = Path(args.fighter_stats) if args.fighter_stats else paths["fighter_stats"]
    output_path = Path(args.output) if args.output else paths["reports_dir"] / "no_odds_prediction_packet.csv"

    if not stats_path.exists():
        raise SystemExit(
            f"No fighter stats CSV found: {stats_path}\n"
            "Run no-odds prep first:\n"
            f".\\.venv-win\\Scripts\\python.exe scripts\\run_event_pipeline.py --manifest {args.manifest} --skip-odds --quiet-children"
        )

    side_model_path = Path(args.side_model) if args.side_model else default_side_model_path(ROOT)
    confidence_model_path = Path(args.confidence_model) if args.confidence_model else default_confidence_model_path(ROOT)
    side_model_bundle = load_side_model(side_model_path) if side_model_path.exists() else None
    confidence_model_bundle = load_confidence_model(confidence_model_path) if confidence_model_path.exists() else None

    packet = build_no_odds_prediction_packet(
        manifest,
        load_fighter_stats(stats_path),
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    packet.to_csv(output_path, index=False)
    if not args.quiet:
        _print_summary(packet, output_path)


if __name__ == "__main__":
    main()
