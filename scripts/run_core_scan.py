from __future__ import annotations

import argparse
import itertools
import os
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

from data_sources.odds_api import load_odds_csv
from data_sources.storage import save_odds_snapshot, save_tracked_picks
from features.fighter_features import build_fight_features, load_fighter_stats
from models.ev import expected_value, implied_probability, probability_to_american
from models.projection import project_fight_probabilities
from models.prop_outcomes import (
    default_prop_outcome_model_path,
    load_prop_outcome_model,
    predict_prop_probability_from_fight_row,
)
from models.threshold_policy import (
    DEFAULT_MIN_MODEL_PROB,
    default_threshold_policy_path,
    load_threshold_policy,
    resolve_scan_thresholds,
)
from normalization.odds import normalize_odds_frame
from scripts.event_manifest import DEFAULT_MAIN_CARD_FIGHT_COUNT, derived_paths, load_manifest


CORE_COLUMNS = [
    "event_name",
    "fight",
    "decision",
    "pick",
    "opponent",
    "book",
    "sportsbook_line",
    "fair_line",
    "model_prob",
    "raw_model_prob",
    "market_prob",
    "anchor_cap",
    "implied_prob",
    "edge",
    "expected_value",
    "confidence",
    "data_quality",
    "no_bet_reason",
    "lean_strength",
    "lean_action",
    "pick_gym_name",
    "pick_gym_tier",
    "pick_gym_record",
    "opponent_gym_name",
    "opponent_gym_tier",
    "opponent_gym_record",
    "camp_summary",
    "top_reasons",
    "risk_flags",
    "watch_for",
    "context_summary",
    "news_summary",
    "fighter_a",
    "fighter_b",
    "selection",
    "market",
]

CORE_PROP_COLUMNS = [
    "event_id",
    "event_name",
    "start_time",
    "fight",
    "is_main_card",
    "decision",
    "prop",
    "book",
    "sportsbook_line",
    "fair_line",
    "model_prob",
    "implied_prob",
    "edge",
    "expected_value",
    "confidence",
    "data_quality",
    "no_bet_reason",
    "fighter_a",
    "fighter_b",
    "selection",
    "market",
]

MAIN_CARD_ONLY_PROP_MARKETS = {"knockdown", "takedown"}
SUPPORTED_PROP_MARKETS = {
    "fight_goes_to_decision",
    "fight_doesnt_go_to_decision",
    "inside_distance",
    "submission",
    "ko_tko",
    "fight_ends_by_submission",
    "fight_ends_by_ko_tko",
    "knockdown",
    "takedown",
    "by_decision",
}

CORE_PARLAY_COLUMNS = [
    "parlay",
    "leg_count",
    "legs",
    "decimal_odds",
    "american_odds",
    "model_prob",
    "implied_prob",
    "expected_value",
]

ANSI_COLORS = {
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a market-anchored UFC core board.")
    parser.add_argument("--manifest", help="Optional event manifest. Used to infer input/output paths.")
    parser.add_argument("--odds", help="Current moneyline odds CSV. Defaults from --manifest.")
    parser.add_argument("--prop-odds", help="Optional modeled prop odds CSV. Defaults from --manifest.")
    parser.add_argument("--fighter-stats", help="Cached fighter stats CSV. Defaults from --manifest.")
    parser.add_argument("--lean-board", help="Optional cached lean board CSV used for explanations.")
    parser.add_argument("--output", help="Output CSV path. Defaults to cards/<slug>/reports/core_board.csv.")
    parser.add_argument("--props-output", help="Output CSV path for optional core props.")
    parser.add_argument("--parlays-output", help="Output CSV path for optional core parlays.")
    parser.add_argument(
        "--book",
        default=os.getenv("ODDS_API_BOOKMAKER", "fanduel"),
        help="Bookmaker to use when the odds CSV contains multiple books. Use 'any' to keep all books.",
    )
    parser.add_argument("--min-edge", type=float, default=float(os.getenv("CORE_MIN_EDGE", "0.03")))
    parser.add_argument("--min-model-prob", type=float, default=float(os.getenv("CORE_MIN_MODEL_PROB", str(DEFAULT_MIN_MODEL_PROB))))
    parser.add_argument("--min-prop-edge", type=float, default=float(os.getenv("CORE_MIN_PROP_EDGE", "0.05")))
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=float(os.getenv("CORE_MIN_CONFIDENCE", "0.60")),
    )
    parser.add_argument(
        "--min-stats-completeness",
        type=float,
        default=float(os.getenv("CORE_MIN_STATS_COMPLETENESS", "0.80")),
    )
    parser.add_argument(
        "--max-bets",
        type=int,
        default=int(os.getenv("CORE_MAX_BETS", "3")),
        help="Maximum bet candidates to allow on the board. Use 0 for no cap.",
    )
    parser.add_argument(
        "--max-market-move",
        type=float,
        default=float(os.getenv("CORE_MAX_MARKET_MOVE", "0.10")),
        help="Default max probability move away from no-vig market probability.",
    )
    parser.add_argument(
        "--elite-market-move",
        type=float,
        default=float(os.getenv("CORE_ELITE_MARKET_MOVE", "0.12")),
        help="Max market move for elite confidence/data rows.",
    )
    parser.add_argument(
        "--low-confidence-market-move",
        type=float,
        default=float(os.getenv("CORE_LOW_CONFIDENCE_MARKET_MOVE", "0.06")),
        help="Max market move for lower-confidence or incomplete-data rows.",
    )
    parser.add_argument("--include-props", action="store_true", help="Also write a filtered core props board.")
    parser.add_argument("--include-parlays", action="store_true", help="Also write a positive-EV core parlay board.")
    parser.add_argument(
        "--threshold-policy",
        help="Optional JSON path for optimized thresholds. Defaults to models/threshold_policy.json when present.",
    )
    parser.add_argument(
        "--prop-model",
        help="Optional pickle path for trained takedown/knockdown prop models. Defaults to models/prop_outcome_model.pkl when present.",
    )
    parser.add_argument(
        "--prop-thresholds",
        help="Optional prop threshold CSV. Defaults to cards/<slug>/reports/prop_model_thresholds.csv when present.",
    )
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="SQLite DB path for prop odds snapshots and tracked core props.")
    parser.add_argument("--no-prop-odds-archive", action="store_true", help="Do not archive priced prop odds rows into SQLite.")
    parser.add_argument("--no-track-core-props", action="store_true", help="Do not track BET rows from core props in SQLite.")
    parser.add_argument("--max-props", type=int, default=int(os.getenv("CORE_MAX_PROPS", "5")))
    parser.add_argument("--max-parlays", type=int, default=int(os.getenv("CORE_MAX_PARLAYS", "3")))
    parser.add_argument("--parlay-min-legs", type=int, default=int(os.getenv("CORE_PARLAY_MIN_LEGS", "2")))
    parser.add_argument("--parlay-max-legs", type=int, default=int(os.getenv("CORE_PARLAY_MAX_LEGS", "2")))
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default=os.getenv("CORE_COLOR", "auto"),
        help="Colorize console output. CSV files stay plain.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, Path, Path | None, Path, Path | None, Path | None]:
    paths: dict[str, Path] = {}
    if args.manifest:
        paths = derived_paths(load_manifest(args.manifest))

    odds_path = Path(args.odds) if args.odds else paths.get("oddsapi_odds")
    prop_odds_path = Path(args.prop_odds) if args.prop_odds else paths.get("modeled_market_odds")
    stats_path = Path(args.fighter_stats) if args.fighter_stats else paths.get("fighter_stats")
    lean_board_path = Path(args.lean_board) if args.lean_board else paths.get("lean_board")
    output_path = Path(args.output) if args.output else paths.get("core_board")
    props_output_path = Path(args.props_output) if args.props_output else paths.get("core_props")
    parlays_output_path = Path(args.parlays_output) if args.parlays_output else paths.get("core_parlays")

    missing: list[str] = []
    if odds_path is None:
        missing.append("--odds")
    if stats_path is None:
        missing.append("--fighter-stats")
    if output_path is None:
        missing.append("--output")
    if missing:
        raise SystemExit(f"Missing required paths: {', '.join(missing)} or provide --manifest")

    return odds_path, prop_odds_path, stats_path, lean_board_path, output_path, props_output_path, parlays_output_path


def _validate_core_input_files(odds_path: Path, stats_path: Path, *, manifest_path: str | None = None) -> None:
    if not odds_path.exists():
        message = (
            f"No odds CSV found: {odds_path}\n"
            "run_core_card.ps1 requires live moneyline odds before it can build a core board."
        )
        if manifest_path:
            message += (
                "\n\nFor the no-odds card view, run:\n"
                f".\\.venv-win\\Scripts\\python.exe scripts\\print_card_preview.py --manifest {manifest_path}"
            )
        else:
            message += "\n\nCreate/fetch an odds CSV and pass it with --odds."
        raise SystemExit(message)

    if not stats_path.exists():
        message = (
            f"No fighter stats CSV found: {stats_path}\n"
            "Run the no-odds prep first, then rerun the core board after odds are available."
        )
        if manifest_path:
            message += (
                "\n\nPrep command:\n"
                f".\\.venv-win\\Scripts\\python.exe scripts\\run_event_pipeline.py --manifest {manifest_path} --skip-odds --quiet-children"
            )
        raise SystemExit(message)


def _prepare_moneyline_odds(path: Path, book: str) -> pd.DataFrame:
    raw = load_odds_csv(path)
    if "market" in raw.columns:
        raw = raw.loc[raw["market"].astype(str).str.strip().str.lower() == "moneyline"].copy()
    if raw.empty:
        raise ValueError(f"No moneyline odds found in {path}")

    requested_book = str(book or "").strip().lower()
    if requested_book and requested_book != "any" and "book" in raw.columns:
        book_rows = raw.loc[raw["book"].fillna("").astype(str).str.strip().str.lower() == requested_book].copy()
        if not book_rows.empty:
            raw = book_rows

    normalized = normalize_odds_frame(raw)
    normalized = normalized.loc[
        normalized["market"].eq("moneyline") & normalized["selection"].isin(["fighter_a", "fighter_b"])
    ].copy()
    if normalized.empty:
        raise ValueError(f"No valid moneyline rows found in {path}")
    return normalized


def _load_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _safe_probability_to_american(probability: float) -> int:
    clipped = max(0.001, min(0.999, float(probability)))
    return probability_to_american(clipped)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return default if pd.isna(numeric) else numeric


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def _truthy_flag(value: object, *, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "main", "main_card"}:
        return True
    if text in {"0", "false", "no", "n", "prelim", "prelims"}:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return default


def _use_color(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _colorize(value: object, color: str, *, enabled: bool) -> str:
    text = _safe_text(value)
    return _colorize_plain(text, color, enabled=enabled)


def _colorize_plain(text: str, color: str, *, enabled: bool) -> str:
    if not enabled or not text:
        return text
    return f"{ANSI_COLORS[color]}{text}{ANSI_COLORS['reset']}"


def _decision_color(decision: object) -> str:
    return "green" if _safe_text(decision).upper() == "BET" else "gray"


def _edge_color(value: object) -> str:
    edge = _safe_float(value, 0.0)
    if edge >= 0.08:
        return "green"
    if edge >= 0.03:
        return "yellow"
    return "red"


def _confidence_color(value: object) -> str:
    confidence = _safe_float(value, 0.0)
    if confidence >= 0.80:
        return "green"
    if confidence >= 0.65:
        return "yellow"
    return "red"


def _gym_tier_color(value: object) -> str:
    tier = _safe_text(value).upper()
    if tier in {"S", "A"}:
        return "green"
    if tier in {"B", "C"}:
        return "yellow"
    if tier in {"D", "F"}:
        return "red"
    return "gray"


def _risk_color(value: object) -> str:
    text = _safe_text(value, "none").lower()
    if text == "none":
        return "green"
    if any(flag in text for flag in ["fallback", "injury", "market disagreement", "data quality"]):
        return "red"
    return "yellow"


def _decimal_to_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100))
    return int(round(-100 / max(decimal_odds - 1.0, 0.001)))


def _market_anchor_cap(
    row: pd.Series,
    *,
    max_market_move: float,
    elite_market_move: float,
    low_confidence_market_move: float,
) -> float:
    confidence = _safe_float(row.get("model_confidence", 0.0), 0.0)
    data_quality = _safe_float(row.get("data_quality", 0.0), 0.0)
    fallback_penalty = _safe_float(row.get("fallback_penalty", 0.0), 0.0)
    if confidence >= 0.85 and data_quality >= 0.95 and fallback_penalty <= 0.0:
        return max(float(max_market_move), float(elite_market_move))
    if confidence < 0.70 or data_quality < 0.85 or fallback_penalty > 0.0:
        return min(float(max_market_move), float(low_confidence_market_move))
    return float(max_market_move)


def _anchored_selection_prob(
    row: pd.Series,
    *,
    max_market_move: float,
    elite_market_move: float,
    low_confidence_market_move: float,
) -> tuple[float, float, float, float]:
    raw_prob = _safe_float(row.get("model_projected_win_prob"), 0.5)
    market_prob = _safe_float(row.get("market_target_fair_prob"), _safe_float(row.get("implied_prob"), raw_prob))
    cap = _market_anchor_cap(
        row,
        max_market_move=max_market_move,
        elite_market_move=elite_market_move,
        low_confidence_market_move=low_confidence_market_move,
    )
    delta = max(-cap, min(cap, raw_prob - market_prob))
    anchored_prob = max(0.01, min(0.99, market_prob + delta))
    return anchored_prob, raw_prob, market_prob, cap


def _opponent_name(row: pd.Series) -> str:
    return str(row["fighter_b"] if row["selection"] == "fighter_a" else row["fighter_a"])


def _side_prefix(row: pd.Series) -> str:
    return "a" if str(row.get("selection", "")) == "fighter_a" else "b"


def _other_prefix(row: pd.Series) -> str:
    return "b" if _side_prefix(row) == "a" else "a"


def _side_value(row: pd.Series, suffix: str, default: object = "") -> object:
    return row.get(f"{_side_prefix(row)}_{suffix}", default)


def _opponent_value(row: pd.Series, suffix: str, default: object = "") -> object:
    return row.get(f"{_other_prefix(row)}_{suffix}", default)


def _oriented(row: pd.Series, column: str, default: float = 0.0) -> float:
    value = _safe_float(row.get(column, default), default)
    return value if str(row.get("selection", "")) == "fighter_a" else -value


def _tier_label(tier: object) -> str:
    text = _safe_text(tier).upper()
    return f"{text}-tier" if text else "unranked"


def _camp_summary(row: pd.Series) -> str:
    pick = _safe_text(row.get("pick"))
    opponent = _safe_text(row.get("opponent"))
    pick_gym = _safe_text(row.get("pick_gym_name"), "unknown camp")
    opponent_gym = _safe_text(row.get("opponent_gym_name"), "unknown camp")
    return (
        f"{pick} {pick_gym} ({_tier_label(row.get('pick_gym_tier'))}) vs "
        f"{opponent} {opponent_gym} ({_tier_label(row.get('opponent_gym_tier'))})"
    )


def _derived_top_reasons(row: pd.Series) -> str:
    reasons: list[tuple[float, str]] = []
    striking = _oriented(row, "matchup_striking_edge")
    strike_margin = _oriented(row, "strike_margin_diff")
    grappling = _oriented(row, "matchup_grappling_edge")
    control = _oriented(row, "matchup_control_edge")
    reach = _oriented(row, "reach_diff")
    age = -_oriented(row, "age_diff")
    schedule = _oriented(row, "schedule_strength_diff")

    if abs(strike_margin) >= 0.5:
        reasons.append((abs(strike_margin), f"striking edge ({strike_margin:+.2f}/min)"))
    if abs(striking) >= 0.75:
        reasons.append((abs(striking), f"matchup striking edge ({striking:+.2f})"))
    if abs(grappling) >= 0.20:
        reasons.append((abs(grappling), f"grappling edge ({grappling:+.2f})"))
    if abs(control) >= 0.25:
        reasons.append((abs(control), f"control edge ({control:+.2f})"))
    if abs(reach) >= 2.0:
        reasons.append((abs(reach) / 2.0, f"reach advantage ({reach:+.0f} in)"))
    if abs(age) >= 4.0:
        reasons.append((abs(age) / 4.0, f"age edge ({age:+.1f}y)"))
    if abs(schedule) >= 0.08:
        reasons.append((abs(schedule) * 10.0, f"schedule strength ({schedule:+.2f})"))

    if not reasons:
        return "No clear driver edge"
    return ", ".join(reason for _, reason in sorted(reasons, reverse=True)[:3])


def _derived_risk_flags(row: pd.Series) -> str:
    risks: list[str] = []
    if abs(_safe_float(row.get("raw_model_prob"), 0.5) - _safe_float(row.get("market_prob"), 0.5)) >= 0.12:
        risks.append("market disagreement")
    if _safe_float(row.get("data_quality"), 1.0) < 0.85:
        risks.append("data quality")
    if _safe_float(row.get("fallback_penalty"), 0.0) > 0.0:
        risks.append("fallback stats")
    if _safe_float(_side_value(row, "ufc_fight_count", 0.0), 0.0) < 3:
        risks.append("thin UFC sample")
    if _safe_text(_side_value(row, "news_radar_label", "")):
        risks.append("news watch")
    return ", ".join(dict.fromkeys(risks)) if risks else "none"


def _news_summary(row: pd.Series) -> str:
    items: list[str] = []
    for fighter, prefix in [(row.get("pick"), _side_prefix(row)), (row.get("opponent"), _other_prefix(row))]:
        label = _safe_text(row.get(f"{prefix}_news_radar_label", ""))
        summary = _safe_text(row.get(f"{prefix}_news_radar_summary", ""))
        notes = summary or _safe_text(row.get(f"{prefix}_context_notes", ""))
        if label or notes:
            label_text = f"{label}: " if label else ""
            items.append(f"{_safe_text(fighter)} {label_text}{notes}".strip())
    return " | ".join(items)


def _context_summary(row: pd.Series) -> str:
    pick = _safe_text(row.get("pick"))
    opponent = _safe_text(row.get("opponent"))
    pick_age = _safe_float(_side_value(row, "age_years", 0.0), 0.0)
    opponent_age = _safe_float(_opponent_value(row, "age_years", 0.0), 0.0)
    pick_days = _safe_float(_side_value(row, "days_since_last_fight", 999.0), 999.0)
    opponent_days = _safe_float(_opponent_value(row, "days_since_last_fight", 999.0), 999.0)
    pick_sample = _safe_float(_side_value(row, "ufc_fight_count", 0.0), 0.0)
    opponent_sample = _safe_float(_opponent_value(row, "ufc_fight_count", 0.0), 0.0)

    notes: list[str] = []
    if pick_age > 0 and opponent_age > 0:
        younger = pick if pick_age < opponent_age else opponent
        notes.append(f"Age: {younger} younger by {abs(pick_age - opponent_age):.1f}y")
    if pick_days < 999 and opponent_days < 999:
        active = pick if pick_days < opponent_days else opponent
        notes.append(f"Layoff: {active} more active by {abs(pick_days - opponent_days):.0f}d")
    notes.append(f"UFC sample: {pick} {pick_sample:.0f} vs {opponent} {opponent_sample:.0f}")
    return " | ".join(notes)


def _watch_for(row: pd.Series) -> str:
    pick = _safe_text(row.get("pick"))
    opponent = _safe_text(row.get("opponent"))
    reasons = _safe_text(row.get("top_reasons"), "the strongest model phase")
    risks = _safe_text(row.get("risk_flags"), "none")
    first_reason = reasons.split(",")[0]
    note = f"{pick} needs to make the fight about {first_reason} against {opponent}."
    if risks != "none":
        note += f" Main caution: {risks.split(',')[0]}."
    else:
        note += f" If {opponent} keeps the fight away from that phase, the edge tightens."
    return note


def _lean_strength(edge: float, confidence: float, decision: str) -> str:
    if decision == "BET" and edge >= 0.08 and confidence >= 0.72:
        return "Strong Lean"
    if edge >= 0.05 and confidence >= 0.64:
        return "Lean"
    if edge >= 0.02:
        return "Slight Lean"
    return "Pass"


def _lean_lookup(lean_board: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, object]]:
    if lean_board is None or lean_board.empty:
        return {}
    lookup: dict[tuple[str, str], dict[str, object]] = {}
    for row in lean_board.to_dict("records"):
        fight = _safe_text(row.get("fight"))
        lean_side = _safe_text(row.get("lean_side"))
        if fight and lean_side:
            lookup[(fight, lean_side)] = row
    return lookup


def _attach_context(board: pd.DataFrame, lean_board: pd.DataFrame | None) -> pd.DataFrame:
    if board.empty:
        return board.copy()
    enriched = board.copy()
    lean_rows = _lean_lookup(lean_board)

    for index, row in enriched.iterrows():
        pick_prefix = _side_prefix(row)
        opponent_prefix = _other_prefix(row)
        for output_name, source_name, prefix in [
            ("pick_gym_name", "gym_name", pick_prefix),
            ("pick_gym_tier", "gym_tier", pick_prefix),
            ("pick_gym_record", "gym_record", pick_prefix),
            ("opponent_gym_name", "gym_name", opponent_prefix),
            ("opponent_gym_tier", "gym_tier", opponent_prefix),
            ("opponent_gym_record", "gym_record", opponent_prefix),
        ]:
            enriched.loc[index, output_name] = _safe_text(row.get(f"{prefix}_{source_name}", ""))

        lean = lean_rows.get((_safe_text(row.get("fight")), _safe_text(row.get("pick"))), {})
        edge = _safe_float(row.get("edge"), 0.0)
        confidence = _safe_float(row.get("confidence"), 0.0)
        decision = _safe_text(row.get("decision"), "PASS")
        enriched.loc[index, "lean_strength"] = _safe_text(lean.get("lean_strength"), _lean_strength(edge, confidence, decision))
        enriched.loc[index, "lean_action"] = _safe_text(lean.get("lean_action"), "Bet now" if decision == "BET" else "Pass")
        enriched.loc[index, "camp_summary"] = _safe_text(lean.get("camp_summary"), _camp_summary(enriched.loc[index]))
        enriched.loc[index, "top_reasons"] = _safe_text(lean.get("top_reasons"), _derived_top_reasons(enriched.loc[index]))
        enriched.loc[index, "risk_flags"] = _safe_text(lean.get("risk_flags"), _derived_risk_flags(enriched.loc[index]))
        enriched.loc[index, "context_summary"] = _safe_text(lean.get("context_summary"), _context_summary(enriched.loc[index]))
        enriched.loc[index, "news_summary"] = _news_summary(enriched.loc[index])
        enriched.loc[index, "watch_for"] = _safe_text(lean.get("watch_for"), _watch_for(enriched.loc[index]))
    return enriched


def _reason_list(
    row: pd.Series,
    *,
    min_edge: float,
    min_model_prob: float,
    min_confidence: float,
    min_stats_completeness: float,
) -> list[str]:
    reasons: list[str] = []
    if float(row["edge"]) < min_edge:
        reasons.append(f"edge below {min_edge:.1%}")
    if float(row["model_prob"]) < min_model_prob:
        reasons.append(f"model probability below {min_model_prob:.1%}")
    if float(row["confidence"]) < min_confidence:
        reasons.append(f"confidence below {min_confidence:.2f}")
    if float(row["data_quality"]) < min_stats_completeness:
        reasons.append(f"stats below {min_stats_completeness:.2f}")
    if float(row.get("fallback_penalty", 0.0) or 0.0) > 0:
        reasons.append("fallback stats")
    return reasons


def build_core_board(
    odds: pd.DataFrame,
    fighter_stats: pd.DataFrame,
    *,
    lean_board: pd.DataFrame | None = None,
    min_edge: float,
    min_model_prob: float = DEFAULT_MIN_MODEL_PROB,
    min_confidence: float,
    min_stats_completeness: float,
    max_bets: int = 3,
    max_market_move: float = 0.10,
    elite_market_move: float = 0.12,
    low_confidence_market_move: float = 0.06,
) -> pd.DataFrame:
    features = build_fight_features(odds, fighter_stats)
    scored = project_fight_probabilities(features)
    if scored.empty:
        return pd.DataFrame(columns=CORE_COLUMNS)

    scored = scored.copy()
    anchored = scored.apply(
        lambda row: _anchored_selection_prob(
            row,
            max_market_move=max_market_move,
            elite_market_move=elite_market_move,
            low_confidence_market_move=low_confidence_market_move,
        ),
        axis=1,
        result_type="expand",
    )
    anchored.columns = ["model_prob", "raw_model_prob", "market_prob", "anchor_cap"]
    scored = pd.concat([scored, anchored], axis=1)
    scored["implied_prob"] = scored["american_odds"].apply(implied_probability).astype(float)
    scored["edge"] = scored["model_prob"] - scored["implied_prob"]
    scored["expected_value"] = scored.apply(
        lambda row: expected_value(float(row["model_prob"]), int(row["american_odds"])),
        axis=1,
    )
    scored["fair_line"] = scored["model_prob"].apply(_safe_probability_to_american)
    scored["sportsbook_line"] = scored["american_odds"].astype(int)
    scored["confidence"] = pd.to_numeric(scored.get("model_confidence", 0.0), errors="coerce").fillna(0.0)
    scored["data_quality"] = pd.to_numeric(scored.get("data_quality", 0.0), errors="coerce").fillna(0.0)
    scored["fight"] = scored["fighter_a"].astype(str) + " vs " + scored["fighter_b"].astype(str)
    scored["pick"] = scored["selection_name"].astype(str)
    scored["opponent"] = scored.apply(_opponent_name, axis=1)

    best_rows: list[pd.Series] = []
    for _, fight_rows in scored.groupby(["event_id", "fighter_a", "fighter_b"], dropna=False, sort=False):
        best_rows.append(fight_rows.sort_values(["edge", "expected_value"], ascending=False).iloc[0])

    board = pd.DataFrame(best_rows).reset_index(drop=True)
    no_bet_reasons = [
        _reason_list(
            row,
            min_edge=min_edge,
            min_model_prob=min_model_prob,
            min_confidence=min_confidence,
            min_stats_completeness=min_stats_completeness,
        )
        for _, row in board.iterrows()
    ]
    if max_bets > 0:
        qualifying_indexes = [index for index, reasons in enumerate(no_bet_reasons) if not reasons]
        ranked_indexes = sorted(qualifying_indexes, key=lambda index: float(board.loc[index, "edge"]), reverse=True)
        for index in ranked_indexes[max_bets:]:
            no_bet_reasons[index].append(f"outside top {max_bets} core edges")
    board["no_bet_reason"] = [", ".join(reasons) for reasons in no_bet_reasons]
    board["decision"] = ["BET" if not reasons else "PASS" for reasons in no_bet_reasons]
    board = _attach_context(board, lean_board)

    output = board.loc[:, [column for column in CORE_COLUMNS if column in board.columns]].copy()
    for column in [
        "model_prob",
        "raw_model_prob",
        "market_prob",
        "anchor_cap",
        "implied_prob",
        "edge",
        "expected_value",
        "confidence",
        "data_quality",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce").round(4)
    output = output.sort_values(["decision", "edge"], ascending=[True, False]).reset_index(drop=True)
    return output


def _fight_lookup(scored_moneylines: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    lookup: dict[tuple[str, str, str], pd.Series] = {}
    if scored_moneylines.empty:
        return lookup
    for _, row in scored_moneylines.drop_duplicates(["event_id", "fighter_a", "fighter_b"]).iterrows():
        lookup[(str(row["event_id"]), str(row["fighter_a"]), str(row["fighter_b"]))] = row
    return lookup


def _selection_prefix(selection: object) -> str:
    return "a" if str(selection).strip() == "fighter_a" else "b"


def _side_stat(row: pd.Series, selection: object, suffix: str, default: object = 0.0) -> object:
    letter_prefix = _selection_prefix(selection)
    fighter_prefix = "fighter_a" if letter_prefix == "a" else "fighter_b"
    for column in (f"{letter_prefix}_{suffix}", f"{fighter_prefix}_{suffix}"):
        if column in row.index:
            value = row.get(column)
            if not pd.isna(value):
                return value
    return default


def _opponent_selection(selection: object) -> str:
    return "fighter_b" if str(selection).strip() == "fighter_a" else "fighter_a"


def _knockdown_prop_probability(row: pd.Series, selection: object) -> float:
    knockdown_avg = _safe_float(_side_stat(row, selection, "knockdown_avg"), 0.0)
    if knockdown_avg <= 0.0:
        return 0.0

    opponent = _opponent_selection(selection)
    scheduled_rounds = _safe_float(row.get("scheduled_rounds"), 3.0)
    exposure = max(0.75, min(1.45, scheduled_rounds / 3.0))
    own_ko_rate = _safe_float(_side_stat(row, selection, "ko_win_rate"), 0.0)
    opponent_ko_loss_rate = _safe_float(_side_stat(row, opponent, "ko_loss_rate"), 0.0)
    own_strike_rate = _safe_float(_side_stat(row, selection, "sig_strikes_landed_per_min"), 0.0)
    opponent_absorbed_rate = _safe_float(_side_stat(row, opponent, "sig_strikes_absorbed_per_min"), 0.0)
    distance_share = _safe_float(_side_stat(row, selection, "distance_strike_share"), 0.55)

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


def _takedown_prop_probability(row: pd.Series, selection: object) -> float:
    takedown_avg = _safe_float(_side_stat(row, selection, "takedown_avg"), 0.0)
    if takedown_avg <= 0.0:
        return 0.0

    opponent = _opponent_selection(selection)
    scheduled_rounds = _safe_float(row.get("scheduled_rounds"), 3.0)
    exposure = max(0.75, min(1.45, scheduled_rounds / 3.0))
    opponent_td_def = _safe_float(_side_stat(row, opponent, "takedown_defense_pct"), 68.0)
    recent_grappling = _safe_float(_side_stat(row, selection, "recent_grappling_rate"), 0.0)
    control_avg = _safe_float(_side_stat(row, selection, "control_avg"), 0.0)
    recent_control = _safe_float(_side_stat(row, selection, "recent_control_avg"), 0.0)
    grappling_edge = _safe_float(row.get("matchup_grappling_edge"), 0.0)
    if str(selection).strip() == "fighter_b":
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


def _prop_label(prop_row: pd.Series) -> str:
    market = str(prop_row.get("market", "") or "").strip()
    selection = str(prop_row.get("selection", "") or "").strip()
    fighter_a = _safe_text(prop_row.get("fighter_a"))
    fighter_b = _safe_text(prop_row.get("fighter_b"))
    fighter = fighter_a if selection == "fighter_a" else fighter_b if selection == "fighter_b" else ""
    if market == "fight_goes_to_decision":
        return "Fight goes to decision"
    if market == "fight_doesnt_go_to_decision":
        return "Fight doesn't go the distance"
    if market == "fight_ends_by_submission":
        return "Fight ends by submission"
    if market == "fight_ends_by_ko_tko":
        return "Fight ends by KO/TKO"
    if market == "inside_distance" and fighter:
        return f"{fighter} inside distance"
    if market == "submission" and fighter:
        return f"{fighter} by submission"
    if market == "ko_tko" and fighter:
        return f"{fighter} by KO/TKO"
    if market == "knockdown" and fighter:
        return f"{fighter} knockdown"
    if market == "takedown" and fighter:
        return f"{fighter} takedown"
    if market == "by_decision" and fighter:
        return f"{fighter} by decision"
    return _safe_text(prop_row.get("selection_name")) or f"{market} {selection}".strip()


def _prop_probability(
    prop_row: pd.Series,
    fight_row: pd.Series,
    *,
    prop_model_bundle: dict[str, object] | None = None,
) -> float | None:
    market = str(prop_row.get("market", "") or "")
    selection = str(prop_row.get("selection", "") or "")
    trained_selection = selection if selection in {"fighter_a", "fighter_b"} else "fighter_a"
    try:
        trained_probability = predict_prop_probability_from_fight_row(
            prop_model_bundle,
            fight_row,
            market=market,
            selection=trained_selection,
        )
    except Exception:
        trained_probability = None
    if trained_probability is not None:
        return trained_probability
    if market == "fight_goes_to_decision":
        return float(fight_row.get("projected_decision_prob", 0.0) or 0.0)
    if market == "fight_doesnt_go_to_decision":
        return float(fight_row.get("projected_finish_prob", 0.0) or 0.0)
    if market == "inside_distance" and selection == "fighter_a":
        return float(fight_row.get("fighter_a_inside_distance_prob", 0.0) or 0.0)
    if market == "inside_distance" and selection == "fighter_b":
        return float(fight_row.get("fighter_b_inside_distance_prob", 0.0) or 0.0)
    if market == "submission" and selection == "fighter_a":
        return float(fight_row.get("fighter_a_submission_prob", 0.0) or 0.0)
    if market == "submission" and selection == "fighter_b":
        return float(fight_row.get("fighter_b_submission_prob", 0.0) or 0.0)
    if market == "ko_tko" and selection == "fighter_a":
        return float(fight_row.get("fighter_a_ko_tko_prob", 0.0) or 0.0)
    if market == "ko_tko" and selection == "fighter_b":
        return float(fight_row.get("fighter_b_ko_tko_prob", 0.0) or 0.0)
    if market == "fight_ends_by_submission":
        return float(fight_row.get("fighter_a_submission_prob", 0.0) or 0.0) + float(
            fight_row.get("fighter_b_submission_prob", 0.0) or 0.0
        )
    if market == "fight_ends_by_ko_tko":
        return float(fight_row.get("fighter_a_ko_tko_prob", 0.0) or 0.0) + float(
            fight_row.get("fighter_b_ko_tko_prob", 0.0) or 0.0
        )
    if market == "knockdown" and selection in {"fighter_a", "fighter_b"}:
        return _knockdown_prop_probability(fight_row, selection)
    if market == "takedown" and selection in {"fighter_a", "fighter_b"}:
        return _takedown_prop_probability(fight_row, selection)
    if market == "by_decision" and selection == "fighter_a":
        return float(fight_row.get("fighter_a_by_decision_prob", 0.0) or 0.0)
    if market == "by_decision" and selection == "fighter_b":
        return float(fight_row.get("fighter_b_by_decision_prob", 0.0) or 0.0)
    return None


def _score_moneylines_for_props(odds: pd.DataFrame, fighter_stats: pd.DataFrame) -> pd.DataFrame:
    return project_fight_probabilities(build_fight_features(odds, fighter_stats))


def _default_prop_thresholds_path(manifest_path: str | None) -> Path | None:
    if not manifest_path:
        return None
    return derived_paths(load_manifest(manifest_path)).get("prop_model_thresholds")


def load_prop_threshold_gates(path: str | Path | None) -> dict[str, dict[str, object]]:
    if path is None or not Path(path).exists():
        return {}
    frame = pd.read_csv(path)
    if frame.empty or "market" not in frame.columns:
        return {}
    gates: dict[str, dict[str, object]] = {}
    for market, market_rows in frame.groupby(frame["market"].astype(str), dropna=False):
        recommended = market_rows.loc[
            market_rows.get("is_recommended", pd.Series(0, index=market_rows.index)).astype(str).isin(["1", "true", "True"])
        ].copy()
        if recommended.empty:
            gates[str(market)] = {
                "blocked": True,
                "reason": f"{market} market blocked, no reliable holdout threshold yet",
            }
            continue
        threshold = pd.to_numeric(recommended.iloc[0].get("min_model_prob"), errors="coerce")
        if pd.isna(threshold):
            continue
        gates[str(market)] = {
            "blocked": False,
            "min_model_prob": float(threshold),
            "reason": f"{market} probability below learned threshold {float(threshold):.0%}",
        }
    return gates


def _prop_threshold_gate_reason(
    market: object,
    model_prob: float,
    gates: dict[str, dict[str, object]] | None,
) -> str:
    gate = (gates or {}).get(str(market))
    if not gate:
        return ""
    if bool(gate.get("blocked", False)):
        return str(gate.get("reason", "prop market blocked by learned threshold policy"))
    min_model_prob = _safe_float(gate.get("min_model_prob"), 0.0)
    if float(model_prob) < float(min_model_prob):
        return str(gate.get("reason", f"probability below learned threshold {float(min_model_prob):.0%}"))
    return ""


def _archive_prop_odds(prop_odds: pd.DataFrame, db_path: str | Path | None) -> int:
    if db_path is None or prop_odds.empty:
        return 0
    snapshot = prop_odds.copy()
    if "market" not in snapshot.columns:
        return 0
    snapshot = snapshot.loc[snapshot["market"].astype(str).ne("moneyline")].copy()
    snapshot["american_odds"] = pd.to_numeric(snapshot.get("american_odds"), errors="coerce")
    snapshot = snapshot.loc[snapshot["american_odds"].notna()].copy()
    if snapshot.empty:
        return 0
    try:
        return save_odds_snapshot(normalize_odds_frame(snapshot), db_path)
    except Exception:
        return 0


def _track_core_prop_bets(props: pd.DataFrame, db_path: str | Path | None) -> int:
    if db_path is None or props.empty:
        return 0
    bets = props.loc[props["decision"].astype(str).eq("BET")].copy()
    if bets.empty:
        return 0
    tracked = pd.DataFrame(
        {
            "event_id": bets.get("event_id", ""),
            "event_name": bets.get("event_name", ""),
            "start_time": bets.get("start_time", ""),
            "fighter_a": bets.get("fighter_a", ""),
            "fighter_b": bets.get("fighter_b", ""),
            "market": bets.get("market", ""),
            "selection": bets.get("selection", ""),
            "selection_name": bets.get("prop", ""),
            "book": bets.get("book", ""),
            "american_odds": bets.get("sportsbook_line", ""),
            "model_projected_win_prob": bets.get("model_prob", ""),
            "implied_prob": bets.get("implied_prob", ""),
            "edge": bets.get("edge", ""),
            "expected_value": bets.get("expected_value", ""),
            "suggested_stake": 1.0,
            "raw_suggested_stake": 1.0,
            "model_confidence": bets.get("confidence", ""),
            "data_quality": bets.get("data_quality", ""),
            "recommended_action": "Bettable now",
            "chosen_value_expression": bets.get("prop", ""),
            "expression_pick_source": "core_props",
            "chosen_expression_odds": bets.get("sportsbook_line", ""),
            "chosen_expression_prob": bets.get("model_prob", ""),
            "chosen_expression_implied_prob": bets.get("implied_prob", ""),
            "chosen_expression_edge": bets.get("edge", ""),
            "chosen_expression_expected_value": bets.get("expected_value", ""),
            "chosen_expression_stake": 1.0,
            "raw_chosen_expression_stake": 1.0,
            "tracked_market_key": bets.get("market", ""),
            "tracked_selection_key": bets.get("selection", ""),
            "grade_status": "pending",
        }
    )
    try:
        return save_tracked_picks(tracked, db_path)
    except Exception:
        return 0


def build_core_props(
    prop_odds: pd.DataFrame,
    scored_moneylines: pd.DataFrame,
    *,
    min_edge: float,
    min_confidence: float,
    min_stats_completeness: float,
    max_props: int,
    prop_model_bundle: dict[str, object] | None = None,
    prop_threshold_gates: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    if prop_odds.empty or scored_moneylines.empty:
        return pd.DataFrame(columns=CORE_PROP_COLUMNS)
    props = prop_odds.copy()
    props = props.loc[props["market"].astype(str).ne("moneyline")].copy()
    props = props.loc[props["market"].astype(str).isin(SUPPORTED_PROP_MARKETS)].copy()
    props = _with_prop_main_card_flags(props)
    props = props.loc[
        ~props["market"].astype(str).isin(MAIN_CARD_ONLY_PROP_MARKETS)
        | props["is_main_card"].apply(lambda value: _truthy_flag(value, default=False))
    ].copy()
    props["american_odds"] = pd.to_numeric(props["american_odds"], errors="coerce")
    props = props.loc[props["american_odds"].notna()].copy()
    if props.empty:
        return pd.DataFrame(columns=CORE_PROP_COLUMNS)

    lookup = _fight_lookup(scored_moneylines)
    rows: list[dict[str, object]] = []
    for _, prop in props.iterrows():
        fight_row = lookup.get((str(prop["event_id"]), str(prop["fighter_a"]), str(prop["fighter_b"])))
        if fight_row is None:
            continue
        model_prob = _prop_probability(prop, fight_row, prop_model_bundle=prop_model_bundle)
        if model_prob is None:
            continue
        american_odds = int(float(prop["american_odds"]))
        implied_prob = implied_probability(american_odds)
        edge = float(model_prob) - implied_prob
        confidence = float(fight_row.get("model_confidence", 0.0) or 0.0)
        data_quality = float(fight_row.get("data_quality", 0.0) or 0.0)
        reasons: list[str] = []
        threshold_reason = _prop_threshold_gate_reason(prop["market"], float(model_prob), prop_threshold_gates)
        if threshold_reason:
            reasons.append(threshold_reason)
        if edge < min_edge:
            reasons.append(f"edge below {min_edge:.1%}")
        if confidence < min_confidence:
            reasons.append(f"confidence below {min_confidence:.2f}")
        if data_quality < min_stats_completeness:
            reasons.append(f"stats below {min_stats_completeness:.2f}")
        rows.append(
            {
                "event_id": prop.get("event_id", ""),
                "event_name": prop["event_name"],
                "start_time": prop.get("start_time", ""),
                "fight": f"{prop['fighter_a']} vs {prop['fighter_b']}",
                "is_main_card": int(_truthy_flag(prop.get("is_main_card"), default=False)),
                "decision": "BET" if not reasons else "PASS",
                "prop": _prop_label(prop),
                "book": prop.get("book", ""),
                "sportsbook_line": american_odds,
                "fair_line": _safe_probability_to_american(float(model_prob)),
                "model_prob": round(float(model_prob), 4),
                "implied_prob": round(float(implied_prob), 4),
                "edge": round(float(edge), 4),
                "expected_value": round(expected_value(float(model_prob), american_odds), 4),
                "confidence": round(confidence, 4),
                "data_quality": round(data_quality, 4),
                "no_bet_reason": ", ".join(reasons),
                "fighter_a": prop["fighter_a"],
                "fighter_b": prop["fighter_b"],
                "selection": prop["selection"],
                "market": prop["market"],
            }
        )

    output = pd.DataFrame(rows, columns=CORE_PROP_COLUMNS)
    if output.empty:
        return output
    qualifying = output.loc[output["decision"].eq("BET")].sort_values("edge", ascending=False)
    if max_props > 0:
        keep_indexes = set(qualifying.head(max_props).index)
        output.loc[output["decision"].eq("BET") & ~output.index.isin(keep_indexes), "decision"] = "PASS"
        output.loc[
            output["no_bet_reason"].eq("") & output["decision"].eq("PASS"),
            "no_bet_reason",
        ] = f"outside top {max_props} core props"
    return output.sort_values(["decision", "edge"], ascending=[True, False]).reset_index(drop=True)


def _with_prop_main_card_flags(props: pd.DataFrame) -> pd.DataFrame:
    output = props.copy()
    if output.empty:
        output["is_main_card"] = pd.Series(dtype=int)
        return output
    if "is_main_card" in output.columns:
        output["is_main_card"] = output["is_main_card"].apply(lambda value: int(_truthy_flag(value, default=False)))
        return output
    fight_keys = (
        output[["event_id", "fighter_a", "fighter_b"]]
        .drop_duplicates()
        .head(DEFAULT_MAIN_CARD_FIGHT_COUNT)
        .apply(lambda row: (str(row["event_id"]), str(row["fighter_a"]), str(row["fighter_b"])), axis=1)
    )
    main_card_keys = set(fight_keys.tolist())
    output["is_main_card"] = output.apply(
        lambda row: int((str(row["event_id"]), str(row["fighter_a"]), str(row["fighter_b"])) in main_card_keys),
        axis=1,
    )
    return output


def _core_prop_legs_for_parlays(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        return pd.DataFrame()
    legs = props.loc[props["decision"].eq("BET")].copy()
    if legs.empty:
        return pd.DataFrame()
    legs["pick"] = legs["prop"]
    return legs


def _core_leg_fight_key(leg: dict[str, object]) -> str:
    fighter_a = _safe_text(leg.get("fighter_a"))
    fighter_b = _safe_text(leg.get("fighter_b"))
    if fighter_a or fighter_b:
        return f"{fighter_a}||{fighter_b}"
    fight = _safe_text(leg.get("fight"))
    if fight:
        return fight
    return ""


def build_core_parlays(
    core_board: pd.DataFrame,
    *,
    min_legs: int,
    max_legs: int,
    max_parlays: int,
) -> pd.DataFrame:
    legs = core_board.loc[core_board["decision"].eq("BET")].copy()
    if legs.empty:
        return pd.DataFrame(columns=CORE_PARLAY_COLUMNS)
    min_legs = max(2, int(min_legs))
    max_legs = max(min_legs, int(max_legs))

    rows: list[dict[str, object]] = []
    for leg_count in range(min_legs, max_legs + 1):
        for combo in itertools.combinations(legs.to_dict("records"), leg_count):
            fight_keys = [_core_leg_fight_key(leg) for leg in combo]
            keyed_fights = [key for key in fight_keys if key]
            if len(keyed_fights) != len(set(keyed_fights)):
                continue
            decimal_odds = 1.0
            model_prob = 1.0
            labels: list[str] = []
            for leg in combo:
                decimal_odds *= 1.0 / float(leg["implied_prob"])
                model_prob *= float(leg["model_prob"])
                labels.append(f"{leg['pick']} ({leg['sportsbook_line']})")
            implied_prob = 1.0 / decimal_odds
            ev = (model_prob * (decimal_odds - 1.0)) - (1.0 - model_prob)
            if ev <= 0:
                continue
            rows.append(
                {
                    "parlay": " / ".join(str(leg["pick"]) for leg in combo),
                    "leg_count": leg_count,
                    "legs": " | ".join(labels),
                    "decimal_odds": round(decimal_odds, 3),
                    "american_odds": _decimal_to_american(decimal_odds),
                    "model_prob": round(model_prob, 4),
                    "implied_prob": round(implied_prob, 4),
                    "expected_value": round(ev, 4),
                }
            )

    output = pd.DataFrame(rows, columns=CORE_PARLAY_COLUMNS)
    if output.empty:
        return output
    return output.sort_values("expected_value", ascending=False).head(max_parlays).reset_index(drop=True)


def _format_price(value: object) -> str:
    try:
        odds = int(float(value))
    except (TypeError, ValueError):
        return _safe_text(value)
    return f"+{odds}" if odds > 0 else str(odds)


def _format_pct(value: object) -> str:
    return f"{_safe_float(value, 0.0) * 100:.1f}%"


def _bet_rows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty or "decision" not in frame.columns:
        return pd.DataFrame()
    bets = frame.loc[frame["decision"].astype(str).str.upper().eq("BET")].copy()
    if bets.empty:
        return bets
    sort_columns = [column for column in ["expected_value", "edge"] if column in bets.columns]
    if sort_columns:
        bets = bets.sort_values(sort_columns, ascending=[False] * len(sort_columns))
    return bets.reset_index(drop=True)


def _instruction_metric_summary(row: pd.Series) -> str:
    return (
        f"line {_format_price(row.get('sportsbook_line'))}, fair {_format_price(row.get('fair_line'))}, "
        f"model {_format_pct(row.get('model_prob'))}, edge {_format_pct(row.get('edge'))}, "
        f"EV {_safe_float(row.get('expected_value'), 0.0):+.3f}"
    )


def _instruction_book(row: pd.Series) -> str:
    book = _safe_text(row.get("book"))
    return f" at {book}" if book else ""


def format_direct_betting_instructions(
    board: pd.DataFrame,
    props: pd.DataFrame | None = None,
    *,
    props_scanned: bool = False,
) -> str:
    lines = [
        "BET DIRECTLY",
        "------------",
        "These are the rows to bet directly. Do not bet PASS rows.",
        "",
        "Moneyline:",
    ]

    moneyline_bets = _bet_rows(board)
    if moneyline_bets.empty:
        lines.append("- No direct moneyline bets.")
    else:
        for index, row in moneyline_bets.iterrows():
            lines.append(
                f"{index + 1}. BET {_safe_text(row.get('pick'))}{_instruction_book(row)} "
                f"vs {_safe_text(row.get('opponent'))} | {_instruction_metric_summary(row)}"
            )

    lines.extend(["", "Priced props:"])
    if not props_scanned:
        lines.append("- Props were not scanned. Run .\\scripts\\run_core_card.ps1 --include-props to include priced props.")
    else:
        prop_bets = _bet_rows(props)
        if prop_bets.empty:
            lines.append("- No direct priced prop bets qualified.")
        else:
            for index, row in prop_bets.iterrows():
                main_card_note = " | main card" if _truthy_flag(row.get("is_main_card"), default=False) else ""
                lines.append(
                    f"{index + 1}. BET {_safe_text(row.get('prop'))}{_instruction_book(row)} | "
                    f"{_safe_text(row.get('fight'))}{main_card_note} | {_instruction_metric_summary(row)}"
                )

    lines.extend(
        [
            "",
            "Prop markets checked when priced:",
            "- Fight doesn't go the distance",
            "- Fighter inside distance",
            "- Fighter by KO/TKO",
            "- Fighter by submission",
            "- Fight ends by submission",
            "- Fight ends by KO/TKO",
            "- Fighter knockdowns (main card only)",
            "- Fighter takedowns (main card only)",
            "",
            "Detailed board follows:",
            "",
        ]
    )
    return "\n".join(lines)


def _print_colored_core_board(board: pd.DataFrame, *, color_mode: str) -> None:
    color_enabled = _use_color(color_mode)
    if board.empty:
        return

    for _, row in board.iterrows():
        decision = _safe_text(row.get("decision"))
        pick = _safe_text(row.get("pick"))
        opponent = _safe_text(row.get("opponent"))
        edge = _format_pct(row.get("edge"))
        confidence = _format_pct(row.get("confidence"))
        line = _format_price(row.get("sportsbook_line"))
        fair = _format_price(row.get("fair_line"))
        pick_tier = _safe_text(row.get("pick_gym_tier"), "?")
        opponent_tier = _safe_text(row.get("opponent_gym_tier"), "?")

        header = (
            f"{_colorize(decision, _decision_color(decision), enabled=color_enabled)} "
            f"{_colorize(pick, 'bold', enabled=color_enabled)} over {opponent} "
            f"{_colorize(line, 'cyan', enabled=color_enabled)} "
            f"(fair {fair}, edge {_colorize(edge, _edge_color(row.get('edge')), enabled=color_enabled)}, "
            f"conf {_colorize(confidence, _confidence_color(row.get('confidence')), enabled=color_enabled)})"
        )
        print(header)
        print(
            "  "
            f"Gym: {_safe_text(row.get('pick_gym_name'), 'unknown')} "
            f"({_colorize(pick_tier, _gym_tier_color(pick_tier), enabled=color_enabled)}) vs "
            f"{_safe_text(row.get('opponent_gym_name'), 'unknown')} "
            f"({_colorize(opponent_tier, _gym_tier_color(opponent_tier), enabled=color_enabled)})"
        )
        print(f"  Drivers: {_colorize(row.get('top_reasons'), 'cyan', enabled=color_enabled)}")
        risk_text = _safe_text(row.get("risk_flags"), "none")
        print(f"  Risks: {_colorize_plain(risk_text, _risk_color(risk_text), enabled=color_enabled)}")
        news = _safe_text(row.get("news_summary"))
        if news:
            print(f"  News: {_colorize(news, 'yellow', enabled=color_enabled)}")
        print(f"  Watch: {_safe_text(row.get('watch_for'))}")
        no_bet = _safe_text(row.get("no_bet_reason"))
        if no_bet:
            print(f"  Pass reason: {_colorize(no_bet, 'yellow', enabled=color_enabled)}")
        print()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    (
        odds_path,
        prop_odds_path,
        stats_path,
        lean_board_path,
        output_path,
        props_output_path,
        parlays_output_path,
    ) = _resolve_paths(args)

    _validate_core_input_files(odds_path, stats_path, manifest_path=args.manifest)
    threshold_policy_path = Path(args.threshold_policy) if args.threshold_policy else default_threshold_policy_path(ROOT)
    threshold_policy = load_threshold_policy(threshold_policy_path if threshold_policy_path.exists() else None)
    threshold_settings = resolve_scan_thresholds(
        min_edge=args.min_edge,
        min_model_prob=args.min_model_prob,
        min_model_confidence=args.min_confidence,
        min_stats_completeness=args.min_stats_completeness,
        exclude_fallback_rows=True,
        policy=threshold_policy,
    )
    min_edge = float(threshold_settings["min_edge"])
    min_model_prob = float(threshold_settings["min_model_prob"])
    min_confidence = float(threshold_settings["min_model_confidence"])
    min_stats_completeness = float(threshold_settings["min_stats_completeness"])
    prop_model_path = Path(args.prop_model) if args.prop_model else default_prop_outcome_model_path(ROOT)
    prop_model_bundle: dict[str, object] | None = None
    prop_model_loaded = False
    if prop_model_path.exists():
        try:
            prop_model_bundle = load_prop_outcome_model(prop_model_path)
            prop_model_loaded = True
        except Exception as exc:
            if not args.quiet:
                print(f"Prop model load skipped: {exc}")
    prop_thresholds_path = Path(args.prop_thresholds) if args.prop_thresholds else _default_prop_thresholds_path(args.manifest)
    prop_threshold_gates = load_prop_threshold_gates(prop_thresholds_path)

    odds = _prepare_moneyline_odds(odds_path, args.book)
    fighter_stats = load_fighter_stats(stats_path)
    lean_board = _load_optional_csv(lean_board_path)
    board = build_core_board(
        odds,
        fighter_stats,
        lean_board=lean_board,
        min_edge=min_edge,
        min_model_prob=min_model_prob,
        min_confidence=min_confidence,
        min_stats_completeness=min_stats_completeness,
        max_bets=args.max_bets,
        max_market_move=args.max_market_move,
        elite_market_move=args.elite_market_move,
        low_confidence_market_move=args.low_confidence_market_move,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.to_csv(output_path, index=False)

    props = pd.DataFrame(columns=CORE_PROP_COLUMNS)
    archived_prop_odds = 0
    tracked_core_props = 0
    if args.include_props:
        if prop_odds_path is None or not prop_odds_path.exists():
            raise SystemExit("Missing prop odds path. Pass --prop-odds or use --manifest with modeled_market_odds.csv.")
        if props_output_path is None:
            raise SystemExit("Missing --props-output or --manifest.")
        prop_odds = load_odds_csv(prop_odds_path)
        if not args.no_prop_odds_archive:
            archived_prop_odds = _archive_prop_odds(prop_odds, args.db)
        scored_moneylines = _score_moneylines_for_props(odds, fighter_stats)
        props = build_core_props(
            prop_odds,
            scored_moneylines,
            min_edge=args.min_prop_edge,
            min_confidence=max(min_confidence, 0.70),
            min_stats_completeness=max(min_stats_completeness, 0.85),
            max_props=args.max_props,
            prop_model_bundle=prop_model_bundle,
            prop_threshold_gates=prop_threshold_gates,
        )
        props_output_path.parent.mkdir(parents=True, exist_ok=True)
        props.to_csv(props_output_path, index=False)
        if not args.no_track_core_props:
            tracked_core_props = _track_core_prop_bets(props, args.db)

    parlays = pd.DataFrame(columns=CORE_PARLAY_COLUMNS)
    if args.include_parlays:
        if parlays_output_path is None:
            raise SystemExit("Missing --parlays-output or --manifest.")
        parlay_legs = board
        if args.include_props and not props.empty:
            parlay_legs = pd.concat([board, _core_prop_legs_for_parlays(props)], ignore_index=True, sort=False)
        parlays = build_core_parlays(
            parlay_legs,
            min_legs=args.parlay_min_legs,
            max_legs=args.parlay_max_legs,
            max_parlays=args.max_parlays,
        )
        parlays_output_path.parent.mkdir(parents=True, exist_ok=True)
        parlays.to_csv(parlays_output_path, index=False)

    if not args.quiet:
        bets = int(board["decision"].eq("BET").sum()) if not board.empty else 0
        if bool(threshold_settings.get("policy_applied")):
            print(f"Threshold policy: {threshold_settings.get('policy_summary', '')}")
        print(f"Core board: {bets} bet candidates, {len(board) - bets} passes")
        print(f"Saved core board to {output_path}")
        if args.include_props:
            prop_bets = int(props["decision"].eq("BET").sum()) if not props.empty else 0
            print(f"Core props: {prop_bets} bet candidates")
            if prop_model_loaded:
                print(f"Prop model: {prop_model_path}")
            else:
                print("Prop model: heuristic fallback for props")
            if prop_threshold_gates:
                print(f"Prop thresholds: {prop_thresholds_path}")
            if not args.no_prop_odds_archive:
                print(f"Archived prop odds rows: {archived_prop_odds}")
            if not args.no_track_core_props:
                print(f"Tracked core prop bets: {tracked_core_props}")
            print(f"Saved core props to {props_output_path}")
        if args.include_parlays:
            print(f"Core parlays: {len(parlays)} positive-EV combinations")
            print(f"Saved core parlays to {parlays_output_path}")
        print()
        print(format_direct_betting_instructions(board, props if args.include_props else None, props_scanned=args.include_props), end="")
        _print_colored_core_board(board, color_mode=args.color)


if __name__ == "__main__":
    main()
