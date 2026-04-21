from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.evaluator import evaluate_backtest, write_summary_csv
from data_sources.odds_api import load_odds_csv
from data_sources.storage import save_backtest_run
from features.fighter_features import build_fight_features, load_fighter_stats
from models.confidence import default_confidence_model_path, load_confidence_model
from models.projection import project_fight_probabilities
from models.side import default_side_model_path, load_side_model
from normalization.odds import normalize_odds_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest projected UFC selections against historical outcomes.")
    parser.add_argument("--input", required=True, help="Path to historical odds CSV.")
    parser.add_argument("--fighter-stats", required=True, help="Path to fighter stats CSV.")
    parser.add_argument("--output", required=True, help="Path to summary CSV.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path.")
    parser.add_argument("--side-model", help="Optional calibrated side-model pickle path.")
    parser.add_argument("--confidence-model", help="Optional calibrated confidence-model pickle path.")
    return parser.parse_args()


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    min_edge = float(os.getenv("MIN_EDGE", "0.03"))
    bankroll = float(os.getenv("BANKROLL", "1000"))
    fractional_kelly = float(os.getenv("FRACTIONAL_KELLY", "0.25"))

    historical = load_odds_csv(args.input)
    odds = normalize_odds_frame(historical.drop(columns=["actual_result"]))
    odds["actual_result"] = historical["actual_result"].astype(str).str.strip()
    fighter_stats = load_fighter_stats(args.fighter_stats)
    features = build_fight_features(odds, fighter_stats)
    side_model_path = Path(args.side_model) if args.side_model else default_side_model_path(ROOT)
    confidence_model_path = Path(args.confidence_model) if args.confidence_model else default_confidence_model_path(ROOT)
    side_model_bundle = load_side_model(side_model_path) if side_model_path.exists() else None
    confidence_model_bundle = load_confidence_model(confidence_model_path) if confidence_model_path.exists() else None
    projected = project_fight_probabilities(
        features,
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )
    projected["actual_result"] = odds["actual_result"]

    detailed_report, summary = evaluate_backtest(projected, min_edge, bankroll, fractional_kelly)
    write_summary_csv(summary, args.output)
    save_backtest_run(summary, args.db)

    print(detailed_report[["event_name", "selection_name", "american_odds", "edge", "stake", "profit"]].to_string(index=False))
    print(f"\nBacktest summary: {summary}")
    print(f"Saved backtest summary to {args.output}")


if __name__ == "__main__":
    main()
