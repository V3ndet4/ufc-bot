from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import load_odds_csv
from features.fighter_features import build_fight_features, load_fighter_stats
from models.confidence import default_confidence_model_path, load_confidence_model
from models.projection import project_fight_probabilities
from models.side import default_side_model_path, load_side_model
from normalization.odds import normalize_odds_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project win probabilities from fighter stats.")
    parser.add_argument("--odds", required=True, help="Path to the odds CSV input.")
    parser.add_argument("--fighter-stats", required=True, help="Path to the fighter stats CSV.")
    parser.add_argument("--output", required=True, help="Output CSV path for projected probabilities.")
    parser.add_argument("--side-model", help="Optional calibrated side-model pickle path.")
    parser.add_argument("--confidence-model", help="Optional calibrated confidence-model pickle path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    odds = normalize_odds_frame(load_odds_csv(args.odds))
    fighter_stats = load_fighter_stats(args.fighter_stats)
    features = build_fight_features(odds, fighter_stats)
    side_model_path = Path(args.side_model) if args.side_model else default_side_model_path(ROOT)
    confidence_model_path = Path(args.confidence_model) if args.confidence_model else default_confidence_model_path(ROOT)
    side_model_bundle = load_side_model(side_model_path) if side_model_path.exists() else None
    confidence_model_bundle = load_confidence_model(confidence_model_path) if confidence_model_path.exists() else None
    projections = project_fight_probabilities(
        features,
        side_model_bundle=side_model_bundle,
        confidence_model_bundle=confidence_model_bundle,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "event_name",
        "selection_name",
        "american_odds",
        "projected_fighter_a_win_prob",
        "model_projected_win_prob",
        "model_confidence",
    ]
    projections[columns].to_csv(output_path, index=False)
    if projections.empty:
        print("No active odds rows available to project.")
    else:
        print(projections[columns].to_string(index=False))
    print(f"\nSaved projected probabilities to {output_path}")


if __name__ == "__main__":
    main()
