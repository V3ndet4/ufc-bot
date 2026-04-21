from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_training import build_historical_projection_dataset, load_historical_training_datasets
from data_sources.odds_api import load_odds_csv
from data_sources.storage import load_tracked_picks
from models.confidence import (
    default_confidence_model_path,
    save_confidence_model,
    train_confidence_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a calibrated fight-level confidence model.")
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="SQLite database path.")
    parser.add_argument("--input-csv", help="Optional prebuilt selection-level training CSV.")
    parser.add_argument("--historical-odds", help="Optional historical odds CSV used to build a leak-safe training set.")
    parser.add_argument(
        "--greco-cache-dir",
        default=str(ROOT / ".tmp" / "external_ufc_history"),
        help="Cache directory containing the Greco UFC history CSVs.",
    )
    parser.add_argument(
        "--date-tolerance-days",
        type=int,
        default=3,
        help="Max allowed date gap when matching historical odds fights to Greco bouts.",
    )
    parser.add_argument(
        "--output",
        default=str(default_confidence_model_path(ROOT)),
        help="Output pickle path for the trained confidence model.",
    )
    parser.add_argument("--dataset-output", help="Optional CSV path for the labeled confidence training dataset.")
    parser.add_argument(
        "--min-samples",
        type=int,
        default=40,
        help="Minimum number of graded fights required before training succeeds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _load_training_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.input_csv:
        return pd.read_csv(args.input_csv), pd.DataFrame()
    if args.historical_odds:
        datasets = load_historical_training_datasets(args.greco_cache_dir)
        projected, unmatched = build_historical_projection_dataset(
            load_odds_csv(args.historical_odds),
            fight_results=datasets["fight_results"],
            fight_stats=datasets["fight_stats"],
            event_details=datasets["event_details"],
            fighter_tott=datasets["fighter_tott"],
            date_tolerance_days=args.date_tolerance_days,
        )
        return projected, unmatched
    return load_tracked_picks(args.db), pd.DataFrame()


def main() -> None:
    args = parse_args()
    tracked, unmatched = _load_training_source(args)
    try:
        bundle, training = train_confidence_model(
            tracked,
            min_samples=args.min_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    saved_path = save_confidence_model(bundle, args.output)

    if args.dataset_output:
        dataset_path = Path(args.dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        training.to_csv(dataset_path, index=False)

    if not args.quiet:
        print(f"trained_rows: {bundle['training_rows']}")
        print(f"positive_rate: {bundle['positive_rate']:.3f}")
        print(f"in_sample_auc: {bundle['in_sample_auc']:.3f}")
        print(f"in_sample_brier: {bundle['in_sample_brier']:.3f}")
        print(f"model: {saved_path}")
        if not unmatched.empty:
            print(f"unmatched_fights: {len(unmatched)}")
        if args.dataset_output:
            print(f"dataset: {args.dataset_output}")


if __name__ == "__main__":
    main()
