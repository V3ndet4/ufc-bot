from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_training import (
    build_historical_projection_dataset,
    load_historical_alias_overrides,
    load_historical_training_datasets,
    write_historical_unmatched_reports,
)
from data_sources.odds_api import load_odds_csv
from data_sources.storage import load_tracked_picks
from models.side import default_side_model_path, save_side_model, train_side_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a calibrated moneyline side model from graded tracked picks.")
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="SQLite database path.")
    parser.add_argument("--input-csv", help="Optional prebuilt selection-level training CSV.")
    parser.add_argument(
        "--historical-odds",
        default=str(ROOT / "data" / "historical_market_odds.csv"),
        help="Historical odds CSV used to build a leak-safe training set. Defaults to data/historical_market_odds.csv.",
    )
    parser.add_argument(
        "--skip-historical-odds",
        action="store_true",
        help="Ignore the historical market archive and train only from tracked picks / --input-csv.",
    )
    parser.add_argument(
        "--greco-cache-dir",
        default=str(ROOT / ".tmp" / "external_ufc_history"),
        help="Cache directory containing the Greco UFC history CSVs.",
    )
    parser.add_argument(
        "--alias-overrides",
        default=str(ROOT / "data" / "historical_fighter_alias_overrides.csv"),
        help="Optional CSV mapping source_name to canonical_name for historical Greco joins.",
    )
    parser.add_argument(
        "--date-tolerance-days",
        type=int,
        default=3,
        help="Max allowed date gap when matching historical odds fights to Greco bouts.",
    )
    parser.add_argument(
        "--output",
        default=str(default_side_model_path(ROOT)),
        help="Output pickle path for the trained side model.",
    )
    parser.add_argument("--dataset-output", help="Optional CSV path for the labeled training dataset.")
    parser.add_argument(
        "--unmatched-fights-output",
        default=str(ROOT / "reports" / "historical_training_unmatched_fights.csv"),
        help="CSV path for unmatched historical fight rows when building from --historical-odds.",
    )
    parser.add_argument(
        "--unmatched-fighters-output",
        default=str(ROOT / "reports" / "historical_training_unmatched_fighters.csv"),
        help="CSV path for aggregated unmatched fighter rows when building from --historical-odds.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum number of graded moneyline rows required before training succeeds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _load_training_source(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.input_csv:
        return pd.read_csv(args.input_csv), pd.DataFrame()
    historical_odds_path = Path(args.historical_odds) if args.historical_odds else None
    if not args.skip_historical_odds and historical_odds_path is not None and historical_odds_path.exists():
        datasets = load_historical_training_datasets(args.greco_cache_dir)
        alias_overrides = load_historical_alias_overrides(args.alias_overrides)
        projected, unmatched = build_historical_projection_dataset(
            load_odds_csv(historical_odds_path),
            fight_results=datasets["fight_results"],
            fight_stats=datasets["fight_stats"],
            event_details=datasets["event_details"],
            fighter_tott=datasets["fighter_tott"],
            date_tolerance_days=args.date_tolerance_days,
            alias_overrides=alias_overrides,
        )
        return projected, unmatched
    return load_tracked_picks(args.db), pd.DataFrame()


def main() -> None:
    args = parse_args()
    tracked, unmatched = _load_training_source(args)
    unmatched_fights_path: Path | None = None
    unmatched_fighters_path: Path | None = None

    historical_odds_path = Path(args.historical_odds) if args.historical_odds else None
    if (
        not args.input_csv
        and not args.skip_historical_odds
        and historical_odds_path is not None
        and historical_odds_path.exists()
    ):
        unmatched_fights_path, unmatched_fighters_path = write_historical_unmatched_reports(
            unmatched,
            unmatched_fights_output=args.unmatched_fights_output,
            unmatched_fighters_output=args.unmatched_fighters_output,
        )
    try:
        bundle, training = train_side_model(
            tracked,
            min_samples=args.min_samples,
        )
    except ValueError as exc:
        if unmatched_fights_path is not None:
            print(f"unmatched_fights_report: {unmatched_fights_path}")
        if unmatched_fighters_path is not None:
            print(f"unmatched_fighters_report: {unmatched_fighters_path}")
        raise SystemExit(str(exc))
    saved_path = save_side_model(bundle, args.output)

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
        if unmatched_fights_path is not None:
            print(f"unmatched_fights_report: {unmatched_fights_path}")
        if unmatched_fighters_path is not None:
            print(f"unmatched_fighters_report: {unmatched_fighters_path}")
        if args.dataset_output:
            print(f"dataset: {args.dataset_output}")


if __name__ == "__main__":
    main()
