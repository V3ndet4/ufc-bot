from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks
from models.selective import (
    DEFAULT_MIN_CLV_IMPROVEMENT,
    default_selective_model_path,
    save_selective_model,
    train_selective_clv_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a selective CLV model from graded tracked picks.")
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="SQLite database path.")
    parser.add_argument(
        "--output",
        default=str(default_selective_model_path(ROOT)),
        help="Output pickle path for the trained model.",
    )
    parser.add_argument("--dataset-output", help="Optional CSV path for the labeled training dataset.")
    parser.add_argument(
        "--min-clv-improvement",
        type=float,
        default=DEFAULT_MIN_CLV_IMPROVEMENT,
        help="Minimum implied-probability CLV improvement used to label positive examples.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=50,
        help="Minimum number of graded rows required before training succeeds.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracked = load_tracked_picks(args.db)
    try:
        bundle, training = train_selective_clv_model(
            tracked,
            min_clv_improvement=args.min_clv_improvement,
            min_samples=args.min_samples,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    saved_path = save_selective_model(bundle, args.output)

    if args.dataset_output:
        dataset_path = Path(args.dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        training.to_csv(dataset_path, index=False)

    if not args.quiet:
        print(f"trained_rows: {bundle['training_rows']}")
        print(f"positive_rate: {bundle['positive_rate']:.3f}")
        print(f"in_sample_auc: {bundle['in_sample_auc']:.3f}")
        print(f"model: {saved_path}")
        if args.dataset_output:
            print(f"dataset: {args.dataset_output}")


if __name__ == "__main__":
    main()
