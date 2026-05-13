from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.prop_outcomes import (
    build_prop_outcome_history_frame,
    default_prop_outcome_history_path,
    load_cached_external_history_datasets,
)
from models.prop_outcomes import (
    default_prop_outcome_model_path,
    save_prop_outcome_model,
    train_prop_outcome_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train calibrated 1+ takedown / 1+ knockdown prop models from cached UFCStats history."
    )
    parser.add_argument(
        "--greco-cache-dir",
        default=str(ROOT / ".tmp" / "external_ufc_history"),
        help="Cache directory containing Greco UFC history CSVs.",
    )
    parser.add_argument(
        "--input-csv",
        help="Optional prebuilt prop outcome history CSV. When omitted, the cached Greco CSVs are used.",
    )
    parser.add_argument(
        "--dataset-output",
        default=str(default_prop_outcome_history_path(ROOT)),
        help="CSV path for the built prop outcome history dataset.",
    )
    parser.add_argument(
        "--output",
        default=str(default_prop_outcome_model_path(ROOT)),
        help="Output pickle path for the trained prop outcome model.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=400,
        help="Minimum labeled rows required per prop market.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress training summary output.")
    return parser.parse_args()


def _load_or_build_training_frame(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv:
        return pd.read_csv(args.input_csv)

    datasets = load_cached_external_history_datasets(args.greco_cache_dir)
    training = build_prop_outcome_history_frame(
        fight_results=datasets["fight_results"],
        fight_stats=datasets["fight_stats"],
        event_details=datasets["event_details"],
    )
    if args.dataset_output:
        dataset_path = Path(args.dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        training.to_csv(dataset_path, index=False)
    return training


def main() -> None:
    args = parse_args()
    training_source = _load_or_build_training_frame(args)
    try:
        bundle, training = train_prop_outcome_model(training_source, min_samples=args.min_samples)
    except ValueError as exc:
        raise SystemExit(str(exc))
    saved_path = save_prop_outcome_model(bundle, args.output)

    if args.input_csv and args.dataset_output:
        dataset_path = Path(args.dataset_output)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        training.to_csv(dataset_path, index=False)

    if not args.quiet:
        print(f"trained_rows: {bundle['training_rows']}")
        for market, details in bundle["markets"].items():
            print(
                f"{market}: rows {details['training_rows']} | "
                f"positive_rate {details['positive_rate']:.3f} | "
                f"auc {details['in_sample_auc']:.3f} | "
                f"brier {details['in_sample_brier']:.3f}"
            )
        if bundle.get("skipped"):
            for market, reason in bundle["skipped"].items():
                print(f"{market} skipped: {reason}")
        print(f"model: {saved_path}")
        if args.dataset_output:
            print(f"dataset: {args.dataset_output}")


if __name__ == "__main__":
    main()
