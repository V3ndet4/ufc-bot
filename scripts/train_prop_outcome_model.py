from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
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
        description="Train calibrated UFC prop outcome models from cached UFCStats history."
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
        "--experiment-log",
        default=str(ROOT / "data" / "model_experiments.csv"),
        help="Append per-market training metrics to this CSV for model experiment tracking.",
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
    experiment_path = _append_experiment_log(
        bundle,
        model_path=saved_path,
        dataset_path=args.dataset_output or args.input_csv or "",
        experiment_log=args.experiment_log,
        min_samples=args.min_samples,
    )

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
        print(f"experiment_log: {experiment_path}")
        if args.dataset_output:
            print(f"dataset: {args.dataset_output}")


def _append_experiment_log(
    bundle: dict[str, object],
    *,
    model_path: Path,
    dataset_path: str,
    experiment_log: str,
    min_samples: int,
) -> Path:
    output_path = Path(experiment_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rows: list[dict[str, object]] = []
    markets = bundle.get("markets", {})
    if isinstance(markets, dict):
        for market, details in markets.items():
            if not isinstance(details, dict):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "model_version": bundle.get("model_version", ""),
                    "model_path": _project_path(model_path),
                    "dataset_path": _project_path(dataset_path),
                    "market": market,
                    "target": details.get("target", ""),
                    "training_rows": details.get("training_rows", ""),
                    "positive_rate": details.get("positive_rate", ""),
                    "in_sample_auc": details.get("in_sample_auc", ""),
                    "in_sample_brier": details.get("in_sample_brier", ""),
                    "feature_count": len(bundle.get("numeric_features", [])),
                    "min_samples": min_samples,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "run_id",
                "created_at",
                "model_version",
                "model_path",
                "dataset_path",
                "market",
                "target",
                "training_rows",
                "positive_rate",
                "in_sample_auc",
                "in_sample_brier",
                "feature_count",
                "min_samples",
            ]
        )
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
    frame.to_csv(output_path, index=False)
    return output_path


def _project_path(path: str | Path) -> str:
    if not path:
        return ""
    resolved = Path(path)
    if not resolved.is_absolute():
        return str(resolved)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


if __name__ == "__main__":
    main()
