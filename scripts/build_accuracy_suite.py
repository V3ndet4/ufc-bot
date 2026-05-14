from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_snapshot_history, load_tracked_picks
from features.fighter_features import load_fighter_stats
from models.accuracy import (
    build_calibration_report,
    build_current_quality_report,
    build_market_accuracy_report,
    build_odds_movement_clv_report,
    build_fighter_identity_report,
    build_prop_clv_market_report,
    build_prop_market_readiness_report,
    build_prop_odds_inventory_report,
    build_prop_odds_archive_report,
    build_postmortem_code_report,
    build_prediction_snapshot,
    build_prop_model_backtest_predictions,
    build_prop_model_calibration_report,
    build_prop_model_family_report,
    build_prop_model_market_report,
    build_prop_model_walk_forward_predictions,
    build_prop_threshold_report,
    build_quality_gate_report,
    build_segment_performance_report,
    build_style_matchup_diagnostics,
    build_tracked_clv_report,
    normalize_tracked_pick_predictions,
    upsert_prediction_snapshot_archive,
)
from scripts.event_manifest import derived_paths, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UFC model accuracy, calibration, and quality-gate reports.")
    parser.add_argument("--manifest", required=True, help="Event manifest JSON.")
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="Tracked-picks SQLite database.")
    parser.add_argument(
        "--snapshot-archive",
        default=str(ROOT / "data" / "prediction_snapshots.csv"),
        help="Global pre-fight prediction snapshot archive.",
    )
    parser.add_argument(
        "--prop-history",
        default=str(ROOT / "data" / "prop_outcome_history.csv"),
        help="Historical prop outcome dataset built by scripts/train_prop_outcome_model.py.",
    )
    parser.add_argument("--min-gate-samples", type=int, default=5, help="Minimum graded picks before trusting a segment gate.")
    parser.add_argument("--min-prop-train-samples", type=int, default=400, help="Minimum older rows used to train each prop backtest model.")
    parser.add_argument("--min-prop-threshold-samples", type=int, default=50, help="Minimum holdout rows before recommending a prop probability threshold.")
    parser.add_argument("--min-prop-readiness-samples", type=int, default=500, help="Minimum walk-forward prop outcomes before marking a prop market bettable.")
    parser.add_argument("--min-prop-readiness-archive-fights", type=int, default=20, help="Minimum archived priced fights before marking a prop market bettable.")
    parser.add_argument("--min-prop-clv-samples", type=int, default=5, help="Minimum tracked prop CLV samples before marking a prop market bettable.")
    parser.add_argument("--min-prop-positive-clv-pct", type=float, default=52.0, help="Minimum positive CLV rate before marking a prop market bettable.")
    parser.add_argument("--min-prop-walk-forward-test-samples", type=int, default=200, help="Minimum rows per prop walk-forward test fold.")
    parser.add_argument("--prop-walk-forward-folds", type=int, default=3, help="Number of chronological prop walk-forward folds.")
    parser.add_argument("--prop-holdout-fraction", type=float, default=0.25, help="Most recent fraction of historical prop rows reserved for out-of-sample testing.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _load_alias_overrides() -> pd.DataFrame:
    frames = [
        _read_csv_if_exists(ROOT / "data" / "fighter_alias_overrides.csv"),
        _read_csv_if_exists(ROOT / "data" / "historical_fighter_alias_overrides.csv"),
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["source_name", "canonical_name", "notes"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in ["source_name", "canonical_name", "notes"]:
        if column not in combined.columns:
            combined[column] = ""
    return combined[["source_name", "canonical_name", "notes"]].drop_duplicates(subset=["source_name"], keep="last")


def _print_prop_accuracy_summary(prop_market_accuracy: pd.DataFrame, prop_thresholds: pd.DataFrame) -> None:
    if prop_market_accuracy.empty:
        print("Prop holdout accuracy: no historical prop backtest available")
        return
    print()
    print("Prop holdout accuracy")
    print("---------------------")
    for row in prop_market_accuracy.loc[prop_market_accuracy["market"].astype(str).ne("all")].to_dict("records"):
        print(
            f"{row.get('market')}: n={row.get('graded_props')}, "
            f"hit {float(row.get('hit_rate', 0.0)):.1%}, "
            f"avg prob {float(row.get('avg_model_prob', 0.0)):.1%}, "
            f"brier {float(row.get('brier', 0.0)):.3f}"
        )
    recommended = prop_thresholds.loc[prop_thresholds.get("is_recommended", pd.Series(dtype=object)).astype(str).eq("1")]
    if recommended.empty:
        print("Recommended prop thresholds: no threshold has enough reliable holdout sample yet")
        return
    recommendation_text = ", ".join(
        f"{row.get('market')} >= {float(row.get('min_model_prob', 0.0)):.0%}"
        for row in recommended.to_dict("records")
    )
    print(f"Recommended prop thresholds: {recommendation_text}")


def _print_prop_readiness_summary(prop_market_readiness: pd.DataFrame) -> None:
    if prop_market_readiness.empty:
        print("Prop market readiness: no readiness report available")
        return
    print()
    print("Prop market readiness")
    print("---------------------")
    for row in prop_market_readiness.to_dict("records"):
        print(
            f"{row.get('market')}: {row.get('market_action')} "
            f"(sample {row.get('outcome_samples')}, archive fights {row.get('archive_fights')}, "
            f"CLV sample {row.get('clv_samples')}) - "
            f"{row.get('readiness_reason')}"
        )


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)
    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    fighter_stats = load_fighter_stats(paths["fighter_stats"]) if paths["fighter_stats"].exists() else pd.DataFrame()
    no_odds_packet = _read_csv_if_exists(reports_dir / "no_odds_prediction_packet.csv")
    lean_board = _read_csv_if_exists(paths["lean_board"])
    fight_report = _read_csv_if_exists(paths["report"])
    current_prop_odds = _read_csv_if_exists(paths["modeled_market_odds"])
    alias_overrides = _load_alias_overrides()

    snapshot = build_prediction_snapshot(
        manifest=manifest,
        fighter_stats=fighter_stats,
        no_odds_packet=no_odds_packet,
        lean_board=lean_board,
        fight_report=fight_report,
    )
    snapshot_path = reports_dir / "prediction_snapshot.csv"
    snapshot.to_csv(snapshot_path, index=False)
    archive_path = upsert_prediction_snapshot_archive(snapshot, args.snapshot_archive)

    tracked = load_tracked_picks(args.db) if Path(args.db).exists() else pd.DataFrame()
    prediction_history = normalize_tracked_pick_predictions(tracked)
    calibration = build_calibration_report(prediction_history)
    segment_performance = build_segment_performance_report(prediction_history)
    market_accuracy = build_market_accuracy_report(prediction_history)
    prop_bet_market_accuracy = market_accuracy.loc[
        ~market_accuracy["market"].astype(str).isin(["all", "moneyline", "unknown"])
    ].copy()
    quality_gates = build_quality_gate_report(segment_performance, min_samples=args.min_gate_samples)
    current_quality = build_current_quality_report(snapshot, quality_gates)
    style_matchups = build_style_matchup_diagnostics(snapshot, lean_board=lean_board, fight_report=fight_report)
    postmortem_codes = build_postmortem_code_report(prediction_history)
    prop_history = _read_csv_if_exists(Path(args.prop_history))
    prop_backtest = build_prop_model_backtest_predictions(
        prop_history,
        holdout_fraction=args.prop_holdout_fraction,
        min_train_samples=args.min_prop_train_samples,
    )
    prop_walk_forward = build_prop_model_walk_forward_predictions(
        prop_history,
        folds=args.prop_walk_forward_folds,
        min_train_samples=args.min_prop_train_samples,
        min_test_samples=args.min_prop_walk_forward_test_samples,
    )
    prop_market_accuracy = build_prop_model_market_report(prop_backtest)
    prop_family_accuracy = build_prop_model_family_report(prop_backtest)
    prop_walk_forward_market_accuracy = build_prop_model_market_report(prop_walk_forward)
    prop_walk_forward_family_accuracy = build_prop_model_family_report(prop_walk_forward)
    prop_calibration = build_prop_model_calibration_report(prop_backtest)
    prop_thresholds = build_prop_threshold_report(
        prop_backtest,
        min_samples=args.min_prop_threshold_samples,
    )
    snapshot_history = load_snapshot_history(args.db) if Path(args.db).exists() else pd.DataFrame()
    prop_odds_archive = build_prop_odds_archive_report(snapshot_history)
    fighter_identity = build_fighter_identity_report(manifest, fighter_stats, prop_history, alias_overrides=alias_overrides)
    prop_odds_inventory = build_prop_odds_inventory_report(snapshot_history, current_prop_odds=current_prop_odds)
    prop_clv_market_quality = build_prop_clv_market_report(
        prediction_history,
        min_samples=args.min_prop_clv_samples,
        min_positive_clv_pct=args.min_prop_positive_clv_pct,
    )
    readiness_accuracy = prop_walk_forward_market_accuracy if not prop_walk_forward_market_accuracy.empty else prop_market_accuracy
    prop_market_readiness = build_prop_market_readiness_report(
        readiness_accuracy,
        prop_thresholds,
        prop_odds_inventory,
        prop_clv_market_quality,
        min_model_samples=args.min_prop_readiness_samples,
        min_archive_fights=args.min_prop_readiness_archive_fights,
        min_clv_samples=args.min_prop_clv_samples,
        min_positive_clv_pct=args.min_prop_positive_clv_pct,
    )
    odds_movement_clv = build_odds_movement_clv_report(snapshot_history)
    tracked_clv = build_tracked_clv_report(prediction_history)

    calibration_path = reports_dir / "accuracy_calibration.csv"
    segment_path = reports_dir / "segment_performance.csv"
    market_accuracy_path = reports_dir / "market_accuracy.csv"
    prop_bet_market_accuracy_path = reports_dir / "prop_bet_market_accuracy.csv"
    gates_path = reports_dir / "segment_quality_gates.csv"
    global_gates_path = ROOT / "data" / "segment_quality_gates.csv"
    current_quality_path = reports_dir / "current_prediction_quality.csv"
    style_path = reports_dir / "style_matchup_diagnostics.csv"
    postmortem_path = reports_dir / "accuracy_postmortem_codes.csv"
    prop_backtest_path = reports_dir / "prop_model_backtest_predictions.csv"
    prop_walk_forward_path = reports_dir / "prop_model_walk_forward_predictions.csv"
    prop_market_path = reports_dir / "prop_model_market_accuracy.csv"
    prop_family_path = reports_dir / "prop_model_family_accuracy.csv"
    prop_walk_forward_market_path = reports_dir / "prop_walk_forward_market_accuracy.csv"
    prop_walk_forward_family_path = reports_dir / "prop_walk_forward_family_accuracy.csv"
    prop_calibration_path = reports_dir / "prop_model_calibration.csv"
    prop_thresholds_path = reports_dir / "prop_model_thresholds.csv"
    prop_odds_archive_path = reports_dir / "prop_odds_archive_summary.csv"
    fighter_identity_path = reports_dir / "fighter_identity_report.csv"
    prop_odds_inventory_path = reports_dir / "prop_odds_inventory.csv"
    prop_clv_market_quality_path = reports_dir / "prop_clv_market_quality.csv"
    prop_market_readiness_path = reports_dir / "prop_market_readiness.csv"
    odds_movement_clv_path = reports_dir / "odds_movement_clv.csv"
    tracked_clv_path = reports_dir / "tracked_clv.csv"

    calibration.to_csv(calibration_path, index=False)
    segment_performance.to_csv(segment_path, index=False)
    market_accuracy.to_csv(market_accuracy_path, index=False)
    prop_bet_market_accuracy.to_csv(prop_bet_market_accuracy_path, index=False)
    quality_gates.to_csv(gates_path, index=False)
    global_gates_path.parent.mkdir(parents=True, exist_ok=True)
    quality_gates.to_csv(global_gates_path, index=False)
    current_quality.to_csv(current_quality_path, index=False)
    style_matchups.to_csv(style_path, index=False)
    postmortem_codes.to_csv(postmortem_path, index=False)
    prop_backtest.to_csv(prop_backtest_path, index=False)
    prop_walk_forward.to_csv(prop_walk_forward_path, index=False)
    prop_market_accuracy.to_csv(prop_market_path, index=False)
    prop_family_accuracy.to_csv(prop_family_path, index=False)
    prop_walk_forward_market_accuracy.to_csv(prop_walk_forward_market_path, index=False)
    prop_walk_forward_family_accuracy.to_csv(prop_walk_forward_family_path, index=False)
    prop_calibration.to_csv(prop_calibration_path, index=False)
    prop_thresholds.to_csv(prop_thresholds_path, index=False)
    prop_odds_archive.to_csv(prop_odds_archive_path, index=False)
    fighter_identity.to_csv(fighter_identity_path, index=False)
    prop_odds_inventory.to_csv(prop_odds_inventory_path, index=False)
    prop_clv_market_quality.to_csv(prop_clv_market_quality_path, index=False)
    prop_market_readiness.to_csv(prop_market_readiness_path, index=False)
    odds_movement_clv.to_csv(odds_movement_clv_path, index=False)
    tracked_clv.to_csv(tracked_clv_path, index=False)

    if not args.quiet:
        print(f"Saved prediction snapshot to {snapshot_path}")
        print(f"Updated global snapshot archive at {archive_path}")
        print(f"Saved calibration report to {calibration_path}")
        print(f"Saved segment performance to {segment_path}")
        print(f"Saved market accuracy to {market_accuracy_path}")
        print(f"Saved prop bet market accuracy to {prop_bet_market_accuracy_path}")
        print(f"Saved quality gates to {gates_path}")
        print(f"Saved current prediction quality to {current_quality_path}")
        print(f"Saved style matchup diagnostics to {style_path}")
        print(f"Saved postmortem codes to {postmortem_path}")
        print(f"Saved prop backtest predictions to {prop_backtest_path}")
        print(f"Saved prop walk-forward predictions to {prop_walk_forward_path}")
        print(f"Saved prop market accuracy to {prop_market_path}")
        print(f"Saved prop family accuracy to {prop_family_path}")
        print(f"Saved prop walk-forward market accuracy to {prop_walk_forward_market_path}")
        print(f"Saved prop walk-forward family accuracy to {prop_walk_forward_family_path}")
        print(f"Saved prop calibration to {prop_calibration_path}")
        print(f"Saved prop thresholds to {prop_thresholds_path}")
        print(f"Saved prop odds archive summary to {prop_odds_archive_path}")
        print(f"Saved fighter identity report to {fighter_identity_path}")
        print(f"Saved prop odds inventory to {prop_odds_inventory_path}")
        print(f"Saved prop CLV market quality to {prop_clv_market_quality_path}")
        print(f"Saved prop market readiness to {prop_market_readiness_path}")
        print(f"Saved odds movement CLV to {odds_movement_clv_path}")
        print(f"Saved tracked CLV to {tracked_clv_path}")
        _print_prop_accuracy_summary(prop_market_accuracy, prop_thresholds)
        _print_prop_readiness_summary(prop_market_readiness)


if __name__ == "__main__":
    main()
