from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks
from features.fighter_features import load_fighter_stats
from models.accuracy import (
    build_calibration_report,
    build_current_quality_report,
    build_postmortem_code_report,
    build_prediction_snapshot,
    build_quality_gate_report,
    build_segment_performance_report,
    build_style_matchup_diagnostics,
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
    parser.add_argument("--min-gate-samples", type=int, default=5, help="Minimum graded picks before trusting a segment gate.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


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
    quality_gates = build_quality_gate_report(segment_performance, min_samples=args.min_gate_samples)
    current_quality = build_current_quality_report(snapshot, quality_gates)
    style_matchups = build_style_matchup_diagnostics(snapshot, lean_board=lean_board, fight_report=fight_report)
    postmortem_codes = build_postmortem_code_report(prediction_history)

    calibration_path = reports_dir / "accuracy_calibration.csv"
    segment_path = reports_dir / "segment_performance.csv"
    gates_path = reports_dir / "segment_quality_gates.csv"
    global_gates_path = ROOT / "data" / "segment_quality_gates.csv"
    current_quality_path = reports_dir / "current_prediction_quality.csv"
    style_path = reports_dir / "style_matchup_diagnostics.csv"
    postmortem_path = reports_dir / "accuracy_postmortem_codes.csv"

    calibration.to_csv(calibration_path, index=False)
    segment_performance.to_csv(segment_path, index=False)
    quality_gates.to_csv(gates_path, index=False)
    global_gates_path.parent.mkdir(parents=True, exist_ok=True)
    quality_gates.to_csv(global_gates_path, index=False)
    current_quality.to_csv(current_quality_path, index=False)
    style_matchups.to_csv(style_path, index=False)
    postmortem_codes.to_csv(postmortem_path, index=False)

    if not args.quiet:
        print(f"Saved prediction snapshot to {snapshot_path}")
        print(f"Updated global snapshot archive at {archive_path}")
        print(f"Saved calibration report to {calibration_path}")
        print(f"Saved segment performance to {segment_path}")
        print(f"Saved quality gates to {gates_path}")
        print(f"Saved current prediction quality to {current_quality_path}")
        print(f"Saved style matchup diagnostics to {style_path}")
        print(f"Saved postmortem codes to {postmortem_path}")


if __name__ == "__main__":
    main()
