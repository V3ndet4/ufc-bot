from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.event_manifest import current_event_manifest_path, derived_paths, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UFC bot accuracy upgrade loop: refresh history features, retrain prop models, and rebuild accuracy reports."
    )
    parser.add_argument(
        "--manifest",
        default=str(current_event_manifest_path(ROOT)),
        help="Event manifest JSON. Defaults to events/current_event.txt.",
    )
    parser.add_argument(
        "--greco-cache-dir",
        default=str(ROOT / ".tmp" / "external_ufc_history"),
        help="Cache directory for external UFC history CSVs.",
    )
    parser.add_argument("--db", default=str(ROOT / "data" / "ufc_betting.db"), help="SQLite tracked-picks database.")
    parser.add_argument("--skip-external-refresh", action="store_true", help="Skip live external UFC history enrichment for the current card.")
    parser.add_argument("--quiet-children", action="store_true", help="Suppress child command output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)

    if not args.skip_external_refresh and paths["fighter_stats"].exists():
        _run(
            [
                sys.executable,
                "scripts/enrich_fighter_stats_with_external_ufc_history.py",
                "--input",
                str(paths["fighter_stats"]),
                "--output",
                str(paths["fighter_stats"]),
                "--cache-dir",
                args.greco_cache_dir,
                *_quiet_flag(args.quiet_children),
            ]
        )

    _run(
        [
            sys.executable,
            "scripts/train_prop_outcome_model.py",
            "--greco-cache-dir",
            args.greco_cache_dir,
            "--dataset-output",
            str(ROOT / "data" / "prop_outcome_history.csv"),
            "--output",
            str(ROOT / "models" / "prop_outcome_model.pkl"),
            *_quiet_flag(args.quiet_children),
        ]
    )

    _run(
        [
            sys.executable,
            "scripts/build_accuracy_suite.py",
            "--manifest",
            args.manifest,
            "--db",
            args.db,
            *_quiet_flag(args.quiet_children),
        ]
    )

    if not args.quiet_children:
        print("Accuracy upgrade complete.")
        print(f"Prop model: {ROOT / 'models' / 'prop_outcome_model.pkl'}")
        print(f"Prop thresholds: {paths['prop_model_thresholds']}")
        print(f"Prop readiness: {paths['prop_market_readiness']}")
        print(f"Prop CLV quality: {paths['prop_clv_market_quality']}")
        print(f"Fighter identity: {paths['fighter_identity_report']}")
        print(f"Tracked CLV: {paths['tracked_clv']}")
        print(f"Odds movement CLV: {paths['odds_movement_clv']}")


def _quiet_flag(enabled: bool) -> list[str]:
    return ["--quiet"] if enabled else []


def _run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
