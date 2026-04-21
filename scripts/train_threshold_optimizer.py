from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks
from models.threshold_policy import build_threshold_policy, default_threshold_policy_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a threshold policy from graded tracked picks.")
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "ufc_betting.db"),
        help="SQLite database path containing tracked picks.",
    )
    parser.add_argument(
        "--output",
        default=str(default_threshold_policy_path(ROOT)),
        help="JSON output path for the threshold policy.",
    )
    parser.add_argument(
        "--min-graded-bets",
        type=int,
        default=8,
        help="Minimum graded bets required before the optimizer can move off the baseline policy.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracked = load_tracked_picks(args.db)
    policy = build_threshold_policy(tracked, min_graded_bets=args.min_graded_bets)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    if not args.quiet:
        selected = policy.get("selected", {})
        print(
            "Threshold policy "
            f"{policy.get('status', 'unknown')}: "
            f"edge>={float(selected.get('min_edge', 0.0)):.1%}, "
            f"confidence>={float(selected.get('min_model_confidence', 0.0)):.2f}, "
            f"stats>={float(selected.get('min_stats_completeness', 0.0)):.2f}, "
            f"fallback={'off' if bool(selected.get('exclude_fallback_rows', False)) else 'on'}"
        )
        print(f"Saved threshold policy to {output_path}")


if __name__ == "__main__":
    main()
