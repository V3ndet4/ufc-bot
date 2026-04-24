from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_archive import write_historical_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a historical market archive from completed card folders.")
    parser.add_argument(
        "--cards-root",
        default=str(ROOT / "cards"),
        help="Cards root directory to scan.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "data" / "historical_market_odds.csv"),
        help="Output CSV path for the combined historical market archive.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(ROOT / "reports" / "historical_market_archive_summary.csv"),
        help="Optional summary CSV path.",
    )
    parser.add_argument(
        "--snapshot-db",
        default=str(ROOT / "data" / "ufc_betting.db"),
        help="Optional SQLite DB path used to recover missing closing odds from stored odds snapshots.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path, archive, summary = write_historical_archive(
        args.cards_root,
        output_path=args.output,
        summary_output_path=args.summary_output,
        snapshot_db_path=args.snapshot_db,
    )

    if not args.quiet:
        print(f"rows_written: {len(archive)}")
        markets_written = int(archive["market"].nunique()) if not archive.empty and "market" in archive.columns else 0
        print(f"fights_written: {archive[['event_id', 'fighter_a', 'fighter_b']].drop_duplicates().shape[0] if not archive.empty else 0}")
        print(f"markets_written: {markets_written}")
        if not summary.empty:
            ok_cards = int((summary["status"] == "ok").sum()) if "status" in summary.columns else 0
            print(f"cards_with_exportable_results: {ok_cards}")
            skipped = int(summary.get("skipped_missing_odds", 0).fillna(0).sum()) if "skipped_missing_odds" in summary.columns else 0
            print(f"fights_skipped_missing_odds: {skipped}")
        print(f"archive: {output_path}")
        if args.summary_output:
            print(f"summary: {args.summary_output}")


if __name__ == "__main__":
    main()
