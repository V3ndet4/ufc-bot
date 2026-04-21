from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_archive import write_historical_archive
from data_sources.storage import grade_pending_picks, save_fight_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import fight results and auto-grade tracked UFC picks.")
    parser.add_argument("--results", required=True, help="Path to results CSV.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path.")
    parser.add_argument("--output", help="Optional CSV path for graded picks.")
    parser.add_argument("--event-id", help="Optional event id filter.")
    parser.add_argument(
        "--cards-root",
        default=str(ROOT / "cards"),
        help="Cards root scanned to refresh the historical moneyline archive.",
    )
    parser.add_argument(
        "--historical-archive-output",
        default=str(ROOT / "data" / "historical_market_odds.csv"),
        help="Output CSV path for the combined historical market archive.",
    )
    parser.add_argument(
        "--historical-archive-summary-output",
        default=str(ROOT / "reports" / "historical_market_archive_summary.csv"),
        help="Summary CSV path for the historical market archive refresh.",
    )
    parser.add_argument(
        "--skip-historical-archive",
        action="store_true",
        help="Skip refreshing the historical market archive after grading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = pd.read_csv(args.results)
    imported = save_fight_results(results, args.db)
    graded = grade_pending_picks(args.db, event_id=args.event_id)
    archive_rows = 0
    archive_fights = 0
    archive_path: str | None = None
    archive_summary_path: str | None = None
    if not args.skip_historical_archive:
        output_path, archive, _summary = write_historical_archive(
            args.cards_root,
            output_path=args.historical_archive_output,
            summary_output_path=args.historical_archive_summary_output,
        )
        archive_rows = int(len(archive))
        archive_fights = (
            int(archive[["event_id", "fighter_a", "fighter_b"]].drop_duplicates().shape[0])
            if not archive.empty
            else 0
        )
        archive_path = str(output_path)
        archive_summary_path = args.historical_archive_summary_output

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        graded.to_csv(output_path, index=False)

    wins = int((graded.get("actual_result") == "win").sum()) if not graded.empty else 0
    losses = int((graded.get("actual_result") == "loss").sum()) if not graded.empty else 0
    pushes = int((graded.get("actual_result") == "push").sum()) if not graded.empty else 0
    total_profit = round(float(graded.get("profit", pd.Series(dtype=float)).sum()), 2) if not graded.empty else 0.0
    avg_clv = round(float(graded["clv_delta"].dropna().mean()), 2) if not graded.empty and "clv_delta" in graded.columns and graded["clv_delta"].notna().any() else 0.0

    print(f"Imported {imported} fight results into {args.db}")
    print(f"Graded {len(graded)} tracked picks | wins {wins} | losses {losses} | pushes {pushes} | profit {total_profit:+.2f}")
    if not graded.empty and graded["clv_delta"].notna().any():
        print(f"Average CLV delta: {avg_clv:+.2f} odds points")
    if archive_path is not None:
        print(f"Historical market archive refreshed | fights {archive_fights} | rows {archive_rows}")
        print(f"Archive: {archive_path}")
        if archive_summary_path:
            print(f"Archive summary: {archive_summary_path}")
    if args.output:
        print(f"Saved graded picks to {args.output}")


if __name__ == "__main__":
    main()
