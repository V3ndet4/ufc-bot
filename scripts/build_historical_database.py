"""
Build Historical UFC Database

Master script to collect:
1. All UFC fights (outcomes, methods, dates)
2. Historical odds (open/close lines)
3. Fighter stats as of each fight date

Usage:
    python scripts/build_historical_database.py --start-year 2020 --end-year 2026

This is the foundation for training data. Run this first before modeling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_sources.ufc_history import UFCHistoricalCollector


def main():
    parser = argparse.ArgumentParser(
        description="Build historical UFC database for model training"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/historical_ufc.db",
        help="Path to SQLite database"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="Start collecting from this year"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2026,
        help="Collect up to this year"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit to N events (for testing)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between requests (be nice to servers)"
    )
    parser.add_argument(
        "--skip-fights",
        action="store_true",
        help="Skip fight collection (if already have fights)"
    )
    parser.add_argument(
        "--skip-odds",
        action="store_true",
        help="Skip odds collection"
    )
    parser.add_argument(
        "--skip-fighter-stats",
        action="store_true",
        help="Skip fighter stats collection"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UFC HISTORICAL DATABASE BUILDER")
    print("=" * 60)
    print(f"Database: {args.db_path}")
    print(f"Year range: {args.start_year} - {args.end_year}")
    print()
    
    # Step 1: Collect fights
    if not args.skip_fights:
        print("\n[1/3] Collecting UFC fight history...")
        print("-" * 60)
        
        collector = UFCHistoricalCollector(
            db_path=args.db_path,
            delay=args.delay
        )
        
        # Collect events
        events = collector.fetch_event_list()
        
        # Filter by year
        # Note: This is simplified - real filtering would parse event dates
        filtered_events = events[:args.limit] if args.limit else events
        
        print(f"Found {len(filtered_events)} events to process")
        
        total_fights = 0
        for i, event in enumerate(filtered_events):
            print(f"\n[{i+1}/{len(filtered_events)}] {event['event_name']}")
            
            try:
                fights = collector.fetch_event_details(event['event_url'])
                collector.save_fights(fights)
                total_fights += len(fights)
                print(f"  ✓ Saved {len(fights)} fights")
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        print(f"\n✓ Step 1 complete: {total_fights} fights saved")
    else:
        print("\n[1/3] Skipping fight collection (--skip-fights)")
    
    # Step 2: Collect odds
    if not args.skip_odds:
        print("\n[2/3] Collecting historical odds...")
        print("-" * 60)
        print("Note: This requires BestFightOdds scraping")
        print("This is complex - may need manual intervention for now")
        print("For now, use --skip-odds and we'll add this later")
    else:
        print("\n[2/3] Skipping odds collection (--skip-odds)")
    
    # Step 3: Build fighter stats
    if not args.skip_fighter_stats:
        print("\n[3/3] Building historical fighter stats...")
        print("-" * 60)
        print("Note: This requires scraping every fighter page")
        print("This will take a LONG time for full history")
        print("For now, use --skip-fighter-stats and we'll batch this")
    else:
        print("\n[3/3] Skipping fighter stats (--skip-fighter-stats)")
    
    print("\n" + "=" * 60)
    print("BUILD COMPLETE")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Check database: sqlite3 data/historical_ufc.db")
    print("2. Query fights: SELECT COUNT(*) FROM fights;")
    print("3. When ready, run with --limit 10 to test odds/fighter collection")


if __name__ == "__main__":
    main()
