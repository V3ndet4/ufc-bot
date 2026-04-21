from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_sources.bestfightodds import (
    enrich_odds_template_from_bestfightodds,
    enrich_with_bestfightodds_history,
    fetch_html,
    write_odds_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill an odds template with current BestFightOdds moneylines.")
    parser.add_argument("--template", required=True, help="Path to the existing odds template CSV.")
    parser.add_argument("--event-url", required=True, help="BestFightOdds event URL.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--book",
        default="consensus",
        help="Specific sportsbook to use, or 'consensus' for median implied probability.",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also enrich rows with BestFightOdds fighter-history open and current-range fields.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if a fighter from the template cannot be matched on the BestFightOdds event page.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = pd.read_csv(args.template)
    html = fetch_html(args.event_url)
    enriched = enrich_odds_template_from_bestfightodds(
        template,
        html,
        book_preference=args.book,
        strict=args.strict,
    )
    if args.include_history:
        enriched = enrich_with_bestfightodds_history(enriched, html)
    output_path = write_odds_csv(enriched, args.output)
    if not args.quiet:
        print(f"Filled {len(enriched)} rows from BestFightOdds")
        print(f"Saved odds to {output_path}")


if __name__ == "__main__":
    main()
