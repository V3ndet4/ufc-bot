import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.bestfightodds import (
    extract_text_lines,
    enrich_odds_template_from_bestfightodds,
    enrich_with_bestfightodds_history,
    parse_book_order,
    parse_fighter_moneyline,
)


EVENT_HTML = """
<html>
  <body>
    <a href="/fighters/Movsar-Evloev-8457">Movsar Evloev</a>
    <a href="/fighters/Lerone-Murphy-9433">Lerone Murphy</a>
    <div>FanDuel Caesars BetMGM BetRivers BetWay Unibet DraftKings Bet365 PointsBet Props</div>
    <div>Lerone Murphy +196 +205 +200 +200 +200 92</div>
    <div>Movsar Evloev -260 -250 -250 -250 -250 92</div>
    <div>Luke Riley -230 -220 -225 -225 40</div>
    <div>Michael Aswell +190 +180 +185 +185 40</div>
  </body>
</html>
"""

HISTORY_HTML = """
<html>
  <body>
    <div>Movsar Evloev</div>
    <div>-250</div>
    <div>-260</div>
    <div>...</div>
    <div>-240</div>
    <div>0%</div>
    <div>UFC London</div>
    <div>Lerone Murphy</div>
    <div>Mar 21st 2026</div>
  </body>
</html>
"""


class BestFightOddsTests(unittest.TestCase):
    def test_parse_book_order(self) -> None:
        books = parse_book_order(extract_text_lines(EVENT_HTML))
        self.assertEqual(books[:3], ["FanDuel", "Caesars", "BetMGM"])

    def test_parse_fighter_moneyline(self) -> None:
        books = ["FanDuel", "Caesars", "BetMGM", "BetRivers"]
        odds = parse_fighter_moneyline(extract_text_lines(EVENT_HTML), "Movsar Evloev", books)
        self.assertEqual(odds["FanDuel"], -260)
        self.assertEqual(odds["BetMGM"], -250)

    def test_enrich_template_uses_consensus_odds(self) -> None:
        template = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "UFC London",
                    "start_time": "2026-03-21T13:00:00-04:00",
                    "fighter_a": "Movsar Evloev",
                    "fighter_b": "Lerone Murphy",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Movsar Evloev",
                    "book": "manual",
                    "american_odds": 0,
                },
                {
                    "event_id": "e1",
                    "event_name": "UFC London",
                    "start_time": "2026-03-21T13:00:00-04:00",
                    "fighter_a": "Movsar Evloev",
                    "fighter_b": "Lerone Murphy",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "selection_name": "Lerone Murphy",
                    "book": "manual",
                    "american_odds": 0,
                },
            ]
        )
        enriched = enrich_odds_template_from_bestfightodds(template, EVENT_HTML)
        self.assertEqual(enriched.loc[0, "book"], "bestfightodds_consensus")
        self.assertLess(enriched.loc[0, "american_odds"], 0)
        self.assertGreater(enriched.loc[1, "american_odds"], 0)

    def test_enrich_history_adds_open_and_range_fields(self) -> None:
        template = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "UFC Fight Night: Evloev vs. Murphy",
                    "start_time": "2026-03-21T13:00:00-04:00",
                    "fighter_a": "Movsar Evloev",
                    "fighter_b": "Lerone Murphy",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Movsar Evloev",
                    "book": "manual",
                    "american_odds": -250,
                }
            ]
        )

        import data_sources.bestfightodds as bfo

        original_fetch = bfo.fetch_html
        bfo.fetch_html = lambda url, session=None: HISTORY_HTML
        try:
            enriched = enrich_with_bestfightodds_history(template, EVENT_HTML)
        finally:
            bfo.fetch_html = original_fetch

        self.assertEqual(int(enriched.loc[0, "open_american_odds"]), -250)
        self.assertEqual(int(enriched.loc[0, "current_best_range_low"]), -260)
        self.assertEqual(int(enriched.loc[0, "current_best_range_high"]), -240)

    def test_enrich_template_can_leave_unmatched_rows_untouched(self) -> None:
        template = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "UFC London",
                    "start_time": "2026-03-21T13:00:00-04:00",
                    "fighter_a": "Jane Stone",
                    "fighter_b": "Lerone Murphy",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Jane Stone",
                    "book": "manual",
                    "american_odds": 0,
                }
            ]
        )

        enriched = enrich_odds_template_from_bestfightodds(template, EVENT_HTML, strict=False)
        self.assertEqual(int(enriched.loc[0, "american_odds"]), 0)
        self.assertEqual(enriched.loc[0, "book"], "manual")


if __name__ == "__main__":
    unittest.main()
