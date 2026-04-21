import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_event_results import (
    build_closing_odds_lookup,
    build_results_frame,
    select_matching_event,
)


class _StubScraper:
    def __init__(self, fights_by_url):
        self.fights_by_url = fights_by_url

    def get_event_fights(self, event_url: str):
        return self.fights_by_url[event_url]


class FetchEventResultsTests(unittest.TestCase):
    def test_build_results_frame_aligns_winner_side_to_manifest_order(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "Test Event",
            "start_time": "2026-04-18T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
            ],
        }
        event_fights = [
            {
                "fighter_a": "Beta",
                "fighter_b": "Alpha",
                "winner": "Beta",
                "method": "KO/TKO",
            }
        ]
        snapshots = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "american_odds": 120,
                    "snapshot_time": "2026-04-18T19:00:00Z",
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "american_odds": -140,
                    "snapshot_time": "2026-04-18T19:00:00Z",
                },
            ]
        )

        results = build_results_frame(manifest, event_fights, snapshot_history=snapshots)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.loc[0, "winner_name"], "Beta")
        self.assertEqual(results.loc[0, "winner_side"], "fighter_b")
        self.assertEqual(int(results.loc[0, "ended_inside_distance"]), 1)
        self.assertEqual(int(results.loc[0, "closing_fighter_a_odds"]), 120)
        self.assertEqual(int(results.loc[0, "closing_fighter_b_odds"]), -140)

    def test_build_closing_odds_lookup_prefers_latest_pre_event_snapshot(self) -> None:
        snapshots = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "american_odds": 130,
                    "snapshot_time": "2026-04-18T18:00:00Z",
                },
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "american_odds": 115,
                    "snapshot_time": "2026-04-18T19:30:00Z",
                },
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "american_odds": 105,
                    "snapshot_time": "2026-04-18T20:30:00Z",
                },
            ]
        )

        lookup = build_closing_odds_lookup(snapshots, start_time="2026-04-18T20:00:00Z")

        self.assertEqual(lookup["alpha||beta"]["closing_fighter_a_odds"], 115)

    def test_build_closing_odds_lookup_includes_decision_props(self) -> None:
        snapshots = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_goes_to_decision",
                    "selection": "fight_goes_to_decision",
                    "american_odds": 125,
                    "snapshot_time": "2026-04-18T19:30:00Z",
                },
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_doesnt_go_to_decision",
                    "selection": "fight_doesnt_go_to_decision",
                    "american_odds": -155,
                    "snapshot_time": "2026-04-18T19:30:00Z",
                },
            ]
        )

        lookup = build_closing_odds_lookup(snapshots, start_time="2026-04-18T20:00:00Z")

        self.assertEqual(lookup["alpha||beta"]["closing_fight_goes_to_decision_odds"], 125)
        self.assertEqual(lookup["alpha||beta"]["closing_fight_doesnt_go_to_decision_odds"], -155)

    def test_select_matching_event_prefers_highest_fight_overlap(self) -> None:
        manifest = {
            "event_name": "UFC Fight Night: Burns vs. Malott",
            "start_time": "2026-04-18T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
                {"fighter_a": "Gamma", "fighter_b": "Delta"},
            ],
        }
        events = [
            {"name": "Wrong Event", "url": "event-1", "date": "April 18, 2026"},
            {"name": "UFC Fight Night: Burns vs. Malott", "url": "event-2", "date": "April 18, 2026"},
        ]
        fights_by_url = {
            "event-1": [{"fighter_a": "Alpha", "fighter_b": "Beta", "winner": "Alpha", "method": "Decision"}],
            "event-2": [
                {"fighter_a": "Alpha", "fighter_b": "Beta", "winner": "Alpha", "method": "Decision"},
                {"fighter_a": "Gamma", "fighter_b": "Delta", "winner": "Delta", "method": "Submission"},
            ],
        }
        scraper = _StubScraper(fights_by_url)

        event, fights = select_matching_event(manifest, scraper=type("Proxy", (), {
            "get_event_list": lambda self: events,
            "get_event_fights": scraper.get_event_fights,
        })())

        self.assertEqual(event["url"], "event-2")
        self.assertEqual(len(fights), 2)

    def test_select_matching_event_rejects_future_manifest_before_fetching_events(self) -> None:
        manifest = {
            "event_name": "Future Event",
            "start_time": "2099-04-25T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
            ],
        }

        class _FutureProxy:
            def get_event_list(self):
                raise AssertionError("get_event_list should not be called for future events")

            def get_event_fights(self, event_url: str):
                raise AssertionError("get_event_fights should not be called for future events")

        with self.assertRaisesRegex(ValueError, r"2099-04-25.*future relative to local date"):
            select_matching_event(manifest, scraper=_FutureProxy())


if __name__ == "__main__":
    unittest.main()
