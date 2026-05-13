import sys
import unittest
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.fetch_event_results import (
    _parse_official_ufc_event_fights,
    _parse_tapology_event_fights,
    build_closing_odds_lookup,
    build_results_frame,
    fetch_results_for_manifest,
    select_matching_event,
)


class _StubScraper:
    def __init__(self, fights_by_url):
        self.fights_by_url = fights_by_url

    def get_event_fights(self, event_url: str):
        return self.fights_by_url[event_url]


class FetchEventResultsTests(unittest.TestCase):
    def test_parse_official_ufc_event_fights_reads_winner_and_method(self) -> None:
        html = """
        <html><body>
        <div class="c-listing-fight">
          <div class="c-listing-fight__corner--red">
            <div class="c-listing-fight__outcome-wrapper">Win</div>
          </div>
          <div class="c-listing-fight__corner--blue">
            <div class="c-listing-fight__outcome-wrapper">Loss</div>
          </div>
          <div class="c-listing-fight__corner-name c-listing-fight__corner-name--red">Alpha Fighter</div>
          <div class="c-listing-fight__corner-name c-listing-fight__corner-name--blue">Beta Fighter</div>
          <div class="c-listing-fight__result">
            <div class="c-listing-fight__result-label">Round</div>
            <div class="c-listing-fight__result-text round">3</div>
          </div>
          <div class="c-listing-fight__result">
            <div class="c-listing-fight__result-label">Method</div>
            <div class="c-listing-fight__result-text">Decision - Unanimous</div>
          </div>
        </div>
        </body></html>
        """

        fights = _parse_official_ufc_event_fights(html)

        self.assertEqual(len(fights), 1)
        self.assertEqual(fights[0]["fighter_a"], "Alpha Fighter")
        self.assertEqual(fights[0]["fighter_b"], "Beta Fighter")
        self.assertEqual(fights[0]["winner"], "Alpha Fighter")
        self.assertEqual(fights[0]["method"], "Decision - Unanimous")

    def test_parse_tapology_event_fights_reads_decision_and_finish_rows(self) -> None:
        html = """
        <html><body>
        <h2>Fight Card</h2>
        <div>Decision, Unanimous  3 Rounds, 15:00 Total</div>
        <div>W</div>
        <div>Alpha Fighter</div>
        <div>Up to 11-0</div>
        <div>Main Card</div>
        <div>155</div>
        <div>3 x 5</div>
        <div>L</div>
        <div>Beta Fighter</div>
        <a href="/fightcenter/bouts/1">Matchup Page</a>
        <div>Submission, Rear Naked Choke  2:14 Round 1 of 3</div>
        <div>W</div>
        <div>Gamma Fighter</div>
        <div>Prelim</div>
        <div>170</div>
        <div>3 x 5</div>
        <div>L</div>
        <div>Delta Fighter</div>
        <a href="/fightcenter/bouts/2">Matchup Page</a>
        </body></html>
        """

        fights = _parse_tapology_event_fights(html)

        self.assertEqual(len(fights), 2)
        self.assertEqual(fights[0]["method"], "Decision, Unanimous")
        self.assertEqual(fights[0]["winner"], "Alpha Fighter")
        self.assertEqual(fights[1]["method"], "Submission, Rear Naked Choke")
        self.assertEqual(fights[1]["winner"], "Gamma Fighter")

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

    def test_build_results_frame_marks_decision_abbreviations_as_decisions(self) -> None:
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
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "winner": "Alpha",
                "method": "U-DEC",
            }
        ]

        results = build_results_frame(manifest, event_fights)

        self.assertEqual(len(results), 1)
        self.assertEqual(int(results.loc[0, "went_decision"]), 1)
        self.assertEqual(int(results.loc[0, "ended_inside_distance"]), 0)

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

    def test_fetch_results_for_manifest_falls_back_to_espn_history_when_ufcstats_is_unavailable(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "UFC Fight Night: Alpha vs. Beta",
            "start_time": "2026-04-25T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
            ],
        }

        class _FailingScraper:
            def get_event_list(self):
                raise requests.HTTPError("502 Server Error")

            def get_event_fights(self, event_url: str):
                raise AssertionError("get_event_fights should not be reached when event lookup fails")

        temp_dir = ROOT / ".tmp-test-fetch-event-results"
        temp_dir.mkdir(exist_ok=True)
        fighter_stats_path = temp_dir / "fighter_stats.csv"
        pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "source_url": "https://www.espn.com/mma/fighter/_/id/1/alpha",
                }
            ]
        ).to_csv(fighter_stats_path, index=False)

        def _history_fetcher(url: str) -> pd.DataFrame:
            self.assertEqual(url, "https://www.espn.com/mma/fighter/_/id/1/alpha")
            return pd.DataFrame(
                [
                    {
                        "date": pd.Timestamp("2026-04-25"),
                        "opponent": "Beta",
                        "event": "UFC Fight Night: Alpha vs. Beta",
                        "result": "W",
                        "result_code": "W",
                        "decision_type": "SUB",
                        "round_number": 2,
                        "minutes": 7.5,
                    }
                ]
            )

        try:
            results, metadata = fetch_results_for_manifest(
                manifest,
                scraper=_FailingScraper(),
                fighter_stats_path=fighter_stats_path,
                espn_history_fetcher=_history_fetcher,
            )
        finally:
            fighter_stats_path.unlink(missing_ok=True)
            temp_dir.rmdir()

        self.assertEqual(metadata["results_source"], "espn_fighter_history")
        self.assertEqual(len(results), 1)
        self.assertEqual(results.loc[0, "winner_name"], "Alpha")
        self.assertEqual(results.loc[0, "winner_side"], "fighter_a")
        self.assertEqual(results.loc[0, "method"], "SUB")
        self.assertEqual(int(results.loc[0, "ended_inside_distance"]), 1)

    def test_fetch_results_for_manifest_uses_tapology_fallback_when_ufcstats_fails(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "UFC Fight Night: Alpha vs. Beta",
            "start_time": "2026-04-25T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
            ],
        }

        class _FailingScraper:
            def get_event_list(self):
                raise requests.HTTPError("502 Server Error")

            def get_event_fights(self, event_url: str):
                raise AssertionError("get_event_fights should not be reached when event lookup fails")

        search_html = """
        <html><body>
        <table>
          <tr>
            <td><a href="/fightcenter/events/123-test-event">UFC Fight Night</a></td>
            <td>Alpha vs. Beta</td>
            <td>2026.04.25</td>
            <td>12</td>
          </tr>
        </table>
        </body></html>
        """
        event_html = """
        <html><body>
        <h2>Fight Card</h2>
        <div>Decision, Split  3 Rounds, 15:00 Total</div>
        <div>W</div>
        <div>Alpha</div>
        <div>Up to 10-0</div>
        <div>Main Card</div>
        <div>155</div>
        <div>3 x 5</div>
        <div>L</div>
        <div>Beta</div>
        <a href="/fightcenter/bouts/1">Matchup Page</a>
        </body></html>
        """

        def _html_fetcher(url: str) -> str:
            if "fightcenter/events/123-test-event" in url:
                return event_html
            if "tapology.com/search" in url:
                return search_html
            raise AssertionError(f"unexpected URL {url}")

        results, metadata = fetch_results_for_manifest(
            manifest,
            scraper=_FailingScraper(),
            tapology_html_fetcher=_html_fetcher,
        )

        self.assertEqual(metadata["results_source"], "tapology")
        self.assertEqual(results.loc[0, "winner_name"], "Alpha")
        self.assertEqual(int(results.loc[0, "went_decision"]), 1)
        self.assertEqual(int(results.loc[0, "ended_inside_distance"]), 0)
        self.assertEqual(results.loc[0, "method"], "Decision, Split")

    def test_fetch_results_for_manifest_uses_official_ufc_fallback_when_ufcstats_fails(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "UFC Fight Night: Alpha vs. Beta",
            "start_time": "2026-04-25T20:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta"},
            ],
            "ufc_event_url": "https://www.ufc.com/event/ufc-fight-night-april-25-2026",
        }

        class _FailingScraper:
            def get_event_list(self):
                raise requests.HTTPError("502 Server Error")

            def get_event_fights(self, event_url: str):
                raise AssertionError("get_event_fights should not be reached when event lookup fails")

        event_html = """
        <html><body>
        <div class="c-listing-fight">
          <div class="c-listing-fight__corner--red">
            <div class="c-listing-fight__outcome-wrapper">Loss</div>
          </div>
          <div class="c-listing-fight__corner--blue">
            <div class="c-listing-fight__outcome-wrapper">Win</div>
          </div>
          <div class="c-listing-fight__corner-name c-listing-fight__corner-name--red">Alpha</div>
          <div class="c-listing-fight__corner-name c-listing-fight__corner-name--blue">Beta</div>
          <div class="c-listing-fight__result">
            <div class="c-listing-fight__result-label">Method</div>
            <div class="c-listing-fight__result-text">Submission</div>
          </div>
        </div>
        </body></html>
        """

        def _html_fetcher(url: str) -> str:
            self.assertEqual(url, manifest["ufc_event_url"])
            return event_html

        results, metadata = fetch_results_for_manifest(
            manifest,
            scraper=_FailingScraper(),
            ufc_html_fetcher=_html_fetcher,
        )

        self.assertEqual(metadata["results_source"], "ufc_official")
        self.assertEqual(results.loc[0, "winner_name"], "Beta")
        self.assertEqual(int(results.loc[0, "went_decision"]), 0)
        self.assertEqual(int(results.loc[0, "ended_inside_distance"]), 1)
        self.assertEqual(results.loc[0, "method"], "Submission")

    def test_build_results_frame_marks_replacement_opponent_fights(self) -> None:
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
                "fighter_a": "Alpha",
                "fighter_b": "Gamma",
                "winner": "Alpha",
                "method": "DEC",
            }
        ]

        results = build_results_frame(manifest, event_fights)

        self.assertEqual(len(results), 1)
        self.assertEqual(results.loc[0, "result_status"], "replacement_opponent")
        self.assertEqual(results.loc[0, "result_match_status"], "replacement_opponent")
        self.assertEqual(results.loc[0, "winner_name"], "Alpha")
        self.assertEqual(results.loc[0, "actual_winner_name"], "Alpha")
        self.assertEqual(results.loc[0, "actual_fighter_a"], "Alpha")
        self.assertEqual(results.loc[0, "actual_fighter_b"], "Gamma")

    def test_build_results_frame_treats_punctuation_and_accents_as_same_fighters(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "Test Event",
            "start_time": "2026-05-09T21:00:00-04:00",
            "fights": [
                {"fighter_a": "Alexander Volkov", "fighter_b": "Waldo Cortes-Acosta"},
                {"fighter_a": "Joel Alvarez", "fighter_b": "Yaroslav Amosov"},
            ],
        }
        event_fights = [
            {
                "fighter_a": "Alexander Volkov",
                "fighter_b": "Waldo Cortes Acosta",
                "winner": "Alexander Volkov",
                "method": "Decision - Unanimous",
            },
            {
                "fighter_a": "Joel Álvarez",
                "fighter_b": "Yaroslav Amosov",
                "winner": "Yaroslav Amosov",
                "method": "Submission",
            },
        ]

        results = build_results_frame(manifest, event_fights)

        self.assertEqual(list(results["result_status"]), ["official", "official"])
        self.assertEqual(list(results["result_match_status"]), ["exact", "exact"])
        self.assertEqual(list(results["winner_side"]), ["fighter_a", "fighter_b"])


if __name__ == "__main__":
    unittest.main()
