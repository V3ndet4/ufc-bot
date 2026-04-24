import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_snapshot_history, save_odds_snapshot
from scripts import fetch_the_odds_api_odds


class FetchTheOddsApiOddsTests(unittest.TestCase):
    def test_main_saves_snapshot_by_default(self) -> None:
        template_path = ROOT / "tests" / "_tmp_fetch_odds_template.csv"
        output_path = ROOT / "tests" / "_tmp_fetch_odds_output.csv"
        db_path = ROOT / "tests" / "_tmp_fetch_odds.db"
        template_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,manual,0",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,manual,0",
                ]
            ),
            encoding="utf-8",
        )
        events = [
            {
                "id": "fight-1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Alpha", "price": -120},
                                    {"name": "Beta", "price": 102},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        try:
            with patch.object(fetch_the_odds_api_odds, "load_api_key", return_value="test-key"), patch.object(
                fetch_the_odds_api_odds, "fetch_the_odds_api_events", return_value=events
            ), patch.object(
                fetch_the_odds_api_odds, "fetch_modeled_market_snapshots", return_value=pd.DataFrame()
            ), patch.object(
                sys,
                "argv",
                [
                    "fetch_the_odds_api_odds.py",
                    "--template",
                    str(template_path),
                    "--output",
                    str(output_path),
                    "--db",
                    str(db_path),
                    "--bookmaker",
                    "fanduel",
                ],
            ):
                fetch_the_odds_api_odds.main()
            output_exists = output_path.exists()
            snapshots = load_snapshot_history(db_path)
        finally:
            template_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            db_path.unlink(missing_ok=True)

        self.assertTrue(output_exists)
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(set(snapshots["book"].astype(str)), {"fanduel"})

    def test_main_saves_modeled_market_snapshots(self) -> None:
        template_path = ROOT / "tests" / "_tmp_fetch_odds_template.csv"
        output_path = ROOT / "tests" / "_tmp_fetch_odds_output.csv"
        db_path = ROOT / "tests" / "_tmp_fetch_odds.db"
        template_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,manual,0",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,manual,0",
                ]
            ),
            encoding="utf-8",
        )
        events = [
            {
                "id": "fight-1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Alpha", "price": -120},
                                    {"name": "Beta", "price": 102},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        modeled_snapshots = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_goes_to_decision",
                    "selection": "fight_goes_to_decision",
                    "selection_name": "Fight goes to decision",
                    "book": "fanduel",
                    "american_odds": 110,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_doesnt_go_to_decision",
                    "selection": "fight_doesnt_go_to_decision",
                    "selection_name": "Fight doesn't go to decision",
                    "book": "fanduel",
                    "american_odds": -140,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "fanduel",
                    "american_odds": 210,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "selection": "fighter_b",
                    "selection_name": "Beta",
                    "book": "fanduel",
                    "american_odds": 280,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "by_decision",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "fanduel",
                    "american_odds": 325,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "by_decision",
                    "selection": "fighter_b",
                    "selection_name": "Beta",
                    "book": "fanduel",
                    "american_odds": 400,
                },
            ]
        )
        try:
            with patch.object(fetch_the_odds_api_odds, "load_api_key", return_value="test-key"), patch.object(
                fetch_the_odds_api_odds, "fetch_the_odds_api_events", return_value=events
            ), patch.object(
                fetch_the_odds_api_odds, "fetch_modeled_market_snapshots", return_value=modeled_snapshots
            ), patch.object(
                sys,
                "argv",
                [
                    "fetch_the_odds_api_odds.py",
                    "--template",
                    str(template_path),
                    "--output",
                    str(output_path),
                    "--db",
                    str(db_path),
                    "--bookmaker",
                    "fanduel",
                ],
            ):
                fetch_the_odds_api_odds.main()
            snapshots = load_snapshot_history(db_path)
        finally:
            template_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            db_path.unlink(missing_ok=True)

        self.assertEqual(set(snapshots["market"].astype(str)), {
            "moneyline",
            "fight_goes_to_decision",
            "fight_doesnt_go_to_decision",
            "inside_distance",
            "by_decision",
        })
        self.assertEqual(
            int(
                snapshots.loc[
                    (snapshots["market"].astype(str) == "fight_goes_to_decision")
                    & (snapshots["selection"].astype(str) == "fight_goes_to_decision"),
                    "american_odds",
                ].iloc[0]
            ),
            110,
        )

    def test_main_fills_modeled_market_output_from_full_template(self) -> None:
        template_path = ROOT / "tests" / "_tmp_fetch_odds_template.csv"
        output_path = ROOT / "tests" / "_tmp_fetch_odds_output.csv"
        template_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,fight_goes_to_decision,fight_goes_to_decision,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,inside_distance,fighter_a,manual,",
                ]
            ),
            encoding="utf-8",
        )
        events = [
            {
                "id": "fight-1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Alpha", "price": -120},
                                    {"name": "Beta", "price": 102},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        modeled_snapshots = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_goes_to_decision",
                    "selection": "fight_goes_to_decision",
                    "selection_name": "Fight goes to decision",
                    "book": "fanduel",
                    "american_odds": 110,
                    "odds_api_event_id": "fight-1",
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "fanduel",
                    "american_odds": 210,
                    "odds_api_event_id": "fight-1",
                },
            ]
        )
        try:
            with patch.object(fetch_the_odds_api_odds, "load_api_key", return_value="test-key"), patch.object(
                fetch_the_odds_api_odds, "fetch_the_odds_api_events", return_value=events
            ), patch.object(
                fetch_the_odds_api_odds, "fetch_modeled_market_snapshots", return_value=modeled_snapshots
            ), patch.object(
                sys,
                "argv",
                [
                    "fetch_the_odds_api_odds.py",
                    "--template",
                    str(template_path),
                    "--output",
                    str(output_path),
                    "--bookmaker",
                    "fanduel",
                    "--no-snapshot",
                ],
            ):
                fetch_the_odds_api_odds.main()
            output_frame = pd.read_csv(output_path)
        finally:
            template_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertEqual(
            set(output_frame["market"].astype(str)),
            {"moneyline", "fight_goes_to_decision", "inside_distance"},
        )
        self.assertEqual(set(output_frame["book"].astype(str)), {"fanduel"})
        self.assertEqual(set(output_frame["odds_api_event_id"].astype(str)), {"fight-1"})

    def test_main_skips_snapshot_when_flagged(self) -> None:
        template_path = ROOT / "tests" / "_tmp_fetch_odds_template.csv"
        output_path = ROOT / "tests" / "_tmp_fetch_odds_output.csv"
        db_path = ROOT / "tests" / "_tmp_fetch_odds.db"
        template_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,manual,0",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,manual,0",
                ]
            ),
            encoding="utf-8",
        )
        events = [
            {
                "id": "fight-1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Alpha", "price": -120},
                                    {"name": "Beta", "price": 102},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        try:
            with patch.object(fetch_the_odds_api_odds, "load_api_key", return_value="test-key"), patch.object(
                fetch_the_odds_api_odds, "fetch_the_odds_api_events", return_value=events
            ), patch.object(
                sys,
                "argv",
                [
                    "fetch_the_odds_api_odds.py",
                    "--template",
                    str(template_path),
                    "--output",
                    str(output_path),
                    "--db",
                    str(db_path),
                    "--bookmaker",
                    "fanduel",
                    "--no-snapshot",
                ],
            ):
                fetch_the_odds_api_odds.main()
        finally:
            template_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            db_path.unlink(missing_ok=True)

        self.assertFalse(db_path.exists())

    def test_main_reuses_snapshot_history_for_fights_missing_from_live_feed(self) -> None:
        template_path = ROOT / "tests" / "_tmp_fetch_odds_template.csv"
        output_path = ROOT / "tests" / "_tmp_fetch_odds_output.csv"
        db_path = ROOT / "tests" / "_tmp_fetch_odds.db"
        template_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Gamma,Delta,moneyline,fighter_a,manual,",
                    "e1,Test Event,2026-04-13T20:00:00Z,Gamma,Delta,moneyline,fighter_b,manual,",
                ]
            ),
            encoding="utf-8",
        )
        save_odds_snapshot(
            pd.DataFrame(
                [
                    {
                        "event_id": "e1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-13T20:00:00Z",
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "selection_name": "Gamma",
                        "book": "fanduel",
                        "american_odds": 118,
                    },
                    {
                        "event_id": "e1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-13T20:00:00Z",
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "selection_name": "Delta",
                        "book": "fanduel",
                        "american_odds": -138,
                    },
                ]
            ),
            db_path,
        )
        events = [
            {
                "id": "fight-1",
                "home_team": "Alpha",
                "away_team": "Beta",
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Alpha", "price": -120},
                                    {"name": "Beta", "price": 102},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        try:
            with patch.object(fetch_the_odds_api_odds, "load_api_key", return_value="test-key"), patch.object(
                fetch_the_odds_api_odds, "fetch_the_odds_api_events", return_value=events
            ), patch.object(
                fetch_the_odds_api_odds, "fetch_modeled_market_snapshots", return_value=pd.DataFrame()
            ), patch.object(
                sys,
                "argv",
                [
                    "fetch_the_odds_api_odds.py",
                    "--template",
                    str(template_path),
                    "--output",
                    str(output_path),
                    "--db",
                    str(db_path),
                    "--bookmaker",
                    "fanduel",
                ],
            ):
                fetch_the_odds_api_odds.main()
            output_frame = pd.read_csv(output_path)
            snapshots = load_snapshot_history(db_path)
        finally:
            template_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            db_path.unlink(missing_ok=True)

        self.assertEqual(len(output_frame), 4)
        alpha_rows = output_frame.loc[output_frame["fighter_a"].astype(str) == "Alpha"].reset_index(drop=True)
        gamma_rows = output_frame.loc[output_frame["fighter_a"].astype(str) == "Gamma"].reset_index(drop=True)
        self.assertEqual(set(alpha_rows["odds_source"].astype(str)), {"live_api"})
        self.assertEqual(set(gamma_rows["odds_source"].astype(str)), {"snapshot_history"})
        self.assertEqual(set(gamma_rows["snapshot_fallback_used"].astype(int)), {1})
        self.assertEqual(int(gamma_rows.loc[0, "american_odds"]), 118)
        self.assertEqual(int(gamma_rows.loc[1, "american_odds"]), -138)
        self.assertEqual(len(snapshots), 4)


if __name__ == "__main__":
    unittest.main()
