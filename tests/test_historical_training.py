import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_training import (
    build_historical_projection_dataset,
    build_unmatched_fighter_report,
)


class HistoricalTrainingTests(unittest.TestCase):
    def _sample_fight_results(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Alpha One vs. Gamma Three",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight",
                    "METHOD": "KO/TKO",
                    "ROUND": 2,
                    "TIME": "3:00",
                },
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Beta Two vs. Delta Four",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight",
                    "METHOD": "Decision - Unanimous",
                    "ROUND": 3,
                    "TIME": "5:00",
                },
                {
                    "EVENT": "UFC Test 2",
                    "BOUT": "Alpha One vs. Beta Two",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight",
                    "METHOD": "Decision - Unanimous",
                    "ROUND": 3,
                    "TIME": "5:00",
                },
            ]
        )

    def _sample_event_details(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"EVENT": "UFC Test 1", "DATE": "2024-01-01"},
                {"EVENT": "UFC Test 2", "DATE": "2024-04-01"},
            ]
        )

    def _sample_fight_stats(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Alpha One vs. Gamma Three",
                    "FIGHTER": "Alpha One",
                    "KD": 1,
                    "SIG.STR.": "30 of 60",
                    "TD": "2 of 5",
                    "SUB.ATT": 1,
                    "CTRL": "2:00",
                    "HEAD": "18 of 36",
                    "BODY": "6 of 12",
                    "LEG": "6 of 12",
                    "DISTANCE": "22 of 44",
                    "CLINCH": "5 of 10",
                    "GROUND": "3 of 6",
                },
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Alpha One vs. Gamma Three",
                    "FIGHTER": "Gamma Three",
                    "KD": 0,
                    "SIG.STR.": "18 of 48",
                    "TD": "0 of 2",
                    "SUB.ATT": 0,
                    "CTRL": "0:45",
                    "HEAD": "10 of 28",
                    "BODY": "4 of 10",
                    "LEG": "4 of 10",
                    "DISTANCE": "14 of 36",
                    "CLINCH": "3 of 8",
                    "GROUND": "1 of 4",
                },
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Beta Two vs. Delta Four",
                    "FIGHTER": "Beta Two",
                    "KD": 0,
                    "SIG.STR.": "24 of 52",
                    "TD": "1 of 3",
                    "SUB.ATT": 0,
                    "CTRL": "1:30",
                    "HEAD": "14 of 32",
                    "BODY": "4 of 10",
                    "LEG": "6 of 10",
                    "DISTANCE": "17 of 37",
                    "CLINCH": "5 of 9",
                    "GROUND": "2 of 6",
                },
                {
                    "EVENT": "UFC Test 1",
                    "BOUT": "Beta Two vs. Delta Four",
                    "FIGHTER": "Delta Four",
                    "KD": 0,
                    "SIG.STR.": "16 of 44",
                    "TD": "0 of 1",
                    "SUB.ATT": 0,
                    "CTRL": "0:20",
                    "HEAD": "9 of 24",
                    "BODY": "3 of 10",
                    "LEG": "4 of 10",
                    "DISTANCE": "12 of 34",
                    "CLINCH": "3 of 7",
                    "GROUND": "1 of 3",
                },
                {
                    "EVENT": "UFC Test 2",
                    "BOUT": "Alpha One vs. Beta Two",
                    "FIGHTER": "Alpha One",
                    "KD": 0,
                    "SIG.STR.": "20 of 44",
                    "TD": "1 of 3",
                    "SUB.ATT": 0,
                    "CTRL": "1:00",
                    "HEAD": "12 of 28",
                    "BODY": "4 of 8",
                    "LEG": "4 of 8",
                    "DISTANCE": "16 of 34",
                    "CLINCH": "3 of 6",
                    "GROUND": "1 of 4",
                },
                {
                    "EVENT": "UFC Test 2",
                    "BOUT": "Alpha One vs. Beta Two",
                    "FIGHTER": "Beta Two",
                    "KD": 0,
                    "SIG.STR.": "18 of 42",
                    "TD": "1 of 2",
                    "SUB.ATT": 0,
                    "CTRL": "1:10",
                    "HEAD": "10 of 24",
                    "BODY": "4 of 8",
                    "LEG": "4 of 10",
                    "DISTANCE": "14 of 32",
                    "CLINCH": "3 of 6",
                    "GROUND": "1 of 4",
                },
            ]
        )

    def _sample_fighter_tott(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"FIGHTER": "Alpha One", "HEIGHT": "5' 11\"", "WEIGHT": "155 lbs.", "REACH": "72\"", "STANCE": "Orthodox", "DOB": "1992-01-01", "URL": "alpha"},
                {"FIGHTER": "Beta Two", "HEIGHT": "5' 10\"", "WEIGHT": "155 lbs.", "REACH": "71\"", "STANCE": "Southpaw", "DOB": "1991-06-01", "URL": "beta"},
                {"FIGHTER": "Gamma Three", "HEIGHT": "5' 9\"", "WEIGHT": "155 lbs.", "REACH": "70\"", "STANCE": "Orthodox", "DOB": "1990-01-01", "URL": "gamma"},
                {"FIGHTER": "Delta Four", "HEIGHT": "5' 8\"", "WEIGHT": "155 lbs.", "REACH": "69\"", "STANCE": "Orthodox", "DOB": "1989-01-01", "URL": "delta"},
            ]
        )

    def test_build_historical_projection_dataset_uses_only_prior_fights(self) -> None:
        historical_odds = pd.DataFrame(
            [
                {
                    "event_id": "ufc-test-2",
                    "event_name": "UFC Test 2",
                    "start_time": "2024-04-01T20:00:00Z",
                    "fighter_a": "Alpha One",
                    "fighter_b": "Beta Two",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -120,
                    "actual_result": "win",
                },
                {
                    "event_id": "ufc-test-2",
                    "event_name": "UFC Test 2",
                    "start_time": "2024-04-01T20:00:00Z",
                    "fighter_a": "Alpha One",
                    "fighter_b": "Beta Two",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": 100,
                    "actual_result": "loss",
                },
            ]
        )

        projected, unmatched = build_historical_projection_dataset(
            historical_odds,
            fight_results=self._sample_fight_results(),
            fight_stats=self._sample_fight_stats(),
            event_details=self._sample_event_details(),
            fighter_tott=self._sample_fighter_tott(),
        )

        self.assertTrue(unmatched.empty)
        self.assertEqual(len(projected), 2)
        self.assertEqual(projected.loc[0, "fighter_a"], "Alpha One")
        self.assertEqual(projected.loc[0, "fighter_b"], "Beta Two")
        self.assertEqual(int(projected.loc[0, "a_wins"]), 1)
        self.assertEqual(int(projected.loc[0, "b_wins"]), 1)
        self.assertEqual(int(projected.loc[0, "a_ufc_fight_count"]), 1)
        self.assertEqual(int(projected.loc[0, "b_ufc_fight_count"]), 1)
        self.assertAlmostEqual(float(projected.loc[0, "a_days_since_last_fight"]), 91.0, places=0)
        self.assertIn("Orthodox", str(projected.loc[0, "a_history_style_label"]))
        self.assertEqual(projected.loc[0, "actual_result"], "win")

    def test_build_historical_projection_dataset_applies_alias_overrides(self) -> None:
        historical_odds = pd.DataFrame(
            [
                {
                    "event_id": "ufc-test-2",
                    "event_name": "UFC Test 2",
                    "start_time": "2024-04-01T20:00:00Z",
                    "fighter_a": "A. One",
                    "fighter_b": "Beta Two",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -120,
                    "actual_result": "win",
                },
                {
                    "event_id": "ufc-test-2",
                    "event_name": "UFC Test 2",
                    "start_time": "2024-04-01T20:00:00Z",
                    "fighter_a": "A. One",
                    "fighter_b": "Beta Two",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": 100,
                    "actual_result": "loss",
                },
            ]
        )
        alias_overrides = pd.DataFrame(
            [
                {
                    "source_name": "A. One",
                    "canonical_name": "Alpha One",
                    "notes": "test alias",
                }
            ]
        )

        projected, unmatched = build_historical_projection_dataset(
            historical_odds,
            fight_results=self._sample_fight_results(),
            fight_stats=self._sample_fight_stats(),
            event_details=self._sample_event_details(),
            fighter_tott=self._sample_fighter_tott(),
            alias_overrides=alias_overrides,
        )

        self.assertTrue(unmatched.empty)
        self.assertEqual(len(projected), 2)
        self.assertEqual(projected.loc[0, "fighter_a"], "A. One")

    def test_build_unmatched_fighter_report_groups_alias_context(self) -> None:
        unmatched_fights = pd.DataFrame(
            [
                {
                    "event_id": "ufc-test-9",
                    "event_name": "UFC Test 9",
                    "start_time": "2024-06-01T20:00:00Z",
                    "fighter_a": "A. One",
                    "fighter_b": "Beta Two",
                    "resolved_fighter_a": "Alpha One",
                    "resolved_fighter_b": "Beta Two",
                    "fighter_a_key": "alpha one",
                    "fighter_b_key": "beta two",
                    "fighter_a_alias_applied": True,
                    "fighter_b_alias_applied": False,
                    "reason": "date_out_of_tolerance",
                    "candidate_count": 1,
                    "nearest_event": "UFC Test 2",
                    "nearest_bout": "Alpha One vs. Beta Two",
                    "nearest_date": "2024-04-01",
                    "nearest_date_gap_days": 61,
                }
            ]
        )

        fighter_report = build_unmatched_fighter_report(unmatched_fights)

        self.assertEqual(len(fighter_report), 2)
        alpha_row = fighter_report.loc[fighter_report["source_name"] == "A. One"].iloc[0]
        self.assertEqual(alpha_row["resolved_name"], "Alpha One")
        self.assertTrue(bool(alpha_row["alias_applied"]))
        self.assertEqual(int(alpha_row["unmatched_fight_count"]), 1)
        self.assertIn("date_out_of_tolerance", str(alpha_row["reasons"]))
        self.assertEqual(alpha_row["nearest_event"], "UFC Test 2")


if __name__ == "__main__":
    unittest.main()
