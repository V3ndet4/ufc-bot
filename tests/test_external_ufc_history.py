import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import _days_since
from data_sources.external_ufc_history import (
    build_external_ufc_history_features,
    merge_external_ufc_history_into_fighter_stats,
)


class ExternalUfcHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.event_details = pd.DataFrame(
            [
                {"EVENT": "UFC Alpha 3", "URL": "http://example.com/3", "DATE": "April 1, 2026", "LOCATION": "USA"},
                {"EVENT": "UFC Alpha 2", "URL": "http://example.com/2", "DATE": "January 1, 2026", "LOCATION": "USA"},
                {"EVENT": "UFC Alpha 1", "URL": "http://example.com/1", "DATE": "July 1, 2025", "LOCATION": "USA"},
            ]
        )
        self.fight_results = pd.DataFrame(
            [
                {
                    "EVENT": "UFC Alpha 3",
                    "BOUT": "Alpha vs. Beta",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight Bout",
                    "METHOD": "Decision - Unanimous",
                    "ROUND": 3,
                    "TIME": "5:00",
                },
                {
                    "EVENT": "UFC Alpha 2",
                    "BOUT": "Alpha vs. Gamma",
                    "OUTCOME": "L/W",
                    "WEIGHTCLASS": "Lightweight Bout",
                    "METHOD": "KO/TKO",
                    "ROUND": 2,
                    "TIME": "1:00",
                },
                {
                    "EVENT": "UFC Alpha 1",
                    "BOUT": "Alpha vs. Delta",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight Bout",
                    "METHOD": "Submission",
                    "ROUND": 1,
                    "TIME": "3:00",
                },
            ]
        )
        self.fight_stats = pd.DataFrame(
            [
                {
                    "EVENT": "UFC Alpha 3",
                    "BOUT": "Alpha vs. Beta",
                    "ROUND": "Round 1",
                    "FIGHTER": "Alpha",
                    "KD": 0,
                    "SIG.STR.": "30 of 60",
                    "SIG.STR. %": "50%",
                    "TOTAL STR.": "35 of 70",
                    "TD": "1 of 3",
                    "TD %": "33%",
                    "SUB.ATT": 0,
                    "REV.": 0,
                    "CTRL": "4:00",
                    "HEAD": "20 of 40",
                    "BODY": "5 of 10",
                    "LEG": "5 of 10",
                    "DISTANCE": "25 of 50",
                    "CLINCH": "5 of 10",
                    "GROUND": "0 of 0",
                },
                {
                    "EVENT": "UFC Alpha 3",
                    "BOUT": "Alpha vs. Beta",
                    "ROUND": "Round 1",
                    "FIGHTER": "Beta",
                    "KD": 0,
                    "SIG.STR.": "20 of 50",
                    "SIG.STR. %": "40%",
                    "TOTAL STR.": "24 of 55",
                    "TD": "0 of 2",
                    "TD %": "0%",
                    "SUB.ATT": 0,
                    "REV.": 0,
                    "CTRL": "0:30",
                    "HEAD": "12 of 30",
                    "BODY": "4 of 10",
                    "LEG": "4 of 10",
                    "DISTANCE": "18 of 42",
                    "CLINCH": "2 of 8",
                    "GROUND": "0 of 0",
                },
                {
                    "EVENT": "UFC Alpha 2",
                    "BOUT": "Alpha vs. Gamma",
                    "ROUND": "Round 1",
                    "FIGHTER": "Alpha",
                    "KD": 0,
                    "SIG.STR.": "10 of 25",
                    "SIG.STR. %": "40%",
                    "TOTAL STR.": "12 of 29",
                    "TD": "2 of 4",
                    "TD %": "50%",
                    "SUB.ATT": 1,
                    "REV.": 0,
                    "CTRL": "2:30",
                    "HEAD": "7 of 16",
                    "BODY": "2 of 4",
                    "LEG": "1 of 5",
                    "DISTANCE": "8 of 20",
                    "CLINCH": "2 of 5",
                    "GROUND": "0 of 0",
                },
                {
                    "EVENT": "UFC Alpha 2",
                    "BOUT": "Alpha vs. Gamma",
                    "ROUND": "Round 1",
                    "FIGHTER": "Gamma",
                    "KD": 1,
                    "SIG.STR.": "15 of 30",
                    "SIG.STR. %": "50%",
                    "TOTAL STR.": "16 of 32",
                    "TD": "0 of 1",
                    "TD %": "0%",
                    "SUB.ATT": 0,
                    "REV.": 0,
                    "CTRL": "1:00",
                    "HEAD": "9 of 18",
                    "BODY": "3 of 6",
                    "LEG": "3 of 6",
                    "DISTANCE": "13 of 26",
                    "CLINCH": "2 of 4",
                    "GROUND": "0 of 0",
                },
                {
                    "EVENT": "UFC Alpha 1",
                    "BOUT": "Alpha vs. Delta",
                    "ROUND": "Round 1",
                    "FIGHTER": "Alpha",
                    "KD": 0,
                    "SIG.STR.": "8 of 12",
                    "SIG.STR. %": "67%",
                    "TOTAL STR.": "9 of 13",
                    "TD": "0 of 0",
                    "TD %": "0%",
                    "SUB.ATT": 2,
                    "REV.": 0,
                    "CTRL": "1:30",
                    "HEAD": "5 of 8",
                    "BODY": "1 of 2",
                    "LEG": "2 of 2",
                    "DISTANCE": "8 of 12",
                    "CLINCH": "0 of 0",
                    "GROUND": "0 of 0",
                },
                {
                    "EVENT": "UFC Alpha 1",
                    "BOUT": "Alpha vs. Delta",
                    "ROUND": "Round 1",
                    "FIGHTER": "Delta",
                    "KD": 0,
                    "SIG.STR.": "3 of 10",
                    "SIG.STR. %": "30%",
                    "TOTAL STR.": "4 of 12",
                    "TD": "0 of 0",
                    "TD %": "0%",
                    "SUB.ATT": 0,
                    "REV.": 0,
                    "CTRL": "0:15",
                    "HEAD": "2 of 6",
                    "BODY": "1 of 2",
                    "LEG": "0 of 2",
                    "DISTANCE": "3 of 10",
                    "CLINCH": "0 of 0",
                    "GROUND": "0 of 0",
                },
            ]
        )
        self.fighter_tott = pd.DataFrame(
            [
                {"FIGHTER": "Alpha", "HEIGHT": "6' 0\"", "WEIGHT": "155 lbs.", "REACH": "74\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1990", "URL": "http://ufcstats.com/fighter-details/alpha"},
                {"FIGHTER": "Beta", "HEIGHT": "5' 10\"", "WEIGHT": "155 lbs.", "REACH": "71\"", "STANCE": "Southpaw", "DOB": "Jan 01, 1992", "URL": "http://ufcstats.com/fighter-details/beta"},
                {"FIGHTER": "Gamma", "HEIGHT": "5' 11\"", "WEIGHT": "155 lbs.", "REACH": "72\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1991", "URL": "http://ufcstats.com/fighter-details/gamma"},
                {"FIGHTER": "Delta", "HEIGHT": "5' 9\"", "WEIGHT": "155 lbs.", "REACH": "70\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1993", "URL": "http://ufcstats.com/fighter-details/delta"},
                {"FIGHTER": "Prospect", "HEIGHT": "6' 1\"", "WEIGHT": "170 lbs.", "REACH": "76\"", "STANCE": "Orthodox", "DOB": "Jan 01, 2000", "URL": "http://ufcstats.com/fighter-details/prospect"},
                {"FIGHTER": "Casey Clone", "HEIGHT": "6' 0\"", "WEIGHT": "170 lbs.", "REACH": "74\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1994", "URL": "http://ufcstats.com/fighter-details/casey-1"},
                {"FIGHTER": "Casey Clone", "HEIGHT": "6' 0\"", "WEIGHT": "170 lbs.", "REACH": "74\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1995", "URL": "http://ufcstats.com/fighter-details/casey-2"},
            ]
        )

    def test_build_external_history_features_derives_recent_metrics_and_debut_flag(self) -> None:
        features = build_external_ufc_history_features(
            fight_results=self.fight_results,
            fight_stats=self.fight_stats,
            event_details=self.event_details,
            fighter_tott=self.fighter_tott,
        ).set_index("fighter_key")

        alpha = features.loc["alpha"]
        self.assertAlmostEqual(alpha["recent_result_score"], 0.333, places=3)
        self.assertEqual(alpha["losses_in_row"], 0.0)
        self.assertEqual(alpha["ufc_fight_count"], 3.0)
        self.assertEqual(alpha["ufc_debut_flag"], 0.0)
        self.assertAlmostEqual(alpha["first_round_finish_rate"], 0.333, places=3)
        self.assertAlmostEqual(alpha["recent_strike_margin_per_min"], 0.417, places=3)
        self.assertAlmostEqual(alpha["recent_grappling_rate"], 1.875, places=3)
        self.assertAlmostEqual(alpha["control_avg"], 5.0, places=3)
        self.assertAlmostEqual(alpha["recent_control_avg"], 5.0, places=3)
        self.assertAlmostEqual(alpha["submission_avg"], 1.875, places=3)
        self.assertAlmostEqual(alpha["strike_accuracy_pct"], 49.48, places=2)
        self.assertAlmostEqual(alpha["strike_defense_pct"], 57.78, places=2)
        self.assertAlmostEqual(alpha["takedown_accuracy_pct"], 42.86, places=2)
        self.assertAlmostEqual(alpha["takedown_defense_pct"], 100.0, places=2)
        self.assertAlmostEqual(alpha["distance_strike_share"], 0.8542, places=4)
        self.assertAlmostEqual(alpha["clinch_strike_share"], 0.1458, places=4)
        self.assertAlmostEqual(alpha["head_strike_share"], 0.6667, places=4)
        self.assertAlmostEqual(alpha["recency_weighted_strike_margin"], 0.413, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_grappling_rate"], 2.055, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_control_avg"], 5.514, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_strike_pace"], 4.053, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_result_score"], 0.359, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_finish_win_rate"], 0.227, places=3)
        self.assertAlmostEqual(alpha["recency_weighted_finish_loss_rate"], 0.320, places=3)
        self.assertAlmostEqual(alpha["recent_damage_score"], 1.9, places=3)
        self.assertAlmostEqual(alpha["opponent_avg_win_rate"], 0.333, places=3)
        self.assertAlmostEqual(alpha["opponent_avg_recent_result_score"], -0.333, places=3)
        self.assertAlmostEqual(alpha["opponent_avg_finish_win_rate"], 0.333, places=3)
        self.assertAlmostEqual(alpha["opponent_quality_score"], 0.291, places=3)
        self.assertAlmostEqual(alpha["recent_opponent_quality_score"], 0.291, places=3)
        self.assertEqual(alpha["days_since_last_fight"], float(_days_since(pd.Timestamp("2026-04-01"))))

        prospect = features.loc["prospect"]
        self.assertEqual(prospect["ufc_fight_count"], 0.0)
        self.assertEqual(prospect["ufc_debut_flag"], 1.0)
        self.assertEqual(prospect["days_since_last_fight"], 999.0)
        self.assertEqual(prospect["control_avg"], 0.0)
        self.assertEqual(prospect["recent_control_avg"], 0.0)
        self.assertEqual(prospect["opponent_avg_win_rate"], 0.5)
        self.assertEqual(prospect["opponent_quality_score"], 0.5)
        self.assertEqual(prospect["recent_opponent_quality_score"], 0.5)
        self.assertEqual(prospect["height_in"], 73.0)
        self.assertEqual(prospect["reach_in"], 76.0)
        self.assertGreater(prospect["age_years"], 20.0)
        self.assertEqual(prospect["stance"], "Orthodox")

        self.assertNotIn("casey clone", features.index)

    def test_merge_external_history_only_fills_missing_columns(self) -> None:
        features = build_external_ufc_history_features(
            fight_results=self.fight_results,
            fight_stats=self.fight_stats,
            event_details=self.event_details,
            fighter_tott=self.fighter_tott,
        )
        fighter_stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 4.8,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.4,
                    "takedown_defense_pct": 70,
                    "recent_result_score": 0.75,
                    "age_years": 0.0,
                    "stance": "",
                },
                {
                    "fighter_name": "Prospect",
                    "wins": 5,
                    "losses": 0,
                    "height_in": 0,
                    "reach_in": 0,
                    "sig_strikes_landed_per_min": 3.5,
                    "sig_strikes_absorbed_per_min": 2.0,
                    "takedown_avg": 1.0,
                    "takedown_defense_pct": 65,
                    "age_years": 0.0,
                    "stance": "",
                },
            ]
        )

        enriched = merge_external_ufc_history_into_fighter_stats(fighter_stats, features).set_index("fighter_name")

        self.assertEqual(enriched.loc["Alpha", "recent_result_score"], 0.75)
        self.assertEqual(enriched.loc["Alpha", "ufc_fight_count"], 3.0)
        self.assertEqual(enriched.loc["Alpha", "control_avg"], 5.0)
        self.assertEqual(enriched.loc["Alpha", "recent_control_avg"], 5.0)
        self.assertAlmostEqual(enriched.loc["Alpha", "strike_accuracy_pct"], 49.48, places=2)
        self.assertAlmostEqual(enriched.loc["Alpha", "strike_defense_pct"], 57.78, places=2)
        self.assertAlmostEqual(enriched.loc["Alpha", "takedown_accuracy_pct"], 42.86, places=2)
        self.assertAlmostEqual(enriched.loc["Alpha", "opponent_quality_score"], 0.291, places=3)
        self.assertAlmostEqual(enriched.loc["Alpha", "recent_opponent_quality_score"], 0.291, places=3)
        self.assertGreater(enriched.loc["Alpha", "age_years"], 30.0)
        self.assertEqual(enriched.loc["Alpha", "stance"], "Orthodox")

        self.assertEqual(enriched.loc["Prospect", "ufc_fight_count"], 0.0)
        self.assertEqual(enriched.loc["Prospect", "ufc_debut_flag"], 1.0)
        self.assertEqual(enriched.loc["Prospect", "days_since_last_fight"], 999.0)
        self.assertEqual(enriched.loc["Prospect", "control_avg"], 0.0)
        self.assertEqual(enriched.loc["Prospect", "recent_control_avg"], 0.0)
        self.assertEqual(enriched.loc["Prospect", "opponent_quality_score"], 0.5)
        self.assertEqual(enriched.loc["Prospect", "height_in"], 73.0)
        self.assertEqual(enriched.loc["Prospect", "reach_in"], 76.0)
        self.assertEqual(enriched.loc["Prospect", "stance"], "Orthodox")

    def test_merge_external_history_replaces_espn_proxy_takedown_defense(self) -> None:
        features = build_external_ufc_history_features(
            fight_results=self.fight_results,
            fight_stats=self.fight_stats,
            event_details=self.event_details,
            fighter_tott=self.fighter_tott,
        )
        fighter_stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 4.8,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.4,
                    "takedown_defense_pct": 42.86,
                    "data_notes": "ESPN does not expose takedown defense directly; takedown_accuracy_pct is used as a proxy.",
                }
            ]
        )

        enriched = merge_external_ufc_history_into_fighter_stats(fighter_stats, features).set_index("fighter_name")

        self.assertAlmostEqual(enriched.loc["Alpha", "takedown_defense_pct"], 100.0, places=2)

    def test_build_external_history_features_derives_round_trend_metrics_from_multi_round_fights(self) -> None:
        event_details = pd.DataFrame(
            [
                {"EVENT": "UFC Trend 1", "URL": "http://example.com/1", "DATE": "April 1, 2026", "LOCATION": "USA"},
            ]
        )
        fight_results = pd.DataFrame(
            [
                {
                    "EVENT": "UFC Trend 1",
                    "BOUT": "Alpha vs. Beta",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Welterweight Bout",
                    "METHOD": "Decision - Unanimous",
                    "ROUND": 3,
                    "TIME": "5:00",
                },
            ]
        )
        fight_stats = pd.DataFrame(
            [
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 1", "FIGHTER": "Alpha", "KD": 0, "SIG.STR.": "20 of 40", "SIG.STR. %": "50%", "TOTAL STR.": "24 of 46", "TD": "2 of 4", "TD %": "50%", "SUB.ATT": 0, "REV.": 0, "CTRL": "3:00", "HEAD": "12 of 24", "BODY": "4 of 8", "LEG": "4 of 8", "DISTANCE": "16 of 32", "CLINCH": "4 of 8", "GROUND": "0 of 0"},
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 1", "FIGHTER": "Beta", "KD": 0, "SIG.STR.": "10 of 30", "SIG.STR. %": "33%", "TOTAL STR.": "12 of 34", "TD": "0 of 1", "TD %": "0%", "SUB.ATT": 0, "REV.": 0, "CTRL": "0:20", "HEAD": "6 of 18", "BODY": "2 of 6", "LEG": "2 of 6", "DISTANCE": "8 of 24", "CLINCH": "2 of 6", "GROUND": "0 of 0"},
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 2", "FIGHTER": "Alpha", "KD": 0, "SIG.STR.": "15 of 30", "SIG.STR. %": "50%", "TOTAL STR.": "18 of 34", "TD": "0 of 1", "TD %": "0%", "SUB.ATT": 0, "REV.": 0, "CTRL": "0:30", "HEAD": "9 of 18", "BODY": "3 of 6", "LEG": "3 of 6", "DISTANCE": "12 of 24", "CLINCH": "3 of 6", "GROUND": "0 of 0"},
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 2", "FIGHTER": "Beta", "KD": 0, "SIG.STR.": "12 of 28", "SIG.STR. %": "43%", "TOTAL STR.": "14 of 31", "TD": "1 of 2", "TD %": "50%", "SUB.ATT": 0, "REV.": 0, "CTRL": "1:30", "HEAD": "8 of 18", "BODY": "2 of 4", "LEG": "2 of 6", "DISTANCE": "10 of 22", "CLINCH": "2 of 6", "GROUND": "0 of 0"},
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 3", "FIGHTER": "Alpha", "KD": 0, "SIG.STR.": "10 of 15", "SIG.STR. %": "67%", "TOTAL STR.": "12 of 18", "TD": "0 of 0", "TD %": "0%", "SUB.ATT": 0, "REV.": 0, "CTRL": "0:30", "HEAD": "6 of 9", "BODY": "2 of 3", "LEG": "2 of 3", "DISTANCE": "8 of 12", "CLINCH": "2 of 3", "GROUND": "0 of 0"},
                {"EVENT": "UFC Trend 1", "BOUT": "Alpha vs. Beta", "ROUND": "Round 3", "FIGHTER": "Beta", "KD": 0, "SIG.STR.": "18 of 30", "SIG.STR. %": "60%", "TOTAL STR.": "20 of 34", "TD": "1 of 2", "TD %": "50%", "SUB.ATT": 0, "REV.": 0, "CTRL": "2:00", "HEAD": "12 of 20", "BODY": "3 of 5", "LEG": "3 of 5", "DISTANCE": "14 of 24", "CLINCH": "4 of 6", "GROUND": "0 of 0"},
            ]
        )
        fighter_tott = pd.DataFrame(
            [
                {"FIGHTER": "Alpha", "HEIGHT": "6' 0\"", "WEIGHT": "170 lbs.", "REACH": "74\"", "STANCE": "Orthodox", "DOB": "Jan 01, 1990", "URL": "http://ufcstats.com/fighter-details/alpha"},
                {"FIGHTER": "Beta", "HEIGHT": "6' 0\"", "WEIGHT": "170 lbs.", "REACH": "74\"", "STANCE": "Southpaw", "DOB": "Jan 01, 1991", "URL": "http://ufcstats.com/fighter-details/beta"},
            ]
        )

        features = build_external_ufc_history_features(
            fight_results=fight_results,
            fight_stats=fight_stats,
            event_details=event_details,
            fighter_tott=fighter_tott,
        ).set_index("fighter_key")

        alpha = features.loc["alpha"]
        beta = features.loc["beta"]

        self.assertAlmostEqual(alpha["strike_round_trend"], -2.5, places=3)
        self.assertAlmostEqual(alpha["grappling_round_trend"], -6.0, places=3)
        self.assertAlmostEqual(alpha["control_round_trend"], -7.5, places=3)
        self.assertAlmostEqual(alpha["strike_pace_round_trend"], -3.5, places=3)
        self.assertAlmostEqual(beta["strike_round_trend"], 2.5, places=3)
        self.assertAlmostEqual(beta["grappling_round_trend"], 3.0, places=3)
        self.assertAlmostEqual(beta["control_round_trend"], 4.25, places=3)
        self.assertAlmostEqual(beta["strike_pace_round_trend"], -0.2, places=3)


if __name__ == "__main__":
    unittest.main()
