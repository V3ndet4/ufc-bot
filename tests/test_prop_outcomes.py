import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.prop_outcomes import build_prop_outcome_history_frame
from models.prop_outcomes import (
    predict_prop_probability_from_fight_row,
    train_prop_outcome_model,
)


class PropOutcomeTests(unittest.TestCase):
    def test_history_frame_uses_prior_stats_for_td_kd_labels(self) -> None:
        fight_results = pd.DataFrame(
            [
                {
                    "EVENT": "Event 1",
                    "BOUT": "Alpha vs. Beta",
                    "OUTCOME": "W/L",
                    "WEIGHTCLASS": "Lightweight Bout",
                    "METHOD": "Decision - Unanimous",
                    "ROUND": 1,
                    "TIME": "5:00",
                    "TIME FORMAT": "3 Rnd (5-5-5)",
                },
                {
                    "EVENT": "Event 2",
                    "BOUT": "Alpha vs. Beta",
                    "OUTCOME": "L/W",
                    "WEIGHTCLASS": "Lightweight Bout",
                    "METHOD": "KO/TKO",
                    "ROUND": 1,
                    "TIME": "5:00",
                    "TIME FORMAT": "3 Rnd (5-5-5)",
                },
            ]
        )
        event_details = pd.DataFrame(
            [
                {"EVENT": "Event 1", "DATE": "2026-01-01"},
                {"EVENT": "Event 2", "DATE": "2026-02-01"},
            ]
        )
        fight_stats = pd.DataFrame(
            [
                _stat_row("Event 1", "Alpha", kd=1, td="2 of 3", sig="10 of 20", ctrl="1:00"),
                _stat_row("Event 1", "Beta", kd=0, td="0 of 1", sig="8 of 15", ctrl="0:10"),
                _stat_row("Event 2", "Alpha", kd=0, td="0 of 2", sig="5 of 10", ctrl="0:00"),
                _stat_row("Event 2", "Beta", kd=1, td="1 of 2", sig="9 of 18", ctrl="0:30"),
            ]
        )

        frame = build_prop_outcome_history_frame(
            fight_results=fight_results,
            fight_stats=fight_stats,
            event_details=event_details,
        )

        first_alpha = frame.loc[(frame["event"] == "Event 1") & (frame["fighter_key"] == "alpha")].iloc[0]
        second_alpha = frame.loc[(frame["event"] == "Event 2") & (frame["fighter_key"] == "alpha")].iloc[0]
        second_beta = frame.loc[(frame["event"] == "Event 2") & (frame["fighter_key"] == "beta")].iloc[0]
        self.assertEqual(int(first_alpha["takedown_1plus_target"]), 1)
        self.assertEqual(int(first_alpha["by_decision_target"]), 1)
        self.assertEqual(int(first_alpha["fight_goes_to_decision_target"]), 1)
        self.assertEqual(float(first_alpha["selection_takedown_avg"]), 0.0)
        self.assertEqual(int(second_alpha["knockdown_1plus_target"]), 0)
        self.assertEqual(int(second_alpha["inside_distance_target"]), 0)
        self.assertAlmostEqual(float(second_alpha["selection_takedown_avg"]), 6.0)
        self.assertAlmostEqual(float(second_alpha["selection_knockdown_avg"]), 3.0)
        self.assertEqual(int(second_beta["ko_tko_target"]), 1)
        self.assertEqual(int(second_beta["fight_ends_by_ko_tko_target"]), 1)

    def test_train_and_predict_prop_outcome_model(self) -> None:
        rows = []
        for index in range(20):
            takedown_positive = int(index % 2 == 0)
            knockdown_positive = int(index % 2 == 1)
            rows.append(
                {
                    "scheduled_rounds": 3,
                    "selection_ufc_fight_count": index,
                    "opponent_ufc_fight_count": 10,
                    "selection_takedown_avg": 2.5 if takedown_positive else 0.1,
                    "selection_takedown_accuracy_pct": 55 if takedown_positive else 25,
                    "opponent_takedown_defense_pct": 50 if takedown_positive else 80,
                    "selection_recent_grappling_rate": 1.5 if takedown_positive else 0.0,
                    "selection_control_avg": 3.0 if takedown_positive else 0.0,
                    "selection_recent_control_avg": 3.0 if takedown_positive else 0.0,
                    "selection_matchup_grappling_edge": 2.0 if takedown_positive else -1.0,
                    "selection_knockdown_avg": 0.6 if knockdown_positive else 0.0,
                    "selection_ko_win_rate": 0.4 if knockdown_positive else 0.0,
                    "opponent_ko_loss_rate": 0.3 if knockdown_positive else 0.0,
                    "selection_sig_strikes_landed_per_min": 5.0 if knockdown_positive else 2.0,
                    "opponent_sig_strikes_absorbed_per_min": 5.0 if knockdown_positive else 2.0,
                    "selection_distance_strike_share": 0.7,
                    "selection_clinch_strike_share": 0.1,
                    "selection_ground_strike_share": 0.0,
                    "takedown_1plus_target": takedown_positive,
                    "knockdown_1plus_target": knockdown_positive,
                    "ko_tko_target": knockdown_positive,
                }
            )
        bundle, _ = train_prop_outcome_model(pd.DataFrame(rows), min_samples=10)

        probability = predict_prop_probability_from_fight_row(
            bundle,
            {
                "scheduled_rounds": 3,
                "a_ufc_fight_count": 12,
                "b_ufc_fight_count": 8,
                "a_takedown_avg": 2.2,
                "a_takedown_accuracy_pct": 60,
                "b_takedown_defense_pct": 48,
                "a_recent_grappling_rate": 1.2,
                "a_control_avg": 2.0,
                "a_recent_control_avg": 2.5,
                "matchup_grappling_edge": 1.5,
            },
            market="takedown",
            selection="fighter_a",
        )
        self.assertIsNotNone(probability)
        self.assertGreater(float(probability), 0.0)

        ko_probability = predict_prop_probability_from_fight_row(
            bundle,
            {
                "scheduled_rounds": 3,
                "a_ufc_fight_count": 12,
                "b_ufc_fight_count": 8,
                "a_knockdown_avg": 0.5,
                "a_ko_win_rate": 0.4,
                "b_ko_loss_rate": 0.3,
                "a_sig_strikes_landed_per_min": 5.0,
                "b_sig_strikes_absorbed_per_min": 5.0,
            },
            market="ko_tko",
            selection="fighter_a",
        )
        self.assertIsNotNone(ko_probability)
        self.assertGreater(float(ko_probability), 0.0)


def _stat_row(event: str, fighter: str, *, kd: int, td: str, sig: str, ctrl: str) -> dict[str, object]:
    return {
        "EVENT": event,
        "BOUT": "Alpha vs. Beta",
        "ROUND": "Round 1",
        "FIGHTER": fighter,
        "KD": kd,
        "SIG.STR.": sig,
        "SIG.STR. %": "50%",
        "TOTAL STR.": sig,
        "TD": td,
        "TD %": "50%",
        "SUB.ATT": 0,
        "REV.": 0,
        "CTRL": ctrl,
        "HEAD": sig,
        "BODY": "0 of 0",
        "LEG": "0 of 0",
        "DISTANCE": sig,
        "CLINCH": "0 of 0",
        "GROUND": "0 of 0",
    }


if __name__ == "__main__":
    unittest.main()
