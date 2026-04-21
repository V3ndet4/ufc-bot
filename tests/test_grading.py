import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.grading import grade_tracked_picks


class GradingTests(unittest.TestCase):
    def test_grading_handles_moneyline_and_finish_markets(self) -> None:
        picks = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "chosen_value_expression": "Alpha",
                    "chosen_expression_odds": -110,
                    "chosen_expression_prob": 0.60,
                    "chosen_expression_stake": 100.0,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "chosen_value_expression": "Fight doesn't go to decision",
                    "chosen_expression_odds": +120,
                    "chosen_expression_prob": 0.58,
                    "chosen_expression_stake": 100.0,
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "official",
                    "went_decision": 0,
                    "ended_inside_distance": 1,
                    "closing_fighter_a_odds": -130,
                    "closing_fight_doesnt_go_to_decision_odds": +105,
                }
            ]
        )

        graded = grade_tracked_picks(picks, results)

        self.assertEqual(list(graded["actual_result"]), ["win", "win"])
        self.assertAlmostEqual(float(graded.loc[0, "profit"]), 90.91, places=2)
        self.assertAlmostEqual(float(graded.loc[1, "profit"]), 120.0, places=2)
        self.assertEqual(int(graded.loc[0, "closing_american_odds"]), -130)
        self.assertEqual(int(graded.loc[1, "closing_american_odds"]), 105)


if __name__ == "__main__":
    unittest.main()
