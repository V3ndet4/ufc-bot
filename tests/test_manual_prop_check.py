import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_manual_props import evaluate_manual_props


class ManualPropCheckTests(unittest.TestCase):
    def test_evaluate_manual_props_supports_submission_and_ko_props(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "fighter_a": "Kevin Holland",
                    "fighter_b": "Randy Brown",
                    "fighter_a_model_win_prob": 0.52,
                    "fighter_b_model_win_prob": 0.48,
                    "fighter_a_inside_distance_prob": 0.34,
                    "fighter_b_inside_distance_prob": 0.18,
                    "fighter_a_submission_prob": 0.19,
                    "fighter_b_submission_prob": 0.04,
                    "fighter_a_ko_tko_prob": 0.15,
                    "fighter_b_ko_tko_prob": 0.14,
                    "fighter_a_by_decision_prob": 0.18,
                    "fighter_b_by_decision_prob": 0.30,
                    "fight_goes_to_decision_model_prob": 0.48,
                    "fight_doesnt_go_to_decision_model_prob": 0.52,
                }
            ]
        )
        props = pd.DataFrame(
            [
                {
                    "fighter_a": "Kevin Holland",
                    "fighter_b": "Randy Brown",
                    "selection_name": "Kevin Holland",
                    "prop_type": "submission",
                    "american_odds": 500,
                },
                {
                    "fighter_a": "Kevin Holland",
                    "fighter_b": "Randy Brown",
                    "selection_name": "Kevin Holland",
                    "prop_type": "ko_tko",
                    "american_odds": 650,
                },
            ]
        )

        evaluated = evaluate_manual_props(report, props)

        self.assertEqual(len(evaluated), 2)
        self.assertEqual(evaluated.loc[0, "prop_type"], "submission")
        self.assertAlmostEqual(float(evaluated.loc[0, "model_prob"]), 0.19, places=4)
        self.assertEqual(int(evaluated.loc[0, "fair_american_odds"]), 426)
        self.assertIn(evaluated.loc[0, "verdict"], {"value", "thin", "no_value"})


if __name__ == "__main__":
    unittest.main()
