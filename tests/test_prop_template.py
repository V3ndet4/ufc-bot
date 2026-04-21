import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_prop_template import export_prop_template


class PropTemplateTests(unittest.TestCase):
    def test_export_prop_template_curates_to_realistic_props(self) -> None:
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
                    "projected_decision_prob": 0.48,
                    "projected_finish_prob": 0.52,
                }
            ]
        )

        template = export_prop_template(report, min_model_prob=0.10)

        self.assertTrue((template["fighter_a"] == "Kevin Holland").all())
        self.assertIn("Kevin Holland submission", template["prop_label"].tolist())
        self.assertIn("Kevin Holland moneyline", template["prop_label"].tolist())
        self.assertTrue((template["american_odds"] == "").all())
        self.assertNotIn("Randy Brown submission", template["prop_label"].tolist())

    def test_export_prop_template_full_includes_every_supported_row(self) -> None:
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
                    "projected_decision_prob": 0.48,
                    "projected_finish_prob": 0.52,
                }
            ]
        )

        template = export_prop_template(report, min_model_prob=0.04, full=True)

        self.assertIn("Kevin Holland KO/TKO", template["prop_label"].tolist())
        self.assertIn("Randy Brown submission", template["prop_label"].tolist())


if __name__ == "__main__":
    unittest.main()
