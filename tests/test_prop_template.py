import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_prop_template import build_prop_leans, export_prop_template, format_prop_leans_summary


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
                    "fighter_a_knockdown_avg": 0.45,
                    "fighter_b_knockdown_avg": 0.04,
                    "fighter_a_ko_win_rate": 0.38,
                    "fighter_b_ko_win_rate": 0.12,
                    "fighter_a_ko_loss_rate": 0.02,
                    "fighter_b_ko_loss_rate": 0.16,
                    "fighter_a_sig_strikes_landed_per_min": 5.4,
                    "fighter_b_sig_strikes_landed_per_min": 3.0,
                    "fighter_a_sig_strikes_absorbed_per_min": 3.2,
                    "fighter_b_sig_strikes_absorbed_per_min": 5.0,
                    "fighter_a_distance_strike_share": 0.82,
                    "fighter_b_distance_strike_share": 0.60,
                    "fighter_a_takedown_avg": 2.6,
                    "fighter_b_takedown_avg": 0.3,
                    "fighter_a_takedown_defense_pct": 74.0,
                    "fighter_b_takedown_defense_pct": 58.0,
                    "fighter_a_recent_grappling_rate": 2.4,
                    "fighter_b_recent_grappling_rate": 0.4,
                    "fighter_a_control_avg": 2.2,
                    "fighter_b_control_avg": 0.2,
                    "fighter_a_recent_control_avg": 2.6,
                    "fighter_b_recent_control_avg": 0.2,
                    "matchup_grappling_edge": 1.1,
                    "scheduled_rounds": 3,
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
        self.assertIn("Kevin Holland knockdown", template["prop_label"].tolist())
        self.assertIn("Kevin Holland takedown", template["prop_label"].tolist())
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
                    "fighter_a_knockdown_avg": 0.45,
                    "fighter_b_knockdown_avg": 0.04,
                    "fighter_a_ko_win_rate": 0.38,
                    "fighter_b_ko_win_rate": 0.12,
                    "fighter_a_ko_loss_rate": 0.02,
                    "fighter_b_ko_loss_rate": 0.16,
                    "fighter_a_sig_strikes_landed_per_min": 5.4,
                    "fighter_b_sig_strikes_landed_per_min": 3.0,
                    "fighter_a_sig_strikes_absorbed_per_min": 3.2,
                    "fighter_b_sig_strikes_absorbed_per_min": 5.0,
                    "fighter_a_distance_strike_share": 0.82,
                    "fighter_b_distance_strike_share": 0.60,
                    "fighter_a_takedown_avg": 2.6,
                    "fighter_b_takedown_avg": 0.3,
                    "fighter_a_takedown_defense_pct": 74.0,
                    "fighter_b_takedown_defense_pct": 58.0,
                    "fighter_a_recent_grappling_rate": 2.4,
                    "fighter_b_recent_grappling_rate": 0.4,
                    "fighter_a_control_avg": 2.2,
                    "fighter_b_control_avg": 0.2,
                    "fighter_a_recent_control_avg": 2.6,
                    "fighter_b_recent_control_avg": 0.2,
                    "matchup_grappling_edge": 1.1,
                    "scheduled_rounds": 3,
                    "fight_goes_to_decision_model_prob": 0.48,
                    "fight_doesnt_go_to_decision_model_prob": 0.52,
                    "projected_decision_prob": 0.48,
                    "projected_finish_prob": 0.52,
                }
            ]
        )

        template = export_prop_template(report, min_model_prob=0.04, full=True)

        self.assertIn("Kevin Holland KO/TKO", template["prop_label"].tolist())
        self.assertIn("Kevin Holland knockdown", template["prop_label"].tolist())
        self.assertIn("Kevin Holland takedown", template["prop_label"].tolist())
        self.assertIn("Randy Brown submission", template["prop_label"].tolist())

    def test_knockdown_and_takedown_props_are_main_card_only(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "fighter_a": "Main Bomber",
                    "fighter_b": "Main Wrestler",
                    "is_main_card": 1,
                    "fighter_a_model_win_prob": 0.55,
                    "fighter_b_model_win_prob": 0.45,
                    "fighter_a_knockdown_prop_prob": 0.42,
                    "fighter_b_knockdown_prop_prob": 0.05,
                    "fighter_a_takedown_prop_prob": 0.72,
                    "fighter_b_takedown_prop_prob": 0.12,
                    "fight_goes_to_decision_model_prob": 0.44,
                    "fight_doesnt_go_to_decision_model_prob": 0.56,
                    "projected_decision_prob": 0.44,
                    "projected_finish_prob": 0.56,
                },
                {
                    "fighter_a": "Prelim Bomber",
                    "fighter_b": "Prelim Wrestler",
                    "is_main_card": 0,
                    "fighter_a_model_win_prob": 0.57,
                    "fighter_b_model_win_prob": 0.43,
                    "fighter_a_knockdown_prop_prob": 0.66,
                    "fighter_b_knockdown_prop_prob": 0.04,
                    "fighter_a_takedown_prop_prob": 0.81,
                    "fighter_b_takedown_prop_prob": 0.10,
                    "fight_goes_to_decision_model_prob": 0.46,
                    "fight_doesnt_go_to_decision_model_prob": 0.54,
                    "projected_decision_prob": 0.46,
                    "projected_finish_prob": 0.54,
                },
            ]
        )

        template = export_prop_template(report, min_model_prob=0.04, full=True)
        summary = format_prop_leans_summary(report, limit=5)
        labels = template["prop_label"].tolist()

        self.assertIn("Main Bomber knockdown", labels)
        self.assertIn("Main Bomber takedown", labels)
        self.assertIn("Prelim Bomber moneyline", labels)
        self.assertNotIn("Prelim Bomber knockdown", labels)
        self.assertNotIn("Prelim Bomber takedown", labels)
        self.assertIn("Main Bomber knockdown", summary)
        self.assertIn("Main Bomber takedown", summary)
        self.assertNotIn("Prelim Bomber knockdown", summary)
        self.assertNotIn("Prelim Bomber takedown", summary)

    def test_format_prop_leans_summary_prints_model_only_high_probability_props(self) -> None:
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
                    "fighter_a_knockdown_avg": 0.45,
                    "fighter_b_knockdown_avg": 0.04,
                    "fighter_a_ko_win_rate": 0.38,
                    "fighter_b_ko_win_rate": 0.12,
                    "fighter_a_ko_loss_rate": 0.02,
                    "fighter_b_ko_loss_rate": 0.16,
                    "fighter_a_sig_strikes_landed_per_min": 5.4,
                    "fighter_b_sig_strikes_landed_per_min": 3.0,
                    "fighter_a_sig_strikes_absorbed_per_min": 3.2,
                    "fighter_b_sig_strikes_absorbed_per_min": 5.0,
                    "fighter_a_distance_strike_share": 0.82,
                    "fighter_b_distance_strike_share": 0.60,
                    "fighter_a_takedown_avg": 2.6,
                    "fighter_b_takedown_avg": 0.3,
                    "fighter_a_takedown_defense_pct": 74.0,
                    "fighter_b_takedown_defense_pct": 58.0,
                    "fighter_a_recent_grappling_rate": 2.4,
                    "fighter_b_recent_grappling_rate": 0.4,
                    "fighter_a_control_avg": 2.2,
                    "fighter_b_control_avg": 0.2,
                    "fighter_a_recent_control_avg": 2.6,
                    "fighter_b_recent_control_avg": 0.2,
                    "matchup_grappling_edge": 1.1,
                    "scheduled_rounds": 3,
                    "fight_goes_to_decision_model_prob": 0.48,
                    "fight_doesnt_go_to_decision_model_prob": 0.62,
                    "projected_decision_prob": 0.38,
                    "projected_finish_prob": 0.62,
                },
                {
                    "fighter_a": "Mike Grappler",
                    "fighter_b": "Sam Striker",
                    "fighter_a_model_win_prob": 0.58,
                    "fighter_b_model_win_prob": 0.42,
                    "fighter_a_inside_distance_prob": 0.33,
                    "fighter_b_inside_distance_prob": 0.10,
                    "fighter_a_submission_prob": 0.31,
                    "fighter_b_submission_prob": 0.02,
                    "fighter_a_ko_tko_prob": 0.02,
                    "fighter_b_ko_tko_prob": 0.08,
                    "fighter_a_by_decision_prob": 0.25,
                    "fighter_b_by_decision_prob": 0.32,
                    "fighter_a_knockdown_avg": 0.08,
                    "fighter_b_knockdown_avg": 0.05,
                    "fighter_a_ko_win_rate": 0.06,
                    "fighter_b_ko_win_rate": 0.10,
                    "fighter_a_ko_loss_rate": 0.0,
                    "fighter_b_ko_loss_rate": 0.02,
                    "fighter_a_sig_strikes_landed_per_min": 2.8,
                    "fighter_b_sig_strikes_landed_per_min": 4.0,
                    "fighter_a_sig_strikes_absorbed_per_min": 2.4,
                    "fighter_b_sig_strikes_absorbed_per_min": 3.1,
                    "fighter_a_distance_strike_share": 0.50,
                    "fighter_b_distance_strike_share": 0.64,
                    "fighter_a_takedown_avg": 0.2,
                    "fighter_b_takedown_avg": 0.4,
                    "fighter_a_takedown_defense_pct": 70.0,
                    "fighter_b_takedown_defense_pct": 68.0,
                    "fighter_a_recent_grappling_rate": 0.4,
                    "fighter_b_recent_grappling_rate": 0.6,
                    "fighter_a_control_avg": 0.2,
                    "fighter_b_control_avg": 0.3,
                    "fighter_a_recent_control_avg": 0.2,
                    "fighter_b_recent_control_avg": 0.3,
                    "matchup_grappling_edge": 0.2,
                    "scheduled_rounds": 3,
                    "fight_goes_to_decision_model_prob": 0.47,
                    "fight_doesnt_go_to_decision_model_prob": 0.53,
                    "projected_decision_prob": 0.47,
                    "projected_finish_prob": 0.53,
                },
            ]
        )

        prop_leans = build_prop_leans(report, limit=3)
        summary = format_prop_leans_summary(report, limit=3)

        self.assertIn("Fight doesn't go the distance", prop_leans["prop_label"].tolist())
        self.assertIn("Kevin Holland knockdown", prop_leans["prop_label"].tolist())
        self.assertIn("Kevin Holland takedown", prop_leans["prop_label"].tolist())
        self.assertIn("Prop leans: top", summary)
        self.assertIn("fair", summary)
        self.assertIn("finish lean", summary)
        self.assertIn("knockdown lean", summary)
        self.assertIn("takedown lean", summary)


if __name__ == "__main__":
    unittest.main()
