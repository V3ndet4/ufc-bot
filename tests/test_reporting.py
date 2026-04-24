import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fight_week_report import build_fight_week_report, build_skipped_fights_report, print_report_summary


class ReportingTests(unittest.TestCase):
    def test_build_fight_week_report_outputs_one_row_per_fight(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "open_american_odds": -120,
                    "current_best_range_low": -115,
                    "current_best_range_high": -105,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": 100,
                    "open_american_odds": 105,
                    "current_best_range_low": 100,
                    "current_best_range_high": 110,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "age_years": 31,
                    "height_in": 70,
                    "reach_in": 72,
                    "stance": "Orthodox",
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "ko_win_rate": 0.4,
                    "submission_win_rate": 0.2,
                    "days_since_last_fight": 110,
                    "ufc_fight_count": 8,
                    "ufc_debut_flag": 0,
                    "recent_grappling_rate": 1.8,
                    "control_avg": 4.6,
                    "recent_control_avg": 3.9,
                    "short_notice_acceptance_flag": 1,
                    "short_notice_success_flag": 1,
                    "cardio_fade_flag": 1,
                    "context_notes": "Short camp rumor",
                    "gym_name": "Kill Cliff FC",
                    "gym_tier": "A",
                    "gym_record": "40-10-0",
                    "gym_score": 0.78,
                    "gym_changed_flag": 1,
                    "previous_gym_name": "Sanford MMA",
                },
                {
                    "fighter_name": "Beta",
                    "wins": 9,
                    "losses": 4,
                    "age_years": 28,
                    "height_in": 69,
                    "reach_in": 70,
                    "stance": "Southpaw",
                    "sig_strikes_landed_per_min": 4.0,
                    "sig_strikes_absorbed_per_min": 4.0,
                    "takedown_avg": 0.5,
                    "takedown_defense_pct": 65,
                    "ko_win_rate": 0.2,
                    "submission_win_rate": 0.1,
                    "days_since_last_fight": 260,
                    "ufc_fight_count": 0,
                    "ufc_debut_flag": 1,
                    "recent_grappling_rate": 0.7,
                    "control_avg": 1.2,
                    "recent_control_avg": 0.4,
                    "short_notice_acceptance_flag": 0,
                    "short_notice_success_flag": 0,
                    "cardio_fade_flag": 0,
                    "context_notes": "",
                    "gym_name": "Factory X",
                    "gym_tier": "B",
                    "gym_record": "22-12-0",
                    "gym_score": 0.63,
                },
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_reporting_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_reporting_stats.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)
        try:
            report = build_fight_week_report(odds_path, stats_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)

        self.assertEqual(len(report), 1)
        self.assertIn("fighter_a_model_win_prob", report.columns)
        self.assertIn("fighter_b_model_win_prob", report.columns)
        self.assertIn("projected_finish_prob", report.columns)
        self.assertIn("projected_decision_prob", report.columns)
        self.assertIn("preferred_market_projected_prob", report.columns)
        self.assertEqual(report.loc[0, "fighter_a_context_notes"], "Short camp rumor")
        self.assertIn("fighter_a_height_advantage_in", report.columns)
        self.assertIn("fighter_a_reach_advantage_in", report.columns)
        self.assertIn("fighter_a_age_years", report.columns)
        self.assertIn("fighter_b_age_years", report.columns)
        self.assertIn("fighter_a_stance", report.columns)
        self.assertIn("fighter_b_stance", report.columns)
        self.assertIn("fighter_a_sig_strikes_landed_per_min", report.columns)
        self.assertIn("fighter_b_sig_strikes_landed_per_min", report.columns)
        self.assertIn("fighter_a_sig_strikes_absorbed_per_min", report.columns)
        self.assertIn("fighter_b_sig_strikes_absorbed_per_min", report.columns)
        self.assertIn("fighter_a_days_since_last_fight", report.columns)
        self.assertIn("fighter_b_days_since_last_fight", report.columns)
        self.assertIn("fighter_a_ufc_fight_count", report.columns)
        self.assertIn("fighter_b_ufc_fight_count", report.columns)
        self.assertIn("fighter_a_ufc_debut_flag", report.columns)
        self.assertIn("fighter_b_ufc_debut_flag", report.columns)
        self.assertIn("fighter_a_record_wins", report.columns)
        self.assertIn("fighter_b_record_wins", report.columns)
        self.assertIn("fighter_a_record_losses", report.columns)
        self.assertIn("fighter_b_record_losses", report.columns)
        self.assertIn("fighter_a_record_draws", report.columns)
        self.assertIn("fighter_b_record_draws", report.columns)
        self.assertIn("age_diff", report.columns)
        self.assertIn("experience_diff", report.columns)
        self.assertIn("fighter_a_control_avg", report.columns)
        self.assertIn("fighter_a_recent_control_avg", report.columns)
        self.assertIn("fighter_a_recent_grappling_rate", report.columns)
        self.assertIn("control_diff", report.columns)
        self.assertIn("recent_control_diff", report.columns)
        self.assertIn("grappling_pressure_diff", report.columns)
        self.assertIn("matchup_striking_edge", report.columns)
        self.assertIn("matchup_grappling_edge", report.columns)
        self.assertIn("matchup_control_edge", report.columns)
        self.assertIn("layoff_diff", report.columns)
        self.assertIn("ufc_experience_diff", report.columns)
        self.assertIn("ufc_debut_penalty_diff", report.columns)
        self.assertIn("opponent_quality_diff", report.columns)
        self.assertIn("schedule_strength_diff", report.columns)
        self.assertIn("normalized_strike_margin_diff", report.columns)
        self.assertIn("base_projected_fighter_a_win_prob", report.columns)
        self.assertIn("trained_side_fighter_a_win_prob", report.columns)
        self.assertIn("side_model_blend_weight", report.columns)
        self.assertEqual(int(report.loc[0, "fighter_a_short_notice_acceptance_flag"]), 1)
        self.assertEqual(int(report.loc[0, "fighter_a_short_notice_success_flag"]), 1)
        self.assertEqual(int(report.loc[0, "fighter_a_cardio_fade_flag"]), 1)
        self.assertAlmostEqual(float(report.loc[0, "fighter_a_age_years"]), 31.0)
        self.assertAlmostEqual(float(report.loc[0, "fighter_b_age_years"]), 28.0)
        self.assertEqual(int(report.loc[0, "fighter_a_days_since_last_fight"]), 110)
        self.assertEqual(int(report.loc[0, "fighter_b_days_since_last_fight"]), 260)
        self.assertEqual(int(report.loc[0, "fighter_a_ufc_fight_count"]), 8)
        self.assertEqual(int(report.loc[0, "fighter_b_ufc_fight_count"]), 0)
        self.assertEqual(int(report.loc[0, "fighter_a_ufc_debut_flag"]), 0)
        self.assertEqual(int(report.loc[0, "fighter_b_ufc_debut_flag"]), 1)
        self.assertEqual(report.loc[0, "fighter_a_stance"], "Orthodox")
        self.assertEqual(report.loc[0, "fighter_b_stance"], "Southpaw")
        self.assertIn("preferred_market_expression", report.columns)
        self.assertIn("market_substitution_reason", report.columns)
        self.assertIn("market_style_tags", report.columns)
        self.assertIn("value_expression_winner", report.columns)
        self.assertIn("value_expression_reason", report.columns)
        self.assertIn("fighter_a_gym_name", report.columns)
        self.assertIn("fighter_a_gym_tier", report.columns)
        self.assertIn("fighter_a_previous_gym_name", report.columns)
        self.assertIn("fighter_a_model_context_flags", report.columns)
        self.assertIn("fighter_a_operator_context_flags", report.columns)
        self.assertIn("fighter_a_current_decimal_odds", report.columns)
        self.assertIn("fighter_b_current_decimal_odds", report.columns)
        self.assertIn("speculative_prop_expression", report.columns)
        self.assertIn("speculative_prop_fair_american_odds", report.columns)
        self.assertIn("speculative_prop_reason", report.columns)
        self.assertIn("fighter_a_submission_prob", report.columns)
        self.assertIn("fighter_a_ko_tko_prob", report.columns)
        self.assertIn("chosen_value_expression", report.columns)
        self.assertIn("runner_up_expression", report.columns)
        self.assertIn("historical_overlay_grade", report.columns)
        self.assertIn("fragility_bucket", report.columns)
        self.assertIn("selection_gym_name", report.columns)
        self.assertIn("selection_gym_tier", report.columns)
        self.assertIn("selection_gym_record", report.columns)
        self.assertIn("selection_previous_gym_name", report.columns)
        self.assertIn("selection_control_avg", report.columns)
        self.assertIn("selection_recent_control_avg", report.columns)
        self.assertIn("selection_recent_grappling_rate", report.columns)
        self.assertIn("selection_grappling_pressure_score", report.columns)
        selected_name = str(report.loc[0, "selection_name"])
        expected_gym = "Kill Cliff FC" if selected_name == "Alpha" else "Factory X"
        expected_tier = "A" if selected_name == "Alpha" else "B"
        expected_record = "40-10-0" if selected_name == "Alpha" else "22-12-0"
        expected_previous_gym = "Sanford MMA" if selected_name == "Alpha" else ""
        expected_control = 4.6 if selected_name == "Alpha" else 1.2
        expected_recent_control = 3.9 if selected_name == "Alpha" else 0.4
        expected_recent_grappling = 1.8 if selected_name == "Alpha" else 0.7
        self.assertEqual(report.loc[0, "selection_gym_name"], expected_gym)
        self.assertEqual(report.loc[0, "selection_gym_tier"], expected_tier)
        self.assertEqual(report.loc[0, "selection_gym_record"], expected_record)
        self.assertEqual(report.loc[0, "selection_previous_gym_name"], expected_previous_gym)
        self.assertAlmostEqual(float(report.loc[0, "selection_control_avg"]), expected_control, places=3)
        self.assertAlmostEqual(float(report.loc[0, "selection_recent_control_avg"]), expected_recent_control, places=3)
        self.assertAlmostEqual(float(report.loc[0, "selection_recent_grappling_rate"]), expected_recent_grappling, places=3)

    def test_print_report_summary_uses_action_and_driver_lines(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "fighter_a_model_win_prob": 0.68,
                    "fighter_b_model_win_prob": 0.32,
                    "fighter_a_edge_vs_current": 0.09,
                    "fighter_b_edge_vs_current": -0.08,
                    "preferred_market_expression": "Fight doesn't go to decision",
                    "preferred_market_american_odds": pd.NA,
                    "fighter_a_current_american_odds": -135,
                    "fighter_b_current_american_odds": 118,
                    "fighter_a_reach_advantage_in": 3.0,
                    "fighter_a_height_advantage_in": 2.0,
                    "strike_margin_diff": 1.8,
                    "grappling_diff": 0.7,
                    "control_diff": 2.1,
                    "recent_control_diff": 1.6,
                    "grappling_pressure_diff": 1.1,
                    "first_round_finish_rate_diff": 0.18,
                    "durability_diff": 0.14,
                    "decision_rate_diff": -0.02,
                    "projected_finish_prob": 0.64,
                    "projected_decision_prob": 0.36,
                    "market_blend_weight": 0.18,
                    "model_confidence": 1.0,
                    "data_quality": 1.0,
                    "fighter_a_fallback_used": 0.0,
                    "fighter_b_fallback_used": 0.0,
                    "fighter_a_short_notice_flag": 0.0,
                    "fighter_b_short_notice_flag": 0.0,
                    "fighter_a_cardio_fade_flag": 0.0,
                    "fighter_b_cardio_fade_flag": 0.0,
                    "fighter_a_gym_name": "Kill Cliff FC",
                    "fighter_b_gym_name": "Factory X",
                    "fighter_a_gym_tier": "S",
                    "fighter_b_gym_tier": "B",
                    "fighter_a_gym_record": "70-15-0",
                    "fighter_b_gym_record": "24-18-0",
                    "fighter_a_gym_changed_flag": 1.0,
                    "fighter_b_gym_changed_flag": 0.0,
                    "fighter_a_previous_gym_name": "Sanford MMA",
                    "fighter_b_previous_gym_name": "",
                    "value_expression_winner": "side_only",
                }
            ]
        )

        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            print_report_summary(report)

        output = mock_stdout.getvalue()
        self.assertIn("Action:", output)
        self.assertIn("Decision:", output)
        self.assertIn("Numbers:", output)
        self.assertIn("History:", output)
        self.assertIn("Fragility:", output)
        self.assertIn("Drivers:", output)
        self.assertIn("Camp:", output)
        self.assertIn("moneyline now", output)
        self.assertIn("Fight doesn't go to decision", output)
        self.assertIn("C-tier control edge", output)
        self.assertIn("Kill Cliff FC", output)
        self.assertIn("switched from Sanford MMA", output)

    def test_print_report_summary_colorizes_gym_tier_for_tty(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "fighter_a_model_win_prob": 0.62,
                    "fighter_b_model_win_prob": 0.38,
                    "fighter_a_edge_vs_current": 0.06,
                    "fighter_b_edge_vs_current": -0.05,
                    "fighter_a_current_american_odds": -130,
                    "fighter_b_current_american_odds": 110,
                    "preferred_market_expression": "Alpha moneyline",
                    "preferred_market_american_odds": -130,
                    "preferred_market_projected_prob": 0.62,
                    "preferred_market_implied_prob": 0.5652,
                    "preferred_market_edge": 0.0548,
                    "chosen_value_expression": "Alpha moneyline",
                    "expression_pick_source": "side_market",
                    "chosen_expression_odds": -130,
                    "chosen_expression_prob": 0.62,
                    "chosen_expression_implied_prob": 0.5652,
                    "chosen_expression_edge": 0.0548,
                    "market_blend_weight": 0.18,
                    "model_confidence": 1.0,
                    "data_quality": 1.0,
                    "grappling_diff": 1.55,
                    "recent_control_diff": 3.9,
                    "grappling_pressure_diff": 3.2,
                    "projected_finish_prob": 0.51,
                    "projected_decision_prob": 0.49,
                    "fighter_a_gym_name": "Kill Cliff FC",
                    "fighter_a_gym_tier": "S",
                    "fighter_a_gym_record": "70-15-0",
                    "fighter_a_previous_gym_name": "",
                    "fighter_a_gym_changed_flag": 0.0,
                    "fighter_b_gym_name": "Factory X",
                    "fighter_b_gym_tier": "B",
                    "fighter_b_gym_record": "24-18-0",
                    "fighter_b_previous_gym_name": "",
                    "fighter_b_gym_changed_flag": 0.0,
                    "value_expression_winner": "side_market",
                }
            ]
        )

        class TtyStringIO(io.StringIO):
            def isatty(self) -> bool:
                return True

        with patch("sys.stdout", new_callable=TtyStringIO) as mock_stdout:
            print_report_summary(report)

        output = mock_stdout.getvalue()
        self.assertIn("\x1b[36mS-tier\x1b[0m", output)
        self.assertIn("\x1b[32mA-tier\x1b[0m control edge", output)

    def test_build_skipped_fights_report_marks_partial_prices(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": pd.NA,
                },
            ]
        )

        skipped = build_skipped_fights_report(odds)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.loc[0, "skip_reason"], "missing_fighter_b_price")

    def test_build_skipped_fights_report_marks_fully_unpriced_fights(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "manual",
                    "american_odds": pd.NA,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "manual",
                    "american_odds": pd.NA,
                },
            ]
        )

        skipped = build_skipped_fights_report(odds)
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped.loc[0, "skip_reason"], "no_priced_odds")


if __name__ == "__main__":
    unittest.main()
