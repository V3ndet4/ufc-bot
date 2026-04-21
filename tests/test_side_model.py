import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.side import apply_side_model_adjustments, train_side_model
from scripts.build_lean_board import build_lean_board, format_best_leans_summary, format_full_card_breakdown


class SideModelTests(unittest.TestCase):
    def test_train_side_model_and_apply_adjustments(self) -> None:
        training = pd.DataFrame(
            [
                {
                    "market": "moneyline",
                    "selection": "fighter_a" if index % 2 == 0 else "fighter_b",
                    "book": "Book",
                    "segment_label": "five_round" if index % 3 == 0 else "standard",
                    "american_odds": 110 if index % 2 == 0 else -125,
                    "model_projected_win_prob": 0.64 if index < 6 else 0.42,
                    "implied_prob": 0.48 if index < 6 else 0.56,
                    "edge": 0.16 if index < 6 else -0.06,
                    "model_confidence": 0.82 if index < 6 else 0.58,
                    "data_quality": 0.95 if index < 6 else 0.76,
                    "line_movement_toward_fighter": 0.01 if index < 6 else -0.02,
                    "market_blend_weight": 0.14 if index < 6 else 0.28,
                    "selection_days_since_last_fight": 120 if index < 6 else 330,
                    "selection_ufc_fight_count": 8 if index < 6 else 2,
                    "selection_ufc_debut_flag": 0 if index < 6 else 1,
                    "selection_recent_finish_damage": 0.0 if index < 6 else 1.0,
                    "selection_recent_ko_damage": 0.0 if index < 6 else 1.0,
                    "selection_recent_damage_score": 0.1 if index < 6 else 0.8,
                    "selection_first_round_finish_rate": 0.22 if index < 6 else 0.08,
                    "selection_finish_loss_rate": 0.10 if index < 6 else 0.34,
                    "selection_recent_grappling_rate": 1.4 if index < 6 else 0.3,
                    "selection_control_avg": 3.8 if index < 6 else 0.9,
                    "selection_recent_control_avg": 3.1 if index < 6 else 0.4,
                    "selection_context_instability": 0.0 if index < 6 else 2.0,
                    "selection_gym_score": 0.78 if index < 6 else 0.42,
                    "selection_matchup_striking_edge": 1.1 if index < 6 else -0.6,
                    "selection_matchup_grappling_edge": 1.2 if index < 6 else -0.4,
                    "selection_matchup_control_edge": 0.9 if index < 6 else -0.3,
                    "tracked_market_key": "moneyline",
                    "actual_result": "win" if index < 6 else "loss",
                    "grade_status": "graded",
                }
                for index in range(12)
            ]
        )
        bundle, labeled = train_side_model(training, min_samples=6)

        self.assertGreaterEqual(bundle["training_rows"], 12)
        self.assertIn("side_win_target", labeled.columns)

        live = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "selection": "fighter_a",
                    "american_odds": 120,
                    "projected_fighter_a_win_prob": 0.55,
                    "raw_projected_fighter_a_win_prob": 0.57,
                    "model_projected_win_prob": 0.55,
                    "model_confidence": 0.83,
                    "data_quality": 0.94,
                    "line_movement_toward_fighter": 0.01,
                    "market_blend_weight": 0.16,
                    "a_days_since_last_fight": 110,
                    "b_days_since_last_fight": 280,
                    "a_ufc_fight_count": 7,
                    "b_ufc_fight_count": 2,
                    "a_ufc_debut_flag": 0,
                    "b_ufc_debut_flag": 1,
                    "a_recent_finish_loss_365d": 0.0,
                    "b_recent_finish_loss_365d": 1.0,
                    "a_recent_ko_loss_365d": 0.0,
                    "b_recent_ko_loss_365d": 1.0,
                    "a_recent_damage_score": 0.1,
                    "b_recent_damage_score": 0.7,
                    "a_first_round_finish_rate": 0.20,
                    "b_first_round_finish_rate": 0.08,
                    "a_finish_loss_rate": 0.10,
                    "b_finish_loss_rate": 0.28,
                    "a_recent_grappling_rate": 1.3,
                    "b_recent_grappling_rate": 0.4,
                    "a_control_avg": 4.1,
                    "b_control_avg": 1.0,
                    "a_recent_control_avg": 3.4,
                    "b_recent_control_avg": 0.5,
                    "a_gym_score": 0.8,
                    "b_gym_score": 0.45,
                    "context_stability_diff": 1.0,
                    "matchup_striking_edge": 1.3,
                    "matchup_grappling_edge": 1.1,
                    "matchup_control_edge": 0.9,
                    "segment_label": "five_round",
                    "book": "Book",
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "selection": "fighter_b",
                    "american_odds": -140,
                    "projected_fighter_a_win_prob": 0.55,
                    "raw_projected_fighter_a_win_prob": 0.57,
                    "model_projected_win_prob": 0.45,
                    "model_confidence": 0.83,
                    "data_quality": 0.94,
                    "line_movement_toward_fighter": -0.01,
                    "market_blend_weight": 0.16,
                    "a_days_since_last_fight": 110,
                    "b_days_since_last_fight": 280,
                    "a_ufc_fight_count": 7,
                    "b_ufc_fight_count": 2,
                    "a_ufc_debut_flag": 0,
                    "b_ufc_debut_flag": 1,
                    "a_recent_finish_loss_365d": 0.0,
                    "b_recent_finish_loss_365d": 1.0,
                    "a_recent_ko_loss_365d": 0.0,
                    "b_recent_ko_loss_365d": 1.0,
                    "a_recent_damage_score": 0.1,
                    "b_recent_damage_score": 0.7,
                    "a_first_round_finish_rate": 0.20,
                    "b_first_round_finish_rate": 0.08,
                    "a_finish_loss_rate": 0.10,
                    "b_finish_loss_rate": 0.28,
                    "a_recent_grappling_rate": 1.3,
                    "b_recent_grappling_rate": 0.4,
                    "a_control_avg": 4.1,
                    "b_control_avg": 1.0,
                    "a_recent_control_avg": 3.4,
                    "b_recent_control_avg": 0.5,
                    "a_gym_score": 0.8,
                    "b_gym_score": 0.45,
                    "context_stability_diff": 1.0,
                    "matchup_striking_edge": 1.3,
                    "matchup_grappling_edge": 1.1,
                    "matchup_control_edge": 0.9,
                    "segment_label": "five_round",
                    "book": "Book",
                },
            ]
        )
        adjusted = apply_side_model_adjustments(live, bundle)

        self.assertIn("trained_side_fighter_a_win_prob", adjusted.columns)
        self.assertIn("side_model_blend_weight", adjusted.columns)
        self.assertTrue((adjusted["side_model_blend_weight"] > 0).all())
        self.assertNotAlmostEqual(float(adjusted.loc[0, "projected_fighter_a_win_prob"]), 0.55, places=4)

    def test_build_lean_board_outputs_side_guidance(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "event_name": "Test Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "fighter_a_model_win_prob": 0.63,
                    "fighter_b_model_win_prob": 0.37,
                    "fighter_a_current_american_odds": 120,
                    "fighter_b_current_american_odds": -140,
                    "fighter_a_open_american_odds": 135,
                    "fighter_b_open_american_odds": -155,
                    "model_confidence": 0.8,
                    "fragility_bucket": "low",
                    "strike_margin_diff": 1.3,
                    "matchup_striking_edge": 1.4,
                    "grappling_diff": 0.8,
                    "matchup_grappling_edge": 0.9,
                    "control_diff": 1.6,
                    "recent_control_diff": 1.2,
                    "matchup_control_edge": 1.0,
                    "grappling_pressure_diff": 1.1,
                    "decision_rate_diff": 0.1,
                    "projected_finish_prob": 0.58,
                    "projected_decision_prob": 0.42,
                    "market_blend_weight": 0.16,
                    "fighter_a_gym_tier": "A",
                    "fighter_b_gym_tier": "C",
                    "fighter_a_gym_name": "Kill Cliff FC",
                    "fighter_b_gym_name": "Factory X",
                    "fighter_a_record_wins": 12,
                    "fighter_a_record_losses": 3,
                    "fighter_a_record_draws": 0,
                    "fighter_b_record_wins": 10,
                    "fighter_b_record_losses": 4,
                    "fighter_b_record_draws": 1,
                    "fighter_a_age_years": 30.0,
                    "fighter_b_age_years": 34.0,
                    "fighter_a_stance": "Orthodox",
                    "fighter_b_stance": "Southpaw",
                    "fighter_a_sig_strikes_landed_per_min": 5.4,
                    "fighter_a_sig_strikes_absorbed_per_min": 3.1,
                    "fighter_b_sig_strikes_landed_per_min": 3.6,
                    "fighter_b_sig_strikes_absorbed_per_min": 4.2,
                    "fighter_a_days_since_last_fight": 112,
                    "fighter_b_days_since_last_fight": 280,
                    "fighter_a_ufc_fight_count": 10,
                    "fighter_b_ufc_fight_count": 4,
                    "fighter_a_ufc_debut_flag": 0,
                    "fighter_b_ufc_debut_flag": 0,
                    "fighter_a_recent_grappling_rate": 1.1,
                    "fighter_b_recent_grappling_rate": 0.6,
                    "fighter_a_control_avg": 2.4,
                    "fighter_b_control_avg": 1.1,
                    "fighter_a_ko_win_rate": 0.45,
                    "fighter_b_ko_win_rate": 0.20,
                    "fighter_a_submission_win_rate": 0.15,
                    "fighter_b_submission_win_rate": 0.10,
                    "fighter_a_decision_rate": 0.40,
                    "fighter_b_decision_rate": 0.55,
                    "fighter_a_history_style_label": "Volume striker | Orthodox",
                    "fighter_b_history_style_label": "Control grappler | Southpaw",
                    "gym_score_diff": 0.12,
                }
            ]
        )
        board = build_lean_board(report)
        self.assertEqual(len(board), 1)
        self.assertEqual(board.loc[0, "lean_side"], "Alpha")
        self.assertIn(board.loc[0, "lean_action"], {"Bet now", "Wait for a better number", "Lean only"})
        self.assertIn("top_reasons", board.columns)
        self.assertIn("watch_for", board.columns)
        self.assertIn("context_summary", board.columns)
        self.assertIn("camp_summary", board.columns)
        self.assertIn("pick_style", board.columns)
        self.assertIn("opponent_style", board.columns)
        full_card = format_full_card_breakdown(board)
        best_leans = format_best_leans_summary(board)
        self.assertIn("Full card read: 1 fights", full_card)
        self.assertIn("Look for", full_card)
        self.assertIn("Styles", full_card)
        self.assertIn("Volume striker | Orthodox", full_card)
        self.assertIn("record 12-3", full_card)
        self.assertIn("Lean board: 1 best choices", best_leans)
        self.assertIn("Drivers", best_leans)


if __name__ == "__main__":
    unittest.main()
