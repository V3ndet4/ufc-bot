import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.selective import build_selective_training_frame, predict_selective_clv_prob, train_selective_clv_model


class SelectiveModelTests(unittest.TestCase):
    def test_build_training_frame_uses_implied_probability_clv_target(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "recommended_tier": "A",
                    "recommended_action": "Bettable now",
                    "expression_pick_source": "moneyline",
                    "segment_label": "standard",
                    "book": "fanduel",
                    "chosen_expression_odds": 100,
                    "closing_american_odds": -120,
                    "grade_status": "graded",
                },
                {
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "recommended_tier": "B",
                    "recommended_action": "Watchlist",
                    "expression_pick_source": "moneyline",
                    "segment_label": "standard",
                    "book": "fanduel",
                    "chosen_expression_odds": 100,
                    "closing_american_odds": 130,
                    "grade_status": "graded",
                },
            ]
        )

        training = build_selective_training_frame(frame, min_clv_improvement=0.01)

        self.assertEqual(len(training), 2)
        self.assertGreater(training.loc[0, "clv_implied_delta"], 0.01)
        self.assertLess(training.loc[1, "clv_implied_delta"], 0.0)
        self.assertEqual(training.loc[0, "positive_clv_target"], 1)
        self.assertEqual(training.loc[1, "positive_clv_target"], 0)

    def test_train_and_predict_selective_model(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(60):
            positive = index % 2 == 0
            rows.append(
                {
                    "market": "moneyline",
                    "selection": "fighter_a" if positive else "fighter_b",
                    "recommended_tier": "A" if positive else "C",
                    "recommended_action": "Bettable now" if positive else "Pass",
                    "expression_pick_source": "moneyline",
                    "segment_label": "five_round" if positive else "standard",
                    "book": "fanduel",
                    "american_odds": 110 if positive else 130,
                    "chosen_expression_odds": 110 if positive else 130,
                    "chosen_expression_prob": 0.58 if positive else 0.48,
                    "chosen_expression_implied_prob": 0.476 if positive else 0.435,
                    "chosen_expression_edge": 0.10 if positive else 0.01,
                    "chosen_expression_expected_value": 0.12 if positive else 0.01,
                    "chosen_expression_stake": 22.0 if positive else 5.0,
                    "model_projected_win_prob": 0.58 if positive else 0.48,
                    "implied_prob": 0.476 if positive else 0.435,
                    "edge": 0.10 if positive else 0.01,
                    "expected_value": 0.12 if positive else 0.01,
                    "suggested_stake": 20.0 if positive else 4.0,
                    "model_confidence": 0.82 if positive else 0.58,
                    "data_quality": 0.96 if positive else 0.82,
                    "selection_stats_completeness": 0.96 if positive else 0.82,
                    "selection_fallback_used": 0.0,
                    "line_movement_toward_fighter": 0.03 if positive else -0.01,
                    "market_blend_weight": 0.18 if positive else 0.42,
                    "bet_quality_score": 86.0 if positive else 58.0,
                    "support_count": 5 if positive else 2,
                    "risk_flag_count": 1 if positive else 4,
                    "market_consensus_bookmaker_count": 5.0,
                    "market_overround": 0.03 if positive else 0.07,
                    "price_edge_vs_consensus": 0.03 if positive else -0.02,
                    "is_wmma": 0.0,
                    "is_heavyweight": 0.0,
                    "is_five_round_fight": 1.0 if positive else 0.0,
                    "selection_recent_finish_damage": 0.0 if positive else 2.0,
                    "selection_recent_ko_damage": 0.0 if positive else 1.0,
                    "selection_recent_damage_score": 0.0 if positive else 1.6,
                    "selection_stance_matchup_edge": 0.18 if positive else -0.18,
                    "selection_days_since_last_fight": 120 if positive else 420,
                    "selection_ufc_fight_count": 8 if positive else 2,
                    "selection_ufc_debut_flag": 0.0,
                    "selection_context_instability": 0.0 if positive else 1.0,
                    "selection_first_round_finish_rate": 0.25 if positive else 0.10,
                    "selection_finish_loss_rate": 0.10 if positive else 0.35,
                    "closing_american_odds": -115 if positive else 150,
                    "grade_status": "graded",
                }
            )

        frame = pd.DataFrame(rows)
        bundle, training = train_selective_clv_model(frame, min_samples=20)
        probabilities = predict_selective_clv_prob(training.head(6), bundle)

        self.assertEqual(len(training), 60)
        self.assertGreater(bundle["in_sample_auc"], 0.7)
        self.assertEqual(len(probabilities), 6)
        self.assertTrue(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
