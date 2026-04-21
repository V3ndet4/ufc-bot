import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.confidence import apply_confidence_model, train_confidence_model


class ConfidenceModelTests(unittest.TestCase):
    def test_train_confidence_model_and_apply_predictions(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(16):
            event_id = f"event-{index}"
            fighter_a_prob = 0.68 if index < 8 else 0.58
            actual_winner = "fighter_a" if index % 4 in {0, 1, 2} else "fighter_b"
            for selection in ("fighter_a", "fighter_b"):
                actual_result = "win" if selection == actual_winner else "loss"
                rows.append(
                    {
                        "event_id": event_id,
                        "event_name": f"Event {index}",
                        "start_time": f"2024-0{(index % 8) + 1}-15T20:00:00Z",
                        "fighter_a": f"Alpha {index}",
                        "fighter_b": f"Beta {index}",
                        "selection": selection,
                        "book": "Book",
                        "segment_label": "standard" if index % 3 else "five_round",
                        "projected_fighter_a_win_prob": fighter_a_prob,
                        "baseline_raw_fighter_a_win_prob": fighter_a_prob + 0.02,
                        "heuristic_model_confidence": 0.82 if index < 8 else 0.56,
                        "model_confidence": 0.82 if index < 8 else 0.56,
                        "data_quality": 0.95 if index < 8 else 0.76,
                        "fallback_penalty": 0.0 if index < 8 else 1.0,
                        "fighter_a_current_implied_prob": 0.54 if index < 8 else 0.52,
                        "market_consensus_bookmaker_count": 5 if index < 8 else 2,
                        "market_overround": 0.03 if index < 8 else 0.06,
                        "a_ufc_fight_count": 8 if index < 8 else 2,
                        "b_ufc_fight_count": 6 if index < 8 else 1,
                        "a_ufc_debut_flag": 0,
                        "b_ufc_debut_flag": 0 if index < 8 else 1,
                        "a_short_notice_flag": 0,
                        "b_short_notice_flag": 0 if index < 8 else 1,
                        "a_cardio_fade_flag": 0,
                        "b_cardio_fade_flag": 0 if index < 8 else 1,
                        "a_injury_concern_flag": 0,
                        "b_injury_concern_flag": 0,
                        "a_weight_cut_concern_flag": 0,
                        "b_weight_cut_concern_flag": 0,
                        "a_replacement_fighter_flag": 0,
                        "b_replacement_fighter_flag": 0 if index < 8 else 1,
                        "a_travel_disadvantage_flag": 0,
                        "b_travel_disadvantage_flag": 0,
                        "a_new_gym_flag": 0,
                        "b_new_gym_flag": 0 if index < 8 else 1,
                        "a_camp_change_flag": 0,
                        "b_camp_change_flag": 0,
                        "a_days_since_last_fight": 120 if index < 8 else 300,
                        "b_days_since_last_fight": 140 if index < 8 else 40,
                        "recent_form_diff": 0.8 if index < 8 else -0.1,
                        "recent_strike_form_diff": 1.1 if index < 8 else 0.2,
                        "recent_grappling_form_diff": 0.7 if index < 8 else -0.2,
                        "recent_control_diff": 0.9 if index < 8 else 0.1,
                        "recent_damage_diff": -0.4 if index < 8 else 0.8,
                        "gym_score_diff": 0.18 if index < 8 else 0.02,
                        "strike_margin_last_3_diff": 1.2 if index < 8 else 0.2,
                        "grappling_rate_last_3_diff": 1.0 if index < 8 else 0.1,
                        "control_avg_last_3_diff": 1.4 if index < 8 else 0.0,
                        "result_score_last_3_diff": 1.1 if index < 8 else -0.1,
                        "strike_pace_last_3_diff": 1.5 if index < 8 else 0.2,
                        "distance_strike_share_diff": 0.12 if index < 8 else 0.02,
                        "clinch_strike_share_diff": -0.03 if index < 8 else 0.01,
                        "ground_strike_share_diff": 0.10 if index < 8 else 0.00,
                        "is_wmma": 0,
                        "is_heavyweight": 0,
                        "is_five_round_fight": 1 if index % 3 == 0 else 0,
                        "a_history_style_label": "Volume striker | Orthodox",
                        "b_history_style_label": "Control grappler | Southpaw",
                        "actual_result": actual_result,
                    }
                )

        frame = pd.DataFrame(rows)
        bundle, training = train_confidence_model(frame, min_samples=8)

        self.assertEqual(bundle["training_rows"], 16)
        self.assertIn("confidence_target", training.columns)

        heuristic = pd.Series(0.6, index=frame.index, dtype=float)
        calibrated = apply_confidence_model(frame, bundle, heuristic)

        self.assertEqual(len(calibrated), len(frame))
        self.assertTrue(((calibrated >= 0.2) & (calibrated <= 0.92)).all())
        self.assertNotAlmostEqual(float(calibrated.iloc[0]), 0.6, places=4)


if __name__ == "__main__":
    unittest.main()
