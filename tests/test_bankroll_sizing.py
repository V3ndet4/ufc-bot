import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bankroll.sizing import BankrollGovernorConfig, apply_bankroll_governor


class BankrollSizingTests(unittest.TestCase):
    def test_apply_bankroll_governor_caps_and_exposure(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fight_key": "alpha||beta",
                    "recommended_tier": "A",
                    "recommended_action": "Bettable now",
                    "fragility_bucket": "low",
                    "tracked_market_key": "moneyline",
                    "bet_quality_score": 92.0,
                    "effective_edge": 0.14,
                    "market_blend_weight": 0.15,
                    "historical_overlay_grade": "low_sample",
                    "historical_sample_size": 0,
                    "chosen_expression_stake": 100.0,
                    "suggested_stake": 90.0,
                },
                {
                    "event_id": "e1",
                    "fight_key": "alpha||beta",
                    "recommended_tier": "B",
                    "recommended_action": "Watchlist",
                    "fragility_bucket": "medium",
                    "tracked_market_key": "moneyline",
                    "bet_quality_score": 80.0,
                    "effective_edge": 0.09,
                    "market_blend_weight": 0.10,
                    "historical_overlay_grade": "low_sample",
                    "historical_sample_size": 0,
                    "chosen_expression_stake": 60.0,
                    "suggested_stake": 50.0,
                },
                {
                    "event_id": "e1",
                    "fight_key": "gamma||delta",
                    "recommended_tier": "B",
                    "recommended_action": "Bettable now",
                    "fragility_bucket": "low",
                    "tracked_market_key": "moneyline",
                    "bet_quality_score": 70.0,
                    "effective_edge": 0.07,
                    "market_blend_weight": 0.10,
                    "historical_overlay_grade": "low_sample",
                    "historical_sample_size": 0,
                    "chosen_expression_stake": 50.0,
                    "suggested_stake": 45.0,
                },
            ]
        )

        config = BankrollGovernorConfig(
            max_stake_pct=0.05,
            max_card_exposure_pct=0.08,
            max_fight_exposure_pct=0.06,
            watchlist_multiplier=0.50,
            medium_fragility_multiplier=0.75,
            high_fragility_multiplier=0.40,
            prop_multiplier=0.75,
            disagreement_multiplier=0.80,
            negative_history_multiplier=0.80,
            min_actionable_stake=0.0,
        )

        governed = apply_bankroll_governor(frame, bankroll=1000.0, config=config)

        first = governed.iloc[0]
        second = governed.iloc[1]
        third = governed.iloc[2]

        self.assertEqual(float(first["raw_chosen_expression_stake"]), 100.0)
        self.assertEqual(float(first["chosen_expression_stake"]), 50.0)
        self.assertIn("per_bet_cap", str(first["stake_governor_reason"]))

        self.assertEqual(float(second["chosen_expression_stake"]), 10.0)
        self.assertIn("watchlist_half_stake", str(second["stake_governor_reason"]))
        self.assertIn("fragility_medium_trim", str(second["stake_governor_reason"]))
        self.assertIn("fight_exposure_cap", str(second["stake_governor_reason"]))

        self.assertEqual(float(third["chosen_expression_stake"]), 20.0)
        self.assertIn("card_exposure_cap", str(third["stake_governor_reason"]))

    def test_apply_bankroll_governor_zeroes_model_pass_rows(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "e2",
                    "fight_key": "alpha||beta",
                    "recommended_tier": "C",
                    "recommended_action": "Pass",
                    "fragility_bucket": "high",
                    "tracked_market_key": "inside_distance",
                    "bet_quality_score": 55.0,
                    "effective_edge": 0.03,
                    "market_blend_weight": 0.50,
                    "chosen_expression_stake": 25.0,
                    "suggested_stake": 20.0,
                }
            ]
        )

        governed = apply_bankroll_governor(frame, bankroll=1000.0)
        row = governed.iloc[0]

        self.assertEqual(float(row["raw_chosen_expression_stake"]), 25.0)
        self.assertEqual(float(row["chosen_expression_stake"]), 0.0)
        self.assertEqual(float(row["stake_governor_multiplier"]), 0.0)
        self.assertIn("model_pass", str(row["stake_governor_reason"]))


if __name__ == "__main__":
    unittest.main()
