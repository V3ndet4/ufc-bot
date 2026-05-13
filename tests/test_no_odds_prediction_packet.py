import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_no_odds_prediction_packet import build_no_odds_prediction_packet


class NoOddsPredictionPacketTests(unittest.TestCase):
    def test_packet_outputs_one_lean_per_fight_with_gym_context(self) -> None:
        manifest = {
            "event_id": "e1",
            "event_name": "No Odds Event",
            "start_time": "2026-05-09T21:00:00-04:00",
            "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
        }
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 12,
                    "losses": 2,
                    "draws": 0,
                    "height_in": 72,
                    "reach_in": 76,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 2.0,
                    "takedown_avg": 2.0,
                    "takedown_defense_pct": 80,
                    "stats_completeness": 1.0,
                    "gym_name": "Alpha Gym",
                    "gym_tier": "B",
                    "gym_record": "10-2-0",
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 5,
                    "draws": 0,
                    "height_in": 70,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.0,
                    "sig_strikes_absorbed_per_min": 4.0,
                    "takedown_avg": 0.5,
                    "takedown_defense_pct": 55,
                    "stats_completeness": 1.0,
                    "gym_name": "Beta Gym",
                    "gym_tier": "D",
                    "gym_record": "6-4-0",
                },
            ]
        )

        packet = build_no_odds_prediction_packet(manifest, stats)

        self.assertEqual(len(packet), 1)
        self.assertEqual(packet.loc[0, "fight"], "Alpha vs Beta")
        self.assertEqual(packet.loc[0, "lean_side"], "Alpha")
        self.assertIn(packet.loc[0, "lean_strength"], {"Strong Lean", "Lean", "Slight Lean", "Coin Flip"})
        self.assertEqual(packet.loc[0, "pick_gym_tier"], "B")
        self.assertIn("Alpha Gym", packet.loc[0, "camp_summary"])
        self.assertIn("Beta Gym", packet.loc[0, "camp_summary"])
        self.assertGreater(float(packet.loc[0, "model_prob"]), 0.5)


if __name__ == "__main__":
    unittest.main()
