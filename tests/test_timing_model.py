import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.timing import attach_timing_signals


class TimingModelTests(unittest.TestCase):
    def test_attach_timing_signals_detects_steam_and_bet_now(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "BookA",
                }
            ]
        )
        history = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "BookA",
                    "snapshot_time": "2026-04-01T10:00:00Z",
                    "american_odds": 150,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "BookA",
                    "snapshot_time": "2026-04-01T11:00:00Z",
                    "american_odds": 125,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "BookA",
                    "snapshot_time": "2026-04-01T12:00:00Z",
                    "american_odds": -110,
                },
            ]
        )

        enriched = attach_timing_signals(frame, history)
        row = enriched.iloc[0]

        self.assertEqual(int(row["timing_snapshot_count"]), 3)
        self.assertEqual(int(row["timing_book_count"]), 1)
        self.assertAlmostEqual(float(row["timing_implied_change"]), 0.1238, places=4)
        self.assertAlmostEqual(float(row["timing_score"]), 68.0, places=2)
        self.assertEqual(str(row["timing_signal"]), "steam")
        self.assertEqual(str(row["timing_action"]), "Bet now")
        self.assertIn("line moving toward pick", str(row["timing_reason"]))

    def test_attach_timing_signals_falls_back_without_history(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "event_id": "e2",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Gamma",
                    "book": "BookB",
                    "open_american_odds": 110,
                    "american_odds": 140,
                }
            ]
        )

        enriched = attach_timing_signals(frame)
        row = enriched.iloc[0]

        self.assertEqual(int(row["timing_snapshot_count"]), 0)
        self.assertEqual(str(row["timing_signal"]), "drift")
        self.assertEqual(str(row["timing_action"]), "Monitor")
        self.assertIn("line moving away", str(row["timing_reason"]))


if __name__ == "__main__":
    unittest.main()
