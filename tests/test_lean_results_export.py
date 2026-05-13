import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_lean_board_results import (
    build_lean_board_results,
    build_lean_postmortem_summary,
)


class LeanResultsExportTests(unittest.TestCase):
    def test_build_lean_board_results_and_summary(self) -> None:
        lean_board = pd.DataFrame(
            [
                {
                    "event_name": "Test Event",
                    "fight": "Alpha vs Beta",
                    "lean_side": "Alpha",
                    "opponent_side": "Beta",
                    "lean_strength": "Strong Lean",
                    "lean_action": "Bet now",
                    "current_american_odds": -110,
                },
                {
                    "event_name": "Test Event",
                    "fight": "Gamma vs Delta",
                    "lean_side": "Gamma",
                    "opponent_side": "Delta",
                    "lean_strength": "Coin Flip",
                    "lean_action": "Pass",
                    "current_american_odds": 125,
                },
                {
                    "event_name": "Test Event",
                    "fight": "Iota vs Kappa",
                    "lean_side": "Kappa",
                    "opponent_side": "Iota",
                    "lean_strength": "Strong Lean",
                    "lean_action": "Bet now",
                    "current_american_odds": 145,
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "actual_fighter_a": "Alpha",
                    "actual_fighter_b": "Beta",
                    "winner_name": "Alpha",
                    "actual_winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "official",
                    "result_match_status": "exact",
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "actual_fighter_a": "Gamma",
                    "actual_fighter_b": "Epsilon",
                    "winner_name": "Gamma",
                    "actual_winner_name": "Gamma",
                    "winner_side": "fighter_a",
                    "result_status": "replacement_opponent",
                    "result_match_status": "replacement_opponent",
                },
            ]
        )

        report = build_lean_board_results(lean_board, results)
        summary = build_lean_postmortem_summary(report)

        self.assertEqual(list(report["graded_result"]), ["win", "push", "pending"])
        self.assertEqual(list(report["lean_fighter_actual_outcome"]), ["won", "won", "pending"])
        self.assertEqual(report.loc[1, "actual_fight"], "Gamma vs Epsilon")
        self.assertIn("void", report.loc[1, "grading_note"].lower())

        all_row = summary.loc[summary["bucket"] == "all"].iloc[0]
        replacement_row = summary.loc[summary["bucket"] == "replacement_opponent"].iloc[0]
        self.assertEqual(int(all_row["wins"]), 1)
        self.assertEqual(int(all_row["pushes"]), 1)
        self.assertEqual(int(all_row["pending"]), 1)
        self.assertEqual(int(replacement_row["pushes"]), 1)
        self.assertEqual(int(replacement_row["actual_fighter_wins"]), 1)


if __name__ == "__main__":
    unittest.main()
