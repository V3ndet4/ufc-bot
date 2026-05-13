import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.grading import grade_tracked_picks
from data_sources.storage import load_fight_results, save_fight_results


class GradingTests(unittest.TestCase):
    def test_grading_handles_moneyline_and_finish_markets(self) -> None:
        picks = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "chosen_value_expression": "Alpha",
                    "chosen_expression_odds": -110,
                    "chosen_expression_prob": 0.60,
                    "chosen_expression_stake": 100.0,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "chosen_value_expression": "Fight doesn't go to decision",
                    "chosen_expression_odds": +120,
                    "chosen_expression_prob": 0.58,
                    "chosen_expression_stake": 100.0,
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "official",
                    "went_decision": 0,
                    "ended_inside_distance": 1,
                    "closing_fighter_a_odds": -130,
                    "closing_fight_doesnt_go_to_decision_odds": +105,
                }
            ]
        )

        graded = grade_tracked_picks(picks, results)

        self.assertEqual(list(graded["actual_result"]), ["win", "win"])
        self.assertAlmostEqual(float(graded.loc[0, "profit"]), 90.91, places=2)
        self.assertAlmostEqual(float(graded.loc[1, "profit"]), 120.0, places=2)
        self.assertEqual(int(graded.loc[0, "closing_american_odds"]), -130)
        self.assertEqual(int(graded.loc[1, "closing_american_odds"]), 105)

    def test_grading_pushes_replacement_opponent_results(self) -> None:
        picks = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "chosen_value_expression": "Alpha",
                    "chosen_expression_odds": -110,
                    "chosen_expression_prob": 0.60,
                    "chosen_expression_stake": 100.0,
                }
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "actual_fighter_a": "Alpha",
                    "actual_fighter_b": "Gamma",
                    "winner_name": "Alpha",
                    "actual_winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "replacement_opponent",
                    "went_decision": 1,
                    "ended_inside_distance": 0,
                }
            ]
        )

        graded = grade_tracked_picks(picks, results)

        self.assertEqual(list(graded["actual_result"]), ["push"])
        self.assertAlmostEqual(float(graded.loc[0, "profit"]), 0.0, places=2)

    def test_grading_handles_method_and_stat_props(self) -> None:
        picks = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "submission",
                    "selection": "fighter_a",
                    "selection_name": "Alpha by submission",
                    "book": "Book",
                    "american_odds": 250,
                    "chosen_value_expression": "Alpha by submission",
                    "chosen_expression_odds": 250,
                    "chosen_expression_prob": 0.35,
                    "chosen_expression_stake": 10.0,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_ends_by_submission",
                    "selection": "fight_ends_by_submission",
                    "selection_name": "Fight ends by submission",
                    "book": "Book",
                    "american_odds": 300,
                    "chosen_value_expression": "Fight ends by submission",
                    "chosen_expression_odds": 300,
                    "chosen_expression_prob": 0.30,
                    "chosen_expression_stake": 10.0,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "takedown",
                    "selection": "fighter_a",
                    "selection_name": "Alpha takedown",
                    "book": "Book",
                    "american_odds": -150,
                    "chosen_value_expression": "Alpha takedown",
                    "chosen_expression_odds": -150,
                    "chosen_expression_prob": 0.62,
                    "chosen_expression_stake": 10.0,
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "knockdown",
                    "selection": "fighter_b",
                    "selection_name": "Beta knockdown",
                    "book": "Book",
                    "american_odds": 400,
                    "chosen_value_expression": "Beta knockdown",
                    "chosen_expression_odds": 400,
                    "chosen_expression_prob": 0.20,
                    "chosen_expression_stake": 10.0,
                },
            ]
        )
        results = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "official",
                    "went_decision": 0,
                    "ended_inside_distance": 1,
                    "method": "Submission",
                    "fighter_a_takedowns": 2,
                    "fighter_b_knockdowns": 0,
                    "closing_fighter_a_submission_odds": 220,
                    "closing_fight_ends_by_submission_odds": 280,
                    "closing_fighter_a_takedown_odds": -170,
                    "closing_fighter_b_knockdown_odds": 450,
                }
            ]
        )

        graded = grade_tracked_picks(picks, results)

        self.assertEqual(list(graded["actual_result"]), ["win", "win", "win", "loss"])
        self.assertEqual(int(graded.loc[0, "closing_american_odds"]), 220)
        self.assertEqual(int(graded.loc[2, "closing_american_odds"]), -170)
        self.assertAlmostEqual(float(graded.loc[3, "profit"]), -10.0, places=2)

    def test_save_fight_results_replaces_existing_event_rows(self) -> None:
        first = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "winner_name": "Alpha",
                    "winner_side": "fighter_a",
                    "result_status": "replacement_opponent",
                    "went_decision": 1,
                    "ended_inside_distance": 0,
                    "method": "Decision",
                }
            ]
        )
        second = first.copy()
        second.loc[0, "result_status"] = "official"

        db_path = ROOT / "tests" / "_tmp_fight_results.db"
        try:
            self.assertEqual(save_fight_results(first, db_path), 1)
            self.assertEqual(save_fight_results(second, db_path), 1)
            loaded = load_fight_results(db_path, event_id="e1")
        finally:
            db_path.unlink(missing_ok=True)

        self.assertEqual(len(loaded), 1)
        self.assertEqual(str(loaded.loc[0, "result_status"]), "official")


if __name__ == "__main__":
    unittest.main()
