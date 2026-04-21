import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.decision_support import apply_market_history_coverage, build_market_history_coverage_lookup


class DecisionSupportTests(unittest.TestCase):
    def test_apply_market_history_coverage_defaults_props_to_not_ready_without_archive(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "market": "moneyline",
                    "tracked_market_key": "inside_distance",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                }
            ]
        )

        covered = apply_market_history_coverage(frame, archive_frame=pd.DataFrame())
        row = covered.iloc[0]

        self.assertEqual(int(row["market_history_event_count"]), 0)
        self.assertEqual(int(row["market_history_fight_count"]), 0)
        self.assertFalse(bool(row["market_history_recommendation_ready"]))
        self.assertIn("need 2+ events and 8+ fights", str(row["market_history_note"]))

    def test_build_market_history_coverage_lookup_marks_props_ready_after_multiple_events(self) -> None:
        archive = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e1",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Eta",
                    "fighter_b": "Theta",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Eta",
                    "fighter_b": "Theta",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Iota",
                    "fighter_b": "Kappa",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Iota",
                    "fighter_b": "Kappa",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Lambda",
                    "fighter_b": "Mu",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Lambda",
                    "fighter_b": "Mu",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Nu",
                    "fighter_b": "Xi",
                    "market": "inside_distance",
                    "actual_result": "win",
                },
                {
                    "event_id": "e2",
                    "fighter_a": "Nu",
                    "fighter_b": "Xi",
                    "market": "inside_distance",
                    "actual_result": "loss",
                },
            ]
        )

        lookup = build_market_history_coverage_lookup(archive)
        inside_distance = lookup["inside_distance"]

        self.assertEqual(int(inside_distance["market_history_event_count"]), 2)
        self.assertEqual(int(inside_distance["market_history_fight_count"]), 6)
        self.assertFalse(bool(inside_distance["market_history_recommendation_ready"]))

        expanded = pd.concat(
            [
                archive,
                pd.DataFrame(
                    [
                        {
                            "event_id": "e2",
                            "fighter_a": "Omicron",
                            "fighter_b": "Pi",
                            "market": "inside_distance",
                            "actual_result": "win",
                        },
                        {
                            "event_id": "e2",
                            "fighter_a": "Omicron",
                            "fighter_b": "Pi",
                            "market": "inside_distance",
                            "actual_result": "loss",
                        },
                        {
                            "event_id": "e2",
                            "fighter_a": "Rho",
                            "fighter_b": "Sigma",
                            "market": "inside_distance",
                            "actual_result": "win",
                        },
                        {
                            "event_id": "e2",
                            "fighter_a": "Rho",
                            "fighter_b": "Sigma",
                            "market": "inside_distance",
                            "actual_result": "loss",
                        },
                    ]
                ),
            ],
            ignore_index=True,
        )
        expanded_lookup = build_market_history_coverage_lookup(expanded)
        ready_market = expanded_lookup["inside_distance"]

        self.assertEqual(int(ready_market["market_history_event_count"]), 2)
        self.assertEqual(int(ready_market["market_history_fight_count"]), 8)
        self.assertTrue(bool(ready_market["market_history_recommendation_ready"]))


if __name__ == "__main__":
    unittest.main()
