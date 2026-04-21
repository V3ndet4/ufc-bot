import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.threshold_policy import build_threshold_policy, resolve_scan_thresholds


class ThresholdPolicyTests(unittest.TestCase):
    def test_build_threshold_policy_tightens_when_stronger_slice_is_clear(self) -> None:
        rows: list[dict[str, object]] = []
        for index in range(6):
            rows.append(
                {
                    "chosen_expression_edge": 0.09,
                    "model_confidence": 0.74,
                    "data_quality": 0.96,
                    "selection_stats_completeness": 0.96,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 10.0,
                    "profit": 5.0,
                    "clv_delta": 0.04,
                    "actual_result": "win" if index < 5 else "loss",
                    "grade_status": "graded",
                }
            )
        for index in range(6):
            rows.append(
                {
                    "chosen_expression_edge": 0.03,
                    "model_confidence": 0.58,
                    "data_quality": 0.78,
                    "selection_stats_completeness": 0.78,
                    "selection_fallback_used": 1.0 if index < 3 else 0.0,
                    "chosen_expression_stake": 10.0,
                    "profit": -4.0,
                    "clv_delta": -0.03,
                    "actual_result": "loss",
                    "grade_status": "graded",
                }
            )

        policy = build_threshold_policy(pd.DataFrame(rows), min_graded_bets=4)
        selected = policy["selected"]
        baseline = policy["baseline"]

        self.assertIn(policy["status"], {"optimized", "baseline"})
        self.assertGreaterEqual(float(selected["min_edge"]), 0.03)
        self.assertGreaterEqual(float(selected["min_model_confidence"]), 0.60)
        self.assertTrue(bool(selected["exclude_fallback_rows"]))
        self.assertGreaterEqual(float(selected["roi_pct"]), float(baseline["roi_pct"]))
        self.assertEqual(int(selected["graded_bets"]), 6)

    def test_resolve_scan_thresholds_respects_stricter_existing_floors(self) -> None:
        resolved = resolve_scan_thresholds(
            min_edge=0.06,
            min_model_confidence=0.70,
            min_stats_completeness=0.88,
            exclude_fallback_rows=False,
            policy={
                "selected": {
                    "min_edge": 0.04,
                    "min_model_confidence": 0.65,
                    "min_stats_completeness": 0.82,
                    "exclude_fallback_rows": True,
                }
            },
        )

        self.assertEqual(float(resolved["min_edge"]), 0.06)
        self.assertEqual(float(resolved["min_model_confidence"]), 0.70)
        self.assertEqual(float(resolved["min_stats_completeness"]), 0.88)
        self.assertTrue(bool(resolved["exclude_fallback_rows"]))
        self.assertTrue(bool(resolved["policy_applied"]))

    def test_build_threshold_policy_does_not_choose_higher_score_with_worse_profit(self) -> None:
        rows: list[dict[str, object]] = []

        # Baseline slice: modestly losing, but clearly better than the tighter slice below.
        for index in range(20):
            rows.append(
                {
                    "chosen_expression_edge": 0.04,
                    "model_confidence": 0.62,
                    "data_quality": 0.85,
                    "selection_stats_completeness": 0.85,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -10.0 if index < 12 else 8.0,
                    "clv_delta": 0.12,
                    "actual_result": "loss" if index < 12 else "win",
                    "grade_status": "graded",
                }
            )

        # Tighter slice: higher confidence/edge/CLV, but loses far more money.
        for index in range(12):
            rows.append(
                {
                    "chosen_expression_edge": 0.09,
                    "model_confidence": 0.74,
                    "data_quality": 0.92,
                    "selection_stats_completeness": 0.92,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -35.0 if index < 10 else 12.0,
                    "clv_delta": 0.28,
                    "actual_result": "loss" if index < 10 else "win",
                    "grade_status": "graded",
                }
            )

        policy = build_threshold_policy(pd.DataFrame(rows), min_graded_bets=8)
        selected = policy["selected"]
        baseline = policy["baseline"]

        self.assertEqual(policy["status"], "baseline")
        self.assertEqual(float(selected["total_profit"]), float(baseline["total_profit"]))
        self.assertEqual(float(selected["roi_pct"]), float(baseline["roi_pct"]))

    def test_build_threshold_policy_can_still_choose_higher_profit_slice(self) -> None:
        rows: list[dict[str, object]] = []

        # Early history before the validation window.
        for index in range(8):
            rows.append(
                {
                    "tracked_at": f"2026-01-{index + 1:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.04,
                    "model_confidence": 0.62,
                    "data_quality": 0.85,
                    "selection_stats_completeness": 0.85,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -8.0 if index < 4 else 10.0,
                    "clv_delta": 0.05,
                    "actual_result": "loss" if index < 4 else "win",
                    "grade_status": "graded",
                }
            )

        # Baseline-qualified bets in the validation window that are only mediocre.
        for index in range(10):
            rows.append(
                {
                    "tracked_at": f"2026-02-{index + 1:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.04,
                    "model_confidence": 0.62,
                    "data_quality": 0.85,
                    "selection_stats_completeness": 0.85,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -10.0 if index < 6 else 8.0,
                    "clv_delta": 0.04,
                    "actual_result": "loss" if index < 6 else "win",
                    "grade_status": "graded",
                }
            )

        # Stronger tighter slice that performs clearly better in validation.
        for index in range(10):
            rows.append(
                {
                    "tracked_at": f"2026-03-{index + 1:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.09,
                    "model_confidence": 0.74,
                    "data_quality": 0.92,
                    "selection_stats_completeness": 0.92,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -6.0 if index < 2 else 18.0,
                    "clv_delta": 0.09,
                    "actual_result": "loss" if index < 2 else "win",
                    "grade_status": "graded",
                }
            )

        policy = build_threshold_policy(pd.DataFrame(rows), min_graded_bets=8)
        selected = policy["selected"]
        baseline = policy["baseline"]

        self.assertEqual(policy["status"], "optimized")
        self.assertGreater(float(selected["roi_pct"]), float(baseline["roi_pct"]))
        self.assertGreater(float(selected["score"]), 0.0)

    def test_build_threshold_policy_uses_walk_forward_validation(self) -> None:
        rows: list[dict[str, object]] = []

        # Early bets make the tighter slice look attractive in-sample.
        for index in range(8):
            rows.append(
                {
                    "tracked_at": f"2026-01-{index + 1:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.10,
                    "model_confidence": 0.75,
                    "data_quality": 0.92,
                    "selection_stats_completeness": 0.92,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": 18.0,
                    "clv_delta": 0.10,
                    "actual_result": "win",
                    "grade_status": "graded",
                }
            )

        # Later bets are the true validation period and the tighter slice collapses.
        for index in range(8):
            rows.append(
                {
                    "tracked_at": f"2026-02-{index + 1:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.10,
                    "model_confidence": 0.75,
                    "data_quality": 0.92,
                    "selection_stats_completeness": 0.92,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": -25.0,
                    "clv_delta": 0.08,
                    "actual_result": "loss",
                    "grade_status": "graded",
                }
            )

        # Baseline-only rows with weaker thresholds but better later realized performance.
        for index in range(8):
            rows.append(
                {
                    "tracked_at": f"2026-02-{index + 11:02d}T10:00:00Z",
                    "chosen_expression_edge": 0.04,
                    "model_confidence": 0.62,
                    "data_quality": 0.85,
                    "selection_stats_completeness": 0.85,
                    "selection_fallback_used": 0.0,
                    "chosen_expression_stake": 100.0,
                    "profit": 6.0,
                    "clv_delta": 0.03,
                    "actual_result": "win" if index < 5 else "loss",
                    "grade_status": "graded",
                }
            )

        policy = build_threshold_policy(pd.DataFrame(rows), min_graded_bets=4)

        self.assertEqual(policy["status"], "baseline")
        self.assertGreaterEqual(int(policy["validation_graded_bets"]), 4)
        self.assertLessEqual(
            float(policy["selected_validation"]["roi_pct"]),
            float(policy["baseline_validation"]["roi_pct"]) + 0.01,
        )


if __name__ == "__main__":
    unittest.main()
