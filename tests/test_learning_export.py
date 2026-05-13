import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_learning_report import (
    build_filter_performance_report,
    build_pick_postmortem_report,
    build_pick_postmortem_summary,
    build_learning_report,
    build_learning_summary,
    enrich_with_feedback_buckets,
)
from data_sources.storage import load_tracked_picks, save_tracked_picks


class LearningExportTests(unittest.TestCase):
    def test_learning_report_and_summary(self) -> None:
        tracked = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Event 1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "selection_name": "Alpha",
                    "chosen_value_expression": "Alpha moneyline",
                    "tracked_at": "2026-04-01T10:00:00Z",
                    "american_odds": 120,
                    "chosen_expression_odds": 125,
                    "model_projected_win_prob": 0.58,
                    "chosen_expression_prob": 0.60,
                    "implied_prob": 0.45,
                    "chosen_expression_implied_prob": 0.44,
                    "edge": 0.13,
                    "chosen_expression_edge": 0.16,
                    "suggested_stake": 20.0,
                    "chosen_expression_stake": 25.0,
                    "model_confidence": 0.78,
                    "data_quality": 0.94,
                    "selection_stats_completeness": 0.96,
                    "selection_fallback_used": 0.0,
                    "line_movement_toward_fighter": 0.05,
                    "closing_american_odds": 110,
                    "clv_delta": 15.0,
                    "timing_action": "Wait",
                    "timing_score": 35.0,
                    "timing_signal": "drift",
                    "timing_reason": "line moving away",
                    "news_radar_score": 0.82,
                    "news_radar_label": "red",
                    "news_radar_summary": "camp change, injury watch",
                    "actual_result": "win",
                    "profit": 31.25,
                    "grade_status": "graded",
                    "tracked_market_key": "moneyline",
                    "recommended_tier": "A",
                },
                {
                    "event_id": "e1",
                    "event_name": "Event 1",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "selection_name": "Gamma",
                    "chosen_value_expression": "Gamma by decision",
                    "tracked_at": "2026-04-01T11:00:00Z",
                    "american_odds": 140,
                    "chosen_expression_odds": 220,
                    "model_projected_win_prob": 0.52,
                    "chosen_expression_prob": 0.38,
                    "implied_prob": 0.42,
                    "chosen_expression_implied_prob": 0.31,
                    "edge": 0.10,
                    "chosen_expression_edge": 0.07,
                    "suggested_stake": 10.0,
                    "chosen_expression_stake": 12.5,
                    "model_confidence": 0.61,
                    "data_quality": 0.82,
                    "selection_stats_completeness": 0.79,
                    "selection_fallback_used": 1.0,
                    "line_movement_toward_fighter": -0.03,
                    "closing_american_odds": pd.NA,
                    "clv_delta": pd.NA,
                    "actual_result": "pending",
                    "profit": None,
                    "grade_status": "pending",
                    "tracked_market_key": "by_decision",
                    "recommended_tier": "C",
                },
                {
                    "event_id": "e2",
                    "event_name": "Event 2",
                    "fighter_a": "Epsilon",
                    "fighter_b": "Zeta",
                    "selection_name": "Epsilon",
                    "chosen_value_expression": "Epsilon",
                    "tracked_at": "2026-04-08T11:00:00Z",
                    "american_odds": -145,
                    "chosen_expression_odds": -145,
                    "model_projected_win_prob": 0.68,
                    "chosen_expression_prob": 0.68,
                    "implied_prob": 0.59,
                    "chosen_expression_implied_prob": 0.59,
                    "edge": 0.09,
                    "chosen_expression_edge": 0.09,
                    "suggested_stake": 15.0,
                    "chosen_expression_stake": 15.0,
                    "model_confidence": 0.73,
                    "data_quality": 0.91,
                    "selection_stats_completeness": 0.91,
                    "selection_fallback_used": 0.0,
                    "line_movement_toward_fighter": 0.00,
                    "closing_american_odds": -130,
                    "clv_delta": 15.0,
                    "actual_result": "loss",
                    "profit": -15.0,
                    "grade_status": "graded",
                    "tracked_market_key": "moneyline",
                    "recommended_tier": "B",
                },
            ]
        )

        report = build_learning_report(tracked)
        summary = build_learning_summary(tracked)
        bucketed = enrich_with_feedback_buckets(tracked)
        filter_report = build_filter_performance_report(tracked)

        self.assertEqual(len(report), 3)
        self.assertEqual(report.loc[0, "fight"], "Alpha vs Beta")
        self.assertEqual(float(report.loc[0, "roi_pct"]), 125.0)
        self.assertEqual(float(report.loc[1, "roi_pct"]), 0.0)
        self.assertIn("decimal_line_at_pick", report.columns)
        self.assertIn("decimal_closing_line", report.columns)
        self.assertIn("confidence_at_pick", report.columns)
        self.assertIn("line_movement_bucket", report.columns)
        self.assertEqual(str(report.loc[0, "timing_bucket"]), "wait")
        self.assertEqual(str(report.loc[0, "news_radar_bucket"]), "red")
        self.assertEqual(float(report.loc[0, "decimal_line_at_pick"]), 2.25)
        self.assertEqual(len(summary), 2)
        self.assertEqual(int(summary.loc[0, "bets"]), 2)
        self.assertEqual(int(summary.loc[0, "graded_bets"]), 1)
        self.assertEqual(int(summary.loc[0, "wins"]), 1)
        self.assertEqual(int(summary.loc[0, "pending"]), 1)
        self.assertEqual(float(summary.loc[0, "total_profit"]), 31.25)
        self.assertEqual(bucketed.loc[0, "confidence_bucket"], "0.75_plus")
        self.assertEqual(bucketed.loc[1, "fallback_bucket"], "fallback_used")
        self.assertEqual(bucketed.loc[2, "price_bucket"], "favorite")
        self.assertEqual(bucketed.loc[0, "timing_bucket"], "wait")

        postmortem = build_pick_postmortem_report(tracked)
        postmortem_summary = build_pick_postmortem_summary(tracked)
        self.assertIn("postmortem_bucket", postmortem.columns)
        self.assertIn("root_cause", postmortem.columns)
        self.assertIn("next_action", postmortem.columns)
        self.assertEqual(str(postmortem.loc[0, "root_cause"]), "timing_miss")
        self.assertEqual(str(postmortem.loc[0, "next_action"]), "tighten_timing")
        self.assertTrue(
            ((postmortem_summary["root_cause"] == "timing_miss") & (postmortem_summary["postmortem_bucket"] == "validated")).any()
        )

        confidence_row = filter_report.loc[
            (filter_report["dimension"] == "confidence")
            & (filter_report["bucket"] == "0.75_plus")
        ].iloc[0]
        self.assertEqual(int(confidence_row["wins"]), 1)
        self.assertEqual(float(confidence_row["roi_pct"]), 125.0)

        fallback_row = filter_report.loc[
            (filter_report["dimension"] == "fallback")
            & (filter_report["bucket"] == "fallback_used")
        ].iloc[0]
        self.assertEqual(int(fallback_row["pending"]), 1)
        self.assertEqual(str(fallback_row["recommendation"]), "needs_more_data")

    def test_save_tracked_picks_only_persists_actionable_rows(self) -> None:
        tracked = pd.DataFrame(
            [
                {
                    "event_id": "e3",
                    "event_name": "Event 3",
                    "start_time": "2026-04-18T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "book": "Book",
                    "american_odds": -110,
                    "model_projected_win_prob": 0.60,
                    "implied_prob": 0.52,
                    "edge": 0.08,
                    "expected_value": 0.14,
                    "suggested_stake": 25.0,
                    "recommended_action": "Bettable now",
                    "recommended_tier": "A",
                    "chosen_value_expression": "Alpha",
                    "chosen_expression_odds": -110,
                    "chosen_expression_prob": 0.60,
                    "chosen_expression_implied_prob": 0.52,
                    "chosen_expression_edge": 0.08,
                    "chosen_expression_expected_value": 0.14,
                    "chosen_expression_stake": 25.0,
                },
                {
                    "event_id": "e3",
                    "event_name": "Event 3",
                    "start_time": "2026-04-18T20:00:00Z",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Gamma",
                    "book": "Book",
                    "american_odds": 150,
                    "model_projected_win_prob": 0.44,
                    "implied_prob": 0.40,
                    "edge": 0.04,
                    "expected_value": 0.10,
                    "suggested_stake": 10.0,
                    "recommended_action": "Pass",
                    "recommended_tier": "C",
                    "chosen_value_expression": "Gamma",
                    "chosen_expression_odds": 150,
                    "chosen_expression_prob": 0.44,
                    "chosen_expression_implied_prob": 0.40,
                    "chosen_expression_edge": 0.04,
                    "chosen_expression_expected_value": 0.10,
                    "chosen_expression_stake": 10.0,
                },
            ]
        )

        db_path = ROOT / "tests" / "_tmp_learning_export.db"
        try:
            saved = save_tracked_picks(tracked, db_path)
            loaded = load_tracked_picks(db_path, event_id="e3")
        finally:
            db_path.unlink(missing_ok=True)

        self.assertEqual(saved, 1)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(str(loaded.loc[0, "selection_name"]), "Alpha")
        self.assertEqual(str(loaded.loc[0, "recommended_action"]), "Bettable now")


if __name__ == "__main__":
    unittest.main()
