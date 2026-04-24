import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_operator_dashboard import build_operator_dashboard_html


class OperatorDashboardTests(unittest.TestCase):
    def test_build_operator_dashboard_html_surfaces_exposure_and_policy(self) -> None:
        fight_report = pd.DataFrame([{"event_name": "Test Card"}])
        lean_board = pd.DataFrame(
            [
                {
                    "fight": "Alpha vs Beta",
                    "lean_side": "Alpha",
                    "lean_strength": "Lean",
                    "lean_action": "Bet now",
                    "edge": 0.054,
                    "lean_prob": 0.59,
                    "top_reasons": "A-tier control edge",
                }
            ]
        )
        value_report = pd.DataFrame(
            [
                {
                    "event_name": "Test Card",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "chosen_value_expression": "Alpha moneyline",
                    "chosen_expression_odds": -110,
                    "chosen_expression_stake": 32.0,
                    "raw_chosen_expression_stake": 50.0,
                    "stake_governor_reason": "per_bet_cap",
                    "effective_edge": 0.081,
                    "model_confidence": 0.74,
                    "recommended_tier": "A",
                    "recommended_action": "Bettable now",
                    "risk_flags": "market_disagreement",
                },
                {
                    "event_name": "Test Card",
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                    "chosen_value_expression": "Gamma inside distance",
                    "chosen_expression_odds": 210,
                    "chosen_expression_stake": 0.0,
                    "raw_chosen_expression_stake": 0.0,
                    "stake_governor_reason": "",
                    "effective_edge": 0.044,
                    "model_confidence": 0.61,
                    "recommended_tier": "C",
                    "recommended_action": "Report-only",
                    "risk_flags": "thin_market_history",
                    "tracked_market_key": "inside_distance",
                    "market_history_event_count": 1,
                    "market_history_fight_count": 4,
                    "market_history_recommendation_ready": False,
                    "market_history_note": "need 2+ events and 8+ fights in archive",
                },
            ]
        )
        betting_board = pd.DataFrame()
        passes = pd.DataFrame(
            [
                {
                    "fight": "Gamma vs Delta",
                    "selection_name": "Gamma",
                    "pass_reason": "edge_below_threshold",
                    "risk_flags": "thin_edge",
                }
            ]
        )
        alerts = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha Beta",
                    "alert_summary": "Alpha Beta changed camps ahead of UFC return",
                    "alert_category": "camp_change",
                    "confidence_score": 0.91,
                    "alert_radar_score": 0.82,
                    "published_at": "2026-04-18T15:00:00Z",
                }
            ]
        )
        parlays = pd.DataFrame(
            [
                {
                    "parlay_name": "Top 3-Leg Value Parlay",
                    "american_odds": "+425",
                    "decimal_odds": "5.25",
                    "edge": "7.2%",
                    "expected_value": "11.3%",
                    "parlay_confidence": "0.67",
                    "legs": "Alpha moneyline (-110) | Epsilon moneyline (+120) | Iota moneyline (-125)",
                }
            ]
        )

        html_output = build_operator_dashboard_html(
            fight_report=fight_report,
            lean_board=lean_board,
            value_report=value_report,
            betting_board=betting_board,
            passes=passes,
            alerts=alerts,
            parlays=parlays,
            threshold_policy={
                "status": "optimized",
                "selected": {
                    "min_edge": 0.05,
                    "min_model_confidence": 0.65,
                    "min_stats_completeness": 0.85,
                    "exclude_fallback_rows": True,
                    "graded_bets": 18,
                    "roi_pct": 12.4,
                },
            },
        )

        self.assertIn("Test Card", html_output)
        self.assertIn("Governed Exposure", html_output)
        self.assertIn("$32.00", html_output)
        self.assertIn("$50.00", html_output)
        self.assertIn("Threshold Policy", html_output)
        self.assertIn("Parlay Board", html_output)
        self.assertIn("Market Readiness", html_output)
        self.assertIn("Timing Signals", html_output)
        self.assertIn("Fight Week Radar", html_output)
        self.assertIn("Report-only", html_output)
        self.assertIn("need 2+ events and 8+ fights", html_output)
        self.assertIn("Top 3-Leg Value Parlay", html_output)
        self.assertIn("per_bet_cap", html_output)
        self.assertIn("edge_below_threshold", html_output)
        self.assertIn("Alpha Beta", html_output)


if __name__ == "__main__":
    unittest.main()
