import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_value_scan import _apply_expression_overrides, _print_console_summary


class ValueScanFilterTests(unittest.TestCase):
    def test_print_console_summary_colorizes_tiered_supports_for_tty(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "recommended_tier": "A",
                    "bet_quality_score": 88.4,
                    "effective_suggested_stake": 42.0,
                    "suggested_stake": 42.0,
                    "chosen_value_expression": "Alpha moneyline",
                    "selection_name": "Alpha",
                    "american_odds": 120,
                    "effective_american_odds": 120,
                    "model_projected_win_prob": 0.62,
                    "effective_projected_prob": 0.62,
                    "implied_prob": 0.4545,
                    "effective_implied_prob": 0.4545,
                    "edge": 0.1655,
                    "effective_edge": 0.1655,
                    "expected_value": 0.37,
                    "effective_expected_value": 0.37,
                    "support_signals": "A-tier control edge, B-tier grappling pressure",
                    "risk_flags": "",
                }
            ]
        )

        class TtyStringIO(io.StringIO):
            def isatty(self) -> bool:
                return True

        with patch("sys.stdout", new_callable=TtyStringIO) as mock_stdout:
            _print_console_summary(report, report, "model_projected_win_prob")

        output = mock_stdout.getvalue()
        self.assertIn("\x1b[32mA-tier\x1b[0m control edge", output)
        self.assertIn("\x1b[33mB-tier\x1b[0m grappling pressure", output)

    def test_min_stats_completeness_filter_excludes_low_quality_rows(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 120,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": -140,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "stats_completeness": 0.8,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 9,
                    "losses": 4,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 4.0,
                    "sig_strikes_absorbed_per_min": 4.0,
                    "takedown_avg": 0.5,
                    "takedown_defense_pct": 65,
                    "stats_completeness": 0.5,
                },
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_value_scan_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_value_scan_stats.csv"
        output_path = ROOT / "tests" / "_tmp_value_scan_output.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)

        env = os.environ.copy()
        env["MIN_EDGE"] = "0.0"
        env["MIN_MODEL_CONFIDENCE"] = "0.0"
        env["MIN_STATS_COMPLETENESS"] = "0.75"
        selective_model_path = ROOT / "tests" / "_missing_selective_model.pkl"

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_value_scan.py"),
                    "--input",
                    str(odds_path),
                    "--fighter-stats",
                    str(stats_path),
                    "--output",
                    str(output_path),
                    "--selective-model",
                    str(selective_model_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            report = pd.read_csv(output_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertTrue(completed.returncode == 0)
        self.assertTrue(report.empty)

    def test_value_scan_writes_ranked_report_and_shortlist(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e2",
                    "event_name": "Ranked Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 130,
                },
                {
                    "event_id": "e2",
                    "event_name": "Ranked Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": -150,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 15,
                    "losses": 1,
                    "height_in": 72,
                    "reach_in": 75,
                    "sig_strikes_landed_per_min": 6.2,
                    "sig_strikes_absorbed_per_min": 2.5,
                    "takedown_avg": 2.1,
                    "takedown_defense_pct": 82,
                    "recent_result_score": 1.0,
                    "recent_strike_margin_per_min": 2.5,
                    "recent_grappling_rate": 1.8,
                    "days_since_last_fight": 140,
                    "ufc_fight_count": 8,
                    "ufc_debut_flag": 0,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 6,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.1,
                    "sig_strikes_absorbed_per_min": 4.8,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 58,
                    "recent_result_score": -1.0,
                    "recent_strike_margin_per_min": -1.6,
                    "recent_grappling_rate": 0.3,
                    "days_since_last_fight": 480,
                    "ufc_fight_count": 2,
                    "ufc_debut_flag": 0,
                    "stats_completeness": 0.92,
                    "fallback_used": 0,
                },
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_ranked_value_scan_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_ranked_value_scan_stats.csv"
        output_path = ROOT / "tests" / "_tmp_ranked_value_scan_output.csv"
        shortlist_path = ROOT / "tests" / "_tmp_ranked_value_scan_output_shortlist.csv"
        board_path = ROOT / "tests" / "_tmp_ranked_value_scan_output_board.csv"
        passes_path = ROOT / "tests" / "_tmp_ranked_value_scan_output_passes.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)

        env = os.environ.copy()
        env["MIN_EDGE"] = "0.0"
        env["MIN_MODEL_CONFIDENCE"] = "0.0"
        env["MIN_STATS_COMPLETENESS"] = "0.0"
        selective_model_path = ROOT / "tests" / "_missing_selective_model.pkl"

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_value_scan.py"),
                    "--input",
                    str(odds_path),
                    "--fighter-stats",
                    str(stats_path),
                    "--board-output",
                    str(board_path),
                    "--passes-output",
                    str(passes_path),
                    "--output",
                    str(output_path),
                    "--selective-model",
                    str(selective_model_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            report = pd.read_csv(output_path)
            shortlist = pd.read_csv(shortlist_path)
            board = pd.read_csv(board_path)
            passes = pd.read_csv(passes_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            shortlist_path.unlink(missing_ok=True)
            board_path.unlink(missing_ok=True)
            passes_path.unlink(missing_ok=True)

        self.assertEqual(completed.returncode, 0)
        self.assertIn("bet_quality_score", report.columns)
        self.assertIn("recommended_tier", report.columns)
        self.assertIn("why_it_rates_well", report.columns)
        self.assertIn("risk_flags", report.columns)
        self.assertIn("support_signals", report.columns)
        self.assertIn("selective_clv_prob", report.columns)
        self.assertIn("segment_label", report.columns)
        self.assertIn("historical_overlay_grade", report.columns)
        self.assertIn("fragility_bucket", report.columns)
        self.assertIn("raw_chosen_expression_stake", report.columns)
        self.assertIn("stake_governor_multiplier", report.columns)
        self.assertIn("stake_governor_reason", report.columns)
        self.assertIn("runner_up_expression", report.columns)
        self.assertFalse(report.empty)
        self.assertIn("bet", board.columns)
        self.assertIn("american_line", board.columns)
        self.assertIn("history", board.columns)
        self.assertIn("fragility", board.columns)
        self.assertIn("stake_notes", board.columns)
        self.assertIn("runner_up_bet", board.columns)
        self.assertIn("pass_reason", passes.columns)
        self.assertIn("american_line", passes.columns)
        self.assertIn("history", passes.columns)
        self.assertIn("fragility", passes.columns)
        self.assertIn("model_context_flags", board.columns)
        self.assertIn("operator_context_flags", passes.columns)
        non_empty_lines = pd.to_numeric(board["line"], errors="coerce").dropna()
        self.assertTrue((non_empty_lines > 1.0).all())
        self.assertTrue(set(shortlist["recommended_tier"]).issubset({"A", "B"}))
        self.assertTrue(report["support_signals"].astype(str).str.contains("reach advantage").any())

    def test_value_scan_surfaces_control_edge_in_output(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e_control",
                    "event_name": "Control Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 120,
                },
                {
                    "event_id": "e_control",
                    "event_name": "Control Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": -140,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.0,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.0,
                    "takedown_defense_pct": 75,
                    "submission_avg": 0.6,
                    "recent_grappling_rate": 1.3,
                    "control_avg": 4.8,
                    "recent_control_avg": 4.0,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.0,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.0,
                    "takedown_defense_pct": 75,
                    "submission_avg": 0.1,
                    "recent_grappling_rate": 0.4,
                    "control_avg": 0.9,
                    "recent_control_avg": 0.2,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_control_value_scan_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_control_value_scan_stats.csv"
        output_path = ROOT / "tests" / "_tmp_control_value_scan_output.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)

        env = os.environ.copy()
        env["MIN_EDGE"] = "0.0"
        env["MIN_MODEL_CONFIDENCE"] = "0.0"
        env["MIN_STATS_COMPLETENESS"] = "0.0"
        selective_model_path = ROOT / "tests" / "_missing_selective_model.pkl"

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_value_scan.py"),
                    "--input",
                    str(odds_path),
                    "--fighter-stats",
                    str(stats_path),
                    "--output",
                    str(output_path),
                    "--selective-model",
                    str(selective_model_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            report = pd.read_csv(output_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

        self.assertIn("selection_control_avg", report.columns)
        self.assertIn("selection_recent_control_avg", report.columns)
        self.assertIn("selection_grappling_pressure_score", report.columns)
        self.assertTrue(report["support_signals"].astype(str).str.contains("A-tier control edge").any())

    def test_value_scan_carries_expression_winner_from_fight_report(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e3",
                    "event_name": "Expression Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                },
                {
                    "event_id": "e3",
                    "event_name": "Expression Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": 100,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 15,
                    "losses": 1,
                    "height_in": 72,
                    "reach_in": 75,
                    "sig_strikes_landed_per_min": 6.2,
                    "sig_strikes_absorbed_per_min": 2.5,
                    "takedown_avg": 2.1,
                    "takedown_defense_pct": 82,
                    "stats_completeness": 1.0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 6,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.1,
                    "sig_strikes_absorbed_per_min": 4.8,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 58,
                    "stats_completeness": 1.0,
                },
            ]
        )
        fight_report = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "preferred_market_expression": "Fight doesn't go to decision",
                    "preferred_market_american_odds": -150,
                    "value_expression_winner": "alternative_market",
                    "value_expression_reason": "Test alternative",
                    "speculative_prop_expression": "Alpha inside distance",
                    "speculative_prop_model_prob": 0.19,
                    "speculative_prop_fair_american_odds": 426,
                    "speculative_prop_fair_decimal_odds": 5.26,
                    "speculative_prop_reason": "Aggressive shot only.",
                }
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_expression_value_scan_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_expression_value_scan_stats.csv"
        fight_report_path = ROOT / "tests" / "_tmp_expression_fight_report.csv"
        output_path = ROOT / "tests" / "_tmp_expression_value_scan_output.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)
        fight_report.to_csv(fight_report_path, index=False)

        env = os.environ.copy()
        env["MIN_EDGE"] = "0.0"
        env["MIN_MODEL_CONFIDENCE"] = "0.0"
        env["MIN_STATS_COMPLETENESS"] = "0.0"
        selective_model_path = ROOT / "tests" / "_missing_selective_model.pkl"

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_value_scan.py"),
                    "--input",
                    str(odds_path),
                    "--fighter-stats",
                    str(stats_path),
                    "--fight-report",
                    str(fight_report_path),
                    "--output",
                    str(output_path),
                    "--selective-model",
                    str(selective_model_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            report = pd.read_csv(output_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            fight_report_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            (ROOT / "tests" / "_tmp_expression_value_scan_output_shortlist.csv").unlink(missing_ok=True)

        self.assertIn("chosen_value_expression", report.columns)
        self.assertIn("expression_pick_source", report.columns)
        self.assertIn("chosen_expression_odds", report.columns)
        self.assertIn("expression_rank_score", report.columns)
        self.assertIn("speculative_prop_expression", report.columns)
        self.assertIn("speculative_prop_fair_american_odds", report.columns)
        alpha_row = report.loc[report["selection_name"] == "Alpha"].iloc[0]
        self.assertEqual(alpha_row["chosen_value_expression"], "Fight doesn't go to decision")
        self.assertEqual(int(alpha_row["chosen_expression_odds"]), -150)
        self.assertEqual(alpha_row["expression_pick_source"], "alternative_market")
        self.assertEqual(alpha_row["speculative_prop_expression"], "Alpha inside distance")

        normalized = pd.DataFrame(
            [
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "selection_name": "Alpha",
                    "american_odds": -110,
                    "implied_prob": 0.5238,
                    "model_projected_win_prob": 0.58,
                    "edge": 0.0562,
                },
                {
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "selection_name": "Beta",
                    "american_odds": 100,
                    "implied_prob": 0.5000,
                    "model_projected_win_prob": 0.42,
                    "edge": -0.0800,
                },
            ]
        )
        manual_fight_report_path = ROOT / "tests" / "_tmp_expression_fight_report_manual.csv"
        fight_report.to_csv(manual_fight_report_path, index=False)
        try:
            overrides = _apply_expression_overrides(normalized, str(manual_fight_report_path), "model_projected_win_prob")
        finally:
            manual_fight_report_path.unlink(missing_ok=True)
        beta_row = overrides.loc[overrides["selection_name"] == "Beta"].iloc[0]
        self.assertEqual(beta_row["chosen_value_expression"], "Beta")
        self.assertEqual(beta_row["expression_pick_source"], "side_market")

    def test_value_scan_surfaces_fight_week_context_risks(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e4",
                    "event_name": "Fight Week Event",
                    "start_time": "2026-04-19T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 145,
                },
                {
                    "event_id": "e4",
                    "event_name": "Fight Week Event",
                    "start_time": "2026-04-19T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "book": "Book",
                    "american_odds": -165,
                },
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 14,
                    "losses": 3,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 5.5,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.1,
                    "takedown_defense_pct": 78,
                    "stats_completeness": 1.0,
                    "injury_concern_flag": 1,
                    "weight_cut_concern_flag": 1,
                    "replacement_fighter_flag": 1,
                    "travel_disadvantage_flag": 1,
                    "camp_change_flag": 1,
                    "gym_name": "Kill Cliff FC",
                    "gym_tier": "A",
                    "gym_record": "40-10-0",
                },
                {
                    "fighter_name": "Beta",
                    "wins": 9,
                    "losses": 5,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 4.1,
                    "sig_strikes_absorbed_per_min": 4.4,
                    "takedown_avg": 0.3,
                    "takedown_defense_pct": 61,
                    "stats_completeness": 1.0,
                },
            ]
        )

        odds_path = ROOT / "tests" / "_tmp_context_value_scan_odds.csv"
        stats_path = ROOT / "tests" / "_tmp_context_value_scan_stats.csv"
        output_path = ROOT / "tests" / "_tmp_context_value_scan_output.csv"
        passes_path = ROOT / "tests" / "_tmp_context_value_scan_output_passes.csv"
        odds.to_csv(odds_path, index=False)
        stats.to_csv(stats_path, index=False)

        env = os.environ.copy()
        env["MIN_EDGE"] = "0.0"
        env["MIN_MODEL_CONFIDENCE"] = "0.0"
        env["MIN_STATS_COMPLETENESS"] = "0.0"
        selective_model_path = ROOT / "tests" / "_missing_selective_model.pkl"

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_value_scan.py"),
                    "--input",
                    str(odds_path),
                    "--fighter-stats",
                    str(stats_path),
                    "--passes-output",
                    str(passes_path),
                    "--output",
                    str(output_path),
                    "--selective-model",
                    str(selective_model_path),
                ],
                check=True,
                env=env,
                capture_output=True,
                text=True,
            )
            passes = pd.read_csv(passes_path)
        finally:
            odds_path.unlink(missing_ok=True)
            stats_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            passes_path.unlink(missing_ok=True)
            (ROOT / "tests" / "_tmp_context_value_scan_output_shortlist.csv").unlink(missing_ok=True)

        alpha_row = passes.loc[passes["selection_name"] == "Alpha"].iloc[0]
        risk_flags = str(alpha_row["risk_flags"])
        self.assertIn("injury_concern", risk_flags)
        self.assertIn("weight_cut_concern", risk_flags)
        self.assertIn("late_replacement", risk_flags)
        self.assertIn("travel_disadvantage", risk_flags)
        self.assertIn("camp_change", risk_flags)
        self.assertIn("Kill Cliff FC", str(alpha_row["context_notes"]))
        self.assertIn("A-tier", str(alpha_row["context_notes"]))


if __name__ == "__main__":
    unittest.main()
