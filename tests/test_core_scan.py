import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.fighter_features import build_fight_features
from models.projection import project_fight_probabilities
from normalization.odds import normalize_odds_frame
from scripts.run_core_scan import (
    _validate_core_input_files,
    build_core_board,
    build_core_parlays,
    build_core_props,
    format_direct_betting_instructions,
)


class CoreScanTests(unittest.TestCase):
    def test_validate_core_input_files_explains_missing_odds_for_no_odds_workflow(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            _validate_core_input_files(
                Path("missing_odds.csv"),
                Path("missing_stats.csv"),
                manifest_path="events/example.json",
            )

        message = str(raised.exception)
        self.assertIn("No odds CSV found: missing_odds.csv", message)
        self.assertIn("run_core_card.ps1 requires live moneyline odds", message)
        self.assertIn("scripts\\print_card_preview.py --manifest events/example.json", message)

    def test_core_board_outputs_one_moneyline_decision_per_fight(self) -> None:
        odds = normalize_odds_frame(
            pd.DataFrame(
                [
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "fanduel",
                        "american_odds": 150,
                    },
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "fanduel",
                        "american_odds": -170,
                    },
                ]
            )
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 14,
                    "losses": 2,
                    "height_in": 72,
                    "reach_in": 76,
                    "sig_strikes_landed_per_min": 5.8,
                    "sig_strikes_absorbed_per_min": 2.8,
                    "takedown_avg": 2.2,
                    "takedown_defense_pct": 82,
                    "recent_result_score": 1.0,
                    "recent_strike_margin_per_min": 2.0,
                    "recent_grappling_rate": 1.5,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 5,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.1,
                    "sig_strikes_absorbed_per_min": 4.4,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 58,
                    "recent_result_score": -1.0,
                    "recent_strike_margin_per_min": -1.2,
                    "recent_grappling_rate": 0.2,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
            ]
        )

        board = build_core_board(
            odds,
            stats,
            min_edge=0.0,
            min_confidence=0.0,
            min_stats_completeness=0.0,
        )

        self.assertEqual(len(board), 1)
        self.assertEqual(str(board.loc[0, "fight"]), "Alpha vs Beta")
        self.assertEqual(str(board.loc[0, "pick"]), "Alpha")
        self.assertIn(str(board.loc[0, "decision"]), {"BET", "PASS"})
        self.assertIn("no_bet_reason", board.columns)
        self.assertIn("top_reasons", board.columns)
        self.assertIn("watch_for", board.columns)
        self.assertIn("camp_summary", board.columns)
        self.assertLessEqual(
            abs(float(board.loc[0, "model_prob"]) - float(board.loc[0, "market_prob"])),
            float(board.loc[0, "anchor_cap"]) + 0.0001,
        )

    def test_core_board_uses_cached_lean_explanations_when_available(self) -> None:
        odds = normalize_odds_frame(
            pd.DataFrame(
                [
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "fanduel",
                        "american_odds": 150,
                    },
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "fanduel",
                        "american_odds": -170,
                    },
                ]
            )
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 14,
                    "losses": 2,
                    "height_in": 72,
                    "reach_in": 76,
                    "sig_strikes_landed_per_min": 5.8,
                    "sig_strikes_absorbed_per_min": 2.8,
                    "takedown_avg": 2.2,
                    "takedown_defense_pct": 82,
                    "recent_result_score": 1.0,
                    "recent_strike_margin_per_min": 2.0,
                    "recent_grappling_rate": 1.5,
                    "gym_name": "Alpha Gym",
                    "gym_tier": "A",
                    "gym_record": "10-1-0",
                    "news_radar_label": "amber",
                    "news_radar_summary": "minor injury watch",
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 5,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.1,
                    "sig_strikes_absorbed_per_min": 4.4,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 58,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
            ]
        )
        lean_board = pd.DataFrame(
            [
                {
                    "fight": "Alpha vs Beta",
                    "lean_side": "Alpha",
                    "lean_strength": "Strong Lean",
                    "lean_action": "Bet now",
                    "camp_summary": "Alpha Gym (A-tier) vs Beta unknown camp",
                    "top_reasons": "cached striking edge, cached grappling edge",
                    "risk_flags": "cached risk",
                    "watch_for": "Cached watch note.",
                    "context_summary": "Cached context.",
                }
            ]
        )

        board = build_core_board(
            odds,
            stats,
            lean_board=lean_board,
            min_edge=0.0,
            min_confidence=0.0,
            min_stats_completeness=0.0,
        )

        self.assertEqual(str(board.loc[0, "top_reasons"]), "cached striking edge, cached grappling edge")
        self.assertEqual(str(board.loc[0, "watch_for"]), "Cached watch note.")
        self.assertIn("minor injury watch", str(board.loc[0, "news_summary"]))

    def test_core_props_only_scores_real_prop_prices(self) -> None:
        odds = normalize_odds_frame(
            pd.DataFrame(
                [
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "fanduel",
                        "american_odds": -120,
                    },
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "fanduel",
                        "american_odds": 100,
                    },
                ]
            )
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
                    "takedown_defense_pct": 70,
                    "finish_win_rate": 0.7,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 4,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.0,
                    "sig_strikes_absorbed_per_min": 4.0,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 60,
                    "finish_loss_rate": 0.6,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
            ]
        )
        prop_odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "start_time": "2026-05-02T19:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "fight_doesnt_go_to_decision",
                    "selection": "fight_doesnt_go_to_decision",
                    "selection_name": "Fight doesn't go to decision",
                    "book": "fanduel",
                    "american_odds": 150,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "start_time": "2026-05-02T19:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "inside_distance",
                    "selection": "fighter_a",
                    "selection_name": "Alpha inside distance",
                    "book": "manual",
                    "american_odds": "",
                },
            ]
        )

        scored = project_fight_probabilities(build_fight_features(odds, stats))
        props = build_core_props(
            prop_odds,
            scored,
            min_edge=0.0,
            min_confidence=0.0,
            min_stats_completeness=0.0,
            max_props=5,
        )

        self.assertEqual(len(props), 1)
        self.assertEqual(str(props.loc[0, "prop"]), "Fight doesn't go the distance")

    def test_core_props_scores_expanded_prop_markets_and_main_card_only_specials(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "model_confidence": 0.92,
                    "data_quality": 1.0,
                    "projected_finish_prob": 0.66,
                    "projected_decision_prob": 0.34,
                    "fighter_a_inside_distance_prob": 0.42,
                    "fighter_b_inside_distance_prob": 0.24,
                    "fighter_a_submission_prob": 0.22,
                    "fighter_b_submission_prob": 0.08,
                    "fighter_a_ko_tko_prob": 0.20,
                    "fighter_b_ko_tko_prob": 0.16,
                    "fighter_a_by_decision_prob": 0.18,
                    "fighter_b_by_decision_prob": 0.16,
                    "scheduled_rounds": 3,
                    "a_knockdown_avg": 0.45,
                    "a_ko_win_rate": 0.30,
                    "a_sig_strikes_landed_per_min": 5.0,
                    "a_distance_strike_share": 0.70,
                    "a_takedown_avg": 2.2,
                    "a_recent_grappling_rate": 2.0,
                    "a_control_avg": 2.0,
                    "a_recent_control_avg": 2.2,
                    "b_ko_loss_rate": 0.20,
                    "b_sig_strikes_absorbed_per_min": 4.5,
                    "b_takedown_defense_pct": 58.0,
                    "matchup_grappling_edge": 1.0,
                }
            ]
        )
        prop_odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "submission",
                    "selection": "fighter_a",
                    "book": "fanduel",
                    "american_odds": 450,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "ko_tko",
                    "selection": "fighter_a",
                    "book": "fanduel",
                    "american_odds": 450,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "fight_ends_by_submission",
                    "selection": "fight_ends_by_submission",
                    "book": "fanduel",
                    "american_odds": 350,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "fight_ends_by_ko_tko",
                    "selection": "fight_ends_by_ko_tko",
                    "book": "fanduel",
                    "american_odds": 220,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "knockdown",
                    "selection": "fighter_a",
                    "book": "fanduel",
                    "american_odds": 240,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "takedown",
                    "selection": "fighter_a",
                    "book": "fanduel",
                    "american_odds": -120,
                },
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 0,
                    "market": "knockdown",
                    "selection": "fighter_b",
                    "book": "fanduel",
                    "american_odds": 1000,
                },
            ]
        )

        props = build_core_props(
            prop_odds,
            scored,
            min_edge=0.0,
            min_confidence=0.0,
            min_stats_completeness=0.0,
            max_props=20,
        )

        prop_labels = props["prop"].astype(str).tolist()
        self.assertIn("Alpha by submission", prop_labels)
        self.assertIn("Alpha by KO/TKO", prop_labels)
        self.assertIn("Fight ends by submission", prop_labels)
        self.assertIn("Fight ends by KO/TKO", prop_labels)
        self.assertIn("Alpha knockdown", prop_labels)
        self.assertIn("Alpha takedown", prop_labels)
        self.assertNotIn("Beta knockdown", prop_labels)

    def test_core_props_can_use_trained_td_kd_model(self) -> None:
        odds = normalize_odds_frame(
            pd.DataFrame(
                [
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "fanduel",
                        "american_odds": -130,
                    },
                    {
                        "event_id": "e1",
                        "event_name": "Core Event",
                        "start_time": "2026-05-02T19:00:00Z",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "fanduel",
                        "american_odds": 110,
                    },
                ]
            )
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
                    "takedown_defense_pct": 70,
                    "takedown_accuracy_pct": 55,
                    "recent_grappling_rate": 1.2,
                    "control_avg": 2.0,
                    "recent_control_avg": 2.5,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 4,
                    "height_in": 69,
                    "reach_in": 70,
                    "sig_strikes_landed_per_min": 3.0,
                    "sig_strikes_absorbed_per_min": 4.0,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 60,
                    "stats_completeness": 1.0,
                    "fallback_used": 0,
                },
            ]
        )
        prop_odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Core Event",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "is_main_card": 1,
                    "market": "takedown",
                    "selection": "fighter_a",
                    "book": "fanduel",
                    "american_odds": -120,
                },
            ]
        )
        scored = project_fight_probabilities(build_fight_features(odds, stats))
        props = build_core_props(
            prop_odds,
            scored,
            min_edge=0.0,
            min_confidence=0.0,
            min_stats_completeness=0.0,
            max_props=20,
            prop_model_bundle={"markets": {"takedown": {"pipeline": _FixedPropPipeline(0.95)}}},
        )

        self.assertEqual(len(props), 1)
        self.assertEqual(float(props.loc[0, "model_prob"]), 0.88)

    def test_format_direct_betting_instructions_prints_bet_directly_sections(self) -> None:
        board = pd.DataFrame(
            [
                {
                    "decision": "BET",
                    "pick": "Alpha",
                    "opponent": "Beta",
                    "book": "fanduel",
                    "sportsbook_line": -120,
                    "fair_line": -180,
                    "model_prob": 0.64,
                    "edge": 0.09,
                    "expected_value": 0.14,
                }
            ]
        )
        props = pd.DataFrame(
            [
                {
                    "decision": "BET",
                    "prop": "Fight doesn't go the distance",
                    "book": "fanduel",
                    "fight": "Alpha vs Beta",
                    "is_main_card": 1,
                    "sportsbook_line": 150,
                    "fair_line": 100,
                    "model_prob": 0.50,
                    "edge": 0.10,
                    "expected_value": 0.25,
                }
            ]
        )

        output = format_direct_betting_instructions(board, props, props_scanned=True)

        self.assertIn("BET DIRECTLY", output)
        self.assertIn("BET Alpha at fanduel", output)
        self.assertIn("BET Fight doesn't go the distance at fanduel", output)
        self.assertIn("Fighter knockdowns (main card only)", output)
        self.assertIn("Fighter takedowns (main card only)", output)

    def test_core_parlays_use_only_bet_rows_with_positive_ev(self) -> None:
        core_board = pd.DataFrame(
            [
                {
                    "decision": "BET",
                    "pick": "Alpha",
                    "sportsbook_line": 120,
                    "model_prob": 0.58,
                    "implied_prob": 0.4545,
                },
                {
                    "decision": "BET",
                    "pick": "Gamma",
                    "sportsbook_line": 110,
                    "model_prob": 0.57,
                    "implied_prob": 0.4762,
                },
                {
                    "decision": "PASS",
                    "pick": "Ignored",
                    "sportsbook_line": 100,
                    "model_prob": 0.51,
                    "implied_prob": 0.50,
                },
            ]
        )

        parlays = build_core_parlays(core_board, min_legs=2, max_legs=2, max_parlays=3)

        self.assertEqual(len(parlays), 1)
        self.assertIn("Alpha", str(parlays.loc[0, "parlay"]))
        self.assertIn("Gamma", str(parlays.loc[0, "parlay"]))
        self.assertGreater(float(parlays.loc[0, "expected_value"]), 0.0)

    def test_core_parlays_can_mix_moneylines_and_props_without_same_fight_duplicates(self) -> None:
        legs = pd.DataFrame(
            [
                {
                    "decision": "BET",
                    "pick": "Alpha",
                    "sportsbook_line": -110,
                    "model_prob": 0.58,
                    "implied_prob": 0.5238,
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                },
                {
                    "decision": "BET",
                    "pick": "Alpha by decision",
                    "sportsbook_line": 120,
                    "model_prob": 0.52,
                    "implied_prob": 0.4545,
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                },
                {
                    "decision": "BET",
                    "pick": "Gamma",
                    "sportsbook_line": -125,
                    "model_prob": 0.61,
                    "implied_prob": 0.5556,
                    "fighter_a": "Gamma",
                    "fighter_b": "Delta",
                },
            ]
        )

        parlays = build_core_parlays(legs, min_legs=2, max_legs=2, max_parlays=10)

        self.assertTrue(parlays["legs"].astype(str).str.contains("Alpha by decision").any())
        self.assertFalse(parlays["legs"].astype(str).str.contains("Alpha \\(-110\\).*Alpha by decision", regex=True).any())


class _FixedPropPipeline:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, frame: pd.DataFrame) -> object:
        return pd.DataFrame(
            {
                0: [1.0 - self.probability] * len(frame),
                1: [self.probability] * len(frame),
            }
        ).to_numpy()


if __name__ == "__main__":
    unittest.main()
