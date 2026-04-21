import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.fighter_features import build_fight_features
from models.projection import project_fight_probabilities


class ProjectionTests(unittest.TestCase):
    def test_projection_columns_are_created(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                },
            ]
        )
        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)
        self.assertIn("model_projected_win_prob", projected.columns)
        self.assertIn("projected_finish_prob", projected.columns)
        self.assertIn("projected_decision_prob", projected.columns)
        self.assertIn("baseline_raw_fighter_a_win_prob", projected.columns)
        self.assertIn("heuristic_model_confidence", projected.columns)
        self.assertIn("matchup_striking_edge", features.columns)
        self.assertIn("matchup_grappling_edge", features.columns)
        self.assertIn("matchup_control_edge", features.columns)
        self.assertIn("strike_margin_last_3_diff", features.columns)
        self.assertIn("opponent_quality_diff", features.columns)
        self.assertIn("normalized_strike_margin_diff", features.columns)
        self.assertIn("recency_weighted_strike_margin_diff", features.columns)
        self.assertIn("strike_round_trend_diff", features.columns)
        self.assertIn("fighter_a_submission_prob", projected.columns)
        self.assertIn("fighter_a_ko_tko_prob", projected.columns)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_recency_weighted_and_round_trend_features_improve_projection(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Trend Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "scheduled_rounds": 5,
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 4.6,
                    "sig_strikes_absorbed_per_min": 3.7,
                    "takedown_avg": 1.4,
                    "takedown_defense_pct": 76,
                    "recency_weighted_strike_margin": 1.15,
                    "recency_weighted_grappling_rate": 2.2,
                    "recency_weighted_control_avg": 3.9,
                    "recency_weighted_strike_pace": 4.8,
                    "recency_weighted_result_score": 0.62,
                    "recency_weighted_finish_win_rate": 0.42,
                    "recency_weighted_finish_loss_rate": 0.10,
                    "strike_round_trend": 0.35,
                    "grappling_round_trend": 1.2,
                    "control_round_trend": 1.8,
                    "strike_pace_round_trend": 0.4,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 4.6,
                    "sig_strikes_absorbed_per_min": 3.7,
                    "takedown_avg": 1.4,
                    "takedown_defense_pct": 76,
                    "recency_weighted_strike_margin": -0.10,
                    "recency_weighted_grappling_rate": 0.8,
                    "recency_weighted_control_avg": 0.9,
                    "recency_weighted_strike_pace": 3.7,
                    "recency_weighted_result_score": -0.25,
                    "recency_weighted_finish_win_rate": 0.18,
                    "recency_weighted_finish_loss_rate": 0.32,
                    "strike_round_trend": -0.55,
                    "grappling_round_trend": -0.6,
                    "control_round_trend": -1.1,
                    "strike_pace_round_trend": -0.5,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "recency_weighted_strike_margin_diff"], 0.0)
        self.assertGreater(features.loc[0, "recency_weighted_grappling_rate_diff"], 0.0)
        self.assertGreater(features.loc[0, "recency_weighted_control_diff"], 0.0)
        self.assertGreater(features.loc[0, "recency_weighted_result_score_diff"], 0.0)
        self.assertGreater(features.loc[0, "recency_weighted_durability_diff"], 0.0)
        self.assertGreater(features.loc[0, "strike_round_trend_diff"], 0.0)
        self.assertGreater(features.loc[0, "control_round_trend_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_longer_layoff_is_now_scored_as_an_advantage(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "days_since_last_fight": 320,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "days_since_last_fight": 45,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "layoff_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_cardio_fade_flag_penalizes_full_fight_projection(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "cardio_fade_flag": 1,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "cardio_fade_flag": 0,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertLess(features.loc[0, "cardio_fade_diff"], 0.0)
        self.assertLess(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_gym_score_edge_improves_projection(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "gym_score": 0.82,
                    "gym_fighter_count": 24,
                    "gym_tier": "S",
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "gym_score": 0.50,
                    "gym_fighter_count": 4,
                    "gym_tier": "C",
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "gym_score_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_opponent_strength_normalization_rewards_proven_schedule(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "recent_result_score": 0.4,
                    "opponent_quality_score": 0.78,
                    "recent_opponent_quality_score": 0.80,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "recent_result_score": 0.4,
                    "opponent_quality_score": 0.34,
                    "recent_opponent_quality_score": 0.30,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "opponent_quality_diff"], 0.0)
        self.assertGreater(features.loc[0, "schedule_strength_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_five_round_overlay_rewards_cardio_edge(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Main Event",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "scheduled_rounds": 5,
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 5.4,
                    "sig_strikes_absorbed_per_min": 3.1,
                    "takedown_avg": 1.2,
                    "takedown_defense_pct": 78,
                    "cardio_fade_flag": 0,
                    "ufc_fight_count": 9,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 5.4,
                    "sig_strikes_absorbed_per_min": 3.1,
                    "takedown_avg": 1.2,
                    "takedown_defense_pct": 78,
                    "cardio_fade_flag": 1,
                    "ufc_fight_count": 3,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertEqual(float(features.loc[0, "is_five_round_fight"]), 1.0)
        self.assertIn("five_round", str(features.loc[0, "segment_label"]))
        self.assertGreater(projected.loc[0, "segment_projection_overlay"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_recent_ko_damage_and_stance_edge_support_projection(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Damage Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "sig_strikes_landed_per_min": 4.8,
                    "sig_strikes_absorbed_per_min": 3.7,
                    "takedown_avg": 1.1,
                    "takedown_defense_pct": 74,
                    "stance": "Southpaw",
                    "recent_damage_score": 0.0,
                    "recent_ko_loss_365d": 0,
                    "recent_finish_loss_365d": 0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.8,
                    "sig_strikes_absorbed_per_min": 3.7,
                    "takedown_avg": 1.1,
                    "takedown_defense_pct": 74,
                    "stance": "Orthodox",
                    "recent_damage_score": 1.8,
                    "recent_ko_loss_365d": 1,
                    "recent_finish_loss_365d": 2,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "stance_matchup_diff"], 0.0)
        self.assertGreater(features.loc[0, "recent_ko_damage_diff"], 0.0)
        self.assertGreater(features.loc[0, "recent_damage_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_short_notice_acceptance_only_helps_in_actual_short_notice_spot(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "short_notice_flag": 1,
                    "short_notice_acceptance_flag": 1,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "short_notice_flag": 0,
                    "short_notice_acceptance_flag": 1,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "short_notice_readiness_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_short_notice_success_only_helps_in_actual_short_notice_spot(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "short_notice_flag": 1,
                    "short_notice_success_flag": 1,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "short_notice_flag": 0,
                    "short_notice_success_flag": 1,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "short_notice_success_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_first_round_finish_rate_now_contributes_to_projection(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "first_round_finish_rate": 0.45,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "first_round_finish_rate": 0.05,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "first_round_finish_rate_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)
        self.assertGreater(projected.loc[0, "projected_finish_prob"], 0.5)

    def test_recent_control_edge_now_contributes_to_projection(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Control Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "sig_strikes_landed_per_min": 4.5,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.8,
                    "takedown_defense_pct": 75,
                    "submission_avg": 0.8,
                    "control_avg": 5.0,
                    "recent_control_avg": 4.0,
                    "recent_grappling_rate": 2.0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.5,
                    "sig_strikes_absorbed_per_min": 3.2,
                    "takedown_avg": 1.8,
                    "takedown_defense_pct": 75,
                    "submission_avg": 0.8,
                    "control_avg": 1.0,
                    "recent_control_avg": 0.5,
                    "recent_grappling_rate": 1.0,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(features.loc[0, "control_diff"], 0.0)
        self.assertGreater(features.loc[0, "recent_control_diff"], 0.0)
        self.assertGreater(features.loc[0, "grappling_pressure_diff"], 0.0)
        self.assertGreater(projected.loc[0, "model_projected_win_prob"], 0.5)

    def test_high_combined_control_lowers_finish_projection(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Control Finish Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        baseline_stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 71,
                    "reach_in": 73,
                    "sig_strikes_landed_per_min": 4.9,
                    "sig_strikes_absorbed_per_min": 3.6,
                    "takedown_avg": 1.3,
                    "takedown_defense_pct": 74,
                    "finish_win_rate": 0.45,
                    "finish_loss_rate": 0.22,
                    "decision_rate": 0.33,
                    "first_round_finish_rate": 0.18,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 12,
                    "losses": 3,
                    "height_in": 71,
                    "reach_in": 73,
                    "sig_strikes_landed_per_min": 4.9,
                    "sig_strikes_absorbed_per_min": 3.6,
                    "takedown_avg": 1.3,
                    "takedown_defense_pct": 74,
                    "finish_win_rate": 0.45,
                    "finish_loss_rate": 0.22,
                    "decision_rate": 0.33,
                    "first_round_finish_rate": 0.18,
                },
            ]
        )
        control_heavy_stats = baseline_stats.copy()
        control_heavy_stats["control_avg"] = [5.0, 4.8]
        control_heavy_stats["recent_control_avg"] = [4.2, 4.0]

        baseline_projected = project_fight_probabilities(build_fight_features(odds, baseline_stats))
        control_projected = project_fight_probabilities(build_fight_features(odds, control_heavy_stats))

        self.assertGreater(control_projected.loc[0, "combined_control_avg"], 0.0)
        self.assertLess(
            control_projected.loc[0, "projected_finish_prob"],
            baseline_projected.loc[0, "projected_finish_prob"],
        )

    def test_projection_handles_empty_feature_frame(self) -> None:
        odds = pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "start_time",
                "fighter_a",
                "fighter_b",
                "market",
                "selection",
                "book",
                "american_odds",
                "selection_name",
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
                }
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertTrue(projected.empty)
        self.assertIn("model_projected_win_prob", projected.columns)

    def test_method_split_generates_submission_and_ko_probabilities(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 12,
                    "losses": 2,
                    "height_in": 72,
                    "reach_in": 74,
                    "sig_strikes_landed_per_min": 3.8,
                    "sig_strikes_absorbed_per_min": 2.7,
                    "takedown_avg": 2.6,
                    "takedown_defense_pct": 80,
                    "submission_avg": 1.2,
                    "submission_win_rate": 0.30,
                    "ko_win_rate": 0.05,
                    "submission_loss_rate": 0.05,
                    "ko_loss_rate": 0.05,
                    "finish_win_rate": 0.35,
                    "finish_loss_rate": 0.10,
                    "first_round_finish_rate": 0.10,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 11,
                    "losses": 4,
                    "height_in": 75,
                    "reach_in": 78,
                    "sig_strikes_landed_per_min": 5.6,
                    "sig_strikes_absorbed_per_min": 4.5,
                    "takedown_avg": 0.6,
                    "takedown_defense_pct": 61,
                    "submission_avg": 0.1,
                    "submission_win_rate": 0.05,
                    "ko_win_rate": 0.22,
                    "submission_loss_rate": 0.18,
                    "ko_loss_rate": 0.20,
                    "finish_win_rate": 0.30,
                    "finish_loss_rate": 0.38,
                    "first_round_finish_rate": 0.18,
                },
            ]
        )

        features = build_fight_features(odds, stats)
        projected = project_fight_probabilities(features)

        self.assertGreater(projected.loc[0, "fighter_a_submission_prob"], 0.0)
        self.assertGreater(projected.loc[0, "fighter_a_ko_tko_prob"], 0.0)
        self.assertAlmostEqual(
            float(projected.loc[0, "fighter_a_submission_prob"] + projected.loc[0, "fighter_a_ko_tko_prob"]),
            float(projected.loc[0, "fighter_a_inside_distance_prob"]),
            places=6,
        )

    def test_missing_numeric_espn_fields_do_not_break_feature_build(self) -> None:
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
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": pd.NA,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "stats_completeness": 0.55,
                    "fallback_used": 1,
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
                },
            ]
        )

        features = build_fight_features(odds, stats)

        self.assertEqual(float(features.loc[0, "a_reach_in"]), 0.0)
        self.assertEqual(float(features.loc[0, "fallback_penalty"]), 1.0)

    def test_model_confidence_is_not_maxed_out_by_clean_coin_flip_data(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Balanced Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
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
                    "stats_completeness": 1.0,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 5.0,
                    "sig_strikes_absorbed_per_min": 3.0,
                    "takedown_avg": 1.5,
                    "takedown_defense_pct": 75,
                    "stats_completeness": 1.0,
                },
            ]
        )

        projected = project_fight_probabilities(build_fight_features(odds, stats))

        self.assertGreater(float(projected.loc[0, "model_confidence"]), 0.55)
        self.assertLess(float(projected.loc[0, "model_confidence"]), 0.75)

    def test_model_confidence_drops_with_instability_and_debut_flags(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Confidence Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": -110,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stable_stats = pd.DataFrame(
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
                    "stats_completeness": 1.0,
                    "ufc_fight_count": 7,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.3,
                    "sig_strikes_absorbed_per_min": 3.8,
                    "takedown_avg": 1.1,
                    "takedown_defense_pct": 72,
                    "stats_completeness": 1.0,
                    "ufc_fight_count": 6,
                },
            ]
        )
        unstable_stats = pd.DataFrame(
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
                    "stats_completeness": 1.0,
                    "ufc_fight_count": 0,
                    "ufc_debut_flag": 1,
                    "short_notice_flag": 1,
                    "injury_concern_flag": 1,
                },
                {
                    "fighter_name": "Beta",
                    "wins": 10,
                    "losses": 2,
                    "height_in": 70,
                    "reach_in": 72,
                    "sig_strikes_landed_per_min": 4.3,
                    "sig_strikes_absorbed_per_min": 3.8,
                    "takedown_avg": 1.1,
                    "takedown_defense_pct": 72,
                    "stats_completeness": 1.0,
                    "ufc_fight_count": 0,
                    "ufc_debut_flag": 1,
                    "weight_cut_concern_flag": 1,
                },
            ]
        )

        stable_projected = project_fight_probabilities(build_fight_features(odds, stable_stats))
        unstable_projected = project_fight_probabilities(build_fight_features(odds, unstable_stats))

        self.assertGreater(
            float(stable_projected.loc[0, "model_confidence"]),
            float(unstable_projected.loc[0, "model_confidence"]),
        )

    def test_sparse_matchup_does_not_produce_nan_projection_or_confidence(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Sparse Test",
                    "start_time": "2026-03-21T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Debutant",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "book": "Book",
                    "american_odds": 115,
                    "projected_win_prob": 0.5,
                    "selection_name": "Alpha",
                }
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 20,
                    "losses": 12,
                    "height_in": 71,
                    "reach_in": 76,
                    "age_years": 40.4,
                    "sig_strikes_landed_per_min": 2.405,
                    "sig_strikes_absorbed_per_min": 6.186,
                    "takedown_avg": 0.834,
                    "takedown_defense_pct": 23.53,
                    "recent_grappling_rate": 0.358,
                    "control_avg": 2.381,
                    "recent_control_avg": 1.443,
                    "days_since_last_fight": 282,
                    "ufc_fight_count": 5,
                    "stats_completeness": 1.0,
                    "stance": "Orthodox",
                    "weight_class": "Welterweight",
                },
                {
                    "fighter_name": "Debutant",
                    "wins": 0,
                    "losses": 0,
                    "height_in": 69,
                    "reach_in": 71.5,
                    "age_years": 32.2,
                    "sig_strikes_landed_per_min": 1.499,
                    "sig_strikes_absorbed_per_min": 4.446,
                    "takedown_avg": 0.0,
                    "takedown_defense_pct": 0.0,
                    "recent_grappling_rate": 0.0,
                    "control_avg": 0.0,
                    "recent_control_avg": 0.0,
                    "days_since_last_fight": 188,
                    "ufc_fight_count": 0,
                    "ufc_debut_flag": 1,
                    "stats_completeness": 0.4,
                    "stance": "Orthodox",
                    "weight_class": "Welterweight",
                },
            ]
        )

        projected = project_fight_probabilities(build_fight_features(odds, stats))

        self.assertFalse(pd.isna(projected.loc[0, "projection_score"]))
        self.assertFalse(pd.isna(projected.loc[0, "model_confidence"]))
        self.assertFalse(pd.isna(projected.loc[0, "projected_fighter_a_win_prob"]))
        self.assertGreaterEqual(float(projected.loc[0, "model_confidence"]), 0.35)
        self.assertLessEqual(float(projected.loc[0, "model_confidence"]), 0.92)


if __name__ == "__main__":
    unittest.main()
