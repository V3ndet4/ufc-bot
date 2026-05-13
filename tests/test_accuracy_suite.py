from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.accuracy import (
    build_calibration_report,
    build_current_quality_report,
    build_market_accuracy_report,
    build_prediction_snapshot,
    build_prop_odds_archive_report,
    build_prop_model_backtest_predictions,
    build_prop_model_calibration_report,
    build_prop_model_market_report,
    build_prop_threshold_report,
    build_quality_gate_report,
    build_segment_performance_report,
    build_style_matchup_diagnostics,
    normalize_tracked_pick_predictions,
)


def test_build_prediction_snapshot_from_no_odds_packet_enriches_segments() -> None:
    manifest = {
        "event_id": "e1",
        "event_name": "Event 1",
        "start_time": "2026-05-09T21:00:00-04:00",
    }
    stats = pd.DataFrame(
        [
            {
                "fighter_name": "Alpha",
                "weight_class": "Heavyweight",
                "ufc_fight_count": 2,
                "stats_completeness": 0.9,
                "camp_change_flag": 1,
            },
            {
                "fighter_name": "Beta",
                "weight_class": "Heavyweight",
                "ufc_fight_count": 5,
                "stats_completeness": 1.0,
                "camp_change_flag": 0,
            },
        ]
    )
    packet = pd.DataFrame(
        [
            {
                "event_name": "Event 1",
                "fight": "Alpha vs Beta",
                "lean_side": "Alpha",
                "opponent": "Beta",
                "model_prob": 0.68,
                "confidence": 0.74,
                "data_quality": 0.9,
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "scheduled_rounds": 5,
                "risk_flags": "news watch",
            }
        ]
    )

    snapshot = build_prediction_snapshot(
        manifest=manifest,
        fighter_stats=stats,
        no_odds_packet=packet,
        snapshot_at="2026-05-03T00:00:00+00:00",
    )

    assert len(snapshot) == 1
    assert snapshot.loc[0, "snapshot_source"] == "no_odds_prediction_packet"
    assert snapshot.loc[0, "prediction_mode"] == "model_only"
    assert snapshot.loc[0, "thin_sample_flag"] == 1
    assert snapshot.loc[0, "camp_change_flag"] == 1
    assert snapshot.loc[0, "heavyweight_flag"] == 1
    assert "five_round" in snapshot.loc[0, "segment_label"]


def test_calibration_segments_and_current_gates() -> None:
    tracked = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "selection_name": "Alpha",
                "actual_result": "win",
                "model_projected_win_prob": 0.7,
                "model_confidence": 0.8,
                "data_quality": 0.95,
                "segment_label": "standard",
                "american_odds": -150,
                "market_blend_weight": 0.1,
                "risk_flags": "",
            },
            {
                "event_id": "e2",
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "selection_name": "Gamma",
                "actual_result": "loss",
                "model_projected_win_prob": 0.72,
                "model_confidence": 0.81,
                "data_quality": 0.9,
                "segment_label": "standard",
                "american_odds": -180,
                "market_blend_weight": 0.1,
                "risk_flags": "market disagreement",
            },
        ]
    )

    predictions = normalize_tracked_pick_predictions(tracked)
    calibration = build_calibration_report(predictions)
    segment = build_segment_performance_report(predictions)
    gates = build_quality_gate_report(segment, min_samples=2)
    current = pd.DataFrame(
        [
            {
                "fight": "Alpha vs Beta",
                "lean_side": "Alpha",
                "model_prob": 0.72,
                "confidence": 0.8,
                "data_quality": 0.95,
                "segment_label": "standard",
                "thin_sample_flag": 0,
                "camp_change_flag": 0,
                "market_disagreement": 0.01,
            }
        ]
    )
    current_quality = build_current_quality_report(current, gates)

    assert not calibration.empty
    assert not segment.empty
    market_accuracy = build_market_accuracy_report(predictions)
    assert set(market_accuracy["market"].astype(str)) >= {"all", "unknown"}
    assert "roi_pct" in market_accuracy.columns
    assert not gates.empty
    assert current_quality.loc[0, "gate_action"] in {"trust", "downgrade_confidence", "block_strong_leans", "expand"}
    assert float(current_quality.loc[0, "gated_confidence"]) > 0


def test_style_matchup_diagnostics_uses_lean_board_edges() -> None:
    snapshot = pd.DataFrame([{"fight": "Alpha vs Beta", "lean_side": "Alpha"}])
    lean_board = pd.DataFrame(
        [
            {
                "fight": "Alpha vs Beta",
                "pick_style": "Control grappler",
                "matchup_striking_edge": 0.2,
                "matchup_grappling_edge": 0.8,
                "matchup_control_edge": 1.2,
            }
        ]
    )

    diagnostics = build_style_matchup_diagnostics(snapshot, lean_board=lean_board)

    assert diagnostics.loc[0, "style_summary"] == "Control grappler"
    assert float(diagnostics.loc[0, "grappling_edge"]) == 0.8


def test_prop_model_backtest_reports_out_of_sample_accuracy() -> None:
    history = pd.DataFrame([_prop_history_row(index) for index in range(80)])

    backtest = build_prop_model_backtest_predictions(
        history,
        holdout_fraction=0.25,
        min_train_samples=30,
    )
    market_report = build_prop_model_market_report(backtest)
    calibration = build_prop_model_calibration_report(backtest)
    thresholds = build_prop_threshold_report(backtest, min_samples=3)

    assert not backtest.empty
    assert set(backtest["market"].astype(str)) == {"knockdown", "takedown"}
    assert set(market_report["market"].astype(str)) >= {"all", "knockdown", "takedown"}
    assert not calibration.empty
    assert not thresholds.empty
    assert "threshold_action" in thresholds.columns


def test_prop_odds_archive_report_keeps_open_current_and_closing_candidate() -> None:
    snapshots = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "event_name": "Event",
                "start_time": "2026-05-02T20:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "market": "takedown",
                "selection": "fighter_a",
                "selection_name": "Alpha takedown",
                "book": "fanduel",
                "american_odds": -130,
                "snapshot_time": "2026-05-01T10:00:00Z",
            },
            {
                "event_id": "e1",
                "event_name": "Event",
                "start_time": "2026-05-02T20:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "market": "takedown",
                "selection": "fighter_a",
                "selection_name": "Alpha takedown",
                "book": "fanduel",
                "american_odds": -150,
                "snapshot_time": "2026-05-02T19:00:00Z",
            },
            {
                "event_id": "e1",
                "event_name": "Event",
                "start_time": "2026-05-02T20:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "market": "takedown",
                "selection": "fighter_a",
                "selection_name": "Alpha takedown",
                "book": "fanduel",
                "american_odds": -170,
                "snapshot_time": "2026-05-02T21:00:00Z",
            },
        ]
    )

    report = build_prop_odds_archive_report(snapshots)

    assert len(report) == 1
    assert int(report.loc[0, "open_american_odds"]) == -130
    assert int(report.loc[0, "current_american_odds"]) == -170
    assert int(report.loc[0, "closing_candidate_american_odds"]) == -150


def _prop_history_row(index: int) -> dict[str, object]:
    takedown_hit = int(index % 2 == 0)
    knockdown_hit = int(index % 3 == 0)
    return {
        "event": f"Event {index // 2}",
        "bout": f"Alpha {index} vs Beta {index}",
        "date": f"2025-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}",
        "fighter_key": f"alpha {index}",
        "opponent_key": f"beta {index}",
        "scheduled_rounds": 3,
        "selection_ufc_fight_count": 5 + (index % 6),
        "opponent_ufc_fight_count": 5,
        "selection_takedown_avg": 2.5 if takedown_hit else 0.1,
        "selection_takedown_accuracy_pct": 60 if takedown_hit else 25,
        "opponent_takedown_defense_pct": 48 if takedown_hit else 82,
        "selection_recent_grappling_rate": 1.5 if takedown_hit else 0.0,
        "selection_control_avg": 3.0 if takedown_hit else 0.1,
        "selection_recent_control_avg": 2.8 if takedown_hit else 0.0,
        "selection_matchup_grappling_edge": 2.0 if takedown_hit else -1.0,
        "selection_knockdown_avg": 0.5 if knockdown_hit else 0.0,
        "selection_ko_win_rate": 0.35 if knockdown_hit else 0.05,
        "opponent_ko_loss_rate": 0.25 if knockdown_hit else 0.0,
        "selection_sig_strikes_landed_per_min": 5.0 if knockdown_hit else 2.0,
        "opponent_sig_strikes_absorbed_per_min": 5.0 if knockdown_hit else 2.0,
        "selection_distance_strike_share": 0.7,
        "selection_clinch_strike_share": 0.1,
        "selection_ground_strike_share": 0.05,
        "takedown_1plus_target": takedown_hit,
        "knockdown_1plus_target": knockdown_hit,
    }
