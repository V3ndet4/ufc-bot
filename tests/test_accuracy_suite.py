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
    build_prediction_snapshot,
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
