from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.advanced_accuracy import (
    build_decision_model_report,
    build_elo_rating_report,
    build_finish_hazard_report,
    build_leakage_audit_report,
    build_market_consensus_report,
    build_model_leaderboard_report,
    build_news_reliability_report,
    build_official_context_report,
    build_prediction_uncertainty_report,
    build_scheduled_snapshot_coverage_report,
    uncertainty_band,
)


def test_market_consensus_finds_best_book_and_price_edge() -> None:
    base = {
        "event_id": "e1",
        "event_name": "Event",
        "start_time": "2026-05-23T20:00:00Z",
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "market": "moneyline",
        "selection": "fighter_a",
        "selection_name": "Alpha",
    }
    report = build_market_consensus_report(
        pd.DataFrame([{**base, "book": "book_a", "american_odds": -120}]),
        pd.DataFrame([{**base, "book": "book_b", "american_odds": +110}]),
    )

    assert len(report) == 1
    assert report.loc[0, "consensus_status"] == "multi_book"
    assert report.loc[0, "best_book"] == "book_b"
    assert report.loc[0, "best_american_odds"] == 110
    assert float(report.loc[0, "price_edge_vs_consensus"]) > 0


def test_scheduled_snapshot_coverage_buckets_line_history() -> None:
    history = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "event_name": "Event",
                "start_time": "2026-05-23T20:00:00Z",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "market": "moneyline",
                "selection": "fighter_a",
                "book": "book_a",
                "american_odds": -120,
                "snapshot_time": "2026-05-23T14:30:00Z",
            }
        ]
    )

    report = build_scheduled_snapshot_coverage_report(history)

    assert report.loc[0, "snapshot_bucket"] == "six_hours"
    assert report.loc[0, "snapshot_rows"] == 1
    assert report.loc[0, "coverage_status"] == "thin"


def test_decision_finish_elo_and_leakage_reports_use_historical_rows() -> None:
    history = pd.DataFrame(
        [
            _history_row("Event 1", "Alpha vs Beta", "2026-01-01", "alpha", "W", 1, 0, 0, 0),
            _history_row("Event 1", "Alpha vs Beta", "2026-01-01", "beta", "L", 1, 0, 0, 0),
            _history_row("Event 2", "Gamma vs Delta", "2026-02-01", "gamma", "L", 0, 1, 1, 0),
            _history_row("Event 2", "Gamma vs Delta", "2026-02-01", "delta", "W", 0, 1, 1, 0),
        ]
    )
    manifest = {"event_id": "future", "start_time": "2026-05-23T20:00:00Z", "fights": []}

    decision = build_decision_model_report(history)
    hazard = build_finish_hazard_report(history)
    elo = build_elo_rating_report(history)
    leakage = build_leakage_audit_report(history, manifest)

    assert decision.loc[decision["segment"].eq("all"), "decision_rate"].iloc[0] == 0.5
    assert hazard.loc[hazard["segment"].eq("all"), "estimated_finish_hazard_per_round"].iloc[0] > 0
    assert float(elo.loc[elo["fighter_key"].eq("alpha"), "elo_rating"].iloc[0]) > 1500
    assert set(leakage["status"].astype(str)) == {"pass"}


def test_prediction_uncertainty_news_officials_and_leaderboard_reports() -> None:
    snapshot = pd.DataFrame(
        [
            {
                "fight": "Alpha vs Beta",
                "lean_side": "Alpha",
                "model_prob": 0.62,
                "confidence": 0.85,
                "data_quality": 0.9,
                "current_american_odds": -105,
            }
        ]
    )
    context = pd.DataFrame(
        [
            {
                "fighter_name": "Alpha",
                "news_alert_count": 2,
                "news_alert_confidence": 0.8,
                "news_high_confidence_alerts": 1,
                "news_primary_category": "injury",
                "news_radar_score": 0.7,
            }
        ]
    )
    manifest = {
        "venue": "Apex",
        "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta", "referee": "Ref A", "judges": "Judge A; Judge B; Judge C"}],
    }
    market_accuracy = pd.DataFrame([{"market": "all", "graded_picks": 40, "brier": 0.22, "log_loss": 0.61, "roi_pct": 4.0}])
    prop_accuracy = pd.DataFrame([{"market": "all", "graded_props": 55, "brier": 0.24, "log_loss": 0.64, "roi_pct": 2.0}])
    tracked_clv = pd.DataFrame([{"market": "all", "graded_picks": 40, "roi_pct": 4.0, "avg_clv_edge": 0.02}])

    uncertainty = build_prediction_uncertainty_report(snapshot)
    news = build_news_reliability_report(context, pd.DataFrame())
    officials = build_official_context_report(manifest, pd.DataFrame())
    leaderboard = build_model_leaderboard_report(market_accuracy, prop_accuracy, prop_accuracy, tracked_clv)

    assert float(uncertainty.loc[0, "uncertainty_low"]) < 0.62
    assert float(uncertainty.loc[0, "worst_case_edge"]) > 0
    assert news.loc[0, "reliability_status"] == "trusted_alerts"
    assert officials.loc[0, "context_status"] == "ready"
    assert set(leaderboard["model_name"].astype(str)) >= {"moneyline_value_model", "tracked_clv_process"}
    assert uncertainty_band(0.6, 1.0, 1.0) == (0.545, 0.655)


def _history_row(
    event: str,
    bout: str,
    date: str,
    fighter_key: str,
    result_code: str,
    decision: int,
    inside: int,
    ko: int,
    submission: int,
) -> dict[str, object]:
    return {
        "event": event,
        "bout": bout,
        "date": date,
        "fighter_key": fighter_key,
        "result_code": result_code,
        "scheduled_rounds": 3,
        "weight_class": "Lightweight",
        "selection_ufc_fight_count": 1,
        "selection_takedown_avg": 0.0,
        "selection_recent_grappling_rate": 0.0,
        "selection_control_avg": 0.0,
        "selection_recent_control_avg": 0.0,
        "selection_knockdown_avg": 0.0,
        "selection_submission_avg": 0.0,
        "selection_submission_win_rate": 0.0,
        "selection_finish_win_rate": 0.0,
        "selection_decision_rate": 0.0,
        "selection_recent_damage_score": 0.0,
        "selection_sig_strikes_absorbed_per_min": 0.0,
        "selection_ko_win_rate": 0.0,
        "selection_sig_strikes_landed_per_min": 0.0,
        "fight_goes_to_decision_target": decision,
        "fight_doesnt_go_to_decision_target": inside,
        "inside_distance_target": inside,
        "fight_ends_by_ko_tko_target": ko,
        "fight_ends_by_submission_target": submission,
    }
