from __future__ import annotations

from datetime import timezone

import pandas as pd

from models.ev import implied_probability
from models.prop_outcomes import prop_market_family


def build_market_consensus_report(*odds_frames: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "fight",
        "market",
        "market_family",
        "selection",
        "selection_name",
        "book_count",
        "books",
        "best_book",
        "best_american_odds",
        "worst_american_odds",
        "median_american_odds",
        "consensus_implied_prob",
        "best_implied_prob",
        "price_edge_vs_consensus",
        "consensus_status",
    ]
    working = _combine_odds_frames(*odds_frames)
    if working.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    group_columns = ["event_id", "fight", "market", "selection", "selection_name"]
    for key, group in working.groupby(group_columns, dropna=False):
        books = sorted(set(group["book"].astype(str)))
        odds = pd.to_numeric(group["american_odds"], errors="coerce").dropna()
        if odds.empty:
            continue
        best_index = odds.idxmax()
        consensus_prob = float(group.loc[odds.index, "implied_prob"].median())
        best_implied = implied_probability(int(group.loc[best_index, "american_odds"]))
        book_count = len(books)
        rows.append(
            {
                "event_id": key[0],
                "fight": key[1],
                "market": key[2],
                "market_family": prop_market_family(key[2]),
                "selection": key[3],
                "selection_name": key[4],
                "book_count": book_count,
                "books": ", ".join(books),
                "best_book": group.loc[best_index, "book"],
                "best_american_odds": int(group.loc[best_index, "american_odds"]),
                "worst_american_odds": int(odds.min()),
                "median_american_odds": round(float(odds.median()), 2),
                "consensus_implied_prob": round(consensus_prob, 4),
                "best_implied_prob": round(best_implied, 4),
                "price_edge_vs_consensus": round(consensus_prob - best_implied, 4),
                "consensus_status": "multi_book" if book_count >= 2 else "single_book",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["market", "fight", "selection_name"]).reset_index(drop=True)


def build_scheduled_snapshot_coverage_report(snapshot_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "market_family",
        "snapshot_bucket",
        "snapshot_rows",
        "events",
        "fights",
        "books",
        "first_snapshot_time",
        "latest_snapshot_time",
        "coverage_status",
    ]
    if snapshot_history.empty:
        return pd.DataFrame(columns=columns)
    required = {"market", "fighter_a", "fighter_b", "book", "snapshot_time", "start_time"}
    if not required.issubset(snapshot_history.columns):
        return pd.DataFrame(columns=columns)
    working = snapshot_history.copy()
    working["snapshot_time"] = pd.to_datetime(working["snapshot_time"], errors="coerce", utc=True)
    working["start_time"] = pd.to_datetime(working["start_time"], errors="coerce", utc=True)
    working = working.dropna(subset=["snapshot_time", "start_time"]).copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["market"] = working["market"].fillna("").astype(str)
    working["fight"] = working["fighter_a"].astype(str) + " vs " + working["fighter_b"].astype(str)
    working["snapshot_bucket"] = working.apply(_snapshot_bucket, axis=1)
    rows: list[dict[str, object]] = []
    for (market, bucket), group in working.groupby(["market", "snapshot_bucket"], dropna=False):
        rows.append(
            {
                "market": str(market),
                "market_family": prop_market_family(market),
                "snapshot_bucket": str(bucket),
                "snapshot_rows": int(len(group)),
                "events": int(group.get("event_id", pd.Series("", index=group.index)).nunique()),
                "fights": int(group["fight"].nunique()),
                "books": int(group["book"].nunique()),
                "first_snapshot_time": _timestamp(group["snapshot_time"].min()),
                "latest_snapshot_time": _timestamp(group["snapshot_time"].max()),
                "coverage_status": _snapshot_coverage_status(group),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["market", "snapshot_bucket"]).reset_index(drop=True)


def build_decision_model_report(prop_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "segment",
        "fights",
        "decision_rate",
        "inside_distance_rate",
        "ko_tko_rate",
        "submission_rate",
        "decision_model_status",
    ]
    fights = _unique_fights(prop_history)
    if fights.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for segment, group in _decision_segments(fights):
        rows.append(_decision_row(segment, group))
    return pd.DataFrame(rows, columns=columns).sort_values("segment").reset_index(drop=True)


def build_elo_rating_report(prop_history: pd.DataFrame, *, initial_rating: float = 1500.0, k_factor: float = 32.0) -> pd.DataFrame:
    columns = [
        "fighter_key",
        "elo_rating",
        "elo_fights",
        "elo_wins",
        "elo_losses",
        "latest_fight_date",
        "rating_status",
    ]
    if prop_history.empty:
        return pd.DataFrame(columns=columns)
    history = prop_history.copy()
    history["date"] = pd.to_datetime(history.get("date"), errors="coerce")
    history = history.dropna(subset=["date"]).sort_values(["date", "event", "bout", "fighter_key"]).copy()
    ratings: dict[str, float] = {}
    records: dict[str, dict[str, object]] = {}
    for (_, bout), group in history.groupby(["event", "bout"], sort=False):
        fighters = group[["fighter_key", "result_code", "date"]].dropna(subset=["fighter_key"]).drop_duplicates("fighter_key")
        if len(fighters) != 2:
            continue
        winners = fighters.loc[fighters["result_code"].astype(str).str.upper().eq("W")]
        losers = fighters.loc[fighters["result_code"].astype(str).str.upper().eq("L")]
        if len(winners) != 1 or len(losers) != 1:
            continue
        winner = str(winners.iloc[0]["fighter_key"])
        loser = str(losers.iloc[0]["fighter_key"])
        winner_rating = ratings.get(winner, initial_rating)
        loser_rating = ratings.get(loser, initial_rating)
        expected_winner = 1.0 / (1.0 + 10 ** ((loser_rating - winner_rating) / 400.0))
        delta = k_factor * (1.0 - expected_winner)
        ratings[winner] = winner_rating + delta
        ratings[loser] = loser_rating - delta
        for fighter, won in [(winner, True), (loser, False)]:
            record = records.setdefault(fighter, {"elo_fights": 0, "elo_wins": 0, "elo_losses": 0, "latest_fight_date": ""})
            record["elo_fights"] = int(record["elo_fights"]) + 1
            record["elo_wins"] = int(record["elo_wins"]) + int(won)
            record["elo_losses"] = int(record["elo_losses"]) + int(not won)
            record["latest_fight_date"] = _date(fighters["date"].max())
    rows = [
        {
            "fighter_key": fighter,
            "elo_rating": round(rating, 2),
            **records.get(fighter, {}),
            "rating_status": "active_baseline" if int(records.get(fighter, {}).get("elo_fights", 0)) >= 3 else "thin_history",
        }
        for fighter, rating in ratings.items()
    ]
    return pd.DataFrame(rows, columns=columns).sort_values("elo_rating", ascending=False).reset_index(drop=True)


def build_finish_hazard_report(prop_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "segment",
        "fights",
        "scheduled_rounds",
        "inside_distance_rate",
        "ko_tko_rate",
        "submission_rate",
        "estimated_finish_hazard_per_round",
        "hazard_status",
    ]
    fights = _unique_fights(prop_history)
    if fights.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for segment, group in _finish_segments(fights):
        rounds = float(pd.to_numeric(group.get("scheduled_rounds", 3), errors="coerce").fillna(3).median())
        inside = float(pd.to_numeric(group.get("fight_doesnt_go_to_decision_target", 0), errors="coerce").fillna(0).mean())
        rows.append(
            {
                "segment": segment,
                "fights": int(len(group)),
                "scheduled_rounds": round(rounds, 2),
                "inside_distance_rate": round(inside, 4),
                "ko_tko_rate": round(float(pd.to_numeric(group.get("fight_ends_by_ko_tko_target", 0), errors="coerce").fillna(0).mean()), 4),
                "submission_rate": round(float(pd.to_numeric(group.get("fight_ends_by_submission_target", 0), errors="coerce").fillna(0).mean()), 4),
                "estimated_finish_hazard_per_round": round(inside / max(rounds, 1.0), 4),
                "hazard_status": "ready" if len(group) >= 100 else "thin_sample",
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("segment").reset_index(drop=True)


def build_leakage_audit_report(prop_history: pd.DataFrame, manifest: dict[str, object], snapshot: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = ["check_name", "status", "problem_count", "details"]
    rows: list[dict[str, object]] = []
    start_time = pd.to_datetime(manifest.get("start_time"), errors="coerce", utc=True)
    history = prop_history.copy() if prop_history is not None else pd.DataFrame()
    if not history.empty and "date" in history.columns:
        history["date"] = pd.to_datetime(history["date"], errors="coerce", utc=True)
    future_rows = history.loc[history["date"].ge(start_time)] if not history.empty and not pd.isna(start_time) else pd.DataFrame()
    rows.append(_audit_row("historical_rows_before_event", len(future_rows) == 0, len(future_rows), "prop history rows must predate active card"))

    prior_columns = [
        column
        for column in [
            "selection_takedown_avg",
            "selection_recent_grappling_rate",
            "selection_control_avg",
            "selection_recent_control_avg",
            "selection_knockdown_avg",
            "selection_submission_avg",
            "selection_submission_win_rate",
            "selection_finish_win_rate",
            "selection_decision_rate",
            "selection_recent_damage_score",
            "selection_sig_strikes_absorbed_per_min",
            "selection_ko_win_rate",
            "selection_sig_strikes_landed_per_min",
        ]
        if column in history.columns
    ]
    debut_rows = history.loc[pd.to_numeric(history.get("selection_ufc_fight_count", 0), errors="coerce").fillna(0).eq(0)] if not history.empty else pd.DataFrame()
    leaked_debuts = 0
    if not debut_rows.empty and prior_columns:
        numeric = debut_rows[prior_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        leaked_debuts = int((numeric.abs().sum(axis=1) > 0.0001).sum())
    rows.append(_audit_row("prior_features_zero_for_debuts", leaked_debuts == 0, leaked_debuts, "debut rows should not contain future prior stats"))

    duplicate_count = 0
    if not history.empty:
        duplicate_count = int(history.duplicated().sum())
    rows.append(_audit_row("no_exact_duplicate_history_rows", duplicate_count == 0, duplicate_count, "exact duplicates can overweight old fights"))

    if snapshot is not None and not snapshot.empty:
        snapshot_rows = len(snapshot)
        low_quality = int((pd.to_numeric(snapshot.get("data_quality", 1), errors="coerce").fillna(1) < 0.70).sum())
        rows.append(_audit_row("active_card_snapshot_quality", low_quality == 0, low_quality, f"{snapshot_rows} active-card prediction rows checked"))
    return pd.DataFrame(rows, columns=columns)


def build_prediction_uncertainty_report(snapshot: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fight",
        "selection",
        "model_prob",
        "confidence",
        "data_quality",
        "uncertainty_low",
        "uncertainty_high",
        "uncertainty_width",
        "worst_case_edge",
        "uncertainty_action",
    ]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in snapshot.to_dict("records"):
        probability = _safe_float(row.get("model_prob", row.get("market_aware_prob", 0.5)), 0.5)
        confidence = _safe_float(row.get("confidence", 0.5), 0.5)
        quality = _safe_float(row.get("data_quality", 1.0), 1.0)
        low, high = uncertainty_band(probability, confidence, quality)
        odds = _safe_float(row.get("current_american_odds", ""), None)
        implied = implied_probability(int(odds)) if odds is not None else None
        worst_edge = round(low - implied, 4) if implied is not None else ""
        rows.append(
            {
                "fight": row.get("fight", ""),
                "selection": row.get("lean_side", row.get("selection_name", "")),
                "model_prob": round(probability, 4),
                "confidence": round(confidence, 4),
                "data_quality": round(quality, 4),
                "uncertainty_low": low,
                "uncertainty_high": high,
                "uncertainty_width": round(high - low, 4),
                "worst_case_edge": worst_edge,
                "uncertainty_action": "bettable_range" if worst_edge != "" and float(worst_edge) > 0 else "price_or_pass",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_news_reliability_report(context: pd.DataFrame, alerts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fighter_name",
        "alert_count",
        "avg_confidence_score",
        "high_confidence_alerts",
        "primary_category",
        "radar_score",
        "reliability_status",
    ]
    rows: list[dict[str, object]] = []
    if context is not None and not context.empty and "fighter_name" in context.columns:
        for row in context.fillna("").to_dict("records"):
            alert_count = int(_safe_float(row.get("news_alert_count", 0), 0))
            avg_confidence = _safe_float(row.get("news_alert_confidence", 0), 0)
            rows.append(
                {
                    "fighter_name": row.get("fighter_name", ""),
                    "alert_count": alert_count,
                    "avg_confidence_score": round(float(avg_confidence), 4),
                    "high_confidence_alerts": int(_safe_float(row.get("news_high_confidence_alerts", 0), 0)),
                    "primary_category": row.get("news_primary_category", ""),
                    "radar_score": round(float(_safe_float(row.get("news_radar_score", 0), 0)), 4),
                    "reliability_status": _news_reliability_status(alert_count, avg_confidence),
                }
            )
    if alerts is not None and not alerts.empty and "fighter_name" in alerts.columns:
        existing = {str(row["fighter_name"]) for row in rows}
        for fighter, group in alerts.groupby(alerts["fighter_name"].astype(str), dropna=False):
            if fighter in existing:
                continue
            avg_confidence = float(pd.to_numeric(group.get("confidence_score", 0), errors="coerce").fillna(0).mean())
            rows.append(
                {
                    "fighter_name": fighter,
                    "alert_count": int(len(group)),
                    "avg_confidence_score": round(avg_confidence, 4),
                    "high_confidence_alerts": int((pd.to_numeric(group.get("confidence_score", 0), errors="coerce").fillna(0) >= 0.75).sum()),
                    "primary_category": str(group.get("alert_category", pd.Series("", index=group.index)).mode().iloc[0]) if not group.empty else "",
                    "radar_score": round(float(pd.to_numeric(group.get("alert_radar_score", 0), errors="coerce").fillna(0).max()), 4),
                    "reliability_status": _news_reliability_status(len(group), avg_confidence),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_official_context_report(manifest: dict[str, object], fight_results: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "fight",
        "venue_known",
        "referee_known",
        "judge_data_known",
        "result_context_known",
        "context_status",
    ]
    rows: list[dict[str, object]] = []
    venue_known = int(bool(str(manifest.get("venue", "") or manifest.get("location", "")).strip()))
    results = fight_results if fight_results is not None else pd.DataFrame()
    for fight in manifest.get("fights", []):
        if not isinstance(fight, dict):
            continue
        fight_text = f"{fight.get('fighter_a', '')} vs {fight.get('fighter_b', '')}"
        result_match = pd.DataFrame()
        if not results.empty and {"fighter_a", "fighter_b"}.issubset(results.columns):
            result_match = results.loc[
                results["fighter_a"].astype(str).eq(str(fight.get("fighter_a", "")))
                & results["fighter_b"].astype(str).eq(str(fight.get("fighter_b", "")))
            ]
        referee_known = int("referee" in fight and bool(str(fight.get("referee", "")).strip()))
        judge_data_known = int(any(key in fight for key in ["judges", "judge_1", "judge1"]))
        result_context_known = int(not result_match.empty)
        rows.append(
            {
                "fight": fight_text,
                "venue_known": venue_known,
                "referee_known": referee_known,
                "judge_data_known": judge_data_known,
                "result_context_known": result_context_known,
                "context_status": "ready" if venue_known and referee_known and judge_data_known else "missing_official_context",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_model_leaderboard_report(
    market_accuracy: pd.DataFrame,
    prop_market_accuracy: pd.DataFrame,
    prop_walk_forward_market_accuracy: pd.DataFrame,
    tracked_clv: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "model_name",
        "scope",
        "graded_samples",
        "brier",
        "log_loss",
        "roi_pct",
        "avg_clv_edge",
        "leaderboard_score",
        "promotion_status",
    ]
    rows: list[dict[str, object]] = []
    rows.extend(_leaderboard_rows(market_accuracy, "moneyline_value_model", "tracked_moneyline"))
    rows.extend(_leaderboard_rows(prop_market_accuracy, "prop_holdout_model", "prop_holdout"))
    rows.extend(_leaderboard_rows(prop_walk_forward_market_accuracy, "prop_walk_forward_model", "prop_walk_forward"))
    if tracked_clv is not None and not tracked_clv.empty:
        all_row = tracked_clv.loc[tracked_clv.get("market", pd.Series("", index=tracked_clv.index)).astype(str).eq("all")]
        if not all_row.empty:
            row = all_row.iloc[0]
            rows.append(
                {
                    "model_name": "tracked_clv_process",
                    "scope": "all_tracked",
                    "graded_samples": int(_safe_float(row.get("graded_picks", 0), 0)),
                    "brier": "",
                    "log_loss": "",
                    "roi_pct": row.get("roi_pct", ""),
                    "avg_clv_edge": row.get("avg_clv_edge", ""),
                    "leaderboard_score": _leaderboard_score("", "", row.get("roi_pct", ""), row.get("avg_clv_edge", "")),
                    "promotion_status": "monitor",
                }
            )
    output = pd.DataFrame(rows, columns=columns)
    if output.empty:
        return output
    output["leaderboard_score"] = pd.to_numeric(output["leaderboard_score"], errors="coerce").fillna(-999.0)
    output["promotion_status"] = output.apply(lambda row: _promotion_status(float(row["leaderboard_score"]), int(_safe_float(row["graded_samples"], 0))), axis=1)
    return output.sort_values("leaderboard_score", ascending=False).reset_index(drop=True)


def uncertainty_band(probability: float, confidence: float, data_quality: float, *, is_prop: bool = False) -> tuple[float, float]:
    base = 0.055 + (0.025 if is_prop else 0.0)
    width = base + ((1.0 - max(0.0, min(1.0, confidence))) * 0.10) + ((1.0 - max(0.0, min(1.0, data_quality))) * 0.08)
    probability = max(0.01, min(0.99, float(probability)))
    return round(max(0.01, probability - width), 4), round(min(0.99, probability + width), 4)


def _combine_odds_frames(*odds_frames: pd.DataFrame) -> pd.DataFrame:
    frames = [frame.copy() for frame in odds_frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    working = pd.concat(frames, ignore_index=True, sort=False)
    required = {"event_id", "fighter_a", "fighter_b", "market", "selection", "book", "american_odds"}
    if not required.issubset(working.columns):
        return pd.DataFrame()
    working["american_odds"] = pd.to_numeric(working["american_odds"], errors="coerce")
    working = working.dropna(subset=["american_odds"]).copy()
    if working.empty:
        return pd.DataFrame()
    working["fight"] = working["fighter_a"].astype(str) + " vs " + working["fighter_b"].astype(str)
    if "selection_name" not in working.columns:
        working["selection_name"] = working["selection"]
    working["selection_name"] = working["selection_name"].fillna("").astype(str)
    working["book"] = working["book"].fillna("unknown").astype(str)
    working["implied_prob"] = working["american_odds"].astype(int).apply(implied_probability)
    return working


def _snapshot_bucket(row: pd.Series) -> str:
    hours = (row["start_time"] - row["snapshot_time"]).total_seconds() / 3600.0
    if hours < 0:
        return "post_start"
    if hours <= 1:
        return "close_1h"
    if hours <= 6:
        return "six_hours"
    if hours <= 24:
        return "one_day"
    if hours <= 48:
        return "two_days"
    if hours <= 72:
        return "three_days"
    return "open_early"


def _snapshot_coverage_status(group: pd.DataFrame) -> str:
    fights = int(group["fight"].nunique()) if "fight" in group.columns else 0
    books = int(group["book"].nunique()) if "book" in group.columns else 0
    if fights >= 8 and books >= 2:
        return "strong"
    if fights >= 4:
        return "partial"
    return "thin"


def _unique_fights(prop_history: pd.DataFrame) -> pd.DataFrame:
    if prop_history is None or prop_history.empty:
        return pd.DataFrame()
    working = prop_history.copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
    keys = [column for column in ["event", "bout"] if column in working.columns]
    if not keys:
        return pd.DataFrame()
    return working.drop_duplicates(subset=keys, keep="first").copy()


def _decision_segments(fights: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    segments: list[tuple[str, pd.DataFrame]] = [("all", fights)]
    if "weight_class" in fights.columns:
        segments.extend((f"weight:{weight}", group) for weight, group in fights.groupby(fights["weight_class"].fillna("unknown").astype(str)))
    if "scheduled_rounds" in fights.columns:
        segments.extend((f"rounds:{rounds}", group) for rounds, group in fights.groupby(pd.to_numeric(fights["scheduled_rounds"], errors="coerce").fillna(3).astype(int)))
    return segments


def _finish_segments(fights: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return _decision_segments(fights)


def _decision_row(segment: str, group: pd.DataFrame) -> dict[str, object]:
    decision = pd.to_numeric(group.get("fight_goes_to_decision_target", 0), errors="coerce").fillna(0)
    inside = pd.to_numeric(group.get("fight_doesnt_go_to_decision_target", 0), errors="coerce").fillna(0)
    return {
        "segment": segment,
        "fights": int(len(group)),
        "decision_rate": round(float(decision.mean()), 4) if len(group) else 0,
        "inside_distance_rate": round(float(inside.mean()), 4) if len(group) else 0,
        "ko_tko_rate": round(float(pd.to_numeric(group.get("fight_ends_by_ko_tko_target", 0), errors="coerce").fillna(0).mean()), 4),
        "submission_rate": round(float(pd.to_numeric(group.get("fight_ends_by_submission_target", 0), errors="coerce").fillna(0).mean()), 4),
        "decision_model_status": "ready" if len(group) >= 100 else "thin_sample",
    }


def _audit_row(check_name: str, passed: bool, problem_count: int, details: str) -> dict[str, object]:
    return {"check_name": check_name, "status": "pass" if passed else "fail", "problem_count": int(problem_count), "details": details}


def _news_reliability_status(alert_count: int, avg_confidence: object) -> str:
    confidence = _safe_float(avg_confidence, 0.0)
    if alert_count <= 0:
        return "no_alerts"
    if confidence >= 0.75:
        return "trusted_alerts"
    if confidence >= 0.50:
        return "medium_confidence"
    return "low_confidence_review"


def _leaderboard_rows(frame: pd.DataFrame, model_name: str, scope: str) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, object]] = []
    for row in frame.to_dict("records"):
        label = str(row.get("market", row.get("market_family", "all")))
        if label not in {"all", "moneyline"}:
            continue
        graded = int(_safe_float(row.get("graded_picks", row.get("graded_props", 0)), 0))
        rows.append(
            {
                "model_name": model_name,
                "scope": f"{scope}:{label}",
                "graded_samples": graded,
                "brier": row.get("brier", ""),
                "log_loss": row.get("log_loss", ""),
                "roi_pct": row.get("roi_pct", ""),
                "avg_clv_edge": row.get("avg_clv_edge", ""),
                "leaderboard_score": _leaderboard_score(row.get("brier", ""), row.get("log_loss", ""), row.get("roi_pct", ""), row.get("avg_clv_edge", "")),
                "promotion_status": "monitor",
            }
        )
    return rows


def _leaderboard_score(brier: object, log_loss: object, roi_pct: object, avg_clv_edge: object) -> float:
    score = 0.0
    brier_value = _safe_float(brier, None)
    log_loss_value = _safe_float(log_loss, None)
    roi_value = _safe_float(roi_pct, None)
    clv_value = _safe_float(avg_clv_edge, None)
    if brier_value is not None:
        score += max(0.0, 0.30 - float(brier_value)) * 100
    if log_loss_value is not None:
        score += max(0.0, 0.75 - float(log_loss_value)) * 40
    if roi_value is not None:
        score += float(roi_value) / 10.0
    if clv_value is not None:
        score += float(clv_value) * 100
    return round(score, 4)


def _promotion_status(score: float, samples: int) -> str:
    if samples < 30:
        return "needs_sample"
    if score >= 12:
        return "promote"
    if score >= 4:
        return "monitor"
    return "hold_out"


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _timestamp(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(timezone.utc)
        else:
            stamp = stamp.tz_convert(timezone.utc)
        return stamp.isoformat()
    except (TypeError, ValueError):
        return str(value)


def _date(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).date().isoformat()
    except (TypeError, ValueError):
        return str(value)
