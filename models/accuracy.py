from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from models.ev import implied_probability
from models.prop_outcomes import (
    PROP_MARKET_TARGETS,
    prepare_prop_feature_frame,
    train_prop_outcome_model,
)


SNAPSHOT_COLUMNS = [
    "snapshot_at",
    "event_id",
    "event_name",
    "start_time",
    "fight",
    "fighter_a",
    "fighter_b",
    "snapshot_source",
    "prediction_mode",
    "lean_side",
    "opponent",
    "model_prob",
    "raw_model_prob",
    "market_aware_prob",
    "market_prob",
    "market_blend_weight",
    "market_disagreement",
    "current_american_odds",
    "edge",
    "confidence",
    "data_quality",
    "segment_label",
    "thin_sample_flag",
    "camp_change_flag",
    "low_data_quality_flag",
    "heavyweight_flag",
    "five_round_flag",
    "pick_gym_name",
    "pick_gym_tier",
    "opponent_gym_name",
    "opponent_gym_tier",
    "top_reasons",
    "risk_flags",
    "watch_for",
]

PROP_BACKTEST_COLUMNS = [
    "market",
    "event",
    "bout",
    "date",
    "fighter_key",
    "opponent_key",
    "model_prob",
    "actual_win",
    "probability_bucket",
    "train_rows",
    "test_rows",
]


def probability_bucket(value: object) -> str:
    numeric = _safe_float(value, None)
    if numeric is None:
        return "unknown"
    if numeric >= 0.75:
        return "0.75_plus"
    if numeric >= 0.65:
        return "0.65_to_0.74"
    if numeric >= 0.55:
        return "0.55_to_0.64"
    return "below_0.55"


def confidence_bucket(value: object) -> str:
    numeric = _safe_float(value, None)
    if numeric is None:
        return "unknown"
    if numeric >= 0.80:
        return "0.80_plus"
    if numeric >= 0.70:
        return "0.70_to_0.79"
    if numeric >= 0.60:
        return "0.60_to_0.69"
    return "below_0.60"


def data_quality_bucket(value: object) -> str:
    numeric = _safe_float(value, None)
    if numeric is None:
        return "unknown"
    if numeric >= 0.95:
        return "0.95_plus"
    if numeric >= 0.85:
        return "0.85_to_0.94"
    if numeric >= 0.70:
        return "0.70_to_0.84"
    return "below_0.70"


def market_blend_bucket(value: object) -> str:
    numeric = _safe_float(value, None)
    if numeric is None:
        return "unknown"
    if numeric >= 0.40:
        return "heavy_market_blend"
    if numeric >= 0.20:
        return "moderate_market_blend"
    if numeric > 0:
        return "light_market_blend"
    return "model_only"


def price_bucket(value: object) -> str:
    numeric = _safe_float(value, None)
    if numeric is None:
        return "no_price"
    if numeric <= -200:
        return "heavy_favorite"
    if numeric < 0:
        return "favorite"
    if numeric >= 200:
        return "big_dog"
    return "dog"


def build_prediction_snapshot(
    *,
    manifest: dict[str, object],
    fighter_stats: pd.DataFrame,
    no_odds_packet: pd.DataFrame | None = None,
    lean_board: pd.DataFrame | None = None,
    fight_report: pd.DataFrame | None = None,
    snapshot_at: str | None = None,
) -> pd.DataFrame:
    snapshot_time = snapshot_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    source = _choose_snapshot_source(no_odds_packet=no_odds_packet, lean_board=lean_board)
    if source is None:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    source_name, frame = source
    report_lookup = _fight_report_lookup(fight_report)
    stats_lookup = _fighter_stats_lookup(fighter_stats)
    rows: list[dict[str, object]] = []
    for source_row in frame.to_dict("records"):
        fighter_a = _safe_text(source_row.get("fighter_a", ""))
        fighter_b = _safe_text(source_row.get("fighter_b", ""))
        fight = _safe_text(source_row.get("fight", f"{fighter_a} vs {fighter_b}"))
        lean_side = _safe_text(source_row.get("lean_side", source_row.get("selection_name", "")))
        opponent = _safe_text(source_row.get("opponent", source_row.get("opponent_side", "")))
        model_prob = _safe_float(
            source_row.get("model_prob", source_row.get("lean_prob", source_row.get("model_projected_win_prob", ""))),
            0.5,
        )
        current_odds = _safe_float(source_row.get("current_american_odds", source_row.get("american_odds", "")), None)
        market_prob = implied_probability(int(current_odds)) if current_odds is not None and not math.isnan(current_odds) else None
        report_row = report_lookup.get(_fight_key(fighter_a, fighter_b), {})
        raw_model_prob = _raw_probability_for_pick(report_row, lean_side, fighter_a, fighter_b, model_prob)
        market_blend_weight = _safe_float(
            source_row.get("market_blend_weight", report_row.get("market_blend_weight", 0.0)),
            0.0,
        )
        market_aware_prob = model_prob
        stat_context = _snapshot_stat_context(stats_lookup, fighter_a, fighter_b, lean_side, opponent)
        segment_label = _segment_label(
            stat_context["pick_weight_class"],
            stat_context["opponent_weight_class"],
            _safe_float(source_row.get("scheduled_rounds", report_row.get("scheduled_rounds", 3.0)), 3.0),
            stat_context["min_ufc_sample"],
            stat_context["camp_change_flag"],
        )
        rows.append(
            {
                "snapshot_at": snapshot_time,
                "event_id": manifest.get("event_id", ""),
                "event_name": manifest.get("event_name", source_row.get("event_name", "")),
                "start_time": manifest.get("start_time", ""),
                "fight": fight,
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "snapshot_source": source_name,
                "prediction_mode": "market_aware" if market_blend_weight > 0 else "model_only",
                "lean_side": lean_side,
                "opponent": opponent,
                "model_prob": round(float(model_prob), 6),
                "raw_model_prob": round(float(raw_model_prob), 6),
                "market_aware_prob": round(float(market_aware_prob), 6),
                "market_prob": "" if market_prob is None else round(float(market_prob), 6),
                "market_blend_weight": round(float(market_blend_weight), 6),
                "market_disagreement": "" if market_prob is None else round(abs(float(raw_model_prob) - float(market_prob)), 6),
                "current_american_odds": "" if current_odds is None else int(round(current_odds)),
                "edge": _safe_float(source_row.get("edge", ""), ""),
                "confidence": round(_safe_float(source_row.get("confidence", source_row.get("model_confidence", "")), 0.5), 6),
                "data_quality": round(_safe_float(source_row.get("data_quality", ""), stat_context["data_quality"]), 6),
                "segment_label": segment_label,
                "thin_sample_flag": int(stat_context["min_ufc_sample"] < 3),
                "camp_change_flag": int(stat_context["camp_change_flag"] >= 1),
                "low_data_quality_flag": int(stat_context["data_quality"] < 0.85),
                "heavyweight_flag": int(_is_heavyweight(stat_context["pick_weight_class"]) or _is_heavyweight(stat_context["opponent_weight_class"])),
                "five_round_flag": int(_safe_float(source_row.get("scheduled_rounds", report_row.get("scheduled_rounds", 3.0)), 3.0) >= 5),
                "pick_gym_name": _safe_text(source_row.get("pick_gym_name", "")),
                "pick_gym_tier": _safe_text(source_row.get("pick_gym_tier", "")),
                "opponent_gym_name": _safe_text(source_row.get("opponent_gym_name", "")),
                "opponent_gym_tier": _safe_text(source_row.get("opponent_gym_tier", "")),
                "top_reasons": _safe_text(source_row.get("top_reasons", source_row.get("support_signals", ""))),
                "risk_flags": _safe_text(source_row.get("risk_flags", "")),
                "watch_for": _safe_text(source_row.get("watch_for", "")),
            }
        )
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def upsert_prediction_snapshot_archive(snapshot: pd.DataFrame, archive_path: str | Path) -> Path:
    path = Path(archive_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.empty:
        if not path.exists():
            pd.DataFrame(columns=SNAPSHOT_COLUMNS).to_csv(path, index=False)
        return path
    if path.exists():
        archive = pd.read_csv(path, keep_default_na=False)
    else:
        archive = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    key_columns = ["event_id", "fight", "snapshot_source"]
    archive["_key"] = archive[key_columns].astype(str).agg("||".join, axis=1) if not archive.empty else ""
    snapshot = snapshot.copy()
    snapshot["_key"] = snapshot[key_columns].astype(str).agg("||".join, axis=1)
    archive = archive.loc[~archive["_key"].isin(snapshot["_key"])].drop(columns=["_key"], errors="ignore")
    combined = pd.concat([archive, snapshot.drop(columns=["_key"])], ignore_index=True)
    ordered = [column for column in SNAPSHOT_COLUMNS if column in combined.columns] + [
        column for column in combined.columns if column not in SNAPSHOT_COLUMNS
    ]
    combined.loc[:, ordered].to_csv(path, index=False)
    return path


def normalize_tracked_pick_predictions(tracked: pd.DataFrame) -> pd.DataFrame:
    if tracked.empty:
        return _empty_prediction_history()
    frame = _dedupe_tracked_prediction_rows(tracked)
    actual = frame.get("actual_result", pd.Series("", index=frame.index)).fillna("").astype(str).str.lower()
    frame = frame.loc[actual.isin(["win", "loss"])].copy()
    if frame.empty:
        return _empty_prediction_history()
    frame["actual_win"] = (frame["actual_result"].astype(str).str.lower() == "win").astype(int)
    frame["model_prob"] = pd.to_numeric(frame.get("model_projected_win_prob", pd.Series(0.5, index=frame.index)), errors="coerce").fillna(0.5).clip(0.01, 0.99)
    frame["confidence"] = pd.to_numeric(frame.get("model_confidence", pd.Series(0.5, index=frame.index)), errors="coerce").fillna(0.5)
    frame["data_quality"] = pd.to_numeric(frame.get("data_quality", frame.get("selection_stats_completeness", pd.Series(1.0, index=frame.index))), errors="coerce").fillna(1.0)
    frame["segment_label"] = frame.get("segment_label", pd.Series("standard", index=frame.index)).fillna("standard").astype(str)
    frame["market_blend_weight"] = pd.to_numeric(frame.get("market_blend_weight", pd.Series(0.0, index=frame.index)), errors="coerce").fillna(0.0)
    frame["price_bucket"] = frame.get("american_odds", pd.Series(pd.NA, index=frame.index)).apply(price_bucket)
    frame["probability_bucket"] = frame["model_prob"].apply(probability_bucket)
    frame["confidence_bucket"] = frame["confidence"].apply(confidence_bucket)
    frame["data_quality_bucket"] = frame["data_quality"].apply(data_quality_bucket)
    frame["market_blend_bucket"] = frame["market_blend_weight"].apply(market_blend_bucket)
    frame["risk_bucket"] = frame.get("risk_flags", pd.Series("", index=frame.index)).apply(_risk_bucket)
    frame["prediction_source"] = "tracked_pick"
    return frame.reset_index(drop=True)


def _dedupe_tracked_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["event_id", "fight_key", "tracked_market_key", "tracked_selection_key"]
    if frame.empty or any(column not in frame.columns for column in key_columns):
        return frame.copy()
    working = frame.copy()
    sort_columns = [column for column in ["tracked_at", "pick_id"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, kind="mergesort")
    return working.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def build_calibration_report(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=_calibration_columns())
    rows = [_calibration_row(predictions, "overall", "all")]
    for bucket, bucket_rows in predictions.groupby("probability_bucket", dropna=False):
        rows.append(_calibration_row(bucket_rows, "probability_bucket", str(bucket)))
    for bucket, bucket_rows in predictions.groupby("confidence_bucket", dropna=False):
        rows.append(_calibration_row(bucket_rows, "confidence_bucket", str(bucket)))
    return pd.DataFrame(rows, columns=_calibration_columns())


def build_segment_performance_report(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=_segment_columns())
    dimensions = [
        "segment_label",
        "confidence_bucket",
        "data_quality_bucket",
        "market_blend_bucket",
        "price_bucket",
        "risk_bucket",
        "recommended_tier",
        "tracked_market_key",
    ]
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        if dimension not in predictions.columns:
            continue
        for bucket, bucket_rows in predictions.groupby(dimension, dropna=False):
            rows.append(_segment_row(bucket_rows, dimension, str(bucket)))
    return pd.DataFrame(rows, columns=_segment_columns()).sort_values(by=["dimension", "bucket"]).reset_index(drop=True)


def build_market_accuracy_report(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "graded_picks",
        "wins",
        "losses",
        "avg_model_prob",
        "win_rate",
        "calibration_error",
        "brier",
        "log_loss",
        "total_staked",
        "total_profit",
        "roi_pct",
        "avg_clv_edge",
        "avg_clv_delta",
        "sample_warning",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    market_series = predictions.get("tracked_market_key", pd.Series("unknown", index=predictions.index)).fillna("unknown").astype(str)
    working = predictions.copy()
    working["tracked_market_key"] = market_series.where(market_series.ne(""), "unknown")
    rows: list[dict[str, object]] = []
    for market, market_rows in working.groupby("tracked_market_key", dropna=False):
        rows.append(_market_accuracy_row(market_rows, str(market)))
    rows.append(_market_accuracy_row(working, "all"))
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market"]).reset_index(drop=True)


def build_prop_model_backtest_predictions(
    prop_history: pd.DataFrame,
    *,
    holdout_fraction: float = 0.25,
    min_train_samples: int = 400,
) -> pd.DataFrame:
    if prop_history.empty:
        return pd.DataFrame(columns=PROP_BACKTEST_COLUMNS)
    history = prop_history.copy()
    history["date"] = pd.to_datetime(history.get("date"), errors="coerce")
    history = history.dropna(subset=["date"]).sort_values(["date", "event", "bout", "fighter_key"]).reset_index(drop=True)
    if len(history) < min_train_samples + 20:
        return pd.DataFrame(columns=PROP_BACKTEST_COLUMNS)

    holdout_fraction = max(0.05, min(0.50, float(holdout_fraction)))
    split_index = max(min_train_samples, int(len(history) * (1.0 - holdout_fraction)))
    if split_index >= len(history):
        return pd.DataFrame(columns=PROP_BACKTEST_COLUMNS)

    train_frame = history.iloc[:split_index].copy()
    test_frame = history.iloc[split_index:].copy()
    try:
        bundle, training = train_prop_outcome_model(train_frame, min_samples=min_train_samples)
    except ValueError:
        return pd.DataFrame(columns=PROP_BACKTEST_COLUMNS)

    prepared_test = prepare_prop_feature_frame(test_frame)
    rows: list[dict[str, object]] = []
    for market, model_entry in bundle.get("markets", {}).items():
        target_column = PROP_MARKET_TARGETS.get(str(market))
        if not target_column or target_column not in test_frame.columns:
            continue
        actual = pd.to_numeric(test_frame[target_column], errors="coerce")
        valid_mask = actual.isin([0, 1])
        if not valid_mask.any():
            continue
        market_test = prepared_test.loc[valid_mask].copy()
        probabilities = model_entry["pipeline"].predict_proba(market_test)[:, 1]
        actual_valid = actual.loc[valid_mask].astype(int)
        source_valid = test_frame.loc[valid_mask]
        for index, probability in zip(source_valid.index, probabilities):
            source_row = source_valid.loc[index]
            probability_float = float(max(0.01, min(0.99, probability)))
            rows.append(
                {
                    "market": str(market),
                    "event": source_row.get("event", ""),
                    "bout": source_row.get("bout", ""),
                    "date": source_row.get("date", ""),
                    "fighter_key": source_row.get("fighter_key", ""),
                    "opponent_key": source_row.get("opponent_key", ""),
                    "model_prob": round(probability_float, 4),
                    "actual_win": int(actual_valid.loc[index]),
                    "probability_bucket": probability_bucket(probability_float),
                    "train_rows": int(len(training)),
                    "test_rows": int(len(test_frame)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=PROP_BACKTEST_COLUMNS)
    return pd.DataFrame(rows, columns=PROP_BACKTEST_COLUMNS)


def build_prop_model_market_report(prop_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "graded_props",
        "avg_model_prob",
        "hit_rate",
        "calibration_error",
        "brier",
        "log_loss",
        "sample_warning",
    ]
    if prop_predictions.empty:
        return pd.DataFrame(columns=columns)
    rows = [_prop_market_accuracy_row(prop_predictions, "all")]
    for market, market_rows in prop_predictions.groupby("market", dropna=False):
        rows.append(_prop_market_accuracy_row(market_rows, str(market)))
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market"]).reset_index(drop=True)


def build_prop_model_calibration_report(prop_predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "probability_bucket",
        "graded_props",
        "avg_model_prob",
        "hit_rate",
        "calibration_error",
        "brier",
        "sample_warning",
    ]
    if prop_predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (market, bucket), bucket_rows in prop_predictions.groupby(["market", "probability_bucket"], dropna=False):
        metrics = _prop_metrics(bucket_rows)
        rows.append(
            {
                "market": str(market),
                "probability_bucket": str(bucket),
                "graded_props": int(len(bucket_rows)),
                "avg_model_prob": metrics["avg_model_prob"],
                "hit_rate": metrics["hit_rate"],
                "calibration_error": metrics["calibration_error"],
                "brier": metrics["brier"],
                "sample_warning": _sample_warning(len(bucket_rows)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market", "probability_bucket"]).reset_index(drop=True)


def build_prop_threshold_report(
    prop_predictions: pd.DataFrame,
    *,
    min_samples: int = 50,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    columns = [
        "market",
        "min_model_prob",
        "graded_props",
        "avg_model_prob",
        "hit_rate",
        "calibration_error",
        "brier",
        "threshold_action",
        "is_recommended",
        "sample_warning",
    ]
    if prop_predictions.empty:
        return pd.DataFrame(columns=columns)
    threshold_values = thresholds or [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    rows: list[dict[str, object]] = []
    for market, market_rows in prop_predictions.groupby("market", dropna=False):
        market_threshold_rows: list[dict[str, object]] = []
        for threshold in threshold_values:
            threshold_rows = market_rows.loc[pd.to_numeric(market_rows["model_prob"], errors="coerce").fillna(0.0) >= threshold]
            metrics = _prop_metrics(threshold_rows)
            row = {
                "market": str(market),
                "min_model_prob": round(float(threshold), 2),
                "graded_props": int(len(threshold_rows)),
                "avg_model_prob": metrics["avg_model_prob"],
                "hit_rate": metrics["hit_rate"],
                "calibration_error": metrics["calibration_error"],
                "brier": metrics["brier"],
                "threshold_action": _threshold_action(threshold_rows, metrics, min_samples=min_samples),
                "is_recommended": 0,
                "sample_warning": _sample_warning(len(threshold_rows), min_samples=min_samples),
            }
            market_threshold_rows.append(row)
        _mark_recommended_threshold(market_threshold_rows, min_samples=min_samples)
        rows.extend(market_threshold_rows)
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market", "min_model_prob"]).reset_index(drop=True)


def build_prop_odds_archive_report(snapshot_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "event_name",
        "fight",
        "market",
        "selection",
        "selection_name",
        "book",
        "snapshots",
        "first_snapshot_time",
        "latest_snapshot_time",
        "open_american_odds",
        "current_american_odds",
        "closing_candidate_american_odds",
    ]
    if snapshot_history.empty:
        return pd.DataFrame(columns=columns)
    required = {"event_id", "fighter_a", "fighter_b", "market", "selection", "book", "american_odds"}
    if not required.issubset(snapshot_history.columns):
        return pd.DataFrame(columns=columns)
    working = snapshot_history.copy()
    working["market"] = working["market"].fillna("").astype(str)
    working = working.loc[working["market"].ne("moneyline") & working["market"].ne("")].copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["snapshot_time"] = pd.to_datetime(working.get("snapshot_time"), errors="coerce", utc=True)
    working["start_time"] = pd.to_datetime(working.get("start_time"), errors="coerce", utc=True)
    working["american_odds"] = pd.to_numeric(working["american_odds"], errors="coerce")
    working = working.loc[working["american_odds"].notna()].sort_values("snapshot_time").copy()
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["fight"] = working["fighter_a"].astype(str) + " vs " + working["fighter_b"].astype(str)

    rows: list[dict[str, object]] = []
    group_columns = ["event_id", "event_name", "fight", "market", "selection", "selection_name", "book"]
    for key, rows_frame in working.groupby(group_columns, dropna=False):
        first = rows_frame.iloc[0]
        latest = rows_frame.iloc[-1]
        pre_event = rows_frame.loc[
            rows_frame["start_time"].isna()
            | rows_frame["snapshot_time"].isna()
            | (rows_frame["snapshot_time"] <= rows_frame["start_time"])
        ]
        closing = pre_event.iloc[-1] if not pre_event.empty else latest
        rows.append(
            {
                "event_id": key[0],
                "event_name": key[1],
                "fight": key[2],
                "market": key[3],
                "selection": key[4],
                "selection_name": key[5],
                "book": key[6],
                "snapshots": int(len(rows_frame)),
                "first_snapshot_time": _timestamp_text(first.get("snapshot_time")),
                "latest_snapshot_time": _timestamp_text(latest.get("snapshot_time")),
                "open_american_odds": int(float(first.get("american_odds"))),
                "current_american_odds": int(float(latest.get("american_odds"))),
                "closing_candidate_american_odds": int(float(closing.get("american_odds"))),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(by=["event_id", "fight", "market", "selection"]).reset_index(drop=True)


def build_odds_movement_clv_report(snapshot_history: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "archived_selections",
        "archive_events",
        "archive_fights",
        "books",
        "avg_open_american_odds",
        "avg_current_american_odds",
        "avg_closing_candidate_american_odds",
        "avg_open_implied_prob",
        "avg_current_implied_prob",
        "avg_closing_candidate_implied_prob",
        "avg_open_to_current_implied_move",
        "avg_current_to_close_implied_move",
        "line_shortened_from_open_pct",
        "sample_warning",
    ]
    detail = build_prop_odds_archive_report(snapshot_history)
    if detail.empty:
        return pd.DataFrame(columns=columns)
    working = detail.copy()
    for column in ["open_american_odds", "current_american_odds", "closing_candidate_american_odds"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
        implied_column = column.replace("american_odds", "implied_prob")
        working[implied_column] = working[column].apply(
            lambda value: implied_probability(int(value)) if not pd.isna(value) else pd.NA
        )
    working["open_to_current_implied_move"] = working["current_implied_prob"] - working["open_implied_prob"]
    working["current_to_close_implied_move"] = working["closing_candidate_implied_prob"] - working["current_implied_prob"]
    working["line_shortened_from_open"] = working["open_to_current_implied_move"] > 0

    rows: list[dict[str, object]] = []
    for market, market_rows in working.groupby("market", dropna=False):
        rows.append(
            {
                "market": str(market),
                "archived_selections": int(len(market_rows)),
                "archive_events": int(market_rows["event_id"].nunique()),
                "archive_fights": int(market_rows["fight"].nunique()),
                "books": int(market_rows["book"].nunique()),
                "avg_open_american_odds": round(float(market_rows["open_american_odds"].mean()), 2),
                "avg_current_american_odds": round(float(market_rows["current_american_odds"].mean()), 2),
                "avg_closing_candidate_american_odds": round(float(market_rows["closing_candidate_american_odds"].mean()), 2),
                "avg_open_implied_prob": round(float(market_rows["open_implied_prob"].mean()), 4),
                "avg_current_implied_prob": round(float(market_rows["current_implied_prob"].mean()), 4),
                "avg_closing_candidate_implied_prob": round(float(market_rows["closing_candidate_implied_prob"].mean()), 4),
                "avg_open_to_current_implied_move": round(float(market_rows["open_to_current_implied_move"].mean()), 4),
                "avg_current_to_close_implied_move": round(float(market_rows["current_to_close_implied_move"].mean()), 4),
                "line_shortened_from_open_pct": round(float(market_rows["line_shortened_from_open"].mean()) * 100.0, 2),
                "sample_warning": _sample_warning(len(market_rows)),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market"]).reset_index(drop=True)


def build_tracked_clv_report(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "market",
        "graded_picks",
        "avg_clv_edge",
        "avg_clv_delta",
        "positive_clv_pct",
        "roi_pct",
        "sample_warning",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    working = predictions.copy()
    working["tracked_market_key"] = working.get("tracked_market_key", pd.Series("unknown", index=working.index)).fillna("unknown").astype(str)
    working["clv_edge"] = pd.to_numeric(working.get("clv_edge", pd.Series(pd.NA, index=working.index)), errors="coerce")
    working["clv_delta"] = pd.to_numeric(working.get("clv_delta", pd.Series(pd.NA, index=working.index)), errors="coerce")
    working["stake"] = _numeric_or_default(working, ["chosen_expression_stake", "suggested_stake"], 0.0)
    working["profit"] = _numeric_or_default(working, ["profit"], 0.0)
    rows: list[dict[str, object]] = []
    for market, market_rows in working.groupby("tracked_market_key", dropna=False):
        clv_rows = market_rows.loc[market_rows["clv_edge"].notna()].copy()
        total_staked = float(market_rows["stake"].sum()) if not market_rows.empty else 0.0
        total_profit = float(market_rows["profit"].sum()) if not market_rows.empty else 0.0
        rows.append(
            {
                "market": str(market),
                "graded_picks": int(len(market_rows)),
                "avg_clv_edge": round(float(clv_rows["clv_edge"].mean()), 4) if not clv_rows.empty else "",
                "avg_clv_delta": round(float(clv_rows["clv_delta"].mean()), 2) if not clv_rows.empty else "",
                "positive_clv_pct": round(float((clv_rows["clv_edge"] > 0).mean()) * 100.0, 2) if not clv_rows.empty else "",
                "roi_pct": round((total_profit / total_staked) * 100.0, 2) if total_staked > 0 else "",
                "sample_warning": _sample_warning(len(market_rows)),
            }
        )
    rows.append(_tracked_clv_all_row(working))
    return pd.DataFrame(rows, columns=columns).sort_values(by=["market"]).reset_index(drop=True)


def build_quality_gate_report(segment_performance: pd.DataFrame, *, min_samples: int = 5) -> pd.DataFrame:
    columns = [
        "dimension",
        "bucket",
        "graded_picks",
        "avg_model_prob",
        "win_rate",
        "calibration_error",
        "brier",
        "gate_action",
        "confidence_multiplier",
        "gate_reason",
    ]
    if segment_performance.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in segment_performance.to_dict("records"):
        graded = int(row.get("graded_picks", 0) or 0)
        avg_prob = _safe_float(row.get("avg_model_prob", 0.0), 0.0)
        win_rate = _safe_float(row.get("win_rate", 0.0), 0.0)
        error = _safe_float(row.get("calibration_error", 0.0), 0.0)
        brier = _safe_float(row.get("brier", 0.0), 0.0)
        action = "needs_sample"
        multiplier = 0.92
        reason = f"sample {graded} below {min_samples}"
        if graded >= min_samples:
            action = "trust"
            multiplier = 1.0
            reason = "calibration holding"
            underperformance = win_rate + 0.10 < avg_prob
            soft_underperformance = win_rate + 0.06 < avg_prob
            if brier >= 0.28 or underperformance:
                action = "block_strong_leans"
                multiplier = 0.78
                reason = "bad calibration or underperforming win rate"
                if row.get("dimension") not in {"segment_label", "tracked_market_key"}:
                    action = "downgrade_confidence"
                    multiplier = 0.84
                    reason = "cross-cutting calibration warning"
            elif brier >= 0.24 or error <= -0.08 or soft_underperformance:
                action = "downgrade_confidence"
                multiplier = 0.86
                reason = "soft calibration warning"
            elif brier <= 0.19 and abs(win_rate - avg_prob) <= 0.05:
                action = "expand"
                multiplier = 1.04
                reason = "strong calibration bucket"
        rows.append(
            {
                "dimension": row.get("dimension", ""),
                "bucket": row.get("bucket", ""),
                "graded_picks": graded,
                "avg_model_prob": round(avg_prob, 4),
                "win_rate": round(win_rate, 4),
                "calibration_error": round(error, 4),
                "brier": round(brier, 4),
                "gate_action": action,
                "confidence_multiplier": round(multiplier, 3),
                "gate_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(by=["dimension", "bucket"]).reset_index(drop=True)


def build_current_quality_report(snapshot: pd.DataFrame, quality_gates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "fight",
        "lean_side",
        "model_prob",
        "confidence",
        "gated_confidence",
        "data_quality_score",
        "quality_bucket",
        "segment_label",
        "gate_action",
        "gate_reason",
        "quality_warnings",
    ]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)
    gate_lookup = _gate_lookup(quality_gates)
    rows: list[dict[str, object]] = []
    for row in snapshot.to_dict("records"):
        quality_score, warnings = _current_quality_score(row)
        gate = _resolve_current_gate(row, gate_lookup)
        multiplier = _safe_float(gate.get("confidence_multiplier", 1.0), 1.0)
        confidence = _safe_float(row.get("confidence", 0.5), 0.5)
        rows.append(
            {
                "fight": row.get("fight", ""),
                "lean_side": row.get("lean_side", ""),
                "model_prob": row.get("model_prob", ""),
                "confidence": round(confidence, 4),
                "gated_confidence": round(max(0.2, min(0.95, confidence * multiplier * quality_score)), 4),
                "data_quality_score": round(quality_score, 4),
                "quality_bucket": data_quality_bucket(quality_score),
                "segment_label": row.get("segment_label", ""),
                "gate_action": gate.get("gate_action", "needs_sample"),
                "gate_reason": gate.get("gate_reason", "no historical gate matched"),
                "quality_warnings": ", ".join(warnings) if warnings else "none",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_style_matchup_diagnostics(snapshot: pd.DataFrame, lean_board: pd.DataFrame | None = None, fight_report: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "fight",
        "lean_side",
        "style_summary",
        "striking_edge",
        "grappling_edge",
        "control_edge",
        "durability_note",
        "method_note",
    ]
    if snapshot.empty:
        return pd.DataFrame(columns=columns)
    board_lookup = _generic_fight_lookup(lean_board)
    report_lookup = _generic_fight_lookup(fight_report)
    rows: list[dict[str, object]] = []
    for row in snapshot.to_dict("records"):
        fight = _safe_text(row.get("fight", ""))
        source_row = board_lookup.get(fight, report_lookup.get(fight, {}))
        striking = _safe_float(source_row.get("matchup_striking_edge", source_row.get("striking_diff", "")), 0.0)
        grappling = _safe_float(source_row.get("matchup_grappling_edge", source_row.get("grappling_diff", "")), 0.0)
        control = _safe_float(source_row.get("matchup_control_edge", source_row.get("control_diff", "")), 0.0)
        style_summary = _safe_text(source_row.get("pick_style", "")) or _safe_text(source_row.get("top_reasons", ""))
        rows.append(
            {
                "fight": fight,
                "lean_side": row.get("lean_side", ""),
                "style_summary": style_summary,
                "striking_edge": round(striking, 4),
                "grappling_edge": round(grappling, 4),
                "control_edge": round(control, 4),
                "durability_note": _edge_note("durability", _safe_float(source_row.get("durability_diff", ""), 0.0)),
                "method_note": _safe_text(source_row.get("pick_best_method", source_row.get("method_lean", ""))),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_postmortem_code_report(predictions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id",
        "fight",
        "selection_name",
        "actual_result",
        "model_prob",
        "confidence",
        "postmortem_codes",
        "next_action",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for row in predictions.to_dict("records"):
        result = _safe_text(row.get("actual_result", "")).lower()
        codes: list[str] = []
        if result == "loss":
            if _safe_float(row.get("data_quality", 1.0), 1.0) < 0.85:
                codes.append("data_quality_miss")
            if _safe_float(row.get("market_blend_weight", 0.0), 0.0) >= 0.35:
                codes.append("market_disagreement_miss")
            risk_text = _safe_text(row.get("risk_flags", "")).lower()
            if any(token in risk_text for token in ["camp_change", "short_notice", "injury", "weight_cut", "replacement"]):
                codes.append("context_risk_hit")
            if _safe_float(row.get("model_prob", 0.5), 0.5) >= 0.70:
                codes.append("overconfident_loss")
            if not codes:
                codes.append("model_read_miss")
        elif result == "win":
            codes.append("validated_read")
            if _safe_float(row.get("clv_delta", 0.0), 0.0) < 0:
                codes.append("right_side_bad_price")
        next_action = _postmortem_next_action(codes)
        rows.append(
            {
                "event_id": row.get("event_id", ""),
                "fight": f"{row.get('fighter_a', '')} vs {row.get('fighter_b', '')}",
                "selection_name": row.get("selection_name", ""),
                "actual_result": row.get("actual_result", ""),
                "model_prob": round(_safe_float(row.get("model_prob", 0.5), 0.5), 4),
                "confidence": round(_safe_float(row.get("confidence", 0.5), 0.5), 4),
                "postmortem_codes": ", ".join(codes),
                "next_action": next_action,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _choose_snapshot_source(
    *,
    no_odds_packet: pd.DataFrame | None,
    lean_board: pd.DataFrame | None,
) -> tuple[str, pd.DataFrame] | None:
    if no_odds_packet is not None and not no_odds_packet.empty and (
        lean_board is None or lean_board.empty or len(no_odds_packet) > len(lean_board)
    ):
        return "no_odds_prediction_packet", no_odds_packet.copy()
    if lean_board is not None and not lean_board.empty:
        return "lean_board", lean_board.copy()
    if no_odds_packet is not None and not no_odds_packet.empty:
        return "no_odds_prediction_packet", no_odds_packet.copy()
    return None


def _fighter_stats_lookup(fighter_stats: pd.DataFrame) -> dict[str, dict[str, object]]:
    if fighter_stats.empty or "fighter_name" not in fighter_stats.columns:
        return {}
    return {
        _normalize_name(row["fighter_name"]): row
        for row in fighter_stats.fillna("").to_dict("records")
    }


def _fight_report_lookup(fight_report: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if fight_report is None or fight_report.empty:
        return {}
    return {
        _fight_key(row.get("fighter_a", ""), row.get("fighter_b", "")): row
        for row in fight_report.fillna("").to_dict("records")
    }


def _generic_fight_lookup(frame: pd.DataFrame | None) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "fight" not in frame.columns:
        return {}
    return {
        _safe_text(row.get("fight", "")): row
        for row in frame.fillna("").to_dict("records")
    }


def _snapshot_stat_context(
    stats_lookup: dict[str, dict[str, object]],
    fighter_a: str,
    fighter_b: str,
    lean_side: str,
    opponent: str,
) -> dict[str, object]:
    pick = stats_lookup.get(_normalize_name(lean_side), {})
    opp = stats_lookup.get(_normalize_name(opponent), {})
    if not pick:
        pick = stats_lookup.get(_normalize_name(fighter_a), {})
    if not opp:
        other = fighter_b if _normalize_name(lean_side) == _normalize_name(fighter_a) else fighter_a
        opp = stats_lookup.get(_normalize_name(other), {})
    pick_sample = _safe_float(pick.get("ufc_fight_count", 0), 0.0)
    opp_sample = _safe_float(opp.get("ufc_fight_count", 0), 0.0)
    pick_quality = _safe_float(pick.get("stats_completeness", 1.0), 1.0)
    opp_quality = _safe_float(opp.get("stats_completeness", 1.0), 1.0)
    camp_change = max(_safe_float(pick.get("camp_change_flag", 0), 0.0), _safe_float(opp.get("camp_change_flag", 0), 0.0))
    return {
        "min_ufc_sample": min(pick_sample, opp_sample),
        "data_quality": min(pick_quality, opp_quality),
        "camp_change_flag": camp_change,
        "pick_weight_class": _safe_text(pick.get("weight_class", "")),
        "opponent_weight_class": _safe_text(opp.get("weight_class", "")),
    }


def _raw_probability_for_pick(
    report_row: dict[str, object],
    lean_side: str,
    fighter_a: str,
    fighter_b: str,
    default: float,
) -> float:
    if not report_row:
        return float(default)
    raw_a = _safe_float(report_row.get("fighter_a_raw_model_win_prob", report_row.get("fighter_a_model_win_prob", "")), None)
    if raw_a is None:
        return float(default)
    return float(raw_a) if _normalize_name(lean_side) == _normalize_name(fighter_a) else 1 - float(raw_a)


def _segment_label(
    pick_weight_class: object,
    opponent_weight_class: object,
    scheduled_rounds: float,
    min_ufc_sample: float,
    camp_change_flag: float,
) -> str:
    labels: list[str] = []
    if _is_wmma(pick_weight_class) or _is_wmma(opponent_weight_class):
        labels.append("wmma")
    if _is_heavyweight(pick_weight_class) or _is_heavyweight(opponent_weight_class):
        labels.append("heavyweight")
    if scheduled_rounds >= 5:
        labels.append("five_round")
    if min_ufc_sample < 3:
        labels.append("thin_ufc_sample")
    if camp_change_flag >= 1:
        labels.append("camp_change")
    return "|".join(labels) if labels else "standard"


def _is_wmma(weight_class: object) -> bool:
    return "women" in _safe_text(weight_class).lower()


def _is_heavyweight(weight_class: object) -> bool:
    normalized = _safe_text(weight_class).lower().strip()
    return normalized == "heavyweight"


def _calibration_row(frame: pd.DataFrame, dimension: str, bucket: str) -> dict[str, object]:
    probs = pd.to_numeric(frame["model_prob"], errors="coerce").fillna(0.5).clip(0.01, 0.99)
    actual = pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0)
    avg_prob = float(probs.mean()) if not probs.empty else 0.0
    win_rate = float(actual.mean()) if not actual.empty else 0.0
    return {
        "dimension": dimension,
        "bucket": bucket,
        "graded_picks": int(len(frame)),
        "avg_model_prob": round(avg_prob, 4),
        "win_rate": round(win_rate, 4),
        "calibration_error": round(win_rate - avg_prob, 4),
        "abs_calibration_error": round(abs(win_rate - avg_prob), 4),
        "brier": round(float(((probs - actual) ** 2).mean()), 4),
        "log_loss": round(_log_loss(probs, actual), 4),
    }


def _segment_row(frame: pd.DataFrame, dimension: str, bucket: str) -> dict[str, object]:
    base = _calibration_row(frame, dimension, bucket)
    base["wins"] = int((pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0) == 1).sum())
    base["losses"] = int((pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0) == 0).sum())
    return base


def _market_accuracy_row(frame: pd.DataFrame, market: str) -> dict[str, object]:
    base = _calibration_row(frame, "market", market)
    stake = _numeric_or_default(frame, ["chosen_expression_stake", "suggested_stake"], 0.0)
    profit = _numeric_or_default(frame, ["profit"], 0.0)
    clv_edge = _numeric_or_default(frame, ["clv_edge"], pd.NA)
    clv_delta = _numeric_or_default(frame, ["clv_delta"], pd.NA)
    total_staked = float(stake.sum()) if not stake.empty else 0.0
    total_profit = float(profit.sum()) if not profit.empty else 0.0
    return {
        "market": market,
        "graded_picks": base["graded_picks"],
        "wins": int((pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0) == 1).sum()),
        "losses": int((pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0) == 0).sum()),
        "avg_model_prob": base["avg_model_prob"],
        "win_rate": base["win_rate"],
        "calibration_error": base["calibration_error"],
        "brier": base["brier"],
        "log_loss": base["log_loss"],
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_pct": round((total_profit / total_staked) * 100.0, 2) if total_staked > 0 else "",
        "avg_clv_edge": round(float(clv_edge.dropna().mean()), 4) if clv_edge.dropna().size else "",
        "avg_clv_delta": round(float(clv_delta.dropna().mean()), 2) if clv_delta.dropna().size else "",
        "sample_warning": _sample_warning(len(frame)),
    }


def _tracked_clv_all_row(frame: pd.DataFrame) -> dict[str, object]:
    clv_rows = frame.loc[frame["clv_edge"].notna()].copy()
    total_staked = float(frame["stake"].sum()) if not frame.empty else 0.0
    total_profit = float(frame["profit"].sum()) if not frame.empty else 0.0
    return {
        "market": "all",
        "graded_picks": int(len(frame)),
        "avg_clv_edge": round(float(clv_rows["clv_edge"].mean()), 4) if not clv_rows.empty else "",
        "avg_clv_delta": round(float(clv_rows["clv_delta"].mean()), 2) if not clv_rows.empty else "",
        "positive_clv_pct": round(float((clv_rows["clv_edge"] > 0).mean()) * 100.0, 2) if not clv_rows.empty else "",
        "roi_pct": round((total_profit / total_staked) * 100.0, 2) if total_staked > 0 else "",
        "sample_warning": _sample_warning(len(frame)),
    }


def _prop_market_accuracy_row(frame: pd.DataFrame, market: str) -> dict[str, object]:
    metrics = _prop_metrics(frame)
    return {
        "market": market,
        "graded_props": int(len(frame)),
        "avg_model_prob": metrics["avg_model_prob"],
        "hit_rate": metrics["hit_rate"],
        "calibration_error": metrics["calibration_error"],
        "brier": metrics["brier"],
        "log_loss": metrics["log_loss"],
        "sample_warning": _sample_warning(len(frame)),
    }


def _prop_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "avg_model_prob": 0.0,
            "hit_rate": 0.0,
            "calibration_error": 0.0,
            "brier": 0.0,
            "log_loss": 0.0,
        }
    probs = pd.to_numeric(frame["model_prob"], errors="coerce").fillna(0.5).clip(0.01, 0.99)
    actual = pd.to_numeric(frame["actual_win"], errors="coerce").fillna(0)
    avg_prob = float(probs.mean()) if not probs.empty else 0.0
    hit_rate = float(actual.mean()) if not actual.empty else 0.0
    return {
        "avg_model_prob": round(avg_prob, 4),
        "hit_rate": round(hit_rate, 4),
        "calibration_error": round(hit_rate - avg_prob, 4),
        "brier": round(float(((probs - actual) ** 2).mean()), 4),
        "log_loss": round(_log_loss(probs, actual), 4),
    }


def _numeric_or_default(frame: pd.DataFrame, candidates: list[str], default: object) -> pd.Series:
    output = pd.Series(pd.NA, index=frame.index)
    for column in candidates:
        if column in frame.columns:
            output = output.combine_first(pd.to_numeric(frame[column], errors="coerce"))
    return output.fillna(default)


def _sample_warning(size: int, *, min_samples: int = 30) -> str:
    if size <= 0:
        return "no_sample"
    if size < min_samples:
        return f"small_sample: fewer than {min_samples}"
    if size < min_samples * 2:
        return "moderate_sample"
    return ""


def _timestamp_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    try:
        return pd.Timestamp(value).isoformat()
    except (TypeError, ValueError):
        return str(value)


def _threshold_action(frame: pd.DataFrame, metrics: dict[str, float], *, min_samples: int) -> str:
    if len(frame) < min_samples:
        return "needs_sample"
    if float(metrics["hit_rate"]) + 0.06 < float(metrics["avg_model_prob"]):
        return "tighten_or_block"
    if float(metrics["brier"]) >= 0.24:
        return "tighten"
    if abs(float(metrics["calibration_error"])) <= 0.05:
        return "usable"
    return "monitor"


def _mark_recommended_threshold(rows: list[dict[str, object]], *, min_samples: int) -> None:
    usable = [
        row
        for row in rows
        if int(row["graded_props"]) >= min_samples and str(row["threshold_action"]) in {"usable", "monitor"}
    ]
    if not usable:
        return
    recommended = sorted(
        usable,
        key=lambda row: (
            0 if str(row["threshold_action"]) == "usable" else 1,
            float(row["min_model_prob"]),
        ),
    )[0]
    recommended["is_recommended"] = 1


def _calibration_columns() -> list[str]:
    return [
        "dimension",
        "bucket",
        "graded_picks",
        "avg_model_prob",
        "win_rate",
        "calibration_error",
        "abs_calibration_error",
        "brier",
        "log_loss",
    ]


def _segment_columns() -> list[str]:
    return [*_calibration_columns(), "wins", "losses"]


def _empty_prediction_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_id",
            "fighter_a",
            "fighter_b",
            "selection_name",
            "actual_result",
            "actual_win",
            "model_prob",
            "confidence",
            "data_quality",
            "segment_label",
            "probability_bucket",
            "confidence_bucket",
            "data_quality_bucket",
            "market_blend_bucket",
            "price_bucket",
            "risk_bucket",
        ]
    )


def _log_loss(probs: pd.Series, actual: pd.Series) -> float:
    clipped = probs.clip(0.01, 0.99)
    values = -(actual * clipped.apply(math.log) + (1 - actual) * (1 - clipped).apply(math.log))
    return float(values.mean()) if not values.empty else 0.0


def _gate_lookup(quality_gates: pd.DataFrame) -> dict[tuple[str, str], dict[str, object]]:
    if quality_gates.empty:
        return {}
    return {
        (str(row.get("dimension", "")), str(row.get("bucket", ""))): row
        for row in quality_gates.to_dict("records")
    }


def _resolve_current_gate(row: dict[str, object], gate_lookup: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    candidates = [
        ("segment_label", _safe_text(row.get("segment_label", ""))),
        ("confidence_bucket", confidence_bucket(row.get("confidence", ""))),
        ("data_quality_bucket", data_quality_bucket(row.get("data_quality", ""))),
        ("market_blend_bucket", market_blend_bucket(row.get("market_blend_weight", ""))),
        ("risk_bucket", _risk_bucket(row.get("risk_flags", ""))),
    ]
    matched = [gate_lookup[key] for key in candidates if key in gate_lookup]
    if not matched:
        return {"gate_action": "needs_sample", "confidence_multiplier": 0.92, "gate_reason": "no historical gate matched"}
    priority = {"block_strong_leans": 0, "downgrade_confidence": 1, "needs_sample": 2, "monitor": 3, "trust": 4, "expand": 5}
    return sorted(matched, key=lambda item: priority.get(str(item.get("gate_action", "")), 3))[0]


def _current_quality_score(row: dict[str, object]) -> tuple[float, list[str]]:
    score = _safe_float(row.get("data_quality", 1.0), 1.0)
    warnings: list[str] = []
    if _safe_float(row.get("thin_sample_flag", 0), 0.0) >= 1:
        score -= 0.08
        warnings.append("thin UFC sample")
    if _safe_float(row.get("camp_change_flag", 0), 0.0) >= 1:
        score -= 0.06
        warnings.append("camp change")
    disagreement = _safe_float(row.get("market_disagreement", 0.0), 0.0)
    if disagreement >= 0.12:
        score -= 0.07
        warnings.append("market disagreement")
    if _safe_float(row.get("low_data_quality_flag", 0), 0.0) >= 1:
        score -= 0.10
        warnings.append("low data quality")
    if _safe_float(row.get("heavyweight_flag", 0), 0.0) >= 1:
        score -= 0.03
        warnings.append("heavyweight variance")
    return max(0.45, min(1.0, score)), warnings


def _edge_note(label: str, value: float) -> str:
    if abs(value) < 0.05:
        return f"{label}: neutral"
    return f"{label}: {'for pick' if value > 0 else 'against pick'}"


def _risk_bucket(value: object) -> str:
    text = _safe_text(value).lower()
    if not text or text == "none":
        return "clean"
    if any(token in text for token in ["camp", "injury", "weight", "short", "replacement"]):
        return "context_risk"
    if "market" in text:
        return "market_risk"
    if "data" in text or "sample" in text:
        return "data_risk"
    return "other_risk"


def _postmortem_next_action(codes: list[str]) -> str:
    if any(code in codes for code in ["data_quality_miss"]):
        return "tighten_data_or_lower_confidence"
    if any(code in codes for code in ["context_risk_hit"]):
        return "tighten_context_gate"
    if any(code in codes for code in ["market_disagreement_miss"]):
        return "increase_market_respect"
    if any(code in codes for code in ["overconfident_loss"]):
        return "calibrate_probability_bucket"
    if "right_side_bad_price" in codes:
        return "tighten_entry_price"
    return "keep" if "validated_read" in codes else "review"


def _fight_key(fighter_a: object, fighter_b: object) -> str:
    return f"{_normalize_name(fighter_a)}||{_normalize_name(fighter_b)}"


def _normalize_name(value: object) -> str:
    return " ".join(_safe_text(value).lower().replace("’", "'").split())


def _safe_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def _safe_float(value: object, default: float | str | None = 0.0) -> float | str | None:
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
