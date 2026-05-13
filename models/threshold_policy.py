from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


DEFAULT_MIN_EDGE = 0.03
DEFAULT_MIN_MODEL_PROB = 0.60
DEFAULT_MIN_MODEL_CONFIDENCE = 0.60
DEFAULT_MIN_STATS_COMPLETENESS = 0.80
DEFAULT_EXCLUDE_FALLBACK_ROWS = True
DEFAULT_THRESHOLD_POLICY_PATH = Path("models") / "threshold_policy.json"


def default_threshold_policy_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_THRESHOLD_POLICY_PATH


def load_threshold_policy(path: str | Path | None) -> dict[str, object] | None:
    if not path:
        return None
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    with policy_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_scan_thresholds(
    *,
    min_edge: float,
    min_model_prob: float = DEFAULT_MIN_MODEL_PROB,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
    policy: dict[str, object] | None,
) -> dict[str, object]:
    if not policy:
        return {
            "min_edge": float(min_edge),
            "min_model_prob": float(min_model_prob),
            "min_model_confidence": float(min_model_confidence),
            "min_stats_completeness": float(min_stats_completeness),
            "exclude_fallback_rows": bool(exclude_fallback_rows),
            "policy_applied": False,
            "policy_summary": "",
        }

    selected = policy.get("selected", {})
    if not isinstance(selected, dict):
        selected = {}
    resolved_min_edge = max(float(min_edge), float(selected.get("min_edge", min_edge) or min_edge))
    resolved_min_model_prob = max(
        float(min_model_prob),
        float(selected.get("min_model_prob", min_model_prob) or min_model_prob),
    )
    resolved_min_model_confidence = max(
        float(min_model_confidence),
        float(selected.get("min_model_confidence", min_model_confidence) or min_model_confidence),
    )
    resolved_min_stats_completeness = max(
        float(min_stats_completeness),
        float(selected.get("min_stats_completeness", min_stats_completeness) or min_stats_completeness),
    )
    resolved_exclude_fallback_rows = bool(exclude_fallback_rows) or bool(
        selected.get("exclude_fallback_rows", exclude_fallback_rows)
    )
    summary = (
        f"edge>={resolved_min_edge:.1%}, "
        f"prob>={resolved_min_model_prob:.1%}, "
        f"confidence>={resolved_min_model_confidence:.2f}, "
        f"stats>={resolved_min_stats_completeness:.2f}, "
        f"fallback={'off' if resolved_exclude_fallback_rows else 'on'}"
    )
    return {
        "min_edge": resolved_min_edge,
        "min_model_prob": resolved_min_model_prob,
        "min_model_confidence": resolved_min_model_confidence,
        "min_stats_completeness": resolved_min_stats_completeness,
        "exclude_fallback_rows": resolved_exclude_fallback_rows,
        "policy_applied": True,
        "policy_summary": summary,
    }


def build_threshold_policy(
    frame: pd.DataFrame,
    *,
    min_graded_bets: int = 6,
) -> dict[str, object]:
    normalized = _normalize_training_frame(frame)
    baseline_settings = {
        "min_edge": DEFAULT_MIN_EDGE,
        "min_model_prob": DEFAULT_MIN_MODEL_PROB,
        "min_model_confidence": DEFAULT_MIN_MODEL_CONFIDENCE,
        "min_stats_completeness": DEFAULT_MIN_STATS_COMPLETENESS,
        "exclude_fallback_rows": DEFAULT_EXCLUDE_FALLBACK_ROWS,
    }
    baseline_result = _evaluate_thresholds(normalized, **baseline_settings)
    validation_frame = _build_walk_forward_validation_frame(normalized, min_graded_bets=min_graded_bets)
    baseline_validation_result = _evaluate_thresholds(validation_frame, **baseline_settings)
    min_validation_bets = max(3, min(min_graded_bets, math.ceil(min_graded_bets / 2)))
    if normalized.empty or len(normalized) < min_graded_bets:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "tracked_picks",
            "graded_bets": int(len(normalized)),
            "min_graded_bets": int(min_graded_bets),
            "min_validation_bets": int(min_validation_bets),
            "status": "insufficient_data",
            "baseline": baseline_result,
            "baseline_validation": baseline_validation_result,
            "validation_graded_bets": int(len(validation_frame)),
            "selected": baseline_result,
        }

    best_settings = baseline_settings.copy()
    best_validation_result = baseline_validation_result
    for min_edge in _candidate_values(normalized["effective_edge"], [0.03, 0.04, 0.05, 0.06, 0.08]):
        for min_model_prob in _candidate_values(
            normalized["effective_projected_prob"],
            [0.50, 0.55, 0.60, 0.65, 0.70],
        ):
            for min_model_confidence in _candidate_values(
                normalized["effective_confidence"],
                [0.55, 0.60, 0.65, 0.70, 0.75],
            ):
                for min_stats_completeness in _candidate_values(
                    normalized["effective_stats_completeness"],
                    [0.75, 0.80, 0.85, 0.90],
                ):
                    for exclude_fallback_rows in [True, False]:
                        settings = {
                            "min_edge": min_edge,
                            "min_model_prob": min_model_prob,
                            "min_model_confidence": min_model_confidence,
                            "min_stats_completeness": min_stats_completeness,
                            "exclude_fallback_rows": exclude_fallback_rows,
                        }
                        candidate = _evaluate_thresholds(validation_frame, **settings)
                        if int(candidate["graded_bets"]) < min_validation_bets:
                            continue
                        full_candidate = _evaluate_thresholds(normalized, **settings)
                        if int(full_candidate["graded_bets"]) < min_graded_bets:
                            continue
                        if _is_better_candidate(candidate, best_validation_result):
                            best_validation_result = candidate
                            best_settings = settings

    should_use_baseline = (
        int(best_validation_result["graded_bets"]) < min_validation_bets
        or int(_evaluate_thresholds(normalized, **best_settings)["graded_bets"]) < min_graded_bets
        or not _is_better_candidate(best_validation_result, baseline_validation_result)
        or (
            float(best_validation_result["roi_pct"]) < 0.0
            and float(best_validation_result["avg_clv_delta"]) < 0.0
        )
    )
    selected_settings = baseline_settings if should_use_baseline else best_settings
    selected = _evaluate_thresholds(normalized, **selected_settings)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "tracked_picks",
        "graded_bets": int(len(normalized)),
        "min_graded_bets": int(min_graded_bets),
        "min_validation_bets": int(min_validation_bets),
        "validation_graded_bets": int(len(validation_frame)),
        "status": "baseline" if should_use_baseline else "optimized",
        "sample_warning": _sample_warning(len(normalized)),
        "baseline": baseline_result,
        "baseline_validation": baseline_validation_result,
        "selected_validation": baseline_validation_result if should_use_baseline else best_validation_result,
        "selected": selected,
    }


def _normalize_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    grade_status = working.get("grade_status", pd.Series("graded", index=working.index)).fillna("graded").astype(str).str.lower()
    working = working.loc[grade_status != "pending"].copy()
    if working.empty:
        return working
    working = _dedupe_tracked_training_rows(working)

    working["effective_edge"] = _first_numeric_series(working, ["chosen_expression_edge", "edge"], 0.0)
    probability_columns = ["chosen_expression_prob", "model_projected_win_prob", "projected_win_prob"]
    probability_default = DEFAULT_MIN_MODEL_PROB if any(column in working.columns for column in probability_columns) else 1.0
    working["effective_projected_prob"] = _first_numeric_series(
        working,
        probability_columns,
        probability_default,
    ).clip(0.0, 1.0)
    working["effective_confidence"] = _first_numeric_series(working, ["model_confidence"], 0.0)
    data_quality = _first_numeric_series(working, ["data_quality"], 0.0)
    selection_quality = _first_numeric_series(working, ["selection_stats_completeness"], data_quality)
    working["effective_stats_completeness"] = pd.concat([data_quality, selection_quality], axis=1).min(axis=1)
    fallback_used = _first_numeric_series(working, ["selection_fallback_used"], 0.0)
    fallback_penalty = _first_numeric_series(working, ["fallback_penalty"], 0.0)
    working["effective_fallback_flag"] = ((fallback_used > 0.0) | (fallback_penalty > 0.0)).astype(int)
    working["stake_amount"] = _first_numeric_series(working, ["chosen_expression_stake", "suggested_stake"], 0.0)
    working["profit_amount"] = _first_numeric_series(working, ["profit"], 0.0)
    working["clv_delta"] = _first_numeric_series(working, ["clv_delta"], 0.0)
    working["tracked_at_ts"] = _first_datetime_series(working, ["tracked_at", "start_time"])
    actual_result = working.get("actual_result", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    working["is_win"] = (actual_result == "win").astype(int)
    working["is_loss"] = (actual_result == "loss").astype(int)
    working["is_push"] = (actual_result == "push").astype(int)
    working = working.sort_values(["tracked_at_ts"], kind="stable").reset_index(drop=True)
    return working


def _dedupe_tracked_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["event_id", "fight_key", "tracked_market_key", "tracked_selection_key"]
    if frame.empty or any(column not in frame.columns for column in key_columns):
        return frame.copy()

    working = frame.copy()
    sort_columns = [column for column in ["tracked_at", "pick_id"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, kind="mergesort")
    return working.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)


def _first_numeric_series(frame: pd.DataFrame, columns: list[str], default: float) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _first_datetime_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in frame.columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
            if parsed.notna().any():
                fallback = pd.date_range("2000-01-01", periods=len(frame), freq="min", tz="UTC")
                return parsed.where(parsed.notna(), fallback)
    return pd.date_range("2000-01-01", periods=len(frame), freq="min", tz="UTC")


def _build_walk_forward_validation_frame(frame: pd.DataFrame, *, min_graded_bets: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ordered = frame.sort_values(["tracked_at_ts"], kind="stable").reset_index(drop=True)
    total_rows = len(ordered)
    if total_rows <= min_graded_bets:
        return ordered.iloc[0:0].copy()

    validation_window = max(min_graded_bets, total_rows // 4)
    validation_slices: list[pd.DataFrame] = []
    start = min_graded_bets
    while start < total_rows:
        stop = min(total_rows, start + validation_window)
        validation_slices.append(ordered.iloc[start:stop].copy())
        start = stop

    if not validation_slices:
        return ordered.iloc[0:0].copy()
    return pd.concat(validation_slices, ignore_index=True)


def _candidate_values(series: pd.Series, defaults: list[float]) -> list[float]:
    candidates = {round(float(value), 3) for value in defaults}
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if not cleaned.empty:
        quantiles = cleaned.quantile([0.25, 0.50, 0.65, 0.75]).tolist()
        candidates.update(round(float(value), 3) for value in quantiles)
    return sorted(candidates)


def _evaluate_thresholds(
    frame: pd.DataFrame,
    *,
    min_edge: float,
    min_model_prob: float,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
) -> dict[str, object]:
    if frame.empty:
        return _threshold_result(
            min_edge=min_edge,
            min_model_prob=min_model_prob,
            min_model_confidence=min_model_confidence,
            min_stats_completeness=min_stats_completeness,
            exclude_fallback_rows=exclude_fallback_rows,
            graded_bets=0,
            wins=0,
            losses=0,
            pushes=0,
            win_rate=0.0,
            loss_rate=0.0,
            total_stake=0.0,
            total_profit=0.0,
            roi_pct=0.0,
            avg_clv_delta=0.0,
            avg_edge_at_pick=0.0,
            avg_model_prob_at_pick=0.0,
            score=0.0,
        )
    filtered = frame.loc[
        (frame["effective_edge"] >= min_edge)
        & (frame["effective_projected_prob"] >= min_model_prob)
        & (frame["effective_confidence"] >= min_model_confidence)
        & (frame["effective_stats_completeness"] >= min_stats_completeness)
    ].copy()
    if exclude_fallback_rows:
        filtered = filtered.loc[filtered["effective_fallback_flag"] <= 0].copy()

    graded_bets = int(len(filtered))
    total_stake = float(filtered["stake_amount"].sum()) if graded_bets > 0 else 0.0
    total_profit = float(filtered["profit_amount"].sum()) if graded_bets > 0 else 0.0
    roi_pct = (total_profit / total_stake) * 100 if total_stake > 0 else 0.0
    avg_clv_delta = float(filtered["clv_delta"].mean()) if graded_bets > 0 else 0.0
    avg_edge_at_pick = float(filtered["effective_edge"].mean()) if graded_bets > 0 else 0.0
    avg_model_prob_at_pick = float(filtered["effective_projected_prob"].mean()) if graded_bets > 0 else 0.0
    wins = int(filtered["is_win"].sum()) if graded_bets > 0 else 0
    losses = int(filtered["is_loss"].sum()) if graded_bets > 0 else 0
    pushes = int(filtered["is_push"].sum()) if graded_bets > 0 else 0
    decision_count = wins + losses
    win_rate = (wins / decision_count) if decision_count > 0 else 0.0
    loss_rate = (losses / decision_count) if decision_count > 0 else 0.0
    clv_score = max(-10.0, min(10.0, avg_clv_delta))

    score = (
        (roi_pct * 0.65)
        + (clv_score * 1.2)
        + (min(graded_bets, 30) * 0.50)
        + (avg_edge_at_pick * 100 * 0.15)
        + (avg_model_prob_at_pick * 4.0)
        + (math.log1p(max(0, decision_count)) * 2.0)
    )
    if roi_pct < 0:
        score += roi_pct * 0.25
    if avg_clv_delta < 0:
        score += clv_score * 1.2
    if total_profit < 0:
        score -= 6.0

    return _threshold_result(
        min_edge=min_edge,
        min_model_prob=min_model_prob,
        min_model_confidence=min_model_confidence,
        min_stats_completeness=min_stats_completeness,
        exclude_fallback_rows=exclude_fallback_rows,
        graded_bets=graded_bets,
        wins=wins,
        losses=losses,
        pushes=pushes,
        win_rate=win_rate,
        loss_rate=loss_rate,
        total_stake=total_stake,
        total_profit=total_profit,
        roi_pct=roi_pct,
        avg_clv_delta=avg_clv_delta,
        avg_edge_at_pick=avg_edge_at_pick,
        avg_model_prob_at_pick=avg_model_prob_at_pick,
        score=score,
    )


def _threshold_result(
    *,
    min_edge: float,
    min_model_prob: float,
    min_model_confidence: float,
    min_stats_completeness: float,
    exclude_fallback_rows: bool,
    graded_bets: int,
    wins: int,
    losses: int,
    pushes: int,
    win_rate: float,
    loss_rate: float,
    total_stake: float,
    total_profit: float,
    roi_pct: float,
    avg_clv_delta: float,
    avg_edge_at_pick: float,
    avg_model_prob_at_pick: float,
    score: float,
) -> dict[str, object]:
    return {
        "min_edge": round(float(min_edge), 3),
        "min_model_prob": round(float(min_model_prob), 3),
        "min_model_confidence": round(float(min_model_confidence), 3),
        "min_stats_completeness": round(float(min_stats_completeness), 3),
        "exclude_fallback_rows": bool(exclude_fallback_rows),
        "graded_bets": int(graded_bets),
        "wins": int(wins),
        "losses": int(losses),
        "pushes": int(pushes),
        "win_rate": round(float(win_rate), 4),
        "loss_rate": round(float(loss_rate), 4),
        "total_stake": round(float(total_stake), 2),
        "total_profit": round(float(total_profit), 2),
        "roi_pct": round(float(roi_pct), 2),
        "avg_clv_delta": round(float(avg_clv_delta), 4),
        "avg_edge_at_pick": round(float(avg_edge_at_pick), 4),
        "avg_model_prob_at_pick": round(float(avg_model_prob_at_pick), 4),
        "score": round(float(score), 3),
    }


def _metric(result: dict[str, object], key: str) -> float:
    return float(result.get(key, 0.0) or 0.0)


def _is_better_candidate(candidate: dict[str, object], incumbent: dict[str, object]) -> bool:
    candidate_profit = _metric(candidate, "total_profit")
    incumbent_profit = _metric(incumbent, "total_profit")
    candidate_roi = _metric(candidate, "roi_pct")
    incumbent_roi = _metric(incumbent, "roi_pct")
    candidate_score = _metric(candidate, "score")
    incumbent_score = _metric(incumbent, "score")
    candidate_clv = _metric(candidate, "avg_clv_delta")
    incumbent_clv = _metric(incumbent, "avg_clv_delta")
    candidate_loss_rate = _metric(candidate, "loss_rate")
    incumbent_loss_rate = _metric(incumbent, "loss_rate")
    candidate_bets = int(candidate.get("graded_bets", 0) or 0)
    incumbent_bets = int(incumbent.get("graded_bets", 0) or 0)

    profit_delta = candidate_profit - incumbent_profit
    roi_delta = candidate_roi - incumbent_roi

    # Do not "optimize" into a meaningfully less profitable slice just because
    # CLV, sample size, or a composite score improved.
    if profit_delta < -25.0:
        return False
    if profit_delta < -5.0 and roi_delta <= 0.0:
        return False
    if roi_delta < -5.0:
        return False
    if roi_delta < -3.0 and profit_delta <= 0.0:
        return False
    if candidate_bets < incumbent_bets and roi_delta < -2.0:
        return False
    if candidate_loss_rate > incumbent_loss_rate + 0.15 and profit_delta <= 5.0:
        return False

    # Prefer realized dollars first, then ROI. Composite score and CLV are only
    # tiebreakers once realized outcomes are at least comparable.
    if profit_delta > 25.0:
        return True
    if profit_delta > 5.0 and roi_delta >= -1.0:
        return True
    if roi_delta > 3.0 and profit_delta >= -5.0:
        return True

    if abs(profit_delta) <= 5.0:
        if roi_delta > 1.0:
            return True
        if roi_delta < -1.0:
            return False

    if abs(profit_delta) <= 5.0 and abs(roi_delta) <= 1.0:
        if candidate_score > incumbent_score + 1.0:
            return True
        if candidate_score < incumbent_score - 1.0:
            return False
        if candidate_clv > incumbent_clv + 0.01:
            return True
        if candidate_clv < incumbent_clv - 0.01:
            return False
        if candidate_bets > incumbent_bets:
            return True

    return False


def _sample_warning(row_count: int) -> str:
    if row_count < 20:
        return "data_limited: threshold policy is based on fewer than 20 de-duplicated graded picks"
    return ""
