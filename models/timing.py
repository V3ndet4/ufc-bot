from __future__ import annotations

from typing import Any

import pandas as pd

from models.ev import implied_probability


TIMING_DEFAULT_COLUMNS = [
    "timing_snapshot_count",
    "timing_book_count",
    "timing_open_implied_prob",
    "timing_latest_implied_prob",
    "timing_implied_change",
    "timing_velocity_per_hour",
    "timing_volatility",
    "timing_book_dispersion",
    "timing_score",
    "timing_signal",
    "timing_action",
    "timing_reason",
]


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text or default


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _implied_prob(value: object) -> float:
    try:
        odds = int(float(value))
    except (TypeError, ValueError):
        return 0.0
    return float(implied_probability(odds))


def _select_history(history: pd.DataFrame | None, row: dict[str, Any]) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame()

    working = history.copy()
    for column in ["event_id", "fighter_a", "fighter_b", "market", "selection", "selection_name", "book"]:
        row_value = _safe_text(row.get(column, ""))
        if not row_value or column not in working.columns:
            continue
        candidate = working[column].fillna("").astype(str).str.strip()
        working = working.loc[candidate == row_value].copy()
        if working.empty:
            return working
    if "snapshot_time" in working.columns:
        working["snapshot_time"] = pd.to_datetime(working["snapshot_time"], errors="coerce", utc=True)
        working = working.dropna(subset=["snapshot_time"]).sort_values("snapshot_time")
    return working.reset_index(drop=True)


def _row_fallback_metrics(row: dict[str, Any]) -> dict[str, Any]:
    open_odds = row.get("open_american_odds")
    latest_odds = row.get("american_odds")
    open_implied = _implied_prob(open_odds)
    latest_implied = _implied_prob(latest_odds)
    implied_change = latest_implied - open_implied
    reasons = ["live line only"]
    score = 50.0
    if implied_change >= 0.03:
        score += 12.0
        reasons.append("line moving toward pick")
    elif implied_change <= -0.03:
        score -= 12.0
        reasons.append("line moving away")
    score = _clamp(score, 0.0, 100.0)
    action = "Bet now" if score >= 62.0 and implied_change >= 0.01 else "Monitor"
    signal = "steam" if implied_change >= 0.03 else "drift" if implied_change <= -0.03 else "stable"
    return {
        "timing_snapshot_count": 0,
        "timing_book_count": 0,
        "timing_open_implied_prob": round(open_implied, 4) if open_implied else 0.0,
        "timing_latest_implied_prob": round(latest_implied, 4) if latest_implied else 0.0,
        "timing_implied_change": round(implied_change, 4),
        "timing_velocity_per_hour": 0.0,
        "timing_volatility": abs(round(implied_change, 4)),
        "timing_book_dispersion": 0.0,
        "timing_score": round(score, 2),
        "timing_signal": signal,
        "timing_action": action,
        "timing_reason": ", ".join(dict.fromkeys(reasons)),
    }


def _timing_metrics(row: dict[str, Any], history: pd.DataFrame | None) -> dict[str, Any]:
    working = _select_history(history, row)
    if working.empty:
        return _row_fallback_metrics(row)

    if "american_odds" not in working.columns:
        return _row_fallback_metrics(row)

    working["implied_prob"] = working["american_odds"].apply(_implied_prob)
    if working["implied_prob"].dropna().empty:
        return _row_fallback_metrics(row)

    snapshot_count = int(len(working))
    first = working.iloc[0]
    last = working.iloc[-1]
    first_time = pd.to_datetime(first.get("snapshot_time"), utc=True, errors="coerce")
    last_time = pd.to_datetime(last.get("snapshot_time"), utc=True, errors="coerce")
    span_minutes = max((last_time - first_time).total_seconds() / 60.0, 1.0) if pd.notna(first_time) and pd.notna(last_time) else 1.0

    open_implied = float(working["implied_prob"].iloc[0])
    latest_implied = float(working["implied_prob"].iloc[-1])
    implied_change = latest_implied - open_implied
    volatility = float(working["implied_prob"].max() - working["implied_prob"].min())
    velocity_per_hour = implied_change / max(span_minutes / 60.0, 1.0 / 60.0)

    latest_time = working["snapshot_time"].max() if "snapshot_time" in working.columns else pd.NaT
    latest_slice = working if pd.isna(latest_time) or "snapshot_time" not in working.columns else working.loc[working["snapshot_time"] == latest_time].copy()
    if latest_slice.empty:
        latest_slice = working.tail(1).copy()
    book_count = int(latest_slice["book"].astype(str).nunique()) if "book" in latest_slice.columns else int(len(latest_slice))
    book_dispersion_value = pd.to_numeric(latest_slice["implied_prob"], errors="coerce").dropna().std(ddof=0) if not latest_slice.empty else 0.0
    book_dispersion = 0.0 if pd.isna(book_dispersion_value) else float(book_dispersion_value)

    score = 50.0
    reasons: list[str] = []

    if snapshot_count < 2:
        score -= 10.0
        reasons.append("thin snapshot sample")
    else:
        reasons.append(f"{snapshot_count} snapshots")
    if implied_change >= 0.03:
        score += 18.0
        reasons.append("line moving toward pick")
    elif implied_change <= -0.03:
        score -= 18.0
        reasons.append("line moving away")
    if velocity_per_hour >= 0.02:
        score += 8.0
        reasons.append("positive momentum")
    elif velocity_per_hour <= -0.02:
        score -= 8.0
        reasons.append("negative momentum")
    if volatility <= 0.03:
        score += 6.0
        reasons.append("stable market")
    elif volatility >= 0.08:
        score -= 8.0
        reasons.append("volatile market")
    if book_count >= 2 and book_dispersion <= 0.02:
        score += 4.0
        reasons.append("books aligned")
    elif book_dispersion >= 0.05:
        score -= 6.0
        reasons.append("books split")
    if snapshot_count >= 4 and abs(implied_change) >= 0.02:
        score += 4.0
        reasons.append("enough time-series signal")

    score = _clamp(score, 0.0, 100.0)
    if snapshot_count < 2:
        action = "Monitor"
    elif score >= 68.0 and implied_change >= 0.01:
        action = "Bet now"
    elif score <= 40.0 or implied_change <= -0.03:
        action = "Wait"
    else:
        action = "Monitor"

    if snapshot_count < 2:
        signal = "thin"
    elif implied_change >= 0.03 and score >= 68.0:
        signal = "steam"
    elif implied_change <= -0.03:
        signal = "drift"
    elif abs(implied_change) < 0.02 and volatility <= 0.04:
        signal = "stable"
    else:
        signal = "mixed"

    return {
        "timing_snapshot_count": snapshot_count,
        "timing_book_count": book_count,
        "timing_open_implied_prob": round(open_implied, 4),
        "timing_latest_implied_prob": round(latest_implied, 4),
        "timing_implied_change": round(implied_change, 4),
        "timing_velocity_per_hour": round(velocity_per_hour, 4),
        "timing_volatility": round(volatility, 4),
        "timing_book_dispersion": round(book_dispersion, 4),
        "timing_score": round(score, 2),
        "timing_signal": signal,
        "timing_action": action,
        "timing_reason": ", ".join(dict.fromkeys(reasons)),
    }


def attach_timing_signals(frame: pd.DataFrame, snapshot_history: pd.DataFrame | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    timing_rows = [_timing_metrics(row, snapshot_history) for row in enriched.to_dict("records")]
    timing_frame = pd.DataFrame.from_records(timing_rows, index=enriched.index)
    return pd.concat([enriched, timing_frame], axis=1)
