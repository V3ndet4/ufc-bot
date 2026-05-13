from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_sources.sherdog import normalize_gym_name, normalize_name


DEFAULT_GYM_OVERRIDES_PATH = Path(__file__).resolve().parents[1] / "data" / "fighter_gym_overrides.csv"

GYM_OVERRIDE_COLUMNS = [
    "fighter_name",
    "gym_name",
    "gym_tier",
    "gym_source",
    "gym_page_url",
    "previous_gym_name",
    "gym_changed_flag",
    "new_gym_flag",
    "camp_change_flag",
    "news_alert_count",
    "news_radar_score",
    "news_radar_label",
    "news_radar_summary",
    "context_notes",
    "source_url",
    "source_notes",
    "verified_at",
    "active",
]

GYM_NUMERIC_COLUMNS = [
    "gym_score",
    "gym_fighter_count",
    "gym_total_wins",
    "gym_total_losses",
    "gym_total_draws",
    "gym_win_rate",
    "gym_elite_fighter_count",
    "gym_changed_flag",
]

CONTEXT_NUMERIC_COLUMNS = [
    "new_gym_flag",
    "camp_change_flag",
    "news_alert_count",
    "news_radar_score",
]


def load_fighter_gym_overrides(path: str | Path | None = DEFAULT_GYM_OVERRIDES_PATH) -> pd.DataFrame:
    if not path:
        return _empty_overrides_frame()
    override_path = Path(path)
    if not override_path.exists():
        return _empty_overrides_frame()

    frame = pd.read_csv(override_path, keep_default_na=False)
    if "fighter_name" not in frame.columns:
        raise ValueError("fighter gym overrides CSV must contain fighter_name")

    for column in GYM_OVERRIDE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame["fighter_name"] = frame["fighter_name"].astype(str).str.strip()
    frame = frame.loc[frame["fighter_name"] != ""].copy()
    frame["fighter_name_normalized"] = frame["fighter_name"].map(normalize_name)
    frame["gym_name_normalized"] = frame["gym_name"].map(normalize_gym_name)
    frame["active"] = frame["active"].astype(str).str.strip().str.lower()
    frame = frame.loc[~frame["active"].isin({"0", "false", "no", "inactive"})].copy()
    return frame.reset_index(drop=True)


def apply_fighter_gym_overrides(
    frame: pd.DataFrame,
    overrides: pd.DataFrame | None,
    *,
    timestamp: str | None = None,
) -> pd.DataFrame:
    if frame.empty or overrides is None or overrides.empty:
        return frame.copy()
    if "fighter_name" not in frame.columns:
        return frame.copy()

    now = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    updated = frame.copy()
    updated["fighter_name_normalized"] = updated["fighter_name"].astype(str).str.strip().map(normalize_name)

    for column in [
        "gym_source",
        "gym_name",
        "gym_name_normalized",
        "gym_page_url",
        "gym_tier",
        "previous_gym_name",
        "last_changed_at",
        "last_seen_at",
    ]:
        if column not in updated.columns:
            updated[column] = ""
        updated[column] = updated[column].fillna("").astype(str)
    for column in GYM_NUMERIC_COLUMNS:
        if column not in updated.columns:
            updated[column] = 0.0

    override_lookup = _latest_override_lookup(overrides)
    for row_index, row in updated.iterrows():
        override = override_lookup.get(row["fighter_name_normalized"])
        if override is None:
            continue

        current_gym = _clean_text(row.get("gym_name", ""))
        override_gym = _clean_text(override.get("gym_name", ""))
        if override_gym:
            updated.at[row_index, "gym_name"] = override_gym
            updated.at[row_index, "gym_name_normalized"] = normalize_gym_name(override_gym)

        _set_text_if_present(updated, row_index, override, "gym_source", default="manual_current_camp")
        _set_text_if_present(updated, row_index, override, "gym_page_url")
        _set_text_allow_blank(updated, row_index, override, "gym_tier")

        previous_gym = _clean_text(override.get("previous_gym_name", ""))
        if previous_gym or _clean_text(override.get("gym_changed_flag", "")) != "":
            updated.at[row_index, "previous_gym_name"] = previous_gym
        elif current_gym and override_gym and normalize_gym_name(current_gym) != normalize_gym_name(override_gym):
            updated.at[row_index, "previous_gym_name"] = current_gym

        changed_flag = _int_override_value(override.get("gym_changed_flag", ""))
        if changed_flag is None and current_gym and override_gym and normalize_gym_name(current_gym) != normalize_gym_name(override_gym):
            changed_flag = 1
        if changed_flag is not None:
            updated.at[row_index, "gym_changed_flag"] = max(_safe_float(row.get("gym_changed_flag", 0)), changed_flag)

        if int(float(updated.at[row_index, "gym_changed_flag"] or 0)) >= 1:
            updated.at[row_index, "last_changed_at"] = _clean_text(override.get("verified_at", "")) or now
        updated.at[row_index, "last_seen_at"] = now

        for column in [
            "gym_score",
            "gym_fighter_count",
            "gym_total_wins",
            "gym_total_losses",
            "gym_total_draws",
            "gym_win_rate",
            "gym_elite_fighter_count",
        ]:
            if column in override.index and _clean_text(override.get(column, "")) != "":
                updated.at[row_index, column] = _safe_float(override.get(column, 0), 0.0)

    return updated


def apply_context_gym_overrides(
    frame: pd.DataFrame,
    overrides: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame.empty or overrides is None or overrides.empty:
        return frame.copy()
    if "fighter_name" not in frame.columns:
        return frame.copy()

    updated = frame.copy()
    updated["fighter_name"] = updated["fighter_name"].astype(str).str.strip()
    updated["_fighter_name_normalized"] = updated["fighter_name"].map(normalize_name)

    for column in CONTEXT_NUMERIC_COLUMNS:
        if column not in updated.columns:
            updated[column] = 0
    for column in ["news_radar_label", "news_radar_summary", "context_notes"]:
        if column not in updated.columns:
            updated[column] = ""

    override_lookup = _latest_override_lookup(overrides)
    for row_index, row in updated.iterrows():
        override = override_lookup.get(row["_fighter_name_normalized"])
        if override is None:
            continue

        for column in ["new_gym_flag", "camp_change_flag", "news_alert_count"]:
            override_value = _int_override_value(override.get(column, ""))
            if override_value is not None:
                updated.at[row_index, column] = max(_safe_float(row.get(column, 0), 0), override_value)

        radar_score = _float_override_value(override.get("news_radar_score", ""))
        if radar_score is not None:
            updated.at[row_index, "news_radar_score"] = max(
                _safe_float(row.get("news_radar_score", 0.0), 0.0),
                radar_score,
            )

        _set_text_if_present(updated, row_index, override, "news_radar_label")
        _set_text_if_present(updated, row_index, override, "news_radar_summary")
        updated.at[row_index, "context_notes"] = _merge_notes(
            _clean_text(row.get("context_notes", "")),
            _clean_text(override.get("context_notes", "")),
        )

    updated = updated.drop(columns=["_fighter_name_normalized"])
    return updated


def _empty_overrides_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=GYM_OVERRIDE_COLUMNS)
    frame["fighter_name_normalized"] = pd.Series(dtype="object")
    frame["gym_name_normalized"] = pd.Series(dtype="object")
    return frame


def _latest_override_lookup(overrides: pd.DataFrame) -> dict[str, pd.Series]:
    working = overrides.copy()
    if "fighter_name_normalized" not in working.columns:
        working["fighter_name_normalized"] = working["fighter_name"].map(normalize_name)
    if "verified_at" in working.columns:
        working["_verified_at"] = pd.to_datetime(working["verified_at"], errors="coerce", utc=True)
        working = working.sort_values(by=["fighter_name_normalized", "_verified_at"], na_position="first")
    return {
        str(row["fighter_name_normalized"]): row
        for _, row in working.drop_duplicates(subset=["fighter_name_normalized"], keep="last").iterrows()
    }


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        return default
    return default if pd.isna(numeric) else float(numeric)


def _float_override_value(value: object) -> float | None:
    text = _clean_text(value)
    if text == "":
        return None
    return _safe_float(text, 0.0)


def _int_override_value(value: object) -> int | None:
    numeric = _float_override_value(value)
    return None if numeric is None else int(round(numeric))


def _set_text_if_present(
    frame: pd.DataFrame,
    row_index: int,
    source: pd.Series,
    column: str,
    *,
    default: str = "",
) -> None:
    value = _clean_text(source.get(column, ""))
    if not value and default:
        value = default
    if value:
        frame.at[row_index, column] = value


def _set_text_allow_blank(frame: pd.DataFrame, row_index: int, source: pd.Series, column: str) -> None:
    if column in source.index:
        frame.at[row_index, column] = _clean_text(source.get(column, ""))


def _merge_notes(existing: str, addition: str) -> str:
    existing = _clean_text(existing)
    addition = _clean_text(addition)
    if not addition:
        return existing
    if not existing:
        return addition
    if addition.lower() in existing.lower():
        return existing
    return f"{existing} | {addition}"
