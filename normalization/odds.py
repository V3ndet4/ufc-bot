from __future__ import annotations

import pandas as pd

from data_sources.odds_api import selection_name_for_row

BASE_REQUIRED_COLUMNS = {
    "event_id",
    "event_name",
    "start_time",
    "fighter_a",
    "fighter_b",
    "market",
    "selection",
    "book",
    "american_odds",
}


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = BASE_REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_list}")


def _validate_values(frame: pd.DataFrame) -> None:
    valid_selections = {"fighter_a", "fighter_b", "fight_goes_to_decision", "fight_doesnt_go_to_decision"}
    invalid_selection_mask = ~frame["selection"].isin(valid_selections)
    if invalid_selection_mask.any():
        invalid_rows = frame.loc[invalid_selection_mask, "selection"].tolist()
        raise ValueError(f"Invalid selection values: {invalid_rows}")

    if "projected_win_prob" in frame.columns:
        invalid_prob_mask = (frame["projected_win_prob"] < 0) | (frame["projected_win_prob"] > 1)
        if invalid_prob_mask.any():
            raise ValueError("projected_win_prob must be between 0 and 1")

def normalize_odds_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a raw odds frame into a stable schema."""
    _validate_columns(frame)
    normalized = frame.copy()
    for column in ["event_id", "event_name", "fighter_a", "fighter_b", "market", "book"]:
        normalized[column] = normalized[column].astype(str).str.strip()
    normalized["selection"] = normalized["selection"].astype(str).str.strip()
    normalized["american_odds"] = pd.to_numeric(normalized["american_odds"], errors="coerce")
    normalized = normalized.loc[normalized["american_odds"].notna()].copy()
    if normalized.empty:
        raise ValueError("No rows with valid american_odds were available after normalization")
    normalized["american_odds"] = normalized["american_odds"].astype(int)
    for optional_column in ["open_american_odds", "closing_american_odds"]:
        if optional_column in normalized.columns:
            normalized[optional_column] = pd.to_numeric(normalized[optional_column], errors="coerce").astype("Int64")
    if "projected_win_prob" in normalized.columns:
        normalized["projected_win_prob"] = normalized["projected_win_prob"].astype(float)
    _validate_values(normalized)
    normalized["selection_name"] = normalized.apply(selection_name_for_row, axis=1)
    return normalized
