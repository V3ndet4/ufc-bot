from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.ev import implied_probability


DEFAULT_SIDE_MODEL_PATH = Path("models") / "side_model.pkl"

NUMERIC_FEATURE_COLUMNS = [
    "american_odds",
    "selection_model_base_prob",
    "selection_raw_model_prob",
    "selection_implied_prob",
    "selection_model_edge",
    "model_confidence",
    "data_quality",
    "line_movement_toward_fighter",
    "market_blend_weight",
    "selection_days_since_last_fight",
    "selection_ufc_fight_count",
    "selection_ufc_debut_flag",
    "selection_recent_finish_damage",
    "selection_recent_ko_damage",
    "selection_recent_damage_score",
    "selection_first_round_finish_rate",
    "selection_finish_loss_rate",
    "selection_recent_grappling_rate",
    "selection_control_avg",
    "selection_recent_control_avg",
    "selection_strike_margin_last_3",
    "selection_grappling_rate_last_3",
    "selection_control_avg_last_3",
    "selection_strike_pace_last_3",
    "selection_result_score_last_3",
    "selection_knockdown_avg",
    "selection_distance_strike_share",
    "selection_clinch_strike_share",
    "selection_ground_strike_share",
    "selection_context_instability",
    "selection_gym_score",
    "selection_matchup_striking_edge",
    "selection_matchup_grappling_edge",
    "selection_matchup_control_edge",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "selection",
    "segment_label",
    "book",
    "selection_history_style_label",
    "opponent_history_style_label",
]


def default_side_model_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_SIDE_MODEL_PATH


def _safe_numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _safe_text_series(frame: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _select_series(frame: pd.DataFrame, fighter_a_column: str, fighter_b_column: str, default: float = 0.0) -> pd.Series:
    selection = _safe_text_series(frame, "selection", "fighter_a")
    fighter_a_values = _safe_numeric_series(frame, fighter_a_column, default)
    fighter_b_values = _safe_numeric_series(frame, fighter_b_column, default)
    return fighter_a_values.where(selection == "fighter_a", fighter_b_values)


def _oriented_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    selection = _safe_text_series(frame, "selection", "fighter_a")
    values = _safe_numeric_series(frame, column, default)
    return values.where(selection == "fighter_a", -values)


def prepare_side_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    selection = _safe_text_series(prepared, "selection", "fighter_a")
    prepared["selection"] = selection
    prepared["american_odds"] = _safe_numeric_series(prepared, "american_odds", 0.0)

    if "selection_model_base_prob" not in prepared.columns:
        if "projected_fighter_a_win_prob" in prepared.columns:
            fighter_a_prob = _safe_numeric_series(prepared, "projected_fighter_a_win_prob", 0.5)
            prepared["selection_model_base_prob"] = fighter_a_prob.where(selection == "fighter_a", 1 - fighter_a_prob)
        else:
            prepared["selection_model_base_prob"] = _safe_numeric_series(prepared, "model_projected_win_prob", 0.5)

    if "selection_raw_model_prob" not in prepared.columns:
        if "raw_projected_fighter_a_win_prob" in prepared.columns:
            raw_fighter_a_prob = _safe_numeric_series(prepared, "raw_projected_fighter_a_win_prob", 0.5)
            prepared["selection_raw_model_prob"] = raw_fighter_a_prob.where(selection == "fighter_a", 1 - raw_fighter_a_prob)
        else:
            prepared["selection_raw_model_prob"] = _safe_numeric_series(
                prepared,
                "base_model_projected_win_prob",
                0.0,
            ).where(
                _safe_numeric_series(prepared, "base_model_projected_win_prob", -1.0) >= 0.0,
                _safe_numeric_series(prepared, "model_projected_win_prob", 0.5),
            )

    if "selection_implied_prob" not in prepared.columns:
        if "implied_prob" in prepared.columns:
            prepared["selection_implied_prob"] = _safe_numeric_series(prepared, "implied_prob", 0.5)
        else:
            prepared["selection_implied_prob"] = prepared["american_odds"].apply(
                lambda value: implied_probability(int(value)) if value else 0.5
            )

    if "selection_model_edge" not in prepared.columns:
        if "edge" in prepared.columns:
            prepared["selection_model_edge"] = _safe_numeric_series(prepared, "edge", 0.0)
        else:
            prepared["selection_model_edge"] = prepared["selection_model_base_prob"] - prepared["selection_implied_prob"]

    if "selection_days_since_last_fight" not in prepared.columns:
        prepared["selection_days_since_last_fight"] = _select_series(prepared, "a_days_since_last_fight", "b_days_since_last_fight", 999.0)
    if "selection_ufc_fight_count" not in prepared.columns:
        prepared["selection_ufc_fight_count"] = _select_series(prepared, "a_ufc_fight_count", "b_ufc_fight_count", 0.0)
    if "selection_ufc_debut_flag" not in prepared.columns:
        prepared["selection_ufc_debut_flag"] = _select_series(prepared, "a_ufc_debut_flag", "b_ufc_debut_flag", 0.0)
    if "selection_recent_finish_damage" not in prepared.columns:
        prepared["selection_recent_finish_damage"] = _select_series(prepared, "a_recent_finish_loss_365d", "b_recent_finish_loss_365d", 0.0)
    if "selection_recent_ko_damage" not in prepared.columns:
        prepared["selection_recent_ko_damage"] = _select_series(prepared, "a_recent_ko_loss_365d", "b_recent_ko_loss_365d", 0.0)
    if "selection_recent_damage_score" not in prepared.columns:
        prepared["selection_recent_damage_score"] = _select_series(prepared, "a_recent_damage_score", "b_recent_damage_score", 0.0)
    if "selection_first_round_finish_rate" not in prepared.columns:
        prepared["selection_first_round_finish_rate"] = _select_series(prepared, "a_first_round_finish_rate", "b_first_round_finish_rate", 0.0)
    if "selection_finish_loss_rate" not in prepared.columns:
        prepared["selection_finish_loss_rate"] = _select_series(prepared, "a_finish_loss_rate", "b_finish_loss_rate", 0.0)
    if "selection_recent_grappling_rate" not in prepared.columns:
        prepared["selection_recent_grappling_rate"] = _select_series(prepared, "a_recent_grappling_rate", "b_recent_grappling_rate", 0.0)
    if "selection_control_avg" not in prepared.columns:
        prepared["selection_control_avg"] = _select_series(prepared, "a_control_avg", "b_control_avg", 0.0)
    if "selection_recent_control_avg" not in prepared.columns:
        prepared["selection_recent_control_avg"] = _select_series(prepared, "a_recent_control_avg", "b_recent_control_avg", 0.0)
    if "selection_strike_margin_last_3" not in prepared.columns:
        prepared["selection_strike_margin_last_3"] = _select_series(prepared, "a_strike_margin_last_3", "b_strike_margin_last_3", 0.0)
    if "selection_grappling_rate_last_3" not in prepared.columns:
        prepared["selection_grappling_rate_last_3"] = _select_series(prepared, "a_grappling_rate_last_3", "b_grappling_rate_last_3", 0.0)
    if "selection_control_avg_last_3" not in prepared.columns:
        prepared["selection_control_avg_last_3"] = _select_series(prepared, "a_control_avg_last_3", "b_control_avg_last_3", 0.0)
    if "selection_strike_pace_last_3" not in prepared.columns:
        prepared["selection_strike_pace_last_3"] = _select_series(prepared, "a_strike_pace_last_3", "b_strike_pace_last_3", 0.0)
    if "selection_result_score_last_3" not in prepared.columns:
        prepared["selection_result_score_last_3"] = _select_series(prepared, "a_result_score_last_3", "b_result_score_last_3", 0.0)
    if "selection_knockdown_avg" not in prepared.columns:
        prepared["selection_knockdown_avg"] = _select_series(prepared, "a_knockdown_avg", "b_knockdown_avg", 0.0)
    if "selection_distance_strike_share" not in prepared.columns:
        prepared["selection_distance_strike_share"] = _select_series(prepared, "a_distance_strike_share", "b_distance_strike_share", 0.0)
    if "selection_clinch_strike_share" not in prepared.columns:
        prepared["selection_clinch_strike_share"] = _select_series(prepared, "a_clinch_strike_share", "b_clinch_strike_share", 0.0)
    if "selection_ground_strike_share" not in prepared.columns:
        prepared["selection_ground_strike_share"] = _select_series(prepared, "a_ground_strike_share", "b_ground_strike_share", 0.0)
    if "selection_context_instability" not in prepared.columns:
        if "selection_context_instability" not in prepared.columns and "context_stability_diff" in prepared.columns:
            prepared["selection_context_instability"] = (-_oriented_series(prepared, "context_stability_diff", 0.0)).clip(lower=0.0)
        else:
            prepared["selection_context_instability"] = 0.0
    if "selection_gym_score" not in prepared.columns:
        prepared["selection_gym_score"] = _select_series(prepared, "a_gym_score", "b_gym_score", 0.0)
    if "selection_matchup_striking_edge" not in prepared.columns:
        prepared["selection_matchup_striking_edge"] = _oriented_series(prepared, "matchup_striking_edge", 0.0)
    if "selection_matchup_grappling_edge" not in prepared.columns:
        prepared["selection_matchup_grappling_edge"] = _oriented_series(prepared, "matchup_grappling_edge", 0.0)
    if "selection_matchup_control_edge" not in prepared.columns:
        prepared["selection_matchup_control_edge"] = _oriented_series(prepared, "matchup_control_edge", 0.0)
    if "selection_history_style_label" not in prepared.columns:
        prepared["selection_history_style_label"] = _safe_text_series(prepared, "a_history_style_label", "").where(
            selection == "fighter_a",
            _safe_text_series(prepared, "b_history_style_label", ""),
        )
    if "opponent_history_style_label" not in prepared.columns:
        prepared["opponent_history_style_label"] = _safe_text_series(prepared, "b_history_style_label", "").where(
            selection == "fighter_a",
            _safe_text_series(prepared, "a_history_style_label", ""),
        )

    for column in NUMERIC_FEATURE_COLUMNS:
        prepared[column] = _safe_numeric_series(prepared, column, 0.0)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        prepared[column] = _safe_text_series(prepared, column, "unknown")
    return prepared


def build_side_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    market_key = _safe_text_series(prepared, "tracked_market_key", "moneyline")
    market_fallback = _safe_text_series(prepared, "market", "moneyline")
    actual_result = _safe_text_series(prepared, "actual_result", "")
    grade_status = _safe_text_series(prepared, "grade_status", "")
    working = prepared.loc[
        ((market_key == "moneyline") | (market_fallback == "moneyline"))
        & actual_result.isin(["win", "loss"])
        & (grade_status.str.lower() != "pending")
    ].copy()
    if working.empty:
        return working

    working = prepare_side_feature_frame(working)
    working["side_win_target"] = (actual_result.loc[working.index] == "win").astype(int)
    if "start_time" in working.columns:
        working = working.sort_values("start_time", na_position="last").reset_index(drop=True)
    return working


def _calibration_cv(training: pd.DataFrame, target: pd.Series) -> int | TimeSeriesSplit:
    if "start_time" not in training.columns or training["start_time"].isna().all():
        return 3
    usable_splits = min(5, max(2, len(training) // 12))
    splitter = TimeSeriesSplit(n_splits=usable_splits)
    for train_index, test_index in splitter.split(training):
        if target.iloc[train_index].nunique() < 2 or target.iloc[test_index].nunique() < 2:
            return 3
    return splitter


def train_side_model(
    frame: pd.DataFrame,
    *,
    min_samples: int = 30,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training = build_side_training_frame(frame)
    if len(training) < min_samples:
        raise ValueError(f"Need at least {min_samples} graded moneyline picks; found {len(training)}.")

    target = training["side_win_target"]
    if target.nunique() < 2:
        raise ValueError("Side model needs both win and loss examples.")

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), NUMERIC_FEATURE_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    base_estimator = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", CalibratedClassifierCV(base_estimator, method="sigmoid", cv=_calibration_cv(training, target))),
        ]
    )
    pipeline.fit(training, target)
    probabilities = pipeline.predict_proba(training)[:, 1]
    bundle = {
        "pipeline": pipeline,
        "numeric_features": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "training_rows": int(len(training)),
        "positive_rate": float(target.mean()),
        "in_sample_auc": float(roc_auc_score(target, probabilities)),
        "in_sample_brier": float(brier_score_loss(target, probabilities)),
    }
    return bundle, training


def save_side_model(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_side_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def predict_side_win_prob(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    prepared = prepare_side_feature_frame(frame)
    probabilities = bundle["pipeline"].predict_proba(prepared)[:, 1]
    return pd.Series(probabilities, index=frame.index, dtype=float)


def apply_side_model_adjustments(feature_frame: pd.DataFrame, bundle: dict[str, Any] | None) -> pd.DataFrame:
    scored = feature_frame.copy()
    scored["base_projected_fighter_a_win_prob"] = _safe_numeric_series(scored, "projected_fighter_a_win_prob", 0.5)
    scored["trained_side_fighter_a_win_prob"] = scored["base_projected_fighter_a_win_prob"]
    scored["trained_side_selection_prob"] = _safe_numeric_series(scored, "model_projected_win_prob", 0.5)
    scored["side_model_blend_weight"] = 0.0
    if bundle is None or scored.empty:
        return scored

    selection_probabilities = predict_side_win_prob(scored, bundle)
    scored["trained_side_selection_prob"] = selection_probabilities

    if "event_id" not in scored.columns:
        return scored

    fighter_a_rows = scored["selection"].astype(str) == "fighter_a"
    fighter_b_rows = scored["selection"].astype(str) == "fighter_b"
    fighter_a_probs = (
        scored.loc[fighter_a_rows, ["event_id", "fighter_a", "fighter_b", "trained_side_selection_prob"]]
        .rename(columns={"trained_side_selection_prob": "fighter_a_trained_prob"})
    )
    fighter_b_probs = (
        scored.loc[fighter_b_rows, ["event_id", "fighter_a", "fighter_b", "trained_side_selection_prob"]]
        .rename(columns={"trained_side_selection_prob": "fighter_b_trained_prob"})
    )
    merged = fighter_a_probs.merge(fighter_b_probs, on=["event_id", "fighter_a", "fighter_b"], how="outer")
    if merged.empty:
        return scored

    total = merged["fighter_a_trained_prob"].fillna(0.5) + merged["fighter_b_trained_prob"].fillna(0.5)
    merged["trained_side_fighter_a_win_prob"] = (
        merged["fighter_a_trained_prob"].fillna(0.5) / total.where(total > 0, 1.0)
    ).clip(lower=0.08, upper=0.92)
    merge_keys = ["event_id", "fighter_a", "fighter_b"]
    scored = scored.merge(
        merged.loc[:, [*merge_keys, "trained_side_fighter_a_win_prob"]],
        on=merge_keys,
        how="left",
        suffixes=("", "_derived"),
    )
    if "trained_side_fighter_a_win_prob_derived" in scored.columns:
        scored["trained_side_fighter_a_win_prob"] = scored["trained_side_fighter_a_win_prob_derived"].fillna(
            scored["trained_side_fighter_a_win_prob"]
        )
        scored = scored.drop(columns=["trained_side_fighter_a_win_prob_derived"])

    training_rows = int(bundle.get("training_rows", 0) or 0)
    bundle_weight = 0.18 if training_rows < 60 else 0.24 if training_rows < 120 else 0.30
    confidence = _safe_numeric_series(scored, "model_confidence", 0.5).clip(lower=0.35, upper=1.0)
    data_quality = _safe_numeric_series(scored, "data_quality", 1.0).clip(lower=0.35, upper=1.0)
    scored["side_model_blend_weight"] = (bundle_weight * ((confidence + data_quality) / 2)).clip(lower=0.0, upper=0.35)
    scored["projected_fighter_a_win_prob"] = (
        (scored["base_projected_fighter_a_win_prob"] * (1 - scored["side_model_blend_weight"]))
        + (scored["trained_side_fighter_a_win_prob"] * scored["side_model_blend_weight"])
    ).clip(lower=0.08, upper=0.92)
    scored["model_projected_win_prob"] = scored["projected_fighter_a_win_prob"].where(
        scored["selection"].astype(str) == "fighter_a",
        1 - scored["projected_fighter_a_win_prob"],
    )
    return scored
