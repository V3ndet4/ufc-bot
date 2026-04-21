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


DEFAULT_CONFIDENCE_MODEL_PATH = Path("models") / "confidence_model.pkl"

NUMERIC_FEATURE_COLUMNS = [
    "model_pick_win_prob",
    "baseline_model_pick_win_prob",
    "heuristic_model_confidence",
    "data_quality",
    "fallback_penalty",
    "market_disagreement",
    "market_consensus_bookmaker_count",
    "market_overround",
    "model_pick_margin",
    "min_ufc_sample",
    "max_debut_flag",
    "context_noise_total",
    "layoff_gap_abs",
    "recent_form_gap_abs",
    "recent_strike_gap_abs",
    "recent_grappling_gap_abs",
    "control_gap_abs",
    "damage_gap_abs",
    "gym_score_gap_abs",
    "strike_margin_last_3_gap_abs",
    "grappling_rate_last_3_gap_abs",
    "control_avg_last_3_gap_abs",
    "result_score_last_3_gap_abs",
    "pace_gap_abs",
    "distance_style_gap_abs",
    "clinch_style_gap_abs",
    "ground_style_gap_abs",
    "is_wmma",
    "is_heavyweight",
    "is_five_round_fight",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "segment_label",
    "model_pick_side",
    "model_pick_style",
    "opponent_style",
]


def default_confidence_model_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_CONFIDENCE_MODEL_PATH


def _safe_numeric_series(frame: pd.DataFrame, column: str, default: float | pd.Series = 0.0) -> pd.Series:
    if isinstance(default, pd.Series):
        fallback = pd.to_numeric(default, errors="coerce").reindex(frame.index)
        if column not in frame.columns:
            return fallback.fillna(0.0)
        return pd.to_numeric(frame[column], errors="coerce").fillna(fallback).fillna(0.0)
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _safe_text_series(frame: pd.DataFrame, column: str, default: str = "unknown") -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=object)
    return frame[column].fillna(default).astype(str)


def _group_columns(frame: pd.DataFrame) -> list[str]:
    keys = [column for column in ["event_id", "fighter_a", "fighter_b"] if column in frame.columns]
    if len(keys) == 3:
        return keys
    synthetic = frame.reset_index(drop=False).rename(columns={"index": "row_id"})
    frame["row_id"] = synthetic["row_id"]
    return ["row_id"]


def _collapse_to_fight_level(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    group_columns = _group_columns(working)
    sort_columns = [column for column in ["start_time", "selection"] if column in working.columns]
    if sort_columns:
        working = working.sort_values(sort_columns, na_position="last").reset_index(drop=True)
    fight_level = working.groupby(group_columns, as_index=False, dropna=False).first()
    if "actual_result" in working.columns:
        fighter_a_win = (
            working.loc[working["selection"].astype(str) == "fighter_a", group_columns + ["actual_result"]]
            .rename(columns={"actual_result": "fighter_a_actual_result"})
        )
        fight_level = fight_level.merge(fighter_a_win, on=group_columns, how="left")
        fight_level["actual_winner_side"] = fight_level["fighter_a_actual_result"].astype(str).map(
            {"win": "fighter_a", "loss": "fighter_b"}
        )
        fight_level = fight_level.drop(columns=["fighter_a_actual_result"])
        missing_mask = fight_level["actual_winner_side"].isna()
        if missing_mask.any() and {"selection", "actual_result"}.issubset(fight_level.columns):
            selection = fight_level.loc[missing_mask, "selection"].astype(str)
            actual_result = fight_level.loc[missing_mask, "actual_result"].astype(str)
            inferred = selection.where(actual_result == "win", selection.map({"fighter_a": "fighter_b", "fighter_b": "fighter_a"}))
            fight_level.loc[missing_mask, "actual_winner_side"] = inferred
    return fight_level


def prepare_confidence_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = _collapse_to_fight_level(frame)
    if prepared.empty:
        return prepared

    selection = _safe_text_series(prepared, "selection", "fighter_a")
    fighter_a_prob = _safe_numeric_series(
        prepared,
        "projected_fighter_a_win_prob",
        _safe_numeric_series(prepared, "raw_projected_fighter_a_win_prob", float("nan")),
    ).clip(lower=0.05, upper=0.95)
    if fighter_a_prob.isna().all():
        fighter_a_prob = _safe_numeric_series(prepared, "model_projected_win_prob", 0.5).where(
            selection == "fighter_a",
            1 - _safe_numeric_series(prepared, "model_projected_win_prob", 0.5),
        ).clip(lower=0.05, upper=0.95)
    else:
        fighter_a_prob = fighter_a_prob.fillna(
            _safe_numeric_series(prepared, "model_projected_win_prob", 0.5).where(
                selection == "fighter_a",
                1 - _safe_numeric_series(prepared, "model_projected_win_prob", 0.5),
            )
        ).clip(lower=0.05, upper=0.95)
    baseline_fighter_a_prob = _safe_numeric_series(
        prepared,
        "baseline_raw_fighter_a_win_prob",
        _safe_numeric_series(prepared, "base_model_projected_win_prob", fighter_a_prob),
    ).clip(lower=0.05, upper=0.95)
    heuristic_confidence = _safe_numeric_series(
        prepared,
        "heuristic_model_confidence",
        _safe_numeric_series(prepared, "model_confidence", 0.5),
    )
    fighter_a_market_prob = _safe_numeric_series(prepared, "fighter_a_current_implied_prob", 0.5).clip(lower=0.05, upper=0.95)

    prepared["model_pick_side"] = fighter_a_prob.ge(0.5).map({True: "fighter_a", False: "fighter_b"})
    prepared["model_pick_win_prob"] = fighter_a_prob.where(prepared["model_pick_side"] == "fighter_a", 1 - fighter_a_prob)
    prepared["baseline_model_pick_win_prob"] = baseline_fighter_a_prob.where(
        prepared["model_pick_side"] == "fighter_a",
        1 - baseline_fighter_a_prob,
    )
    prepared["heuristic_model_confidence"] = heuristic_confidence.clip(lower=0.2, upper=0.95)
    prepared["data_quality"] = _safe_numeric_series(prepared, "data_quality", 1.0).clip(lower=0.0, upper=1.0)
    prepared["fallback_penalty"] = _safe_numeric_series(prepared, "fallback_penalty", 0.0).clip(lower=0.0)
    prepared["market_disagreement"] = (
        prepared["baseline_model_pick_win_prob"]
        - fighter_a_market_prob.where(prepared["model_pick_side"] == "fighter_a", 1 - fighter_a_market_prob)
    ).abs()
    prepared["market_consensus_bookmaker_count"] = _safe_numeric_series(prepared, "market_consensus_bookmaker_count", 0.0)
    prepared["market_overround"] = _safe_numeric_series(prepared, "market_overround", 0.0)
    prepared["model_pick_margin"] = (prepared["baseline_model_pick_win_prob"] - 0.5).abs() * 2.0
    prepared["min_ufc_sample"] = pd.concat(
        [
            _safe_numeric_series(prepared, "a_ufc_fight_count", 0.0),
            _safe_numeric_series(prepared, "b_ufc_fight_count", 0.0),
        ],
        axis=1,
    ).min(axis=1)
    prepared["max_debut_flag"] = pd.concat(
        [
            _safe_numeric_series(prepared, "a_ufc_debut_flag", 0.0),
            _safe_numeric_series(prepared, "b_ufc_debut_flag", 0.0),
        ],
        axis=1,
    ).max(axis=1)
    prepared["context_noise_total"] = (
        _safe_numeric_series(prepared, "a_short_notice_flag", 0.0)
        + _safe_numeric_series(prepared, "b_short_notice_flag", 0.0)
        + _safe_numeric_series(prepared, "a_cardio_fade_flag", 0.0)
        + _safe_numeric_series(prepared, "b_cardio_fade_flag", 0.0)
        + _safe_numeric_series(prepared, "a_injury_concern_flag", 0.0)
        + _safe_numeric_series(prepared, "b_injury_concern_flag", 0.0)
        + _safe_numeric_series(prepared, "a_weight_cut_concern_flag", 0.0)
        + _safe_numeric_series(prepared, "b_weight_cut_concern_flag", 0.0)
        + _safe_numeric_series(prepared, "a_replacement_fighter_flag", 0.0)
        + _safe_numeric_series(prepared, "b_replacement_fighter_flag", 0.0)
        + _safe_numeric_series(prepared, "a_travel_disadvantage_flag", 0.0)
        + _safe_numeric_series(prepared, "b_travel_disadvantage_flag", 0.0)
        + _safe_numeric_series(prepared, "a_new_gym_flag", 0.0)
        + _safe_numeric_series(prepared, "b_new_gym_flag", 0.0)
        + _safe_numeric_series(prepared, "a_camp_change_flag", 0.0)
        + _safe_numeric_series(prepared, "b_camp_change_flag", 0.0)
    )
    prepared["layoff_gap_abs"] = (
        _safe_numeric_series(prepared, "a_days_since_last_fight", 999.0)
        - _safe_numeric_series(prepared, "b_days_since_last_fight", 999.0)
    ).abs()
    prepared["recent_form_gap_abs"] = _safe_numeric_series(prepared, "recent_form_diff", 0.0).abs()
    prepared["recent_strike_gap_abs"] = _safe_numeric_series(prepared, "recent_strike_form_diff", 0.0).abs()
    prepared["recent_grappling_gap_abs"] = _safe_numeric_series(prepared, "recent_grappling_form_diff", 0.0).abs()
    prepared["control_gap_abs"] = _safe_numeric_series(prepared, "recent_control_diff", 0.0).abs()
    prepared["damage_gap_abs"] = _safe_numeric_series(prepared, "recent_damage_diff", 0.0).abs()
    prepared["gym_score_gap_abs"] = _safe_numeric_series(prepared, "gym_score_diff", 0.0).abs()
    prepared["strike_margin_last_3_gap_abs"] = _safe_numeric_series(prepared, "strike_margin_last_3_diff", 0.0).abs()
    prepared["grappling_rate_last_3_gap_abs"] = _safe_numeric_series(prepared, "grappling_rate_last_3_diff", 0.0).abs()
    prepared["control_avg_last_3_gap_abs"] = _safe_numeric_series(prepared, "control_avg_last_3_diff", 0.0).abs()
    prepared["result_score_last_3_gap_abs"] = _safe_numeric_series(prepared, "result_score_last_3_diff", 0.0).abs()
    prepared["pace_gap_abs"] = _safe_numeric_series(prepared, "strike_pace_last_3_diff", 0.0).abs()
    prepared["distance_style_gap_abs"] = _safe_numeric_series(prepared, "distance_strike_share_diff", 0.0).abs()
    prepared["clinch_style_gap_abs"] = _safe_numeric_series(prepared, "clinch_strike_share_diff", 0.0).abs()
    prepared["ground_style_gap_abs"] = _safe_numeric_series(prepared, "ground_strike_share_diff", 0.0).abs()
    prepared["is_wmma"] = _safe_numeric_series(prepared, "is_wmma", 0.0)
    prepared["is_heavyweight"] = _safe_numeric_series(prepared, "is_heavyweight", 0.0)
    prepared["is_five_round_fight"] = _safe_numeric_series(prepared, "is_five_round_fight", 0.0)
    prepared["segment_label"] = _safe_text_series(prepared, "segment_label", "standard")
    prepared["model_pick_style"] = _safe_text_series(
        prepared,
        "a_history_style_label",
        "",
    ).where(
        prepared["model_pick_side"] == "fighter_a",
        _safe_text_series(prepared, "b_history_style_label", ""),
    )
    prepared["opponent_style"] = _safe_text_series(
        prepared,
        "b_history_style_label",
        "",
    ).where(
        prepared["model_pick_side"] == "fighter_a",
        _safe_text_series(prepared, "a_history_style_label", ""),
    )
    for column in NUMERIC_FEATURE_COLUMNS:
        prepared[column] = _safe_numeric_series(prepared, column, 0.0)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        prepared[column] = _safe_text_series(prepared, column, "unknown")
    return prepared


def build_confidence_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = prepare_confidence_feature_frame(frame)
    if prepared.empty or "actual_winner_side" not in prepared.columns:
        return prepared.iloc[0:0].copy()
    training = prepared.loc[prepared["actual_winner_side"].isin(["fighter_a", "fighter_b"])].copy()
    if training.empty:
        return training
    training["confidence_target"] = (training["model_pick_side"] == training["actual_winner_side"]).astype(int)
    if "start_time" in training.columns:
        training = training.sort_values("start_time", na_position="last").reset_index(drop=True)
    return training


def _calibration_cv(training: pd.DataFrame, target: pd.Series) -> int | TimeSeriesSplit:
    if "start_time" not in training.columns or training["start_time"].isna().all():
        return 3
    usable_splits = min(5, max(2, len(training) // 12))
    splitter = TimeSeriesSplit(n_splits=usable_splits)
    for train_index, test_index in splitter.split(training):
        if target.iloc[train_index].nunique() < 2 or target.iloc[test_index].nunique() < 2:
            return 3
    return splitter


def train_confidence_model(
    frame: pd.DataFrame,
    *,
    min_samples: int = 40,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training = build_confidence_training_frame(frame)
    if len(training) < min_samples:
        raise ValueError(f"Need at least {min_samples} graded fights for confidence training; found {len(training)}.")

    target = training["confidence_target"]
    if target.nunique() < 2:
        raise ValueError("Confidence model needs both correct and incorrect model-pick examples.")

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


def save_confidence_model(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_confidence_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def predict_confidence(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    prepared = prepare_confidence_feature_frame(frame)
    if prepared.empty:
        return pd.Series(dtype=float, index=frame.index)
    predictions = bundle["pipeline"].predict_proba(prepared)[:, 1]
    prepared = prepared.copy()
    prepared["predicted_model_confidence"] = predictions
    group_columns = [column for column in ["event_id", "fighter_a", "fighter_b"] if column in prepared.columns]
    if len(group_columns) == 3:
        mapped = frame.merge(
            prepared[group_columns + ["predicted_model_confidence"]],
            on=group_columns,
            how="left",
        )["predicted_model_confidence"]
        return pd.Series(mapped.fillna(0.5).to_numpy(), index=frame.index, dtype=float)
    value = float(prepared["predicted_model_confidence"].iloc[0])
    return pd.Series(value, index=frame.index, dtype=float)


def apply_confidence_model(
    frame: pd.DataFrame,
    bundle: dict[str, Any] | None,
    heuristic_confidence: pd.Series,
) -> pd.Series:
    if bundle is None or frame.empty:
        return heuristic_confidence
    learned = predict_confidence(frame, bundle).clip(lower=0.2, upper=0.92)
    training_rows = int(bundle.get("training_rows", 0) or 0)
    bundle_weight = 0.30 if training_rows < 120 else 0.42 if training_rows < 300 else 0.55
    blended = (heuristic_confidence * (1 - bundle_weight)) + (learned * bundle_weight)
    return blended.clip(lower=0.2, upper=0.92)
