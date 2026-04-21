from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.ev import implied_probability


DEFAULT_MIN_CLV_IMPROVEMENT = 0.01
DEFAULT_SELECTIVE_MODEL_PATH = Path("models") / "selective_clv_model.pkl"

NUMERIC_FEATURE_COLUMNS = [
    "american_odds",
    "chosen_expression_odds",
    "model_projected_win_prob",
    "chosen_expression_prob",
    "implied_prob",
    "chosen_expression_implied_prob",
    "edge",
    "chosen_expression_edge",
    "expected_value",
    "chosen_expression_expected_value",
    "suggested_stake",
    "chosen_expression_stake",
    "model_confidence",
    "data_quality",
    "selection_stats_completeness",
    "selection_fallback_used",
    "line_movement_toward_fighter",
    "market_blend_weight",
    "bet_quality_score",
    "support_count",
    "risk_flag_count",
    "market_consensus_bookmaker_count",
    "market_overround",
    "price_edge_vs_consensus",
    "is_wmma",
    "is_heavyweight",
    "is_five_round_fight",
    "selection_recent_finish_damage",
    "selection_recent_ko_damage",
    "selection_recent_damage_score",
    "selection_stance_matchup_edge",
    "selection_days_since_last_fight",
    "selection_ufc_fight_count",
    "selection_ufc_debut_flag",
    "selection_context_instability",
    "selection_first_round_finish_rate",
    "selection_finish_loss_rate",
]

CATEGORICAL_FEATURE_COLUMNS = [
    "market",
    "selection",
    "recommended_tier",
    "recommended_action",
    "expression_pick_source",
    "segment_label",
    "book",
]


def default_selective_model_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_SELECTIVE_MODEL_PATH


def prepare_selective_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    numeric_aliases = {
        "chosen_expression_odds": ["effective_american_odds", "american_odds"],
        "chosen_expression_prob": ["effective_projected_prob", "model_projected_win_prob"],
        "chosen_expression_implied_prob": ["effective_implied_prob", "implied_prob"],
        "chosen_expression_edge": ["effective_edge", "edge"],
        "chosen_expression_expected_value": ["effective_expected_value", "expected_value"],
        "chosen_expression_stake": ["effective_suggested_stake", "suggested_stake"],
    }
    for target_column, fallback_columns in numeric_aliases.items():
        if target_column in prepared.columns:
            continue
        source_column = next((name for name in fallback_columns if name in prepared.columns), None)
        if source_column is not None:
            prepared[target_column] = prepared[source_column]

    for column in NUMERIC_FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = 0.0
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(0.0)

    for column in CATEGORICAL_FEATURE_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = "unknown"
        prepared[column] = prepared[column].fillna("unknown").astype(str)

    return prepared


def build_selective_training_frame(
    frame: pd.DataFrame,
    *,
    min_clv_improvement: float = DEFAULT_MIN_CLV_IMPROVEMENT,
) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        return prepared

    alias_columns = {
        "chosen_expression_odds": ["effective_american_odds", "american_odds"],
    }
    for target_column, fallback_columns in alias_columns.items():
        if target_column in prepared.columns:
            continue
        source_column = next((name for name in fallback_columns if name in prepared.columns), None)
        if source_column is not None:
            prepared[target_column] = prepared[source_column]

    if "closing_american_odds" not in prepared.columns:
        prepared["closing_american_odds"] = pd.NA
    if "grade_status" not in prepared.columns:
        prepared["grade_status"] = ""

    prepared["chosen_expression_odds"] = pd.to_numeric(prepared["chosen_expression_odds"], errors="coerce")
    prepared["closing_american_odds"] = pd.to_numeric(prepared["closing_american_odds"], errors="coerce")

    working = prepared.loc[
        prepared["closing_american_odds"].notna()
        & prepared["chosen_expression_odds"].notna()
        & (prepared["grade_status"].astype(str).str.lower() != "pending")
    ].copy()
    if working.empty:
        return working

    working = prepare_selective_feature_frame(working)
    working["pick_implied_prob"] = working["chosen_expression_odds"].apply(lambda value: implied_probability(int(value)))
    working["close_implied_prob"] = working["closing_american_odds"].apply(lambda value: implied_probability(int(value)))
    working["clv_implied_delta"] = working["close_implied_prob"] - working["pick_implied_prob"]
    working["positive_clv_target"] = (working["clv_implied_delta"] >= float(min_clv_improvement)).astype(int)
    return working


def train_selective_clv_model(
    frame: pd.DataFrame,
    *,
    min_clv_improvement: float = DEFAULT_MIN_CLV_IMPROVEMENT,
    min_samples: int = 50,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training = build_selective_training_frame(frame, min_clv_improvement=min_clv_improvement)
    if len(training) < min_samples:
        raise ValueError(
            f"Need at least {min_samples} graded tracked picks with closing odds; found {len(training)}."
        )

    target = training["positive_clv_target"]
    if target.nunique() < 2:
        raise ValueError("Selective CLV model needs both positive and negative CLV examples.")

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), NUMERIC_FEATURE_COLUMNS),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(training, target)
    in_sample_probabilities = pipeline.predict_proba(training)[:, 1]
    bundle = {
        "pipeline": pipeline,
        "numeric_features": list(NUMERIC_FEATURE_COLUMNS),
        "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
        "min_clv_improvement": float(min_clv_improvement),
        "training_rows": int(len(training)),
        "positive_rate": float(target.mean()),
        "in_sample_auc": float(roc_auc_score(target, in_sample_probabilities)),
    }
    return bundle, training


def save_selective_model(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_selective_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def predict_selective_clv_prob(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    prepared = prepare_selective_feature_frame(frame)
    pipeline = bundle["pipeline"]
    probabilities = pipeline.predict_proba(prepared)[:, 1]
    return pd.Series(probabilities, index=frame.index, dtype=float)
