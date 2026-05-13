from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_PROP_OUTCOME_MODEL_PATH = Path("models") / "prop_outcome_model.pkl"

PROP_MARKET_TARGETS = {
    "takedown": "takedown_1plus_target",
    "knockdown": "knockdown_1plus_target",
}

PROP_MARKET_PROBABILITY_CAPS = {
    "takedown": 0.88,
    "knockdown": 0.72,
}

PROP_NUMERIC_FEATURE_COLUMNS = [
    "scheduled_rounds",
    "selection_ufc_fight_count",
    "opponent_ufc_fight_count",
    "selection_takedown_avg",
    "selection_takedown_accuracy_pct",
    "opponent_takedown_defense_pct",
    "selection_recent_grappling_rate",
    "selection_control_avg",
    "selection_recent_control_avg",
    "selection_matchup_grappling_edge",
    "selection_knockdown_avg",
    "selection_ko_win_rate",
    "opponent_ko_loss_rate",
    "selection_sig_strikes_landed_per_min",
    "opponent_sig_strikes_absorbed_per_min",
    "selection_distance_strike_share",
    "selection_clinch_strike_share",
    "selection_ground_strike_share",
]

PROP_FEATURE_DEFAULTS = {
    "scheduled_rounds": 3.0,
    "opponent_takedown_defense_pct": 68.0,
    "selection_distance_strike_share": 0.55,
    "selection_clinch_strike_share": 0.15,
    "selection_ground_strike_share": 0.10,
}


def default_prop_outcome_model_path(root: str | Path) -> Path:
    return Path(root) / DEFAULT_PROP_OUTCOME_MODEL_PATH


def prepare_prop_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in PROP_NUMERIC_FEATURE_COLUMNS:
        default = PROP_FEATURE_DEFAULTS.get(column, 0.0)
        if column not in prepared.columns:
            prepared[column] = default
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce").fillna(default)
    return prepared


def train_prop_outcome_model(
    frame: pd.DataFrame,
    *,
    min_samples: int = 400,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training = prepare_prop_feature_frame(frame)
    if training.empty:
        raise ValueError("Prop outcome model needs a non-empty historical training frame.")

    markets: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for market, target_column in PROP_MARKET_TARGETS.items():
        if target_column not in training.columns:
            skipped[market] = f"missing target column {target_column}"
            continue
        target = pd.to_numeric(training[target_column], errors="coerce")
        valid_mask = target.isin([0, 1])
        market_training = training.loc[valid_mask].copy()
        target = target.loc[valid_mask].astype(int)
        if len(market_training) < min_samples:
            skipped[market] = f"need at least {min_samples} rows; found {len(market_training)}"
            continue
        if target.nunique() < 2:
            skipped[market] = "needs both positive and negative examples"
            continue

        base_estimator = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", _build_preprocess()),
                ("model", CalibratedClassifierCV(base_estimator, method="sigmoid", cv=_calibration_cv(target))),
            ]
        )
        pipeline.fit(market_training, target)
        probabilities = pipeline.predict_proba(market_training)[:, 1]
        markets[market] = {
            "pipeline": pipeline,
            "target": target_column,
            "training_rows": int(len(market_training)),
            "positive_rate": float(target.mean()),
            "in_sample_auc": _safe_auc(target, probabilities),
            "in_sample_brier": float(brier_score_loss(target, probabilities)),
        }

    if not markets:
        reasons = "; ".join(f"{market}: {reason}" for market, reason in skipped.items())
        raise ValueError(f"No prop outcome models were trained. {reasons}")

    bundle = {
        "model_version": 1,
        "numeric_features": list(PROP_NUMERIC_FEATURE_COLUMNS),
        "markets": markets,
        "training_rows": int(len(training)),
        "skipped": skipped,
    }
    return bundle, training


def _build_preprocess() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("scale", StandardScaler())]), PROP_NUMERIC_FEATURE_COLUMNS),
        ],
        remainder="drop",
    )


def save_prop_outcome_model(bundle: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(bundle, handle)
    return path


def load_prop_outcome_model(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def prop_feature_frame_from_fight_row(fight_row: pd.Series | dict[str, Any], selection: object) -> pd.DataFrame:
    row = fight_row if isinstance(fight_row, pd.Series) else pd.Series(fight_row)
    selection_text = str(selection).strip()
    opponent = "fighter_b" if selection_text == "fighter_a" else "fighter_a"
    grappling_edge = _safe_float(row.get("matchup_grappling_edge"), None)
    if grappling_edge is None:
        grappling_edge = _computed_grappling_edge(row, selection_text)
    elif selection_text == "fighter_b":
        grappling_edge *= -1.0

    features = {
        "scheduled_rounds": _safe_float(row.get("scheduled_rounds"), 3.0),
        "selection_ufc_fight_count": _side_stat(row, selection_text, "ufc_fight_count", 0.0),
        "opponent_ufc_fight_count": _side_stat(row, opponent, "ufc_fight_count", 0.0),
        "selection_takedown_avg": _side_stat(row, selection_text, "takedown_avg", 0.0),
        "selection_takedown_accuracy_pct": _side_stat(row, selection_text, "takedown_accuracy_pct", 50.0),
        "opponent_takedown_defense_pct": _side_stat(row, opponent, "takedown_defense_pct", 68.0),
        "selection_recent_grappling_rate": _side_stat(row, selection_text, "recent_grappling_rate", 0.0),
        "selection_control_avg": _side_stat(row, selection_text, "control_avg", 0.0),
        "selection_recent_control_avg": _side_stat(row, selection_text, "recent_control_avg", 0.0),
        "selection_matchup_grappling_edge": grappling_edge,
        "selection_knockdown_avg": _side_stat(row, selection_text, "knockdown_avg", 0.0),
        "selection_ko_win_rate": _side_stat(row, selection_text, "ko_win_rate", 0.0),
        "opponent_ko_loss_rate": _side_stat(row, opponent, "ko_loss_rate", 0.0),
        "selection_sig_strikes_landed_per_min": _side_stat(row, selection_text, "sig_strikes_landed_per_min", 0.0),
        "opponent_sig_strikes_absorbed_per_min": _side_stat(row, opponent, "sig_strikes_absorbed_per_min", 0.0),
        "selection_distance_strike_share": _side_stat(row, selection_text, "distance_strike_share", 0.55),
        "selection_clinch_strike_share": _side_stat(row, selection_text, "clinch_strike_share", 0.15),
        "selection_ground_strike_share": _side_stat(row, selection_text, "ground_strike_share", 0.10),
    }
    return prepare_prop_feature_frame(pd.DataFrame([features]))


def predict_prop_probability_from_fight_row(
    bundle: dict[str, Any] | None,
    fight_row: pd.Series | dict[str, Any],
    *,
    market: str,
    selection: object,
) -> float | None:
    if not bundle:
        return None
    model_entry = bundle.get("markets", {}).get(str(market))
    if not model_entry:
        return None
    features = prop_feature_frame_from_fight_row(fight_row, selection)
    probability = float(model_entry["pipeline"].predict_proba(features)[:, 1][0])
    cap = PROP_MARKET_PROBABILITY_CAPS.get(str(market), 0.95)
    return round(max(0.0, min(cap, probability)), 4)


def _calibration_cv(target: pd.Series) -> int:
    class_counts = target.value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        raise ValueError("Calibration needs at least two examples in each class.")
    return min(5, max(2, min_class_count))


def _safe_auc(target: pd.Series, probabilities: Any) -> float:
    try:
        return float(roc_auc_score(target, probabilities))
    except ValueError:
        return float("nan")


def _selection_prefix(selection: object) -> str:
    return "a" if str(selection).strip() == "fighter_a" else "b"


def _side_stat(row: pd.Series, selection: object, suffix: str, default: object = 0.0) -> float:
    letter_prefix = _selection_prefix(selection)
    fighter_prefix = "fighter_a" if letter_prefix == "a" else "fighter_b"
    for column in (f"{letter_prefix}_{suffix}", f"{fighter_prefix}_{suffix}"):
        if column in row.index:
            value = row.get(column)
            parsed = _safe_float(value, None)
            if parsed is not None:
                return parsed
    return float(default)


def _computed_grappling_edge(row: pd.Series, selection: str) -> float:
    opponent = "fighter_b" if selection == "fighter_a" else "fighter_a"
    selection_score = (
        _side_stat(row, selection, "takedown_avg", 0.0)
        * max(0.05, (100.0 - _side_stat(row, opponent, "takedown_defense_pct", 68.0)) / 100.0)
        + (_side_stat(row, selection, "recent_grappling_rate", 0.0) * 0.35)
        + (max(5.0, _side_stat(row, selection, "takedown_accuracy_pct", 50.0)) / 100.0)
    )
    opponent_score = (
        _side_stat(row, opponent, "takedown_avg", 0.0)
        * max(0.05, (100.0 - _side_stat(row, selection, "takedown_defense_pct", 68.0)) / 100.0)
        + (_side_stat(row, opponent, "recent_grappling_rate", 0.0) * 0.35)
        + (max(5.0, _side_stat(row, opponent, "takedown_accuracy_pct", 50.0)) / 100.0)
    )
    return selection_score - opponent_score


def _safe_float(value: object, default: float | None = 0.0) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if pd.isna(parsed):
        return default
    return parsed
