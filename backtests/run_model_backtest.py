from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.evaluator import evaluate_backtest, write_summary_csv
from data_sources.storage import save_backtest_run
from models.trainer import TrainingConfig, UFCModelTrainer

EVALUATION_MODE_SCOPES = {
    "in_sample": "in_sample_historical_db",
    "walk_forward": "walk_forward_out_of_sample",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest a UFC fight model against the historical training database."
    )
    parser.add_argument("--db-path", default="data/historical_ufc.db", help="Historical SQLite database path.")
    parser.add_argument(
        "--model-path",
        default="models/ufc_model.pkl",
        help="Saved trainer model pickle path. Used directly for in-sample scoring and as a config source for walk-forward scoring.",
    )
    parser.add_argument(
        "--evaluation-mode",
        default="walk_forward",
        choices=sorted(EVALUATION_MODE_SCOPES),
        help="Backtest mode to run. walk_forward is out-of-sample; in_sample scores the fully trained saved model on the same database.",
    )
    parser.add_argument("--output", default="reports/model_backtest.csv", help="Summary CSV output path.")
    parser.add_argument(
        "--detailed-output",
        help="Optional CSV path for the actionable picks that met the edge threshold.",
    )
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path for backtest summaries.")
    parser.add_argument("--min-edge", type=float, default=0.03, help="Minimum edge threshold for actionable picks.")
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Bankroll used for stake sizing.")
    parser.add_argument(
        "--fractional-kelly",
        type=float,
        default=0.25,
        help="Fraction of Kelly stake to use for the backtest.",
    )
    return parser.parse_args()


def _safe_metric(metric_name: str, truth: pd.Series, probabilities: pd.Series) -> float:
    if truth.empty:
        return float("nan")
    try:
        if metric_name == "accuracy":
            predictions = (probabilities >= 0.5).astype(int)
            return float(accuracy_score(truth, predictions))
        if metric_name == "roc_auc":
            if truth.nunique() < 2:
                return float("nan")
            return float(roc_auc_score(truth, probabilities))
        if metric_name == "log_loss":
            if truth.nunique() < 2:
                return float("nan")
            return float(log_loss(truth, probabilities, labels=[0, 1]))
        if metric_name == "brier":
            return float(brier_score_loss(truth, probabilities))
    except ValueError:
        return float("nan")
    raise ValueError(f"Unsupported metric: {metric_name}")


def _round_metric(value: float) -> float:
    if math.isnan(value):
        return value
    return round(value, 4)


def _coerce_training_config(loaded_config: object) -> TrainingConfig:
    if isinstance(loaded_config, TrainingConfig):
        return loaded_config
    if isinstance(loaded_config, dict):
        return TrainingConfig(**loaded_config)
    return TrainingConfig()


def _load_training_frame(model_path: str | Path, db_path: str | Path) -> tuple[UFCModelTrainer, pd.DataFrame]:
    trainer = UFCModelTrainer(db_path=str(db_path))
    model_file = Path(model_path)
    if model_file.exists():
        saved_bundle = UFCModelTrainer.load_saved_bundle(str(model_file))
        trainer.config = _coerce_training_config(saved_bundle.get("config", {}))

    fights = trainer.engineer_features(trainer.load_training_data())
    fights = fights.dropna(subset=["close_a_odds", "close_b_odds"]).copy()
    if fights.empty:
        raise ValueError("No historical fights with closing odds were available for backtesting.")

    fights["close_a_odds"] = pd.to_numeric(fights["close_a_odds"], errors="raise").astype(int)
    fights["close_b_odds"] = pd.to_numeric(fights["close_b_odds"], errors="raise").astype(int)
    fights = fights.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    trainer.prepare_features(fights)
    return trainer, fights


def _score_fights_in_sample(model_path: str | Path, db_path: str | Path) -> tuple[pd.DataFrame, int]:
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Saved model not found at {model_file}")

    trainer, fights = _load_training_frame(model_file, db_path)
    trainer.load_model(str(model_file))
    fights["predicted_fighter_a_win_prob"] = trainer.predict(fights)
    fights["evaluation_fold"] = 0
    fights["training_rows"] = int(len(fights))
    fights["evaluation_rows"] = int(len(fights))
    return fights, int(len(fights))


def _score_fights_walk_forward(model_path: str | Path, db_path: str | Path) -> tuple[pd.DataFrame, int]:
    trainer, fights = _load_training_frame(model_path, db_path)
    if len(fights) <= trainer.config.cv_splits:
        raise ValueError(
            f"Need more than {trainer.config.cv_splits} fights for walk-forward backtesting; found {len(fights)}."
        )

    X, y = trainer.prepare_features(fights)
    splitter = TimeSeriesSplit(n_splits=trainer.config.cv_splits)
    scored_folds: list[pd.DataFrame] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = trainer.build_estimator(calibration_cv=3)
        model.fit(X[train_idx], y[train_idx])

        fold_frame = fights.iloc[test_idx].copy()
        fold_frame["predicted_fighter_a_win_prob"] = model.predict_proba(X[test_idx])[:, 1]
        fold_frame["evaluation_fold"] = int(fold_index)
        fold_frame["training_rows"] = int(len(train_idx))
        fold_frame["evaluation_rows"] = int(len(test_idx))
        scored_folds.append(fold_frame)

    if not scored_folds:
        raise ValueError("Walk-forward backtest did not produce any scored folds.")

    scored = pd.concat(scored_folds, ignore_index=True)
    scored = scored.sort_values(["event_date", "fight_id"]).reset_index(drop=True)
    return scored, int(len(fights))


def load_scored_fights(
    *,
    model_path: str | Path,
    db_path: str | Path,
    evaluation_mode: str,
) -> tuple[pd.DataFrame, int]:
    if evaluation_mode == "in_sample":
        return _score_fights_in_sample(model_path, db_path)
    if evaluation_mode == "walk_forward":
        return _score_fights_walk_forward(model_path, db_path)
    raise ValueError(f"Unsupported evaluation mode: {evaluation_mode}")


def build_selection_backtest_frame(scored_fights: pd.DataFrame) -> pd.DataFrame:
    extra_columns = [
        column
        for column in ["evaluation_fold", "training_rows", "evaluation_rows"]
        if column in scored_fights.columns
    ]
    base_columns = [
        "fight_id",
        "event_id",
        "event_name",
        "event_date",
        "fighter_a",
        "fighter_b",
        "winner",
        *extra_columns,
    ]

    fighter_a_rows = scored_fights.loc[:, base_columns].copy()
    fighter_a_rows["market"] = "moneyline"
    fighter_a_rows["selection"] = "fighter_a"
    fighter_a_rows["selection_name"] = fighter_a_rows["fighter_a"]
    fighter_a_rows["book"] = "synthetic_close"
    fighter_a_rows["american_odds"] = scored_fights["close_a_odds"].to_numpy()
    fighter_a_rows["model_projected_win_prob"] = scored_fights["predicted_fighter_a_win_prob"].to_numpy()
    fighter_a_rows["actual_result"] = (
        fighter_a_rows["winner"].astype(str) == fighter_a_rows["fighter_a"].astype(str)
    ).map({True: "win", False: "loss"})

    fighter_b_rows = scored_fights.loc[:, base_columns].copy()
    fighter_b_rows["market"] = "moneyline"
    fighter_b_rows["selection"] = "fighter_b"
    fighter_b_rows["selection_name"] = fighter_b_rows["fighter_b"]
    fighter_b_rows["book"] = "synthetic_close"
    fighter_b_rows["american_odds"] = scored_fights["close_b_odds"].to_numpy()
    fighter_b_rows["model_projected_win_prob"] = 1.0 - scored_fights["predicted_fighter_a_win_prob"].to_numpy()
    fighter_b_rows["actual_result"] = (
        fighter_b_rows["winner"].astype(str) == fighter_b_rows["fighter_b"].astype(str)
    ).map({True: "win", False: "loss"})

    selections = pd.concat([fighter_a_rows, fighter_b_rows], ignore_index=True)
    selections = selections.sort_values(["event_date", "fight_id", "selection"]).reset_index(drop=True)
    return selections


def build_model_summary(
    scored_fights: pd.DataFrame,
    actionable_report: pd.DataFrame,
    betting_summary: dict[str, float | int],
    *,
    evaluation_mode: str,
    historical_fights_available: int,
) -> dict[str, float | int]:
    truth = scored_fights["target"].astype(int)
    probabilities = scored_fights["predicted_fighter_a_win_prob"].astype(float)
    summary = dict(betting_summary)
    summary.update(
        {
            "historical_fights_available": int(historical_fights_available),
            "fights_evaluated": int(len(scored_fights)),
            "burn_in_fights": int(max(0, historical_fights_available - len(scored_fights))),
            "selection_rows_evaluated": int(len(scored_fights) * 2),
            "actionable_pick_rate": _round_metric(float(len(actionable_report)) / float(len(scored_fights) * 2)),
            "fight_accuracy": _round_metric(_safe_metric("accuracy", truth, probabilities)),
            "fight_auc": _round_metric(_safe_metric("roc_auc", truth, probabilities)),
            "fight_log_loss": _round_metric(_safe_metric("log_loss", truth, probabilities)),
            "fight_brier": _round_metric(_safe_metric("brier", truth, probabilities)),
            "avg_fighter_a_win_prob": _round_metric(float(probabilities.mean())),
            "avg_winner_prob": _round_metric(
                float(probabilities.where(truth == 1, 1.0 - probabilities).mean())
            ),
            "evaluation_folds": int(scored_fights["evaluation_fold"].nunique()),
            "min_training_rows": int(scored_fights["training_rows"].min()),
            "max_training_rows": int(scored_fights["training_rows"].max()),
            "backtest_scope": EVALUATION_MODE_SCOPES[evaluation_mode],
        }
    )
    return summary


def run_model_backtest(
    *,
    model_path: str | Path,
    db_path: str | Path,
    min_edge: float,
    bankroll: float,
    fractional_kelly: float,
    evaluation_mode: str = "walk_forward",
) -> tuple[pd.DataFrame, dict[str, float | int], pd.DataFrame]:
    scored_fights, historical_fights_available = load_scored_fights(
        model_path=model_path,
        db_path=db_path,
        evaluation_mode=evaluation_mode,
    )
    selections = build_selection_backtest_frame(scored_fights)
    actionable_report, betting_summary = evaluate_backtest(
        selections,
        min_edge=min_edge,
        bankroll=bankroll,
        fractional_kelly=fractional_kelly,
    )
    summary = build_model_summary(
        scored_fights,
        actionable_report,
        betting_summary,
        evaluation_mode=evaluation_mode,
        historical_fights_available=historical_fights_available,
    )
    return actionable_report, summary, selections


def main() -> None:
    args = parse_args()
    actionable_report, summary, _ = run_model_backtest(
        model_path=args.model_path,
        db_path=args.db_path,
        min_edge=args.min_edge,
        bankroll=args.bankroll,
        fractional_kelly=args.fractional_kelly,
        evaluation_mode=args.evaluation_mode,
    )

    write_summary_csv(summary, args.output)
    save_backtest_run(summary, args.db)

    if args.detailed_output:
        detailed_path = Path(args.detailed_output)
        detailed_path.parent.mkdir(parents=True, exist_ok=True)
        actionable_report.to_csv(detailed_path, index=False)
        print(f"Saved actionable picks to {detailed_path}")

    print(f"Evaluation mode: {summary['backtest_scope']}")
    if actionable_report.empty:
        print("No selections met the configured edge threshold.")
    else:
        display_columns = [
            "event_name",
            "selection_name",
            "american_odds",
            "model_projected_win_prob",
            "edge",
            "stake",
            "profit",
        ]
        preview = actionable_report.sort_values(["edge", "model_projected_win_prob"], ascending=[False, False]).head(25)
        print("Top actionable picks by edge:")
        print(preview.loc[:, display_columns].to_string(index=False))

    print(f"\nBacktest summary: {summary}")
    print(f"Saved backtest summary to {args.output}")


if __name__ == "__main__":
    main()
