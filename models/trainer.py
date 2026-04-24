"""
UFC Model Trainer

Trains a logistic regression or XGBoost model on historical UFC data.
Key principle: Use features as of fight time, predict actual outcomes.

Input: Historical database with fights, odds, and fighter snapshots
Output: Trained model + feature importance + backtest results
"""

from __future__ import annotations

import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_type: str = "logistic"  # "logistic" or "xgboost"
    min_fights_per_fighter: int = 3
    calibration: bool = True
    cv_splits: int = 5
    random_state: int = 42


class _TrainerModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):  # type: ignore[override]
        if module == "__main__" and name == "TrainingConfig":
            return TrainingConfig
        return super().find_class(module, name)


def _coerce_training_config(loaded_config: object) -> TrainingConfig:
    if isinstance(loaded_config, TrainingConfig):
        return loaded_config
    if isinstance(loaded_config, dict):
        return TrainingConfig(**loaded_config)
    return TrainingConfig()


class UFCModelTrainer:
    """
    Trains and evaluates UFC prediction models.
    
    Key design decisions:
    - Time-series split (no future data leakage)
    - Features are as of fight time
    - Compare model predictions vs market implied probabilities
    """
    
    def __init__(self, db_path: str = "data/historical_ufc.db", config: TrainingConfig = None):
        self.db_path = Path(db_path)
        self.config = config or TrainingConfig()
        self.model = None
        self.feature_cols = []
        
    def load_training_data(self) -> pd.DataFrame:
        """
        Load labeled training data from database.
        
        Returns DataFrame with:
        - fight_id, event_date
        - event_id, event_name
        - fighter_a, fighter_b
        - all features (reach_diff, age_diff, etc.)
        - target: 1 if fighter_a won, 0 if fighter_b won (filter out draws/NCs)
        - market_implied_prob (from closing odds)
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                f.fight_id,
                f.event_id,
                e.event_name,
                e.event_date,
                f.fighter_a,
                f.fighter_b,
                f.winner,
                f.result_method,
                f.went_decision,
                f.ended_inside_distance,
                fo.close_a_odds,
                fo.close_b_odds,
                -- Fighter A stats snapshot
                h_a.height_in as a_height,
                h_a.reach_in as a_reach,
                h_a.ufc_wins as a_wins,
                h_a.ufc_losses as a_losses,
                h_a.total_ufc_fights as a_total,
                h_a.sig_strikes_landed_per_min as a_ssl,
                h_a.sig_strikes_absorbed_per_min as a_ssa,
                h_a.takedown_avg as a_td,
                h_a.finish_rate as a_finish,
                -- Fighter B stats snapshot
                h_b.height_in as b_height,
                h_b.reach_in as b_reach,
                h_b.ufc_wins as b_wins,
                h_b.ufc_losses as b_losses,
                h_b.total_ufc_fights as b_total,
                h_b.sig_strikes_landed_per_min as b_ssl,
                h_b.sig_strikes_absorbed_per_min as b_ssa,
                h_b.takedown_avg as b_td,
                h_b.finish_rate as b_finish
            FROM fights f
            JOIN events e ON f.event_id = e.event_id
            LEFT JOIN fight_odds fo ON f.fight_id = fo.fight_id
            LEFT JOIN fighter_histories h_a ON f.fight_id = h_a.fight_id 
                AND f.fighter_a = h_a.fighter_name
            LEFT JOIN fighter_histories h_b ON f.fight_id = h_b.fight_id 
                AND f.fighter_b = h_b.fighter_name
            WHERE f.winner IS NOT NULL  -- Exclude draws/NCs
            ORDER BY e.event_date
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"Loaded {len(df)} labeled fights")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from raw data.
        
        This mirrors the logic in features/fighter_features.py
        but for historical data (already joined).
        """
        df = df.copy()
        
        # Physical differences
        df["reach_diff"] = df["a_reach"] - df["b_reach"]
        df["height_diff"] = df["a_height"] - df["b_height"]
        
        # Experience
        df["experience_diff"] = df["a_total"] - df["b_total"]
        df["win_rate_diff"] = (df["a_wins"] / df["a_total"].clip(lower=1)) - \
                              (df["b_wins"] / df["b_total"].clip(lower=1))
        
        # Striking
        df["strike_margin_diff"] = (df["a_ssl"] - df["a_ssa"]) - (df["b_ssl"] - df["b_ssa"])
        df["striking_offense_diff"] = df["a_ssl"] - df["b_ssl"]
        df["striking_defense_diff"] = df["b_ssa"] - df["a_ssa"]
        
        # Grappling
        df["takedown_diff"] = df["a_td"] - df["b_td"]
        
        # Finish rates
        df["finish_rate_diff"] = df["a_finish"] - df["b_finish"]
        
        # Market implied probability (from closing odds)
        def odds_to_implied(odds):
            if pd.isna(odds):
                return 0.5
            if odds > 0:
                return 100 / (odds + 100)
            return abs(odds) / (abs(odds) + 100)
        
        df["implied_prob_a"] = df["close_a_odds"].apply(odds_to_implied)
        df["implied_prob_b"] = df["close_b_odds"].apply(odds_to_implied)
        
        # Target: 1 if fighter_a won, 0 if fighter_b won
        df["target"] = (df["fighter_a"] == df["winner"]).astype(int)
        
        # Filter out rows with missing critical data
        df = df.dropna(subset=["reach_diff", "height_diff", "target"])
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Prepare X, y for sklearn."""
        # Define feature columns
        self.feature_cols = [
            "reach_diff",
            "height_diff",
            "experience_diff",
            "win_rate_diff",
            "strike_margin_diff",
            "striking_offense_diff",
            "striking_defense_diff",
            "takedown_diff",
            "finish_rate_diff",
            "implied_prob_a",  # Include market as a feature
        ]
        
        X = df[self.feature_cols].values
        y = df["target"].values
        
        return X, y

    def build_base_model(self) -> LogisticRegression:
        if self.config.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=self.config.random_state,
            )
        raise NotImplementedError(f"Model type {self.config.model_type} not implemented")

    def build_estimator(self, calibration_cv: int = 3):
        base_model = self.build_base_model()
        if self.config.calibration:
            return CalibratedClassifierCV(base_model, cv=calibration_cv)
        return base_model
    
    def train(self, df: Optional[pd.DataFrame] = None, save_path: Optional[str] = None) -> dict:
        """
        Train model with time-series cross-validation.
        
        TimeSeriesSplit is critical here — we can't use future fights
        to predict past fights.
        """
        if df is None:
            df = self.load_training_data()
            df = self.engineer_features(df)
        
        X, y = self.prepare_features(df)
        
        print(f"Training on {len(y)} fights with {len(self.feature_cols)} features")
        print(f"Features: {self.feature_cols}")
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        cv_scores = {
            "accuracy": [],
            "roc_auc": [],
            "log_loss": [],
            "brier": [],
            "calibration_error": [],
        }
        
        models = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}/{self.config.cv_splits}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = self.build_estimator(calibration_cv=3)
            model.fit(X_train, y_train)
            models.append(model)
            
            # Predictions
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_prob)
            ll = log_loss(y_val, y_prob)
            brier = brier_score_loss(y_val, y_prob)
            
            # Calibration error
            cal_error = np.mean(np.abs(y_prob - y_val))
            
            cv_scores["accuracy"].append(acc)
            cv_scores["roc_auc"].append(auc)
            cv_scores["log_loss"].append(ll)
            cv_scores["brier"].append(brier)
            cv_scores["calibration_error"].append(cal_error)
            
            print(f"  Accuracy: {acc:.3f}, AUC: {auc:.3f}, LogLoss: {ll:.3f}, Brier: {brier:.3f}")
        
        # Train final model on all data
        print("\nTraining final model on all data...")
        self.model = self.build_estimator(calibration_cv=5)
        self.model.fit(X, y)
        
        # Feature importance (coefficients for logistic regression)
        if hasattr(self.model, 'coef_'):
            coefs = self.model.coef_[0]
        elif hasattr(self.model, 'calibrated_classifiers_'):
            coefs = self.model.calibrated_classifiers_[0].estimator.coef_[0]
        else:
            coefs = np.zeros(len(self.feature_cols))
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'coefficient': coefs,
            'abs_coef': np.abs(coefs)
        }).sort_values('abs_coef', ascending=False)
        
        print("\nFeature Importance (Logistic Coefficients):")
        print(importance.to_string(index=False))
        
        # Summary
        results = {
            "cv_mean_accuracy": np.mean(cv_scores["accuracy"]),
            "cv_mean_auc": np.mean(cv_scores["roc_auc"]),
            "cv_mean_logloss": np.mean(cv_scores["log_loss"]),
            "cv_mean_brier": np.mean(cv_scores["brier"]),
            "feature_importance": importance,
            "n_fights": len(y),
            "n_features": len(self.feature_cols),
        }
        
        print("\n" + "=" * 50)
        print("TRAINING RESULTS")
        print("=" * 50)
        print(f"CV Accuracy: {results['cv_mean_accuracy']:.3f}")
        print(f"CV AUC: {results['cv_mean_auc']:.3f}")
        print(f"CV LogLoss: {results['cv_mean_logloss']:.3f}")
        print(f"CV Brier: {results['cv_mean_brier']:.3f}")
        print("=" * 50)
        
        # Save model
        if save_path:
            self.save_model(save_path)
        
        return results
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for new fights."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X = features[self.feature_cols].values
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path: str) -> None:
        """Save trained model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'config': self.config.__dict__,
            }, f)
        print(f"Model saved to {path}")

    @staticmethod
    def load_saved_bundle(path: str) -> dict[str, object]:
        with open(path, 'rb') as f:
            return _TrainerModelUnpickler(f).load()

    def load_model(self, path: str) -> None:
        """Load trained model from disk."""
        saved = self.load_saved_bundle(path)
        self.model = saved['model']
        self.feature_cols = saved['feature_cols']
        self.config = _coerce_training_config(saved.get('config', {}))
        print(f"Model loaded from {path}")


def main():
    """Train model using sample data or database."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train UFC prediction model")
    parser.add_argument("--db-path", default="data/historical_ufc.db")
    parser.add_argument("--save-path", default="models/ufc_model.pkl")
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "xgboost"])
    
    args = parser.parse_args()
    
    config = TrainingConfig(model_type=args.model_type)
    trainer = UFCModelTrainer(db_path=args.db_path, config=config)
    
    results = trainer.train(save_path=args.save_path)
    
    print("\nModel training complete!")


if __name__ == "__main__":
    main()
