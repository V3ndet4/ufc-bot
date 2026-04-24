import random
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.run_model_backtest import run_model_backtest
from scripts.generate_sample_training_data import generate_sample_database
from models.trainer import UFCModelTrainer


class ModelBacktestTests(unittest.TestCase):
    def test_run_model_backtest_walk_forward_is_out_of_sample(self) -> None:
        run_id = uuid4().hex
        db_path = ROOT / "data" / f"test_model_backtest_{run_id}.db"
        model_path = ROOT / "models" / f"test_model_backtest_{run_id}.pkl"

        try:
            random.seed(42)
            generate_sample_database(str(db_path), n_fights=120)

            trainer = UFCModelTrainer(db_path=str(db_path))
            trainer.train(save_path=str(model_path))
            self.assertTrue(model_path.exists())

            report, summary, selections = run_model_backtest(
                model_path=model_path,
                db_path=db_path,
                min_edge=0.0,
                bankroll=1000.0,
                fractional_kelly=0.25,
                evaluation_mode="walk_forward",
            )
        finally:
            model_path.unlink(missing_ok=True)
            db_path.unlink(missing_ok=True)

        self.assertGreater(int(summary["fights_evaluated"]), 0)
        self.assertGreater(int(summary["historical_fights_available"]), int(summary["fights_evaluated"]))
        self.assertGreater(int(summary["burn_in_fights"]), 0)
        self.assertEqual(int(summary["selection_rows_evaluated"]), len(selections))
        self.assertEqual(int(summary["fights_evaluated"]) * 2, len(selections))
        self.assertLessEqual(int(summary["picks"]), len(selections))
        self.assertIn("fight_accuracy", summary)
        self.assertIn("fight_auc", summary)
        self.assertEqual(summary["backtest_scope"], "walk_forward_out_of_sample")
        self.assertEqual(int(summary["evaluation_folds"]), 5)
        self.assertFalse(report.empty)
        self.assertTrue((selections["model_projected_win_prob"] >= 0.0).all())
        self.assertTrue((selections["model_projected_win_prob"] <= 1.0).all())
        self.assertEqual(set(selections["market"]), {"moneyline"})
        self.assertEqual(set(selections["book"]), {"synthetic_close"})
        self.assertTrue((selections["evaluation_fold"] > 0).all())
        self.assertTrue((selections["training_rows"] > 0).all())


if __name__ == "__main__":
    unittest.main()
