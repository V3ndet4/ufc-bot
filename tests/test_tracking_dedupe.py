import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks, save_tracked_picks
from models.threshold_policy import _normalize_training_frame


def _pick(odds: int, edge: float, *, pick_id: int | None = None, profit: float = 0.0) -> dict[str, object]:
    row: dict[str, object] = {
        "event_id": "e1",
        "event_name": "Event",
        "start_time": "2026-05-02T19:00:00Z",
        "fighter_a": "Alpha",
        "fighter_b": "Beta",
        "market": "moneyline",
        "selection": "fighter_a",
        "selection_name": "Alpha",
        "book": "Book",
        "american_odds": odds,
        "model_projected_win_prob": 0.60,
        "implied_prob": 0.50,
        "edge": edge,
        "expected_value": 0.20,
        "suggested_stake": 10.0,
        "chosen_expression_stake": 10.0,
        "recommended_tier": "A",
        "recommended_action": "Bettable now",
        "grade_status": "graded",
        "actual_result": "win" if profit > 0 else "loss",
        "profit": profit,
        "tracked_at": "2026-05-01T12:00:00Z",
        "fight_key": "alpha||beta",
        "tracked_market_key": "moneyline",
        "tracked_selection_key": "fighter_a",
    }
    if pick_id is not None:
        row["pick_id"] = pick_id
    return row


class TrackingDedupeTests(unittest.TestCase):
    def test_save_tracked_picks_replaces_existing_pending_pick(self) -> None:
        db_path = ROOT / "tests" / "_tmp_tracking_dedupe.db"
        db_path.unlink(missing_ok=True)
        try:
            first = pd.DataFrame([_pick(120, 0.08)])
            first["grade_status"] = "pending"
            second = pd.DataFrame([_pick(140, 0.12)])
            second["grade_status"] = "pending"

            self.assertEqual(save_tracked_picks(first, db_path), 1)
            self.assertEqual(save_tracked_picks(second, db_path), 1)

            saved = load_tracked_picks(db_path)
            self.assertEqual(len(saved), 1)
            self.assertEqual(int(saved.loc[0, "american_odds"]), 140)
        finally:
            db_path.unlink(missing_ok=True)

    def test_threshold_training_dedupes_repeated_tracked_pick_keys(self) -> None:
        rows = [
            _pick(120, 0.08, pick_id=1, profit=5.0),
            _pick(140, 0.12, pick_id=2, profit=-10.0),
            {
                **_pick(-110, 0.05, pick_id=3, profit=4.0),
                "fighter_a": "Gamma",
                "fighter_b": "Delta",
                "fight_key": "delta||gamma",
            },
        ]

        normalized = _normalize_training_frame(pd.DataFrame(rows))

        self.assertEqual(len(normalized), 2)
        alpha_row = normalized.loc[normalized["fight_key"].eq("alpha||beta")].iloc[0]
        self.assertEqual(int(alpha_row["pick_id"]), 2)
        self.assertAlmostEqual(float(alpha_row["profit_amount"]), -10.0)


if __name__ == "__main__":
    unittest.main()
