import sys
import unittest
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.historical_archive import build_historical_archive, build_historical_moneyline_archive


class HistoricalArchiveTests(unittest.TestCase):
    def test_build_full_archive_from_completed_cards(self) -> None:
        workspace = ROOT / "tests" / "_tmp_historical_archive"
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            cards_root = workspace / "cards"
            data_dir = cards_root / "test_card" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "winner_name": "Alpha",
                        "winner_side": "fighter_a",
                        "result_status": "official",
                        "went_decision": 0,
                        "ended_inside_distance": 1,
                        "method": "KO/TKO",
                        "closing_fighter_a_odds": -150,
                        "closing_fighter_b_odds": 130,
                        "closing_fight_goes_to_decision_odds": 180,
                        "closing_fight_doesnt_go_to_decision_odds": -220,
                        "closing_fighter_a_inside_distance_odds": 210,
                        "closing_fighter_b_inside_distance_odds": 340,
                        "closing_fighter_a_by_decision_odds": 500,
                        "closing_fighter_b_by_decision_odds": 600,
                    }
                ]
            ).to_csv(data_dir / "results.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-18T20:00:00-04:00",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "scheduled_rounds": 3,
                        "is_title_fight": 0,
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "manual",
                        "american_odds": pd.NA,
                    }
                ]
            ).to_csv(data_dir / "odds_template.csv", index=False)

            archive, summary = build_historical_archive(cards_root)

            self.assertEqual(len(archive), 8)
            self.assertEqual(set(archive["market"]), {
                "moneyline",
                "fight_goes_to_decision",
                "fight_doesnt_go_to_decision",
                "inside_distance",
                "by_decision",
            })
            goes_row = archive.loc[archive["market"] == "fight_goes_to_decision"].iloc[0]
            doesnt_row = archive.loc[archive["market"] == "fight_doesnt_go_to_decision"].iloc[0]
            inside_a_row = archive.loc[
                (archive["market"] == "inside_distance") & (archive["selection"] == "fighter_a")
            ].iloc[0]
            by_decision_a_row = archive.loc[
                (archive["market"] == "by_decision") & (archive["selection"] == "fighter_a")
            ].iloc[0]

            self.assertEqual(goes_row["actual_result"], "loss")
            self.assertEqual(doesnt_row["actual_result"], "win")
            self.assertEqual(inside_a_row["actual_result"], "win")
            self.assertEqual(by_decision_a_row["actual_result"], "loss")
            self.assertEqual(int(summary.loc[0, "rows_written"]), 8)
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)

    def test_build_archive_from_completed_cards(self) -> None:
        workspace = ROOT / "tests" / "_tmp_historical_archive"
        if workspace.exists():
            shutil.rmtree(workspace)
        try:
            cards_root = workspace / "cards"
            data_dir = cards_root / "test_card" / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                [
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "winner_name": "Alpha",
                        "winner_side": "fighter_a",
                        "result_status": "official",
                        "went_decision": 0,
                        "ended_inside_distance": 1,
                        "method": "KO/TKO",
                        "closing_fighter_a_odds": -150,
                        "closing_fighter_b_odds": 130,
                    },
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "winner_name": "",
                        "winner_side": "draw",
                        "result_status": "draw",
                        "went_decision": 1,
                        "ended_inside_distance": 0,
                        "method": "Decision",
                        "closing_fighter_a_odds": pd.NA,
                        "closing_fighter_b_odds": pd.NA,
                    },
                ]
            ).to_csv(data_dir / "results.csv", index=False)

            pd.DataFrame(
                [
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-18T20:00:00-04:00",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "scheduled_rounds": 3,
                        "is_title_fight": 0,
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "manual",
                        "american_odds": pd.NA,
                    },
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-18T20:00:00-04:00",
                        "fighter_a": "Alpha",
                        "fighter_b": "Beta",
                        "scheduled_rounds": 3,
                        "is_title_fight": 0,
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "manual",
                        "american_odds": pd.NA,
                    },
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-18T20:00:00-04:00",
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "scheduled_rounds": 3,
                        "is_title_fight": 0,
                        "market": "moneyline",
                        "selection": "fighter_a",
                        "book": "bestfightodds_consensus",
                        "american_odds": 110,
                    },
                    {
                        "event_id": "event-1",
                        "event_name": "Test Event",
                        "start_time": "2026-04-18T20:00:00-04:00",
                        "fighter_a": "Gamma",
                        "fighter_b": "Delta",
                        "scheduled_rounds": 3,
                        "is_title_fight": 0,
                        "market": "moneyline",
                        "selection": "fighter_b",
                        "book": "bestfightodds_consensus",
                        "american_odds": -130,
                    },
                ]
            ).to_csv(data_dir / "bfo_odds.csv", index=False)

            archive, summary = build_historical_moneyline_archive(cards_root)

            self.assertEqual(len(archive), 4)
            alpha_rows = archive.loc[archive["fighter_a"] == "Alpha"].reset_index(drop=True)
            self.assertEqual(alpha_rows.loc[0, "book"], "results_close")
            self.assertEqual(int(alpha_rows.loc[0, "american_odds"]), -150)
            self.assertEqual(alpha_rows.loc[0, "actual_result"], "win")
            self.assertEqual(alpha_rows.loc[1, "actual_result"], "loss")

            gamma_rows = archive.loc[archive["fighter_a"] == "Gamma"].reset_index(drop=True)
            self.assertEqual(gamma_rows.loc[0, "book"], "bestfightodds_consensus")
            self.assertEqual(int(gamma_rows.loc[0, "odds_is_fallback"]), 1)
            self.assertEqual(gamma_rows.loc[0, "actual_result"], "push")
            self.assertEqual(gamma_rows.loc[1, "actual_result"], "push")

            self.assertEqual(len(summary), 1)
            self.assertEqual(int(summary.loc[0, "rows_written"]), 4)
            self.assertEqual(int(summary.loc[0, "fallback_fights"]), 1)
        finally:
            if workspace.exists():
                shutil.rmtree(workspace)


if __name__ == "__main__":
    unittest.main()
