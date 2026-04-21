import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_line_movement_report import (
    build_line_movement_panels,
    load_moneyline_odds,
    render_line_movement_svg,
    write_per_fight_svgs,
)


class LineMovementReportTests(unittest.TestCase):
    def test_build_line_movement_panels_uses_open_snapshot_and_current(self) -> None:
        odds = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "american_odds": -210,
                    "open_american_odds": -110,
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "american_odds": 175,
                    "open_american_odds": -110,
                },
            ]
        )
        odds["selection_name"] = odds.apply(
            lambda row: row["fighter_a"] if row["selection"] == "fighter_a" else row["fighter_b"],
            axis=1,
        )
        snapshots = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_a",
                    "selection_name": "Alpha",
                    "american_odds": -160,
                    "snapshot_time": "2026-04-10T15:00:00Z",
                },
                {
                    "event_id": "e1",
                    "event_name": "Test Event",
                    "start_time": "2026-04-13T20:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "market": "moneyline",
                    "selection": "fighter_b",
                    "selection_name": "Beta",
                    "american_odds": 140,
                    "snapshot_time": "2026-04-10T15:00:00Z",
                },
            ]
        )

        panels = build_line_movement_panels(odds, snapshots)

        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0]["timeline_labels"][0], "Open")
        self.assertEqual(panels[0]["timeline_labels"][-1], "Current")
        self.assertEqual([point["american_odds"] for point in panels[0]["series"][0]], [-110, -160, -210])
        self.assertEqual([point["american_odds"] for point in panels[0]["series"][1]], [-110, 140, 175])

    def test_render_line_movement_svg_contains_main_labels(self) -> None:
        panels = [
            {
                "event_name": "Test Event",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "start_time": "2026-04-13T20:00:00Z",
                "timeline_labels": ["Open", "Current"],
                "y_min": 0.40,
                "y_max": 0.72,
                "series": [
                    [
                        {
                            "x_key": "Open",
                            "selection_name": "Alpha",
                            "kind": "open",
                            "american_odds": -110,
                            "implied_prob": 0.5238,
                        },
                        {
                            "x_key": "Current",
                            "selection_name": "Alpha",
                            "kind": "current",
                            "american_odds": -210,
                            "implied_prob": 0.6774,
                        },
                    ],
                    [
                        {
                            "x_key": "Open",
                            "selection_name": "Beta",
                            "kind": "open",
                            "american_odds": -110,
                            "implied_prob": 0.5238,
                        },
                        {
                            "x_key": "Current",
                            "selection_name": "Beta",
                            "kind": "current",
                            "american_odds": 175,
                            "implied_prob": 0.3636,
                        },
                    ],
                ],
            }
        ]

        svg = render_line_movement_svg(panels, source_label="FanDuel")

        self.assertIn("UFC Line Movement Board", svg)
        self.assertIn("FanDuel opening to current", svg)
        self.assertIn("History: open + current", svg)
        self.assertIn("Alpha vs Beta", svg)
        self.assertIn("Open", svg)
        self.assertIn("-210 | 1.48", svg)
        self.assertIn("Favorite", svg)
        self.assertIn("-100", svg)
        self.assertIn("+15.4 pts", svg)
        self.assertIn("Became bigger favorite", svg)
        self.assertIn("Favorite to Favorite", svg)
        self.assertIn("67.7%", svg)
        self.assertIn("Current", svg)

    def test_render_line_movement_svg_marks_current_only_history(self) -> None:
        panels = [
            {
                "event_name": "Test Event",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "start_time": "2026-04-13T20:00:00Z",
                "timeline_labels": ["Current"],
                "y_min": 0.20,
                "y_max": 0.80,
                "series": [
                    [
                        {
                            "x_key": "Current",
                            "selection_name": "Alpha",
                            "kind": "current",
                            "american_odds": -210,
                            "implied_prob": 0.6774,
                        }
                    ],
                    [
                        {
                            "x_key": "Current",
                            "selection_name": "Beta",
                            "kind": "current",
                            "american_odds": 175,
                            "implied_prob": 0.3636,
                        }
                    ],
                ],
            }
        ]

        svg = render_line_movement_svg(panels, source_label="FanDuel")

        self.assertIn("History: current only", svg)
        self.assertIn(">Current</text>", svg)

    def test_load_moneyline_odds_filters_to_selected_bookmaker(self) -> None:
        odds_path = ROOT / "tests" / "_tmp_odds_book_filter.csv"
        odds_path.write_text(
            "\n".join(
                [
                    "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds,open_american_odds",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,fanduel,-110,-105",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,fanduel,100,-115",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_a,draftkings,-120,-125",
                    "e1,Test Event,2026-04-13T20:00:00Z,Alpha,Beta,moneyline,fighter_b,draftkings,102,105",
                ]
            ),
            encoding="utf-8",
        )
        try:
            odds = load_moneyline_odds(odds_path, bookmaker="fanduel")
        finally:
            odds_path.unlink(missing_ok=True)

        self.assertEqual(len(odds), 2)
        self.assertEqual(set(odds["book"].astype(str)), {"fanduel"})

    def test_write_per_fight_svgs_creates_named_files(self) -> None:
        panels = [
            {
                "event_name": "Test Event",
                "event_id": "e1",
                "fighter_a": "Alpha",
                "fighter_b": "Beta",
                "start_time": "2026-04-13T20:00:00Z",
                "timeline_labels": ["Open", "Current"],
                "y_min": 0.40,
                "y_max": 0.72,
                "series": [
                    [
                        {"x_key": "Open", "selection_name": "Alpha", "kind": "open", "american_odds": -110, "implied_prob": 0.5238},
                        {"x_key": "Current", "selection_name": "Alpha", "kind": "current", "american_odds": -210, "implied_prob": 0.6774},
                    ],
                    [
                        {"x_key": "Open", "selection_name": "Beta", "kind": "open", "american_odds": -110, "implied_prob": 0.5238},
                        {"x_key": "Current", "selection_name": "Beta", "kind": "current", "american_odds": 175, "implied_prob": 0.3636},
                    ],
                ],
            }
        ]
        output_dir = ROOT / "tests" / "_tmp_line_svgs"
        try:
            written = write_per_fight_svgs(panels, output_dir, source_label="FanDuel")
            self.assertEqual(len(written), 1)
            self.assertTrue(written[0].name.startswith("alpha_vs_beta"))
            self.assertTrue(written[0].exists())
        finally:
            if output_dir.exists():
                for child in output_dir.iterdir():
                    child.unlink(missing_ok=True)
                output_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
