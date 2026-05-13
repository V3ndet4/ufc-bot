import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_event import build_card_preview


class PrepareEventTests(unittest.TestCase):
    def test_build_card_preview_lists_fights_and_context_flags(self) -> None:
        manifest = {
            "event_name": "Test Event",
            "start_time": "2026-05-02T07:00:00-04:00",
            "fights": [
                {"fighter_a": "Alpha", "fighter_b": "Beta", "scheduled_rounds": 5},
                {"fighter_a": "Gamma", "fighter_b": "Delta"},
            ],
        }
        fighter_map = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": "https://www.espn.com/mma/fighter/_/id/1/alpha"},
                {"fighter_name": "Beta", "espn_url": ""},
                {"fighter_name": "Gamma", "espn_url": "https://www.espn.com/mma/fighter/_/id/2/gamma"},
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "wins": 10,
                    "losses": 2,
                    "draws": 0,
                    "height_in": 72.0,
                    "reach_in": 74.0,
                    "sig_strikes_landed_per_min": 5.1,
                    "sig_strikes_absorbed_per_min": 2.8,
                    "takedown_avg": 1.6,
                    "takedown_defense_pct": 78.0,
                    "stance": "Orthodox",
                    "weight_class": "Lightweight",
                },
                {
                    "fighter_name": "Beta",
                    "wins": 8,
                    "losses": 3,
                    "draws": 0,
                    "height_in": 70.0,
                    "reach_in": 71.0,
                    "sig_strikes_landed_per_min": 3.9,
                    "sig_strikes_absorbed_per_min": 3.4,
                    "takedown_avg": 0.4,
                    "takedown_defense_pct": 61.0,
                    "stance": "Southpaw",
                    "weight_class": "Lightweight",
                },
            ]
        )
        gyms = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "gym_name": "Factory X", "gym_tier": "B"},
                {"fighter_name": "Beta", "gym_name": "ATT", "gym_tier": "A"},
            ]
        )
        context = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha",
                    "injury_concern_flag": 1,
                    "new_gym_flag": 0,
                    "news_alert_count": 2,
                    "news_radar_label": "red",
                    "news_radar_summary": "late camp issue",
                    "context_notes": "manual watch",
                },
                {
                    "fighter_name": "Beta",
                    "injury_concern_flag": 0,
                    "new_gym_flag": 0,
                    "news_alert_count": 0,
                    "news_radar_label": "",
                    "news_radar_summary": "",
                    "context_notes": "",
                },
            ]
        )

        preview = build_card_preview(manifest, context, fighter_map, stats, gyms)

        self.assertIn("Event: Test Event", preview)
        self.assertIn("Bouts: 2", preview)
        self.assertIn("Readiness", preview)
        self.assertIn("ESPN links mapped: 2/4 fighters", preview)
        self.assertIn("Offline stats cached: 2/4 fighters", preview)
        self.assertIn("Cached gym profiles: 2/4 fighters", preview)
        self.assertIn("1. Alpha vs Beta (5 rounds) [main event]", preview)
        self.assertIn("2. Gamma vs Delta", preview)
        self.assertIn("Alpha: 10-2-0 | Lightweight | Orthodox", preview)
        self.assertIn("Beta: 8-3-0 | Lightweight | Southpaw", preview)
        self.assertIn("Edges: reach Alpha +3.0\"", preview)
        self.assertIn("Matchup Snapshots", preview)
        self.assertIn("offline stats missing", preview)
        self.assertIn("- Alpha: injury concern | news: 2 alert(s) | red | late camp issue | notes: manual watch", preview)
        self.assertIn("Context Watchlist", preview)

    def test_build_card_preview_uses_alias_lookup_for_stats_and_gyms(self) -> None:
        manifest = {
            "event_name": "Alias Event",
            "start_time": "2026-05-02T07:00:00-04:00",
            "fights": [
                {"fighter_a": "Ben Johnston", "fighter_b": "Wes Schultz"},
            ],
        }
        fighter_map = pd.DataFrame(
            [
                {"fighter_name": "Ben Johnston", "espn_url": "https://www.espn.com/mma/fighter/_/id/5344659/ben-johnston"},
                {"fighter_name": "Wes Schultz", "espn_url": "https://www.espn.com/mma/fighter/_/id/5057023/wes-schultz"},
            ]
        )
        stats = pd.DataFrame(
            [
                {
                    "fighter_name": "Benjamin Johnston",
                    "wins": 5,
                    "losses": 1,
                    "draws": 0,
                    "height_in": 70.0,
                    "reach_in": 72.0,
                    "sig_strikes_landed_per_min": 2.0,
                    "sig_strikes_absorbed_per_min": 1.0,
                    "takedown_avg": 1.4,
                    "takedown_defense_pct": 68.0,
                    "stance": "Orthodox",
                    "weight_class": "Welterweight",
                },
                {
                    "fighter_name": "Wes Schultz",
                    "wins": 4,
                    "losses": 2,
                    "draws": 0,
                    "height_in": 71.0,
                    "reach_in": 73.0,
                    "sig_strikes_landed_per_min": 1.8,
                    "sig_strikes_absorbed_per_min": 1.5,
                    "takedown_avg": 0.6,
                    "takedown_defense_pct": 61.0,
                    "stance": "Southpaw",
                    "weight_class": "Welterweight",
                },
            ]
        )
        gyms = pd.DataFrame(
            [
                {"fighter_name": "Benjamin Johnston", "gym_name": "The Fight Centre"},
                {"fighter_name": "Wes Schultz", "gym_name": "Xtreme Couture"},
            ]
        )

        preview = build_card_preview(
            manifest,
            fighter_map_frame=fighter_map,
            stats_frame=stats,
            gym_frame=gyms,
            alias_lookup={"benjamin johnston": "Ben Johnston"},
        )

        self.assertIn("ESPN links mapped: 2/2 fighters", preview)
        self.assertIn("Offline stats cached: 2/2 fighters", preview)
        self.assertIn("Cached gym profiles: 2/2 fighters", preview)
        self.assertIn("Ben Johnston: 5-1-0 | Welterweight | Orthodox", preview)
        self.assertIn("camp The Fight Centre", preview)


if __name__ == "__main__":
    unittest.main()
