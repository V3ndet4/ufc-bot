import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import (
    _first_round_finish_metrics,
    _outcome_profile_metrics,
    _recent_damage_metrics,
    load_espn_fighter_map,
    merge_espn_url_maps,
    merge_context_into_fighter_map,
    parse_bio_page,
    parse_fight_history,
    parse_stats_tables,
    resolve_espn_fighter_url,
)


BIO_HTML = """
<html>
  <body>
    <h1>Anthony Smith</h1>
    <ul>
      <li>USA</li>
      <li>Light Heavyweight</li>
      <li>HT/WT</li>
      <li>6' 4", 206 lbs</li>
      <li>Stance</li>
      <li>Orthodox</li>
      <li>Reach</li>
      <li>76"</li>
    </ul>
    <div>W-L-D</div>
    <div>38-22-0</div>
  </body>
</html>
"""

OVERVIEW_HTML = """
<table>
  <thead>
    <tr><th>Date</th><th>Opponent</th><th>Res.</th><th>Decision</th><th>Rnd</th><th>Time</th><th>Event</th></tr>
  </thead>
  <tbody>
    <tr><td>Apr 26, 2025</td><td>Zhang Mingyang</td><td>L</td><td>KO/TKO</td><td>1</td><td>2:31</td><td>UFC Fight Night</td></tr>
    <tr><td>Dec 7, 2024</td><td>Dominick Reyes</td><td>L</td><td>U Dec</td><td>3</td><td>5:00</td><td>UFC 310</td></tr>
  </tbody>
</table>
"""

STATS_HTML = """
<table>
  <thead>
    <tr><th>Date</th><th>Opponent</th><th>Event</th><th>Res.</th><th>SSL</th><th>SSA</th></tr>
  </thead>
  <tbody>
    <tr><td>Apr 26, 2025</td><td>Zhang Mingyang</td><td>UFC Fight Night</td><td>L</td><td>10</td><td>21</td></tr>
    <tr><td>Dec 7, 2024</td><td>Dominick Reyes</td><td>UFC 310</td><td>L</td><td>21</td><td>46</td></tr>
  </tbody>
</table>
<table>
  <thead>
    <tr><th>Date</th><th>Opponent</th><th>Event</th><th>Res.</th><th>TDL</th><th>TDA</th></tr>
  </thead>
  <tbody>
    <tr><td>Apr 26, 2025</td><td>Zhang Mingyang</td><td>UFC Fight Night</td><td>L</td><td>0</td><td>1</td></tr>
    <tr><td>Dec 7, 2024</td><td>Dominick Reyes</td><td>UFC 310</td><td>L</td><td>0</td><td>4</td></tr>
  </tbody>
</table>
"""


class EspnParsingTests(unittest.TestCase):
    def test_parse_bio_page_extracts_core_fields(self) -> None:
        fighter = parse_bio_page(BIO_HTML)
        self.assertEqual(fighter["fighter_name"], "Anthony Smith")
        self.assertEqual(fighter["wins"], 38)
        self.assertEqual(fighter["losses"], 22)
        self.assertEqual(fighter["height_in"], 76.0)
        self.assertEqual(fighter["reach_in"], 76.0)
        self.assertEqual(fighter["stance"], "Orthodox")

    def test_parse_stats_tables_extracts_striking_and_clinch(self) -> None:
        stats = parse_stats_tables(STATS_HTML)
        self.assertEqual(list(stats["SSL"]), [10, 21])
        self.assertEqual(list(stats["TDA"]), [1, 4])

    def test_parse_fight_history_computes_minutes(self) -> None:
        history = parse_fight_history(OVERVIEW_HTML)
        self.assertAlmostEqual(history.loc[0, "minutes"], 2 + 31 / 60)
        self.assertAlmostEqual(history.loc[1, "minutes"], 15.0)

    def test_first_round_finish_metrics_counts_round_one_finishes(self) -> None:
        history = pd.DataFrame(
            [
                {"result_code": "W", "round_number": 1, "decision_type": "KO/TKO"},
                {"result_code": "W", "round_number": 1, "decision_type": "Submission"},
                {"result_code": "W", "round_number": 3, "decision_type": "U Dec"},
                {"result_code": "L", "round_number": 1, "decision_type": "KO/TKO"},
            ]
        )
        count, rate = _first_round_finish_metrics(history)
        self.assertEqual(count, 2)
        self.assertEqual(rate, 0.5)

    def test_outcome_profile_metrics_split_ko_and_submission_rates(self) -> None:
        history = pd.DataFrame(
            [
                {"result_code": "W", "decision_type": "KO/TKO"},
                {"result_code": "W", "decision_type": "Submission"},
                {"result_code": "L", "decision_type": "Submission"},
                {"result_code": "L", "decision_type": "U Dec"},
            ]
        )

        metrics = _outcome_profile_metrics(history)
        self.assertEqual(metrics, (0.5, 0.25, 0.25, 0.25, 0.25, 0.0, 0.25))

    def test_recent_damage_metrics_weight_recent_finish_and_ko_losses(self) -> None:
        history = pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-03-01"), "result_code": "L", "decision_type": "KO/TKO"},
                {"date": pd.Timestamp("2025-12-15"), "result_code": "L", "decision_type": "Submission"},
                {"date": pd.Timestamp("2025-08-01"), "result_code": "W", "decision_type": "U Dec"},
                {"date": pd.Timestamp("2024-01-01"), "result_code": "L", "decision_type": "KO/TKO"},
            ]
        )

        metrics = _recent_damage_metrics(history)

        self.assertEqual(metrics[0], 2)
        self.assertEqual(metrics[1], 1)
        self.assertEqual(metrics[2], 2)
        self.assertEqual(metrics[3], 1)
        self.assertGreater(metrics[4], 0.0)

    def test_merge_context_into_fighter_map(self) -> None:
        temp_dir = ROOT / ".tmp-test-espn"
        temp_dir.mkdir(exist_ok=True)
        mapping_path = temp_dir / "mapping.csv"
        context_path = temp_dir / "context.csv"
        try:
            mapping_path.write_text(
                "fighter_name,espn_url\nAnthony Smith,https://www.espn.com/mma/fighter/_/id/2512976/anthony-smith\n",
                encoding="utf-8",
            )
            context_path.write_text(
                "fighter_name,short_notice_flag,short_notice_acceptance_flag,short_notice_success_flag,new_gym_flag,new_contract_flag,cardio_fade_flag,context_notes\nAnthony Smith,1,1,1,0,1,1,test\n",
                encoding="utf-8",
            )
            mapping = load_espn_fighter_map(mapping_path)
            merged = merge_context_into_fighter_map(mapping, context_path)
            self.assertEqual(int(merged.loc[0, "short_notice_flag"]), 1)
            self.assertEqual(int(merged.loc[0, "short_notice_acceptance_flag"]), 1)
            self.assertEqual(int(merged.loc[0, "short_notice_success_flag"]), 1)
            self.assertEqual(int(merged.loc[0, "new_contract_flag"]), 1)
            self.assertEqual(int(merged.loc[0, "cardio_fade_flag"]), 1)
            self.assertEqual(merged.loc[0, "context_notes"], "test")
        finally:
            if mapping_path.exists():
                mapping_path.unlink()
            if context_path.exists():
                context_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

    def test_load_espn_fighter_map_keeps_blank_urls_blank(self) -> None:
        temp_dir = ROOT / ".tmp-test-espn"
        temp_dir.mkdir(exist_ok=True)
        mapping_path = temp_dir / "mapping.csv"
        try:
            mapping_path.write_text(
                "fighter_name,espn_url\nAlpha,\nBeta,https://www.espn.com/mma/fighter/_/id/2/beta\n",
                encoding="utf-8",
            )
            mapping = load_espn_fighter_map(mapping_path)
            self.assertEqual(mapping.loc[0, "espn_url"], "")
            self.assertEqual(mapping.loc[1, "espn_url"], "https://www.espn.com/mma/fighter/_/id/2/beta")
        finally:
            if mapping_path.exists():
                mapping_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

    def test_merge_espn_url_maps_prefers_later_non_blank_urls(self) -> None:
        template = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": ""},
                {"fighter_name": "Beta", "espn_url": ""},
            ]
        )
        cache = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": "https://www.espn.com/mma/fighter/_/id/1/alpha-cache"},
                {"fighter_name": "Beta", "espn_url": "https://www.espn.com/mma/fighter/_/id/2/beta-cache"},
            ]
        )
        existing = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": "https://www.espn.com/mma/fighter/_/id/9/alpha-manual"},
                {"fighter_name": "Beta", "espn_url": ""},
            ]
        )

        merged = merge_espn_url_maps(template, cache, existing)

        self.assertEqual(merged.loc[0, "espn_url"], "https://www.espn.com/mma/fighter/_/id/9/alpha-manual")
        self.assertEqual(merged.loc[1, "espn_url"], "https://www.espn.com/mma/fighter/_/id/2/beta-cache")

    def test_merge_espn_url_maps_applies_alias_lookup(self) -> None:
        template = pd.DataFrame(
            [
                {"fighter_name": "Ben Johnston", "espn_url": ""},
            ]
        )
        cache = pd.DataFrame(
            [
                {"fighter_name": "Benjamin Johnston", "espn_url": "https://www.espn.com/mma/fighter/_/id/5344659/ben-johnston"},
            ]
        )

        merged = merge_espn_url_maps(
            template,
            cache,
            alias_lookup={"benjamin johnston": "Ben Johnston"},
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "fighter_name"], "Ben Johnston")
        self.assertEqual(merged.loc[0, "espn_url"], "https://www.espn.com/mma/fighter/_/id/5344659/ben-johnston")

    @patch("data_sources.espn.search_espn_fighters")
    def test_resolve_espn_fighter_url_prefers_exact_display_name(self, mock_search) -> None:
        mock_search.return_value = [
            {
                "displayName": "Aljamain Sterling",
                "sport": "mma",
                "links": [
                    {
                        "rel": ["playercard", "desktop", "athlete"],
                        "href": "https://www.espn.com/mma/fighter/_/id/3031559/aljamain-sterling",
                    }
                ],
            }
        ]

        url = resolve_espn_fighter_url("Aljamain Sterling")

        self.assertEqual(url, "https://www.espn.com/mma/fighter/_/id/3031559/aljamain-sterling")

    @patch("data_sources.espn.search_espn_fighters")
    def test_resolve_espn_fighter_url_accepts_unique_last_name_match(self, mock_search) -> None:
        mock_search.side_effect = [
            [],
            [
                {
                    "displayName": "Adrian Luna Martinetti",
                    "sport": "mma",
                    "links": [
                        {
                            "rel": ["playercard", "desktop", "athlete"],
                            "href": "https://www.espn.com/mma/fighter/_/id/5310570/adrian-luna-martinetti",
                        }
                    ],
                }
            ],
        ]

        url = resolve_espn_fighter_url("Juan Martinetti")

        self.assertEqual(url, "https://www.espn.com/mma/fighter/_/id/5310570/adrian-luna-martinetti")


if __name__ == "__main__":
    unittest.main()
