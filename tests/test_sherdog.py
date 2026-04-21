import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.sherdog import (
    build_fightfinder_url,
    build_gym_registry,
    merge_fighter_gym_data,
    parse_fightfinder_results_page,
    parse_fighter_profile,
    parse_search_results,
)


FIGHTFINDER_HTML = """
<html>
  <body>
    <table>
      <thead>
        <tr><th>Fighter</th><th>Nickname</th><th>Association</th></tr>
      </thead>
      <tbody>
        <tr><td>Mike Malott</td><td></td><td>Team Alpha Male</td></tr>
        <tr><td>Gilbert Burns</td><td></td><td>Kill Cliff FC</td></tr>
      </tbody>
    </table>
    <a href="/fighter/Mike-Malott-12345">Mike Malott</a>
    <a href="/fighter/Gilbert-Burns-67890">Gilbert Burns</a>
  </body>
</html>
"""

PROFILE_HTML = """
<html>
  <body>
    <h1>Mike Malott</h1>
    <div>ASSOCIATION</div>
    <div>Team Alpha Male</div>
    <div>CLASS</div>
    <div>Welterweight</div>
    <a href="/stats/fightfinder?association=Team+Alpha+Male">Team Alpha Male</a>
    <div>Wins 11 Losses 2 Draws 0</div>
  </body>
</html>
"""

SEARCH_PAYLOAD = {
    "error": "",
    "total_found": 1,
    "limit": 10,
    "time": "0.000",
    "collection": [
        {
            "id": "66313",
            "firstname": "Aljamain",
            "lastname": "Sterling",
            "nickname": "Funk Master",
            "url": "/fighter/Aljamain-Sterling-66313",
            "association": ["Serra-Longo Fight Team"],
            "source": "Fighter",
        }
    ],
}


class SherdogTests(unittest.TestCase):
    def test_parse_search_results_extracts_fighter_and_association(self) -> None:
        results = parse_search_results(SEARCH_PAYLOAD)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["fighter_name"], "Aljamain Sterling")
        self.assertEqual(results[0]["gym_name"], "Serra-Longo Fight Team")
        self.assertEqual(results[0]["sherdog_url"], "https://www.sherdog.com/fighter/Aljamain-Sterling-66313")
        self.assertEqual(results[0]["gym_source"], "sherdog_search")

    def test_parse_fightfinder_results_page_extracts_fighters_and_urls(self) -> None:
        results = parse_fightfinder_results_page(FIGHTFINDER_HTML)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["fighter_name"], "Mike Malott")
        self.assertEqual(results[0]["gym_name"], "Team Alpha Male")
        self.assertEqual(results[0]["sherdog_url"], "https://www.sherdog.com/fighter/Mike-Malott-12345")

    def test_parse_fighter_profile_extracts_association_and_record(self) -> None:
        fighter = parse_fighter_profile(PROFILE_HTML, "https://www.sherdog.com/fighter/Mike-Malott-12345")

        self.assertEqual(fighter["fighter_name"], "Mike Malott")
        self.assertEqual(fighter["gym_source"], "sherdog_profile")
        self.assertEqual(fighter["gym_name"], "Team Alpha Male")
        self.assertEqual(fighter["weight_class"], "Welterweight")
        self.assertEqual(fighter["fighter_wins"], 11)
        self.assertEqual(fighter["fighter_losses"], 2)
        self.assertGreater(fighter["fighter_win_rate"], 0.8)

    def test_build_fightfinder_url_treats_nan_association_as_empty(self) -> None:
        url = build_fightfinder_url(association=float("nan"))

        self.assertIn("association=", url)
        self.assertNotIn("association=nan", url.lower())

    def test_build_gym_registry_aggregates_depth_and_record(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "fighter_name": "Mike Malott",
                    "fighter_name_normalized": "mike malott",
                    "gym_name": "Team Alpha Male",
                    "gym_name_normalized": "team alpha male",
                    "gym_page_url": "https://www.sherdog.com/stats/fightfinder?association=Team+Alpha+Male",
                    "fighter_wins": 11,
                    "fighter_losses": 2,
                    "fighter_draws": 0,
                    "fighter_elite_flag": 1,
                    "profile_last_refreshed_at": "2026-04-18T12:00:00+00:00",
                },
                {
                    "fighter_name": "Fighter Two",
                    "fighter_name_normalized": "fighter two",
                    "gym_name": "Team Alpha Male",
                    "gym_name_normalized": "team alpha male",
                    "gym_page_url": "https://www.sherdog.com/stats/fightfinder?association=Team+Alpha+Male",
                    "fighter_wins": 8,
                    "fighter_losses": 3,
                    "fighter_draws": 0,
                    "fighter_elite_flag": 0,
                    "profile_last_refreshed_at": "2026-04-18T12:00:00+00:00",
                },
            ]
        )

        registry = build_gym_registry(frame)

        self.assertEqual(len(registry), 1)
        self.assertEqual(registry.loc[0, "gym_name"], "Team Alpha Male")
        self.assertEqual(int(registry.loc[0, "gym_fighter_count"]), 2)
        self.assertEqual(registry.loc[0, "gym_record"], "19-5-0")
        self.assertGreater(float(registry.loc[0, "gym_score"]), 0.0)

    def test_merge_fighter_gym_data_carries_auto_switch_flags(self) -> None:
        temp_dir = ROOT / ".tmp-test-sherdog"
        temp_dir.mkdir(exist_ok=True)
        fighter_gyms_path = temp_dir / "fighter_gyms.csv"
        fighter_gyms = pd.DataFrame(
            [
                {
                    "fighter_name": "Mike Malott",
                    "gym_name": "Team Alpha Male",
                    "gym_tier": "A",
                    "gym_score": 0.76,
                    "gym_changed_flag": 1,
                    "previous_gym_name": "Xtreme Couture",
                }
            ]
        )
        fighter_gyms.to_csv(fighter_gyms_path, index=False)

        frame = pd.DataFrame(
            [
                {
                    "fighter_name": "Mike Malott",
                    "new_gym_flag": 0,
                    "camp_change_flag": 0,
                }
            ]
        )

        try:
            merged = merge_fighter_gym_data(frame, fighter_gyms_path)
        finally:
            fighter_gyms_path.unlink(missing_ok=True)
            temp_dir.rmdir()

        self.assertEqual(merged.loc[0, "gym_name"], "Team Alpha Male")
        self.assertEqual(merged.loc[0, "gym_tier"], "A")
        self.assertEqual(int(merged.loc[0, "gym_changed_flag"]), 1)
        self.assertEqual(int(merged.loc[0, "new_gym_flag"]), 1)
        self.assertEqual(int(merged.loc[0, "camp_change_flag"]), 1)


if __name__ == "__main__":
    unittest.main()
