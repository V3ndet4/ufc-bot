import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_espn_fighter_map import _filter_to_event_fighters


class UpdateEspnFighterMapTests(unittest.TestCase):
    def test_filter_to_event_fighters_drops_cache_only_rows(self) -> None:
        mapping = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": ""},
                {"fighter_name": "Beta", "espn_url": ""},
            ]
        )
        merged = pd.DataFrame(
            [
                {"fighter_name": "Alpha", "espn_url": "https://www.espn.com/mma/fighter/_/id/1/alpha"},
                {"fighter_name": "Beta", "espn_url": "https://www.espn.com/mma/fighter/_/id/2/beta"},
                {"fighter_name": "Old Fighter", "espn_url": "https://www.espn.com/mma/fighter/_/id/9/old"},
            ]
        )

        filtered = _filter_to_event_fighters(merged, mapping)

        self.assertEqual(filtered["fighter_name"].tolist(), ["Alpha", "Beta"])
        self.assertEqual(
            filtered["espn_url"].tolist(),
            [
                "https://www.espn.com/mma/fighter/_/id/1/alpha",
                "https://www.espn.com/mma/fighter/_/id/2/beta",
            ],
        )


if __name__ == "__main__":
    unittest.main()
