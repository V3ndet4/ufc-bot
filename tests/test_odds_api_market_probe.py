import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import extract_alternative_market_keys, extract_bookmaker_market_keys


class OddsApiMarketProbeTests(unittest.TestCase):
    def test_extract_bookmaker_market_keys_filters_to_target_bookmaker(self) -> None:
        payload = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {"key": "h2h"},
                        {"key": "fight_goes_distance"},
                        {"key": "fight_ends_inside_distance"},
                    ],
                },
                {
                    "key": "draftkings",
                    "markets": [
                        {"key": "h2h"},
                        {"key": "fighter_a_by_decision"},
                    ],
                },
            ]
        }

        self.assertEqual(
            extract_bookmaker_market_keys(payload, "fanduel"),
            ["fight_ends_inside_distance", "fight_goes_distance", "h2h"],
        )

    def test_extract_alternative_market_keys_excludes_featured_markets(self) -> None:
        payload = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {"key": "h2h"},
                        {"key": "totals"},
                        {"key": "fight_goes_distance"},
                        {"key": "fighter_a_by_decision"},
                    ],
                }
            ]
        }

        self.assertEqual(
            extract_alternative_market_keys(payload, "fanduel"),
            ["fight_goes_distance", "fighter_a_by_decision"],
        )


if __name__ == "__main__":
    unittest.main()
