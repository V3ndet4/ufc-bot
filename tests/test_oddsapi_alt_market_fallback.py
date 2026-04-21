import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fight_week_report import enrich_with_oddsapi_alternative_markets


class OddsApiAltMarketFallbackTests(unittest.TestCase):
    def test_enrich_with_oddsapi_alternative_markets_fills_direct_distance_price(self) -> None:
        report = pd.DataFrame(
            [
                {
                    "odds_api_event_id": "fight-1",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "preferred_market_expression": "Fight goes to decision",
                    "fight_goes_to_decision_odds": pd.NA,
                    "fight_doesnt_go_to_decision_odds": pd.NA,
                    "fighter_a_inside_distance_odds": pd.NA,
                    "fighter_b_inside_distance_odds": pd.NA,
                    "fighter_a_by_decision_odds": pd.NA,
                    "fighter_b_by_decision_odds": pd.NA,
                    "preferred_market_american_odds": pd.NA,
                    "preferred_market_projected_prob": 0.62,
                    "side_market_american_odds": -180,
                    "side_market_projected_prob": 0.68,
                    "market_style_confidence": 0.64,
                    "market_comparison_summary": "No alternative market price found",
                }
            ]
        )

        markets_payload = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {"key": "h2h"},
                        {"key": "fight_goes_distance"},
                    ],
                }
            ]
        }
        odds_payload = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "fight_goes_distance",
                            "outcomes": [
                                {"name": "Yes", "price": 110},
                                {"name": "No", "price": -150},
                            ],
                        }
                    ],
                }
            ]
        }

        with patch("scripts.build_fight_week_report.load_api_key", return_value="test-key"), patch(
            "scripts.build_fight_week_report.fetch_the_odds_api_event_markets",
            return_value=markets_payload,
        ), patch(
            "scripts.build_fight_week_report.fetch_the_odds_api_event_odds",
            return_value=odds_payload,
        ):
            enriched = enrich_with_oddsapi_alternative_markets(report, bookmaker_key="fanduel")

        self.assertEqual(enriched.loc[0, "fight_goes_to_decision_odds"], 110)
        self.assertEqual(enriched.loc[0, "fight_doesnt_go_to_decision_odds"], -150)
        self.assertEqual(enriched.loc[0, "preferred_market_american_odds"], 110)
        self.assertEqual(enriched.loc[0, "market_comparison_summary"], "Side -180 vs alt +110")


if __name__ == "__main__":
    unittest.main()
