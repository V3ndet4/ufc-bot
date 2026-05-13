import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from data_sources.odds_api import OddsApiError, extract_modeled_market_rows, fetch_the_odds_api_events


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, params: dict[str, object], timeout: int) -> _FakeResponse:
        self.calls.append((url, dict(params)))
        return self._responses.pop(0)


class OddsApiTests(unittest.TestCase):
    def test_fetch_events_retries_on_503_then_succeeds(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(503, {"message": "Service Unavailable"}),
                _FakeResponse(200, [{"id": "fight-1", "bookmakers": []}]),
            ]
        )

        with patch("data_sources.odds_api.time.sleep") as sleep_mock:
            payload = fetch_the_odds_api_events(api_key="test-key", session=session)

        self.assertEqual(payload, [{"id": "fight-1", "bookmakers": []}])
        self.assertEqual(len(session.calls), 2)
        sleep_mock.assert_called_once()

    def test_fetch_events_raises_clean_error_after_final_503(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(503, {"message": "Service Unavailable"}),
                _FakeResponse(503, {"message": "Service Unavailable"}),
                _FakeResponse(503, {"message": "Service Unavailable"}),
                _FakeResponse(503, {"message": "Service Unavailable"}),
            ]
        )

        with patch("data_sources.odds_api.time.sleep"):
            with self.assertRaises(OddsApiError) as context:
                fetch_the_odds_api_events(api_key="test-key", session=session)

        self.assertIn("HTTP 503", str(context.exception))

    def test_extract_modeled_market_rows_supports_method_knockdown_and_takedown_props(self) -> None:
        event_rows = pd.DataFrame(
            [
                {
                    "event_id": "e1",
                    "event_name": "Event",
                    "start_time": "2026-05-16T19:00:00Z",
                    "fighter_a": "Alpha",
                    "fighter_b": "Beta",
                    "odds_api_event_id": "fight-1",
                }
            ]
        )
        payload = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "fighter_by_submission",
                            "outcomes": [
                                {"name": "Alpha", "description": "Alpha by submission", "price": 550},
                                {"name": "Beta", "description": "Beta by submission", "price": 700},
                            ],
                        },
                        {
                            "key": "fighter_by_ko_tko",
                            "outcomes": [
                                {"name": "Alpha", "description": "Alpha by KO/TKO", "price": 300},
                                {"name": "Beta", "description": "Beta by KO/TKO", "price": 450},
                            ],
                        },
                        {
                            "key": "fight_ends_by_submission",
                            "outcomes": [{"name": "Yes", "price": 260}, {"name": "No", "price": -330}],
                        },
                        {
                            "key": "fight_ends_by_ko_tko",
                            "outcomes": [{"name": "Yes", "price": 125}, {"name": "No", "price": -145}],
                        },
                        {
                            "key": "fighter_knockdowns",
                            "outcomes": [
                                {"name": "Alpha", "description": "Alpha knockdown", "price": 180},
                                {"name": "Beta", "description": "Beta knockdown", "price": 260},
                            ],
                        },
                        {
                            "key": "fighter_takedowns",
                            "outcomes": [
                                {"name": "Alpha", "description": "Alpha takedown", "price": -110},
                                {"name": "Beta", "description": "Beta takedown", "price": 120},
                            ],
                        },
                    ],
                }
            ]
        }

        rows = extract_modeled_market_rows(payload, bookmaker_key="fanduel", event_rows=event_rows)
        lookup = {
            (str(row.market), str(row.selection)): int(row.american_odds)
            for row in rows.itertuples(index=False)
        }

        self.assertEqual(lookup[("submission", "fighter_a")], 550)
        self.assertEqual(lookup[("ko_tko", "fighter_b")], 450)
        self.assertEqual(lookup[("fight_ends_by_submission", "fight_ends_by_submission")], 260)
        self.assertEqual(lookup[("fight_ends_by_ko_tko", "fight_ends_by_ko_tko")], 125)
        self.assertEqual(lookup[("knockdown", "fighter_a")], 180)
        self.assertEqual(lookup[("takedown", "fighter_b")], 120)


if __name__ == "__main__":
    unittest.main()
