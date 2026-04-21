import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import OddsApiError, fetch_the_odds_api_events


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


if __name__ == "__main__":
    unittest.main()
