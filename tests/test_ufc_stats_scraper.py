import sys
import unittest
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.ufc_stats_scraper import UFCStatsScraper


FIGHT_ROW_HTML = """
<tr class="b-fight-details__table-row">
  <td>
    <p class="b-fight-details__table-text">L</p>
    <p class="b-fight-details__table-text">W</p>
  </td>
  <td>
    <p class="b-fight-details__table-text"><a>Alpha</a></p>
    <p class="b-fight-details__table-text"><a>Beta</a></p>
  </td>
  <td><p class="b-fight-details__table-text">KO/TKO</p></td>
  <td><p class="b-fight-details__table-text">2</p></td>
  <td><p class="b-fight-details__table-text">1:23</p></td>
  <td></td>
  <td></td>
  <td></td>
</tr>
"""


class UFCStatsScraperTests(unittest.TestCase):
    def test_parse_fight_row_handles_second_fighter_win_marker(self) -> None:
        row = BeautifulSoup(FIGHT_ROW_HTML, "html.parser").select_one("tr")
        scraper = UFCStatsScraper(delay=0.0)

        fight = scraper._parse_fight_row(row, "Test Event")

        self.assertIsNotNone(fight)
        self.assertEqual(fight["fighter_a"], "Alpha")
        self.assertEqual(fight["fighter_b"], "Beta")
        self.assertEqual(fight["winner"], "Beta")
        self.assertEqual(fight["method"], "KO/TKO")


if __name__ == "__main__":
    unittest.main()
