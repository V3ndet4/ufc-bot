import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.ufc_stats import parse_fighter_details, parse_fighter_directory, scrape_fighter_stats_for_names


DIRECTORY_HTML = """
<table>
  <tr class="b-statistics__table-row">
    <td><a href="/fighter-details/abc123">Jane</a></td>
    <td>Stone</td>
  </tr>
  <tr class="b-statistics__table-row">
    <td><a href="https://ufcstats.com/fighter-details/def456">Mark</a></td>
    <td>Cole</td>
  </tr>
</table>
"""


DETAIL_HTML = """
<html>
  <body>
    <span class="b-content__title-highlight">Jane Stone</span>
    <span class="b-content__title-record">Record: 12-3-0</span>
    <ul>
      <li class="b-list__box-list-item">Height: 5' 7"</li>
      <li class="b-list__box-list-item">Reach: 69"</li>
      <li class="b-list__box-list-item">SLpM: 5.20</li>
      <li class="b-list__box-list-item">SApM: 3.10</li>
      <li class="b-list__box-list-item">Str. Acc.: 48%</li>
      <li class="b-list__box-list-item">Str. Def: 61%</li>
      <li class="b-list__box-list-item">TD Avg.: 1.40</li>
      <li class="b-list__box-list-item">TD Acc.: 44%</li>
      <li class="b-list__box-list-item">TD Def.: 72%</li>
      <li class="b-list__box-list-item">Sub. Avg.: 0.30</li>
      <li class="b-list__box-list-item">STANCE: Orthodox</li>
      <li class="b-list__box-list-item">DOB: Jan 1, 1995</li>
    </ul>
  </body>
</html>
"""


class UfcStatsParsingTests(unittest.TestCase):
    def test_parse_fighter_directory_returns_names_and_urls(self) -> None:
        fighters = parse_fighter_directory(DIRECTORY_HTML)
        self.assertEqual(len(fighters), 2)
        self.assertEqual(fighters[0].fighter_name, "Jane Stone")
        self.assertEqual(fighters[0].fighter_url, "https://ufcstats.com/fighter-details/abc123")

    def test_parse_fighter_details_normalizes_expected_columns(self) -> None:
        fighter = parse_fighter_details(DETAIL_HTML, "https://ufcstats.com/fighter-details/abc123")
        self.assertEqual(fighter["fighter_name"], "Jane Stone")
        self.assertEqual(fighter["wins"], 12)
        self.assertEqual(fighter["losses"], 3)
        self.assertEqual(fighter["height_in"], 67)
        self.assertEqual(fighter["reach_in"], 69.0)
        self.assertEqual(fighter["sig_strikes_landed_per_min"], 5.2)
        self.assertEqual(fighter["takedown_defense_pct"], 72.0)
        self.assertEqual(fighter["source_url"], "https://ufcstats.com/fighter-details/abc123")

    def test_scrape_fighter_stats_for_names_fetches_requested_subset(self) -> None:
        import data_sources.ufc_stats as ufc_stats

        original_fetch = ufc_stats.fetch_html
        ufc_stats.fetch_html = lambda url, session=None: DIRECTORY_HTML if "fighters?char=s" in url else DETAIL_HTML
        try:
            frame = scrape_fighter_stats_for_names(["Jane Stone"])
        finally:
            ufc_stats.fetch_html = original_fetch

        self.assertEqual(frame["fighter_name"].tolist(), ["Jane Stone"])


if __name__ == "__main__":
    unittest.main()
