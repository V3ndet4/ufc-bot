import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.fight_week_watch import (
    classify_fight_week_entry,
    merge_alerts_into_context,
    parse_google_news_rss,
)


class FightWeekWatchTests(unittest.TestCase):
    def test_parse_google_news_rss_extracts_core_fields(self) -> None:
        xml_text = """
        <rss>
          <channel>
            <item>
              <title>Alpha Beta joins new camp - Sherdog</title>
              <link>https://news.google.com/articles/test</link>
              <pubDate>Sat, 18 Apr 2026 15:00:00 GMT</pubDate>
              <description><![CDATA[<div>Alpha Beta is now training with a new coach.</div>]]></description>
              <source url="https://www.sherdog.com">Sherdog</source>
            </item>
          </channel>
        </rss>
        """

        entries = parse_google_news_rss(xml_text)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source_name"], "Sherdog")
        self.assertEqual(entries[0]["source_url"], "https://www.sherdog.com")
        self.assertIn("Alpha Beta is now training", entries[0]["description"])

    def test_classify_fight_week_entry_detects_camp_change(self) -> None:
        classified = classify_fight_week_entry(
            fighter_name="Alpha Beta",
            gym_name="Kill Cliff FC",
            title="Alpha Beta joins new camp before UFC return",
            summary="Alpha Beta is now training at Kill Cliff FC with a new coach for this camp.",
            published_at=pd.Timestamp("2026-04-18T15:00:00Z"),
            source_name="Sherdog",
            source_url="https://www.sherdog.com",
            article_url="https://www.sherdog.com/news/news/test-1",
        )

        self.assertIsNotNone(classified)
        assert classified is not None
        self.assertEqual(int(classified["camp_change_flag"]), 1)
        self.assertEqual(int(classified["new_gym_flag"]), 1)
        self.assertEqual(classified["confidence_label"], "high")

    def test_merge_alerts_into_context_sets_flags_and_preserves_notes(self) -> None:
        context = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha Beta",
                    "short_notice_flag": 0,
                    "short_notice_acceptance_flag": 0,
                    "short_notice_success_flag": 0,
                    "new_gym_flag": 0,
                    "new_contract_flag": 0,
                    "cardio_fade_flag": 0,
                    "injury_concern_flag": 0,
                    "weight_cut_concern_flag": 0,
                    "replacement_fighter_flag": 0,
                    "travel_disadvantage_flag": 0,
                    "camp_change_flag": 0,
                    "context_notes": "manual note",
                }
            ]
        )
        alerts = pd.DataFrame(
            [
                {
                    "fighter_name": "Alpha Beta",
                    "alert_summary": "camp watch: Alpha Beta joins new camp (Sherdog 2026-04-18)",
                    "published_at": "2026-04-18T15:00:00+00:00",
                    "confidence_score": 0.91,
                    "short_notice_flag": 0,
                    "new_gym_flag": 1,
                    "injury_concern_flag": 0,
                    "weight_cut_concern_flag": 0,
                    "replacement_fighter_flag": 0,
                    "camp_change_flag": 1,
                }
            ]
        )

        merged = merge_alerts_into_context(context, alerts)

        self.assertEqual(int(merged.loc[0, "new_gym_flag"]), 1)
        self.assertEqual(int(merged.loc[0, "camp_change_flag"]), 1)
        self.assertIn("manual note", merged.loc[0, "context_notes"])
        self.assertIn("camp watch:", merged.loc[0, "context_notes"])


if __name__ == "__main__":
    unittest.main()
