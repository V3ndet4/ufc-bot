import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.event_manifest import (
    bestfightodds_event_urls,
    bestfightodds_refresh_url,
    build_context_frame,
    build_fighter_map_frame,
    build_modeled_market_template_frame,
    is_verified_bestfightodds_event_url,
    load_manifest,
    manifest_status_rows,
    merge_existing_context,
    merge_existing_fighter_map,
)


class EventManifestTests(unittest.TestCase):
    def test_bestfightodds_helpers_distinguish_refresh_from_verified_event_urls(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_card",
                    "event_id": "tmp-event",
                    "event_name": "Tmp Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "bestfightodds_refresh_url": "https://www.bestfightodds.com/?desktop=on",
                    "bestfightodds_event_urls": [
                        "https://www.bestfightodds.com/events/tmp-event-12345",
                    ],
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = load_manifest(manifest_path)
            rows = dict(manifest_status_rows(manifest))
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(bestfightodds_refresh_url(manifest), "https://www.bestfightodds.com/?desktop=on")
        self.assertEqual(bestfightodds_event_urls(manifest), ["https://www.bestfightodds.com/events/tmp-event-12345"])
        self.assertTrue(is_verified_bestfightodds_event_url("https://www.bestfightodds.com/events/tmp-event-12345"))
        self.assertFalse(is_verified_bestfightodds_event_url("https://www.bestfightodds.com/?desktop=on"))
        self.assertEqual(rows["bfo_alt_market_status"], "verified")

    def test_merge_existing_context_preserves_manual_flags(self) -> None:
        manifest = {
            "slug": "tmp-card",
            "event_id": "tmp-event",
            "event_name": "Tmp Event",
            "start_time": "2026-04-11T21:00:00-04:00",
            "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
        }
        template = build_context_frame(manifest)
        existing = template.copy()
        existing.loc[existing["fighter_name"] == "Alpha", "new_gym_flag"] = 1
        existing.loc[existing["fighter_name"] == "Alpha", "context_notes"] = "manual watch"

        merged = merge_existing_context(template, existing)

        alpha_row = merged.loc[merged["fighter_name"] == "Alpha"].iloc[0]
        self.assertEqual(int(alpha_row["new_gym_flag"]), 1)
        self.assertEqual(alpha_row["context_notes"], "manual watch")

    def test_merge_existing_fighter_map_preserves_existing_urls(self) -> None:
        manifest = {
            "slug": "tmp-card",
            "event_id": "tmp-event",
            "event_name": "Tmp Event",
            "start_time": "2026-04-11T21:00:00-04:00",
            "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
        }
        template = build_fighter_map_frame(manifest)
        existing = template.copy()
        existing.loc[existing["fighter_name"] == "Alpha", "espn_url"] = "https://www.espn.com/mma/fighter/_/id/1/alpha"

        merged = merge_existing_fighter_map(template, existing)

        alpha_row = merged.loc[merged["fighter_name"] == "Alpha"].iloc[0]
        beta_row = merged.loc[merged["fighter_name"] == "Beta"].iloc[0]
        self.assertEqual(alpha_row["espn_url"], "https://www.espn.com/mma/fighter/_/id/1/alpha")
        self.assertEqual(beta_row["espn_url"], "")

    def test_build_modeled_market_template_frame_emits_supported_markets(self) -> None:
        manifest = {
            "slug": "tmp-card",
            "event_id": "tmp-event",
            "event_name": "Tmp Event",
            "start_time": "2026-04-11T21:00:00-04:00",
            "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
        }

        template = build_modeled_market_template_frame(manifest)

        self.assertEqual(len(template), 8)
        self.assertEqual(
            set(template["market"].astype(str)),
            {
                "moneyline",
                "fight_goes_to_decision",
                "fight_doesnt_go_to_decision",
                "inside_distance",
                "by_decision",
            },
        )
        self.assertEqual(
            set(
                template.loc[template["market"].astype(str) == "inside_distance", "selection"].astype(str)
            ),
            {"fighter_a", "fighter_b"},
        )


if __name__ == "__main__":
    unittest.main()
