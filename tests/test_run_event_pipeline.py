import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_event_pipeline


class RunEventPipelineTests(unittest.TestCase):
    def test_auto_stats_source_uses_ufcstats_for_blank_fighter_map(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            paths["fighter_map"].parent.mkdir(parents=True, exist_ok=True)
            paths["fighter_map"].write_text("fighter_name,espn_url\nAlpha,\nBeta,\n", encoding="utf-8")

            resolved = run_event_pipeline._resolved_stats_source("auto", paths)
        finally:
            manifest_path.unlink(missing_ok=True)
            paths["fighter_map"].unlink(missing_ok=True)

        self.assertEqual(resolved, "ufcstats")

    def test_auto_stats_source_uses_ufcstats_for_partial_fighter_map(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            paths["fighter_map"].parent.mkdir(parents=True, exist_ok=True)
            paths["fighter_map"].write_text(
                "fighter_name,espn_url\nAlpha,https://www.espn.com/mma/fighter/_/id/1/alpha\nBeta,\n",
                encoding="utf-8",
            )

            resolved = run_event_pipeline._resolved_stats_source("auto", paths)
        finally:
            manifest_path.unlink(missing_ok=True)
            paths["fighter_map"].unlink(missing_ok=True)

        self.assertEqual(resolved, "ufcstats")

    def test_auto_stats_source_uses_espn_for_complete_fighter_map(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            paths["fighter_map"].parent.mkdir(parents=True, exist_ok=True)
            paths["fighter_map"].write_text(
                "fighter_name,espn_url\n"
                "Alpha,https://www.espn.com/mma/fighter/_/id/1/alpha\n"
                "Beta,https://www.espn.com/mma/fighter/_/id/2/beta\n",
                encoding="utf-8",
            )

            resolved = run_event_pipeline._resolved_stats_source("auto", paths)
        finally:
            manifest_path.unlink(missing_ok=True)
            paths["fighter_map"].unlink(missing_ok=True)

        self.assertEqual(resolved, "espn")

    def test_fight_week_watch_command_uses_context_and_alert_paths(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            paths["fighter_gyms"].parent.mkdir(parents=True, exist_ok=True)
            paths["fighter_gyms"].write_text("fighter_name,gym_name\nAlpha,Kill Cliff FC\n", encoding="utf-8")
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)):
                run_event_pipeline._run_fight_week_watch(
                    paths,
                    quiet_children=True,
                    lookback_days=7,
                    max_results_per_fighter=3,
                )
        finally:
            manifest_path.unlink(missing_ok=True)
            paths["fighter_gyms"].unlink(missing_ok=True)

        self.assertEqual(len(commands), 1)
        self.assertIn("scripts/update_fight_week_watch.py", commands[0])
        self.assertIn("--context", commands[0])
        self.assertIn(str(paths["context"]), commands[0])
        self.assertIn("--alerts-output", commands[0])
        self.assertIn(str(paths["fight_week_alerts"]), commands[0])
        self.assertIn("--fighter-gyms", commands[0])
        self.assertIn(str(paths["fighter_gyms"]), commands[0])
        self.assertIn("--quiet", commands[0])

    def test_stats_refresh_passes_fighter_gyms_snapshot(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)):
                run_event_pipeline._run_stats_refresh(paths, "espn", quiet_children=True)
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 1)
        self.assertIn("--fighter-gyms", commands[0])
        self.assertIn(str(paths["fighter_gyms"]), commands[0])

    def test_external_history_enrichment_updates_fighter_stats_in_place(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)):
                run_event_pipeline._run_external_history_enrichment(paths, quiet_children=True)
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 1)
        self.assertIn("scripts/enrich_fighter_stats_with_external_ufc_history.py", commands[0])
        self.assertEqual(commands[0][commands[0].index("--input") + 1], str(paths["fighter_stats"]))
        self.assertEqual(commands[0][commands[0].index("--output") + 1], str(paths["fighter_stats"]))
        self.assertIn("--quiet", commands[0])

    def test_stats_refresh_falls_back_to_espn_after_ufcstats_failure(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []

        def fake_run(command: list[str]) -> None:
            commands.append(command)
            if "scripts/fetch_ufc_stats.py" in command:
                raise run_event_pipeline.subprocess.CalledProcessError(1, command)

        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            with patch.object(run_event_pipeline, "_run", side_effect=fake_run), patch.object(
                run_event_pipeline, "_run_espn_map_refresh", return_value=True
            ):
                refreshed = run_event_pipeline._run_stats_refresh(
                    paths,
                    "ufcstats",
                    quiet_children=True,
                    allow_espn_fallback=True,
                )
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertTrue(refreshed)
        self.assertEqual(len(commands), 2)
        self.assertIn("scripts/fetch_ufc_stats.py", commands[0])
        self.assertIn("scripts/fetch_espn_stats.py", commands[1])

    def test_odds_refresh_reuses_existing_odds_after_failure(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            paths["oddsapi_odds"].parent.mkdir(parents=True, exist_ok=True)
            paths["oddsapi_odds"].write_text(
                "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds\n"
                "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_a,fanduel,-110\n"
                "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_b,fanduel,100\n",
                encoding="utf-8",
            )
            with patch.object(
                run_event_pipeline,
                "_run",
                side_effect=run_event_pipeline.subprocess.CalledProcessError(1, ["fetch_the_odds_api_odds.py"]),
            ):
                refreshed = run_event_pipeline._run_odds_refresh(
                    paths,
                    quiet_children=True,
                    odds_source="oddsapi",
                    no_history=False,
                    odds_api_bookmaker="fanduel",
                    bfo_refresh_url="",
                )
        finally:
            manifest_path.unlink(missing_ok=True)
            paths["oddsapi_odds"].unlink(missing_ok=True)

        self.assertTrue(refreshed)

    def test_odds_refresh_returns_false_without_cached_odds_after_failure(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        try:
            manifest = run_event_pipeline.load_manifest(manifest_path)
            paths = run_event_pipeline.derived_paths(manifest)
            with patch.object(
                run_event_pipeline,
                "_run",
                side_effect=run_event_pipeline.subprocess.CalledProcessError(1, ["fetch_the_odds_api_odds.py"]),
            ):
                refreshed = run_event_pipeline._run_odds_refresh(
                    paths,
                    quiet_children=True,
                    odds_source="oddsapi",
                    no_history=False,
                    odds_api_bookmaker="fanduel",
                    bfo_refresh_url="",
                )
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertFalse(refreshed)

    def test_pipeline_runs_fight_week_watch_before_stats_refresh(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)), patch.object(
                run_event_pipeline, "_run_gym_refresh", return_value=True
            ), patch.object(
                run_event_pipeline, "_run_stats_refresh", return_value=True
            ), patch.object(
                run_event_pipeline, "_run_external_history_enrichment", return_value=True
            ), patch.object(
                run_event_pipeline, "_run_espn_map_refresh", return_value=False
            ), patch.object(
                run_event_pipeline, "_has_live_odds", return_value=False
            ), patch.object(
                sys,
                "argv",
                [
                    "run_event_pipeline.py",
                    "--manifest",
                    str(manifest_path),
                    "--skip-prepare",
                    "--skip-odds",
                    "--quiet-children",
                ],
            ):
                run_event_pipeline.main()
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 1)
        self.assertIn("scripts/update_fight_week_watch.py", commands[0])

    def test_pipeline_runs_external_history_enrichment_after_stats_refresh(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        order: list[str] = []
        try:
            with patch.object(run_event_pipeline, "_run_gym_refresh", return_value=True), patch.object(
                run_event_pipeline, "_run_fight_week_watch", return_value=True
            ), patch.object(
                run_event_pipeline, "_run_stats_refresh", side_effect=lambda *args, **kwargs: order.append("stats") or True
            ), patch.object(
                run_event_pipeline,
                "_run_external_history_enrichment",
                side_effect=lambda *args, **kwargs: order.append("external") or True,
            ), patch.object(
                run_event_pipeline, "_run_espn_map_refresh", return_value=False
            ), patch.object(
                run_event_pipeline, "_has_live_odds", return_value=False
            ), patch.object(
                sys,
                "argv",
                [
                    "run_event_pipeline.py",
                    "--manifest",
                    str(manifest_path),
                    "--skip-prepare",
                    "--skip-odds",
                    "--quiet-children",
                ],
            ):
                run_event_pipeline.main()
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(order, ["stats", "external"])

    def test_oddsapi_mode_uses_quiet_child_command(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)), patch.object(
                run_event_pipeline, "_has_live_odds", return_value=False
            ), patch.object(
                sys,
                "argv",
                [
                    "run_event_pipeline.py",
                    "--manifest",
                    str(manifest_path),
                    "--skip-prepare",
                    "--skip-stats",
                    "--odds-source",
                    "oddsapi",
                    "--quiet-children",
                ],
            ):
                run_event_pipeline.main()
        finally:
            manifest_path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 1)
        self.assertIn("scripts/fetch_the_odds_api_odds.py", commands[0])
        self.assertIn("--quiet", commands[0])

    def test_oddsapi_mode_passes_bookmaker_to_line_chart(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        paths = run_event_pipeline.derived_paths(run_event_pipeline.load_manifest(manifest_path))
        paths["fighter_stats"].parent.mkdir(parents=True, exist_ok=True)
        paths["fighter_stats"].write_text("fighter_name,wins,losses,height_in,reach_in,sig_strikes_landed_per_min,sig_strikes_absorbed_per_min,takedown_avg,takedown_defense_pct\n", encoding="utf-8")
        paths["oddsapi_odds"].parent.mkdir(parents=True, exist_ok=True)
        paths["oddsapi_odds"].write_text(
            "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds\n"
            "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_a,fanduel,-110\n"
            "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_b,fanduel,100\n",
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)), patch.object(
                run_event_pipeline, "_run_stats_refresh", return_value=True
            ), patch.object(
                sys,
                "argv",
                [
                    "run_event_pipeline.py",
                    "--manifest",
                    str(manifest_path),
                    "--skip-prepare",
                    "--skip-stats",
                    "--skip-odds",
                    "--odds-source",
                    "oddsapi",
                    "--odds-api-bookmaker",
                    "fanduel",
                    "--quiet-children",
                ],
            ):
                run_event_pipeline.main()
        finally:
            manifest_path.unlink(missing_ok=True)
            for path in [paths["fighter_stats"], paths["oddsapi_odds"]]:
                path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 6)
        self.assertIn("scripts/build_line_movement_report.py", commands[0])
        self.assertIn("--bookmaker", commands[0])
        self.assertIn("fanduel", commands[0])
        self.assertIn("scripts/build_parlay_board.py", commands[4])
        self.assertIn("scripts/build_operator_dashboard.py", commands[5])

    def test_verified_alt_market_urls_flow_to_report_command(self) -> None:
        manifest_path = ROOT / "tests" / "_tmp_pipeline_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "slug": "tmp_pipeline",
                    "event_id": "tmp-pipeline-event",
                    "event_name": "Tmp Pipeline Event",
                    "start_time": "2026-04-11T21:00:00-04:00",
                    "bestfightodds_refresh_url": "https://www.bestfightodds.com/?desktop=on",
                    "bestfightodds_event_urls": ["https://www.bestfightodds.com/events/tmp-pipeline-event-12345"],
                    "fights": [{"fighter_a": "Alpha", "fighter_b": "Beta"}],
                }
            ),
            encoding="utf-8",
        )
        paths = run_event_pipeline.derived_paths(run_event_pipeline.load_manifest(manifest_path))
        paths["fighter_stats"].parent.mkdir(parents=True, exist_ok=True)
        paths["fighter_stats"].write_text("fighter_name,wins,losses,height_in,reach_in,sig_strikes_landed_per_min,sig_strikes_absorbed_per_min,takedown_avg,takedown_defense_pct\n", encoding="utf-8")
        paths["bfo_odds"].parent.mkdir(parents=True, exist_ok=True)
        paths["bfo_odds"].write_text(
            "event_id,event_name,start_time,fighter_a,fighter_b,market,selection,book,american_odds\n"
            "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_a,Book,-110\n"
            "tmp-pipeline-event,Tmp Pipeline Event,2026-04-11T21:00:00-04:00,Alpha,Beta,moneyline,fighter_b,Book,100\n",
            encoding="utf-8",
        )
        commands: list[list[str]] = []
        try:
            with patch.object(run_event_pipeline, "_run", side_effect=lambda command: commands.append(command)), patch.object(
                run_event_pipeline, "_run_stats_refresh", return_value=True
            ), patch.object(
                sys,
                "argv",
                [
                    "run_event_pipeline.py",
                    "--manifest",
                    str(manifest_path),
                    "--skip-prepare",
                    "--skip-stats",
                    "--skip-odds",
                    "--quiet-children",
                ],
            ):
                run_event_pipeline.main()
        finally:
            manifest_path.unlink(missing_ok=True)
            for path in [paths["fighter_stats"], paths["bfo_odds"]]:
                path.unlink(missing_ok=True)

        self.assertEqual(len(commands), 6)
        self.assertIn("scripts/build_line_movement_report.py", commands[0])
        self.assertIn("scripts/build_fight_week_report.py", commands[1])
        self.assertIn("--odds-api-bookmaker", commands[1])
        self.assertIn("--bestfightodds-event-url", commands[1])
        self.assertIn("https://www.bestfightodds.com/events/tmp-pipeline-event-12345", commands[1])
        self.assertIn("--quiet", commands[1])
        self.assertIn("scripts/build_lean_board.py", commands[2])
        self.assertIn("scripts/run_value_scan.py", commands[3])
        self.assertIn("scripts/build_parlay_board.py", commands[4])
        self.assertIn("scripts/build_operator_dashboard.py", commands[5])


if __name__ == "__main__":
    unittest.main()
