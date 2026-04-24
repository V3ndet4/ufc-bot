from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_fight_week_report import _colorize
from scripts.build_lean_board import format_best_leans_summary, format_full_card_breakdown
from scripts.build_parlay_board import format_compact_parlay_summary
from scripts.event_manifest import (
    bestfightodds_event_urls,
    bestfightodds_refresh_url,
    derived_paths,
    load_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UFC event workflow from an event manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing derived files.")
    parser.add_argument("--skip-gyms", action="store_true", help="Reuse existing Sherdog gym data.")
    parser.add_argument("--skip-watch", action="store_true", help="Reuse existing fight-week alert context.")
    parser.add_argument("--skip-stats", action="store_true", help="Reuse existing fighter stats.")
    parser.add_argument(
        "--skip-external-history",
        action="store_true",
        help="Skip the weekly external UFC history enrichment pass after the base fighter stats refresh.",
    )
    parser.add_argument("--skip-odds", action="store_true", help="Reuse existing odds.")
    parser.add_argument("--no-history", action="store_true", help="Skip BestFightOdds history enrichment.")
    parser.add_argument(
        "--odds-source",
        choices=["bfo", "oddsapi"],
        default="bfo",
        help="Odds source used for the refresh and downstream report/scan.",
    )
    parser.add_argument(
        "--odds-api-bookmaker",
        default="fanduel",
        help="Bookmaker key used when --odds-source=oddsapi.",
    )
    parser.add_argument(
        "--stats-source",
        choices=["auto", "espn", "ufcstats"],
        default="auto",
        help="Fighter stats source. 'auto' prefers an ESPN mapping CSV when present, otherwise UFC Stats.",
    )
    parser.add_argument(
        "--gym-refresh-days",
        type=int,
        default=7,
        help="Days before refreshing a Sherdog fighter profile again.",
    )
    parser.add_argument(
        "--gym-association-refresh-days",
        type=int,
        default=30,
        help="Days before refreshing cached gym roster fighters again.",
    )
    parser.add_argument(
        "--gym-max-association-pages",
        type=int,
        default=5,
        help="Max Sherdog Fight Finder pages to scan per gym.",
    )
    parser.add_argument(
        "--watch-lookback-days",
        type=int,
        default=10,
        help="Days of recent coverage to scan for fight-week alerts.",
    )
    parser.add_argument(
        "--watch-max-results-per-fighter",
        type=int,
        default=5,
        help="Max fight-week alerts to retain per fighter.",
    )
    parser.add_argument("--quiet-children", action="store_true", help="Suppress child script console output.")
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _has_live_odds(path: Path) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    if frame.empty or "american_odds" not in frame.columns:
        return False
    numeric = pd.to_numeric(frame["american_odds"], errors="coerce")
    return bool((numeric.notna() & (numeric != 0)).any())


def _fighter_map_counts(paths: dict[str, Path]) -> tuple[int, int]:
    if not paths["fighter_map"].exists():
        return 0, 0
    try:
        fighter_map = pd.read_csv(paths["fighter_map"])
    except Exception:
        return 0, 0
    if fighter_map.empty or "fighter_name" not in fighter_map.columns:
        return 0, 0
    if "espn_url" not in fighter_map.columns:
        fighter_map["espn_url"] = ""
    total = int(fighter_map["fighter_name"].fillna("").astype(str).str.strip().ne("").sum())
    populated = int(fighter_map["espn_url"].fillna("").astype(str).str.strip().ne("").sum())
    return total, populated


def _fighter_map_is_complete(paths: dict[str, Path]) -> bool:
    total, populated = _fighter_map_counts(paths)
    return total > 0 and populated == total


def _resolved_stats_source(stats_source: str, paths: dict[str, Path]) -> str:
    if stats_source == "auto":
        if _fighter_map_is_complete(paths):
            return "espn"
        return "ufcstats"
    return stats_source


def _run_gym_refresh(
    paths: dict[str, Path],
    quiet_children: bool,
    *,
    refresh_days: int,
    association_refresh_days: int,
    max_association_pages: int,
) -> bool:
    command = [
        sys.executable,
        "scripts/update_sherdog_gym_data.py",
        "--fighters-csv",
        str(paths["fighter_list"]),
        "--output",
        str(paths["fighter_gyms"]),
        "--event-registry-output",
        str(paths["gym_registry"]),
        "--global-cache",
        str(ROOT / "data" / "sherdog_fighter_gyms.csv"),
        "--global-registry",
        str(ROOT / "data" / "sherdog_gym_registry.csv"),
        "--refresh-days",
        str(refresh_days),
        "--association-refresh-days",
        str(association_refresh_days),
        "--max-association-pages",
        str(max_association_pages),
    ]
    if quiet_children:
        command.append("--quiet")

    try:
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if paths["fighter_gyms"].exists():
            print(
                "Sherdog gym refresh failed; reusing existing gym snapshot: "
                f"{paths['fighter_gyms']}"
            )
            print(f"Gym refresh error: {exc}")
            return True
        print("Sherdog gym refresh failed; continuing without gym enrichment for this run.")
        print(f"Gym refresh error: {exc}")
        return False


def _run_fight_week_watch(
    paths: dict[str, Path],
    quiet_children: bool,
    *,
    lookback_days: int,
    max_results_per_fighter: int,
) -> bool:
    command = [
        sys.executable,
        "scripts/update_fight_week_watch.py",
        "--fighters-csv",
        str(paths["fighter_list"]),
        "--context",
        str(paths["context"]),
        "--alerts-output",
        str(paths["fight_week_alerts"]),
        "--lookback-days",
        str(lookback_days),
        "--max-results-per-fighter",
        str(max_results_per_fighter),
    ]
    if paths["fighter_gyms"].exists():
        command.extend(["--fighter-gyms", str(paths["fighter_gyms"])])
    if quiet_children:
        command.append("--quiet")

    try:
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if paths["context"].exists():
            print(
                "Fight-week watch refresh failed; reusing existing context file: "
                f"{paths['context']}"
            )
            print(f"Fight-week watch error: {exc}")
            return True
        print("Fight-week watch refresh failed; continuing without fresh alert context.")
        print(f"Fight-week watch error: {exc}")
        return False


def _run_espn_map_refresh(paths: dict[str, Path], quiet_children: bool) -> bool:
    command = [
        sys.executable,
        "scripts/update_espn_fighter_map.py",
        "--mapping",
        str(paths["fighter_map"]),
        "--fighters-csv",
        str(paths["fighter_list"]),
        "--global-cache",
        str(ROOT / "data" / "espn_fighter_map.csv"),
    ]
    if quiet_children:
        command.append("--quiet")

    try:
        _run(command)
    except subprocess.CalledProcessError as exc:
        if paths["fighter_map"].exists():
            print("ESPN fighter map refresh failed; reusing existing fighter map.")
            print(f"ESPN fighter map error: {exc}")
        else:
            print("ESPN fighter map refresh failed; continuing without ESPN auto-mapping.")
            print(f"ESPN fighter map error: {exc}")
        return _fighter_map_is_complete(paths)
    return _fighter_map_is_complete(paths)


def _run_stats_refresh(
    paths: dict[str, Path],
    stats_source: str,
    quiet_children: bool,
    *,
    allow_espn_fallback: bool = False,
) -> bool:
    try:
        if stats_source == "espn":
            command = [
                sys.executable,
                "scripts/fetch_espn_stats.py",
                "--mapping",
                str(paths["fighter_map"]),
                "--context",
                str(paths["context"]),
                "--output",
                str(paths["fighter_stats"]),
                "--fighter-gyms",
                str(paths["fighter_gyms"]),
            ]
        else:
            command = [
                sys.executable,
                "scripts/fetch_ufc_stats.py",
                "--fighters-csv",
                str(paths["fighter_list"]),
                "--context",
                str(paths["context"]),
                "--output",
                str(paths["fighter_stats"]),
                "--fighter-gyms",
                str(paths["fighter_gyms"]),
            ]
        if quiet_children:
            command.append("--quiet")
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if allow_espn_fallback and stats_source != "espn":
            espn_map_ready = _run_espn_map_refresh(paths, quiet_children)
            if espn_map_ready:
                print("UFC Stats refresh failed; retrying with ESPN fighter pages.")
                return _run_stats_refresh(
                    paths,
                    "espn",
                    quiet_children,
                    allow_espn_fallback=False,
                )
        if paths["fighter_stats"].exists():
            print(
                "Fighter stats refresh failed; reusing existing stats file: "
                f"{paths['fighter_stats']}"
            )
            print(f"Stats refresh error: {exc}")
            return True
        print("Fighter stats refresh failed and no cached stats file exists yet.")
        print("Continuing with prep so odds/template files can still be refreshed.")
        print(f"Stats refresh error: {exc}")
        return False


def _run_external_history_enrichment(
    paths: dict[str, Path],
    quiet_children: bool,
) -> bool:
    command = [
        sys.executable,
        "scripts/enrich_fighter_stats_with_external_ufc_history.py",
        "--input",
        str(paths["fighter_stats"]),
        "--output",
        str(paths["fighter_stats"]),
    ]
    if quiet_children:
        command.append("--quiet")

    try:
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if paths["fighter_stats"].exists():
            print(
                "External UFC history enrichment failed; reusing base fighter stats file: "
                f"{paths['fighter_stats']}"
            )
            print(f"External history error: {exc}")
            return True
        print("External UFC history enrichment failed and no fighter stats file is available.")
        print(f"External history error: {exc}")
        return False


def _run_odds_refresh(
    paths: dict[str, Path],
    quiet_children: bool,
    *,
    odds_source: str,
    no_history: bool,
    odds_api_bookmaker: str,
    bfo_refresh_url: str,
) -> bool:
    odds_path = paths["bfo_odds"] if odds_source == "bfo" else paths["oddsapi_odds"]
    source_label = "BestFightOdds" if odds_source == "bfo" else "The Odds API"

    try:
        if odds_source == "bfo":
            if not bfo_refresh_url:
                return odds_path.exists()
            command = [
                sys.executable,
                "scripts/fetch_bestfightodds_event_odds.py",
                "--template",
                str(paths["odds_template"]),
                "--event-url",
                bfo_refresh_url,
                "--output",
                str(paths["bfo_odds"]),
            ]
            if not no_history:
                command.append("--include-history")
        else:
            command = [
                sys.executable,
                "scripts/fetch_the_odds_api_odds.py",
                "--template",
                str(paths["odds_template"]),
                "--bookmaker",
                odds_api_bookmaker,
                "--output",
                str(paths["oddsapi_odds"]),
            ]
        if quiet_children:
            command.append("--quiet")
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if odds_path.exists():
            print(f"{source_label} refresh failed; reusing existing odds file: {odds_path}")
            print(f"Odds refresh error: {exc}")
            return True
        print(f"{source_label} refresh failed and no cached odds file exists yet.")
        print("Continuing without fresh odds for this run.")
        print(f"Odds refresh error: {exc}")
        return False


def _run_modeled_market_odds_refresh(
    paths: dict[str, Path],
    quiet_children: bool,
    *,
    odds_api_bookmaker: str,
) -> bool:
    template_path = paths["modeled_market_template"]
    output_path = paths["modeled_market_odds"]
    if not template_path.exists():
        return output_path.exists()

    command = [
        sys.executable,
        "scripts/fetch_the_odds_api_odds.py",
        "--template",
        str(template_path),
        "--bookmaker",
        odds_api_bookmaker,
        "--output",
        str(output_path),
        "--no-snapshot",
    ]
    if quiet_children:
        command.append("--quiet")

    try:
        _run(command)
        return True
    except subprocess.CalledProcessError as exc:
        if output_path.exists():
            print(f"Modeled-market Odds API refresh failed; reusing existing odds file: {output_path}")
            print(f"Modeled-market refresh error: {exc}")
            return True
        print("Modeled-market Odds API refresh failed; continuing without event-level prop odds artifact.")
        print(f"Modeled-market refresh error: {exc}")
        return False


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)
    odds_path = paths["bfo_odds"] if args.odds_source == "bfo" else paths["oddsapi_odds"]
    bfo_refresh_url = bestfightodds_refresh_url(manifest)
    bfo_alt_urls = bestfightodds_event_urls(manifest)
    full_card_summary_text = ""
    lean_summary_text = ""
    parlay_summary_text = ""

    if not args.skip_prepare:
        command = [sys.executable, "scripts/prepare_event.py", "--manifest", args.manifest]
        if args.quiet_children:
            command.append("--quiet")
        _run(command)

    stats_ready = paths["fighter_stats"].exists()
    if not args.skip_stats:
        if args.stats_source == "auto" and not _fighter_map_is_complete(paths):
            _run_espn_map_refresh(paths, args.quiet_children)
        stats_source = _resolved_stats_source(args.stats_source, paths)
        if not args.skip_gyms:
            _run_gym_refresh(
                paths,
                args.quiet_children,
                refresh_days=args.gym_refresh_days,
                association_refresh_days=args.gym_association_refresh_days,
                max_association_pages=args.gym_max_association_pages,
            )
        if not args.skip_watch:
            _run_fight_week_watch(
                paths,
                args.quiet_children,
                lookback_days=args.watch_lookback_days,
                max_results_per_fighter=args.watch_max_results_per_fighter,
            )
        print(f"Using fighter stats source: {stats_source}")
        stats_ready = _run_stats_refresh(
            paths,
            stats_source,
            args.quiet_children,
            allow_espn_fallback=args.stats_source == "auto",
        )
        if stats_ready and not args.skip_external_history:
            _run_external_history_enrichment(paths, args.quiet_children)
    else:
        stats_source = _resolved_stats_source(args.stats_source, paths)

    if not args.skip_odds:
        _run_odds_refresh(
            paths,
            args.quiet_children,
            odds_source=args.odds_source,
            no_history=args.no_history,
            odds_api_bookmaker=args.odds_api_bookmaker,
            bfo_refresh_url=bfo_refresh_url,
        )
        if args.odds_source == "oddsapi":
            _run_modeled_market_odds_refresh(
                paths,
                args.quiet_children,
                odds_api_bookmaker=args.odds_api_bookmaker,
            )

    if not _has_live_odds(odds_path):
        print("No live odds detected yet; prep files and fighter stats are ready.")
        return

    line_chart_command = [
        sys.executable,
        "scripts/build_line_movement_report.py",
        "--odds",
        str(odds_path),
        "--db",
        str(ROOT / "data" / "ufc_betting.db"),
        "--output",
        str(paths["line_movement"]),
        "--per-fight-dir",
        str(paths["line_movement_fights_dir"]),
    ]
    if args.odds_source == "oddsapi":
        line_chart_command.extend(["--bookmaker", args.odds_api_bookmaker])
    _run(line_chart_command)

    if not stats_ready:
        print("Live odds were found, but fighter stats are not available yet.")
        print("Skipping report and value scan until the stats fetch succeeds.")
        print(f"Saved line movement board to {paths['line_movement']}")
        print(f"Saved per-fight line charts to {paths['line_movement_fights_dir']}")
        return

    report_command = [
        sys.executable,
        "scripts/build_fight_week_report.py",
        "--odds",
        str(odds_path),
        "--fighter-stats",
        str(paths["fighter_stats"]),
        "--skipped-output",
        str(paths["skipped"]),
        "--output",
        str(paths["report"]),
        "--odds-api-bookmaker",
        args.odds_api_bookmaker,
    ]
    db_path = ROOT / "data" / "ufc_betting.db"
    side_model_path = ROOT / "models" / "side_model.pkl"
    confidence_model_path = ROOT / "models" / "confidence_model.pkl"
    selective_model_path = ROOT / "models" / "selective_clv_model.pkl"
    threshold_policy_path = ROOT / "models" / "threshold_policy.json"
    report_command.extend(["--db", str(db_path), "--side-model", str(side_model_path)])
    for event_url in bfo_alt_urls:
        report_command.extend(["--bestfightodds-event-url", event_url])
    if args.quiet_children:
        report_command.append("--quiet")
    if db_path.exists():
        train_side_command = [
            sys.executable,
            "scripts/train_side_model.py",
            "--db",
            str(db_path),
            "--output",
            str(side_model_path),
        ]
        if args.quiet_children:
            train_side_command.append("--quiet")
        completed = subprocess.run(
            train_side_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "").strip()
            print(f"Side model refresh skipped: {reason or f'exit code {completed.returncode}'}")
        train_confidence_command = [
            sys.executable,
            "scripts/train_confidence_model.py",
            "--db",
            str(db_path),
            "--output",
            str(confidence_model_path),
        ]
        if args.quiet_children:
            train_confidence_command.append("--quiet")
        completed = subprocess.run(
            train_confidence_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "").strip()
            print(f"Confidence model refresh skipped: {reason or f'exit code {completed.returncode}'}")
        train_selective_command = [
            sys.executable,
            "scripts/train_selective_clv_model.py",
            "--db",
            str(db_path),
            "--output",
            str(selective_model_path),
        ]
        if args.quiet_children:
            train_selective_command.append("--quiet")
        completed = subprocess.run(
            train_selective_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "").strip()
            print(f"Selective CLV model refresh skipped: {reason or f'exit code {completed.returncode}'}")
        train_threshold_command = [
            sys.executable,
            "scripts/train_threshold_optimizer.py",
            "--db",
            str(db_path),
            "--output",
            str(threshold_policy_path),
        ]
        if args.quiet_children:
            train_threshold_command.append("--quiet")
        completed = subprocess.run(
            train_threshold_command,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            reason = (completed.stderr or completed.stdout or "").strip()
            print(f"Threshold policy refresh skipped: {reason or f'exit code {completed.returncode}'}")
    _run(report_command)

    lean_board_command = [
        sys.executable,
        "scripts/build_lean_board.py",
        "--input",
        str(paths["report"]),
        "--output",
        str(paths["lean_board"]),
    ]
    if args.quiet_children:
        lean_board_command.append("--quiet")
    _run(lean_board_command)
    if args.quiet_children and paths["lean_board"].exists():
        try:
            lean_board = pd.read_csv(paths["lean_board"])
            full_card_summary_text = format_full_card_breakdown(lean_board)
            lean_summary_text = format_best_leans_summary(lean_board)
        except Exception as exc:
            print(f"Lean board summary skipped: {exc}")

    scan_command = [
        sys.executable,
        "scripts/run_value_scan.py",
        "--input",
        str(odds_path),
        "--fighter-stats",
        str(paths["fighter_stats"]),
        "--fight-report",
        str(paths["report"]),
        "--shortlist-output",
        str(paths["shortlist"]),
        "--board-output",
        str(paths["board"]),
        "--passes-output",
        str(paths["passes"]),
        "--output",
        str(paths["value"]),
        "--db",
        str(db_path),
        "--selective-model",
        str(selective_model_path),
        "--side-model",
        str(side_model_path),
    ]
    if threshold_policy_path.exists():
        scan_command.extend(["--threshold-policy", str(threshold_policy_path)])
    if args.quiet_children:
        scan_command.append("--quiet")
    _run(scan_command)

    parlay_command = [
        sys.executable,
        "scripts/build_parlay_board.py",
        "--input",
        str(paths["value"]),
        "--output",
        str(paths["parlays"]),
    ]
    if args.quiet_children:
        parlay_command.append("--quiet")
    _run(parlay_command)
    if args.quiet_children and paths["parlays"].exists():
        try:
            parlays = pd.read_csv(paths["parlays"])
            parlay_summary_text = format_compact_parlay_summary(parlays)
        except Exception as exc:
            print(f"Parlay board summary skipped: {exc}")

    dashboard_command = [
        sys.executable,
        "scripts/build_operator_dashboard.py",
        "--fight-report",
        str(paths["report"]),
        "--lean-board",
        str(paths["lean_board"]),
        "--value-report",
        str(paths["value"]),
        "--betting-board",
        str(paths["board"]),
        "--passes",
        str(paths["passes"]),
        "--alerts",
        str(paths["fight_week_alerts"]),
        "--parlays",
        str(paths["parlays"]),
        "--output",
        str(paths["operator_dashboard"]),
    ]
    if threshold_policy_path.exists():
        dashboard_command.extend(["--threshold-policy", str(threshold_policy_path)])
    if args.quiet_children:
        dashboard_command.append("--quiet")
    _run(dashboard_command)

    if args.quiet_children:
        print()
        print("Generated reports")
        print("-----------------")
        print(f"Line movement: {paths['line_movement']}")
        print(f"Fight charts:  {paths['line_movement_fights_dir']}")
        print(f"Fight report:  {paths['report']}")
        print(f"Lean board:    {paths['lean_board']}")
        print(f"Fight alerts:  {paths['fight_week_alerts']}")
        print(f"Skipped:       {paths['skipped']}")
        print(f"Bet board:     {paths['board']}")
        print(f"Pass reasons:  {paths['passes']}")
        print(f"Value scan:    {paths['value']}")
        print(f"Dashboard:     {paths['operator_dashboard']}")
        print(f"Parlays:       {paths['parlays']}")
        if full_card_summary_text:
            print()
            print(_colorize("Full Card Read", "cyan"))
            print(_colorize("--------------", "gray"))
            print(full_card_summary_text, end="")
        if lean_summary_text:
            print()
            print(_colorize("Lean Board", "green"))
            print(_colorize("----------", "gray"))
            print(lean_summary_text, end="")
        if parlay_summary_text:
            print()
            print(_colorize("Parlay Board", "yellow"))
            print(_colorize("------------", "gray"))
            print(parlay_summary_text, end="")


if __name__ == "__main__":
    main()
