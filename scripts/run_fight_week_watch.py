from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.event_manifest import current_event_manifest_path, derived_paths, load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Sherdog gym data and fight-week news alerts for the active UFC card."
    )
    parser.add_argument(
        "--manifest",
        help="Path to the event manifest JSON. Defaults to events/current_event.txt.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=10,
        help="Days of recent coverage to scan for fight-week alerts.",
    )
    parser.add_argument(
        "--max-results-per-fighter",
        type=int,
        default=5,
        help="Max fight-week alerts to retain per fighter.",
    )
    parser.add_argument(
        "--gym-refresh-days",
        type=int,
        default=7,
        help="Days before a fighter gym profile is refreshed again.",
    )
    parser.add_argument(
        "--gym-association-refresh-days",
        type=int,
        default=30,
        help="Days before cached association fighters are refreshed again.",
    )
    parser.add_argument(
        "--gym-max-association-pages",
        type=int,
        default=5,
        help="Max Sherdog Fight Finder pages to scan per gym.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress child script console output.",
    )
    return parser.parse_args()


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=ROOT)


def _resolve_manifest_path(manifest_path: str | None) -> Path:
    if manifest_path:
        path = Path(manifest_path)
        if not path.is_absolute():
            path = (ROOT / path).resolve()
        return path
    return current_event_manifest_path(ROOT)


def _append_quiet(command: list[str], quiet: bool) -> list[str]:
    if quiet:
        return command + ["--quiet"]
    return command


def main() -> None:
    args = parse_args()
    manifest_path = _resolve_manifest_path(args.manifest)
    manifest = load_manifest(manifest_path)
    paths = derived_paths(manifest)

    print(f"Active event: {manifest['event_name']} ({manifest['event_id']})")
    print(f"Manifest: {manifest_path}")

    prepare_command = [
        sys.executable,
        "scripts/prepare_event.py",
        "--manifest",
        str(manifest_path),
    ]
    _run(_append_quiet(prepare_command, args.quiet))
    print(f"Prepared event workspace at {paths['workspace']}")

    gym_command = [
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
        str(args.gym_refresh_days),
        "--association-refresh-days",
        str(args.gym_association_refresh_days),
        "--max-association-pages",
        str(args.gym_max_association_pages),
    ]
    _run(_append_quiet(gym_command, args.quiet))
    print(f"Refreshed Sherdog gym data at {paths['fighter_gyms']}")

    watch_command = [
        sys.executable,
        "scripts/update_fight_week_watch.py",
        "--fighters-csv",
        str(paths["fighter_list"]),
        "--context",
        str(paths["context"]),
        "--alerts-output",
        str(paths["fight_week_alerts"]),
        "--lookback-days",
        str(args.lookback_days),
        "--max-results-per-fighter",
        str(args.max_results_per_fighter),
    ]
    if paths["fighter_gyms"].exists():
        watch_command.extend(["--fighter-gyms", str(paths["fighter_gyms"])])
    _run(_append_quiet(watch_command, args.quiet))
    print(f"Refreshed fight-week alerts at {paths['fight_week_alerts']}")


if __name__ == "__main__":
    main()
