from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import (
    DEFAULT_BOOKMAKER,
    OddsApiError,
    extract_alternative_market_keys,
    fetch_the_odds_api_event_markets,
    load_api_key,
)
from scripts.event_manifest import current_event_manifest, derived_paths, manifest_status_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List event manifests or set the current event pointer.")
    parser.add_argument("--list", action="store_true", help="List available event manifest files.")
    parser.add_argument("--status", action="store_true", help="Show the current event and artifact readiness.")
    parser.add_argument("--manifest", help="Manifest path relative to the repo root, for example events/ufc_327_prochazka_ulberg.json.")
    parser.add_argument(
        "--probe-oddsapi-alt-markets",
        action="store_true",
        help="Probe FanDuel/The Odds API for alternative-market availability using ids from the current oddsapi CSV.",
    )
    parser.add_argument(
        "--odds-api-bookmaker",
        default=DEFAULT_BOOKMAKER,
        help="Bookmaker key for the live alternative-market probe.",
    )
    return parser.parse_args()


def list_events(events_dir: Path) -> None:
    manifests = sorted(path for path in events_dir.glob("*.json"))
    if not manifests:
        print("No event manifests found.")
        return
    current_pointer = (events_dir / "current_event.txt").read_text(encoding="utf-8").strip() if (events_dir / "current_event.txt").exists() else ""
    for manifest in manifests:
        relative_path = manifest.relative_to(ROOT).as_posix()
        marker = "*" if relative_path == current_pointer else " "
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            label = f"{payload.get('event_name', manifest.stem)} [{payload.get('event_id', 'unknown')}]"
        except Exception:
            label = manifest.stem
        print(f"{marker} {relative_path} | {label}")


def set_current_event(manifest_relative: str, events_dir: Path) -> None:
    manifest_path = (ROOT / manifest_relative).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_relative}")
    if manifest_path.suffix.lower() != ".json":
        raise ValueError("Manifest path must point to a JSON file")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    pointer = events_dir / "current_event.txt"
    pointer.write_text(f"{Path(manifest_relative).as_posix()}\n", encoding="utf-8")
    print(f"Current event set to {manifest_relative}")
    print(f"Event: {payload.get('event_name', manifest_path.stem)}")


def _status_probe_rows(manifest: dict[str, object], bookmaker_key: str) -> list[tuple[str, str]]:
    oddsapi_path = derived_paths(manifest)["oddsapi_odds"]
    if not oddsapi_path.exists():
        return [
            ("oddsapi_alt_market_fights", "0"),
            ("oddsapi_alt_market_status", "oddsapi_odds_missing"),
        ]

    frame = pd.read_csv(oddsapi_path)
    if "odds_api_event_id" not in frame.columns:
        return [
            ("oddsapi_alt_market_fights", "0"),
            ("oddsapi_alt_market_status", "oddsapi_event_id_missing"),
        ]

    event_ids = sorted({str(value).strip() for value in frame["odds_api_event_id"].dropna() if str(value).strip()})
    if not event_ids:
        return [
            ("oddsapi_alt_market_fights", "0"),
            ("oddsapi_alt_market_status", "no_oddsapi_event_ids"),
        ]

    try:
        api_key = load_api_key()
    except OddsApiError:
        return [
            ("oddsapi_alt_market_fights", "0"),
            ("oddsapi_alt_market_status", "odds_api_key_missing"),
        ]

    fight_count_with_alt_markets = 0
    discovered_market_keys: set[str] = set()
    try:
        for event_id in event_ids:
            payload = fetch_the_odds_api_event_markets(api_key=api_key, event_id=event_id)
            alt_keys = extract_alternative_market_keys(payload, bookmaker_key)
            if alt_keys:
                fight_count_with_alt_markets += 1
                discovered_market_keys.update(alt_keys)
    except Exception as exc:
        return [
            ("oddsapi_alt_market_fights", "0"),
            ("oddsapi_alt_market_status", f"probe_failed: {type(exc).__name__}"),
        ]

    return [
        ("oddsapi_alt_market_fights", str(fight_count_with_alt_markets)),
        ("oddsapi_alt_market_status", "available" if fight_count_with_alt_markets else "unavailable"),
        ("oddsapi_alt_market_keys", ",".join(sorted(discovered_market_keys)) if discovered_market_keys else "none"),
    ]


def print_status(*, probe_oddsapi_alt_markets: bool = False, bookmaker_key: str = DEFAULT_BOOKMAKER) -> None:
    manifest = current_event_manifest(ROOT)
    for key, value in manifest_status_rows(manifest):
        print(f"{key}: {value}")
    if probe_oddsapi_alt_markets:
        for key, value in _status_probe_rows(manifest, bookmaker_key):
            print(f"{key}: {value}")


def main() -> None:
    load_dotenv(ROOT / ".env")
    args = parse_args()
    events_dir = ROOT / "events"
    if args.list:
        list_events(events_dir)
        return
    if args.status:
        print_status(
            probe_oddsapi_alt_markets=args.probe_oddsapi_alt_markets,
            bookmaker_key=args.odds_api_bookmaker,
        )
        return
    if not args.manifest:
        raise ValueError("Pass --list, --status, or --manifest")
    set_current_event(args.manifest, events_dir)


if __name__ == "__main__":
    main()
