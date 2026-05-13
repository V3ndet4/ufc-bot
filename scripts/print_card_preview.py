from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.event_manifest import derived_paths, load_manifest
from scripts.prepare_event import _alias_override_lookup, _load_preview_sources, _read_csv_if_exists, build_card_preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the local card preview using cached manifest, stats, gym, and context files.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)
    alias_lookup = _alias_override_lookup()
    fighter_map = _read_csv_if_exists(paths["fighter_map"])
    context = _read_csv_if_exists(paths["context"])
    preview_stats, preview_gyms = _load_preview_sources(paths, alias_lookup)
    print(build_card_preview(manifest, context, fighter_map, preview_stats, preview_gyms, alias_lookup), end="")


if __name__ == "__main__":
    main()
