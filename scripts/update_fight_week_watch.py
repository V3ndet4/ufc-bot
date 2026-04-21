from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.fight_week_watch import (
    collect_fight_week_alerts,
    merge_alerts_into_context,
    write_alerts_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh fight-week alerts and merge detected context into the fighter context CSV.")
    parser.add_argument("--fighters-csv", required=True, help="CSV with a fighter_name column.")
    parser.add_argument("--context", required=True, help="Context CSV to update in place.")
    parser.add_argument("--alerts-output", required=True, help="Output CSV path for fight-week alerts.")
    parser.add_argument("--fighter-gyms", help="Optional fighter gym CSV used to enrich news searches.")
    parser.add_argument("--lookback-days", type=int, default=10, help="How many days of recent fight-week coverage to scan.")
    parser.add_argument("--max-results-per-fighter", type=int, default=5, help="Max alerts to retain per fighter.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _load_fighter_watch_frame(fighters_csv: str | Path, fighter_gyms_csv: str | Path | None) -> pd.DataFrame:
    fighters = pd.read_csv(fighters_csv)
    if "fighter_name" not in fighters.columns:
        raise ValueError("fighters CSV must contain fighter_name")
    watch_frame = fighters[["fighter_name"]].copy()
    watch_frame["fighter_name"] = watch_frame["fighter_name"].astype(str).str.strip()
    watch_frame = watch_frame.loc[watch_frame["fighter_name"] != ""].drop_duplicates(subset=["fighter_name"]).reset_index(drop=True)

    if fighter_gyms_csv and Path(fighter_gyms_csv).exists():
        gyms = pd.read_csv(fighter_gyms_csv)
        if "fighter_name" in gyms.columns:
            gyms = gyms[["fighter_name", "gym_name"]].copy()
            gyms["fighter_name"] = gyms["fighter_name"].astype(str).str.strip()
            gyms["gym_name"] = gyms["gym_name"].fillna("").astype(str).str.strip()
            watch_frame = watch_frame.merge(gyms, on="fighter_name", how="left")
    if "gym_name" not in watch_frame.columns:
        watch_frame["gym_name"] = ""
    else:
        watch_frame["gym_name"] = watch_frame["gym_name"].fillna("").astype(str)
    return watch_frame


def main() -> None:
    args = parse_args()
    watch_frame = _load_fighter_watch_frame(args.fighters_csv, args.fighter_gyms)
    context = pd.read_csv(args.context)
    alerts = collect_fight_week_alerts(
        watch_frame,
        lookback_days=args.lookback_days,
        max_results_per_fighter=args.max_results_per_fighter,
    )
    updated_context = merge_alerts_into_context(context, alerts)

    context_path = Path(args.context)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    updated_context.to_csv(context_path, index=False)
    alerts_path = write_alerts_csv(alerts, args.alerts_output)

    if not args.quiet:
        flagged_fighters = 0 if alerts.empty else int(alerts["fighter_name"].nunique())
        print(f"Fight-week watch: {len(alerts)} alerts | {flagged_fighters} fighters flagged")
        print(f"Updated context at {context_path}")
        print(f"Saved alerts to {alerts_path}")


if __name__ == "__main__":
    main()
