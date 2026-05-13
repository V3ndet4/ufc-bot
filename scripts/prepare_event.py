from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.espn import merge_espn_url_maps
from data_sources.fighter_aliases import (
    build_fighter_alias_lookup,
    fighter_alias_key,
    load_fighter_alias_overrides,
)
from data_sources.gym_overrides import apply_context_gym_overrides, load_fighter_gym_overrides
from scripts.event_manifest import (
    build_context_frame,
    build_fighter_map_frame,
    build_fighter_list_frame,
    build_modeled_market_template_frame,
    build_odds_template_frame,
    derived_paths,
    load_manifest,
    merge_existing_context,
    merge_existing_fighter_map,
)

CONTEXT_FLAG_LABELS = {
    "short_notice_flag": "short notice",
    "short_notice_acceptance_flag": "short-notice acceptance",
    "short_notice_success_flag": "short-notice win",
    "new_gym_flag": "new gym",
    "new_contract_flag": "new contract",
    "cardio_fade_flag": "cardio fade",
    "injury_concern_flag": "injury concern",
    "weight_cut_concern_flag": "weight-cut concern",
    "replacement_fighter_flag": "replacement fighter",
    "travel_disadvantage_flag": "travel disadvantage",
    "camp_change_flag": "camp change",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create fighter list, context, and odds template files from an event manifest.")
    parser.add_argument("--manifest", required=True, help="Path to the event manifest JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _write_csv(frame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return text


def _safe_int(value: object, default: int = 0) -> int:
    if pd.isna(value):
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_name(value: object) -> str:
    return " ".join(_safe_text(value).lower().replace("’", "'").split())


def _alias_override_lookup() -> dict[str, str]:
    alias_overrides = load_fighter_alias_overrides(ROOT / "data" / "fighter_alias_overrides.csv")
    return build_fighter_alias_lookup(alias_overrides)


def _fighter_lookup_key(value: object, alias_lookup: dict[str, str] | None = None) -> str:
    key = fighter_alias_key(value, alias_lookup)
    if key:
        return key
    return _normalize_name(value)


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _combine_preview_frames(
    frames: list[pd.DataFrame | None],
    alias_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()

    combined = pd.concat(valid, ignore_index=True, sort=False)
    if "fighter_name" not in combined.columns:
        return combined
    combined["fighter_name"] = combined["fighter_name"].fillna("").astype(str).str.strip()
    combined = combined.loc[combined["fighter_name"] != ""].copy()
    combined["fighter_name_normalized"] = combined["fighter_name"].map(
        lambda value: _fighter_lookup_key(value, alias_lookup)
    )
    return combined.drop_duplicates(subset=["fighter_name_normalized"], keep="first").reset_index(drop=True)


def _build_fighter_lookup(
    frame: pd.DataFrame | None,
    alias_lookup: dict[str, str] | None = None,
) -> dict[str, dict[str, object]]:
    if frame is None or frame.empty or "fighter_name" not in frame.columns:
        return {}
    working = frame.copy()
    working["fighter_name_normalized"] = working["fighter_name"].map(
        lambda value: _fighter_lookup_key(value, alias_lookup)
    )
    return {
        _safe_text(row.get("fighter_name_normalized")): row
        for row in working.to_dict("records")
        if _safe_text(row.get("fighter_name_normalized"))
    }


def _format_inches(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "?"
    return f"{float(numeric):.1f}\""


def _format_pct(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "?"
    return f"{float(numeric):.0f}%"


def _format_rate(value: object) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "?"
    return f"{float(numeric):.1f}"


def _fighter_has_context_signal(row: dict[str, object] | None) -> bool:
    if not row:
        return False
    if any(_safe_int(row.get(column), 0) > 0 for column in CONTEXT_FLAG_LABELS):
        return True
    if _safe_int(row.get("news_alert_count"), 0) > 0:
        return True
    return any(_safe_text(row.get(column)) for column in ["news_radar_label", "news_radar_summary", "context_notes"])


def _fighter_context_summary(row: dict[str, object] | None) -> str:
    if not row:
        return ""
    parts = [
        label
        for column, label in CONTEXT_FLAG_LABELS.items()
        if _safe_int(row.get(column), 0) > 0
    ]
    news_count = _safe_int(row.get("news_alert_count"), 0)
    news_label = _safe_text(row.get("news_radar_label"))
    news_summary = _safe_text(row.get("news_radar_summary"))
    note = _safe_text(row.get("context_notes"))
    if news_count > 0 or news_label or news_summary:
        news_parts = [part for part in [news_label, news_summary] if part]
        if news_count > 0:
            news_parts.insert(0, f"{news_count} alert(s)")
        parts.append("news: " + " | ".join(news_parts))
    if note:
        parts.append(f"notes: {note}")
    return " | ".join(parts)


def _fighter_record(stats_row: dict[str, object] | None, gym_row: dict[str, object] | None) -> str:
    if stats_row:
        wins = _safe_int(stats_row.get("wins"), -1)
        losses = _safe_int(stats_row.get("losses"), -1)
        draws = _safe_int(stats_row.get("draws"), 0)
        if wins >= 0 and losses >= 0:
            return f"{wins}-{losses}-{draws}"
    if gym_row:
        wins = _safe_int(gym_row.get("fighter_wins"), -1)
        losses = _safe_int(gym_row.get("fighter_losses"), -1)
        draws = _safe_int(gym_row.get("fighter_draws"), 0)
        if wins >= 0 and losses >= 0:
            return f"{wins}-{losses}-{draws}"
    return "record ?"


def _fighter_summary_line(
    fighter_name: str,
    *,
    stats_row: dict[str, object] | None,
    gym_row: dict[str, object] | None,
    map_row: dict[str, object] | None,
) -> str:
    pieces = [f"{fighter_name}: {_fighter_record(stats_row, gym_row)}"]
    weight_class = _safe_text((stats_row or {}).get("weight_class") or (gym_row or {}).get("weight_class"))
    stance = _safe_text((stats_row or {}).get("stance"))
    if weight_class:
        pieces.append(weight_class)
    if stance:
        pieces.append(stance)
    if stats_row:
        pieces.append(f"ht { _format_inches(stats_row.get('height_in')) }")
        pieces.append(f"reach { _format_inches(stats_row.get('reach_in')) }")
        pieces.append(f"SLpM { _format_rate(stats_row.get('sig_strikes_landed_per_min')) }")
        pieces.append(f"SApM { _format_rate(stats_row.get('sig_strikes_absorbed_per_min')) }")
        pieces.append(f"TD avg { _format_rate(stats_row.get('takedown_avg')) }")
        pieces.append(f"TD def { _format_pct(stats_row.get('takedown_defense_pct')) }")
    else:
        pieces.append("offline stats missing")
    gym_name = _safe_text((gym_row or {}).get("gym_name"))
    if gym_name:
        gym_tier = _safe_text((gym_row or {}).get("gym_tier"))
        pieces.append(f"camp {gym_name}{f' ({gym_tier}-tier)' if gym_tier else ''}")
    else:
        pieces.append("camp ?")
    if _safe_text((map_row or {}).get("espn_url")):
        pieces.append("ESPN map ready")
    else:
        pieces.append("ESPN map missing")
    return " | ".join(pieces)


def _metric_edge(label: str, fighter_a: str, fighter_b: str, a_value: object, b_value: object, *, unit: str = "", min_gap: float = 0.1) -> str:
    a_numeric = pd.to_numeric(pd.Series([a_value]), errors="coerce").iloc[0]
    b_numeric = pd.to_numeric(pd.Series([b_value]), errors="coerce").iloc[0]
    if pd.isna(a_numeric) or pd.isna(b_numeric):
        return ""
    diff = float(a_numeric) - float(b_numeric)
    if abs(diff) < min_gap:
        return ""
    leader = fighter_a if diff > 0 else fighter_b
    return f"{label} {leader} +{abs(diff):.1f}{unit}"


def _matchup_edges_line(
    fighter_a: str,
    fighter_b: str,
    a_stats: dict[str, object] | None,
    b_stats: dict[str, object] | None,
) -> str:
    if not a_stats or not b_stats:
        return "Edges: offline stat comparison unavailable."
    strike_margin_a = _safe_float(a_stats.get("sig_strikes_landed_per_min")) - _safe_float(a_stats.get("sig_strikes_absorbed_per_min"))
    strike_margin_b = _safe_float(b_stats.get("sig_strikes_landed_per_min")) - _safe_float(b_stats.get("sig_strikes_absorbed_per_min"))
    edges = [
        _metric_edge("reach", fighter_a, fighter_b, a_stats.get("reach_in"), b_stats.get("reach_in"), unit="\"", min_gap=1.0),
        _metric_edge("height", fighter_a, fighter_b, a_stats.get("height_in"), b_stats.get("height_in"), unit="\"", min_gap=1.0),
        _metric_edge("strike margin", fighter_a, fighter_b, strike_margin_a, strike_margin_b, unit="/min", min_gap=0.5),
        _metric_edge("TD avg", fighter_a, fighter_b, a_stats.get("takedown_avg"), b_stats.get("takedown_avg"), unit="", min_gap=0.4),
        _metric_edge("TD def", fighter_a, fighter_b, a_stats.get("takedown_defense_pct"), b_stats.get("takedown_defense_pct"), unit="%", min_gap=8.0),
    ]
    edges = [edge for edge in edges if edge]
    if not edges:
        return "Edges: no strong offline stat edge."
    return "Edges: " + " | ".join(edges[:4])


def _load_preview_sources(
    paths: dict[str, Path],
    alias_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = _combine_preview_frames(
        [
            _read_csv_if_exists(paths["data_dir"] / "fighter_stats.csv"),
            _read_csv_if_exists(ROOT / "data" / "espn_fighter_stats.csv"),
            _read_csv_if_exists(ROOT / "data" / "real_fighter_stats.csv"),
        ],
        alias_lookup,
    )
    gyms = _combine_preview_frames(
        [
            _read_csv_if_exists(paths["data_dir"] / "fighter_gyms.csv"),
            _read_csv_if_exists(ROOT / "data" / "sherdog_fighter_gyms.csv"),
        ],
        alias_lookup,
    )
    return stats, gyms


def build_card_preview(
    manifest: dict[str, object],
    context_frame: pd.DataFrame | None = None,
    fighter_map_frame: pd.DataFrame | None = None,
    stats_frame: pd.DataFrame | None = None,
    gym_frame: pd.DataFrame | None = None,
    alias_lookup: dict[str, str] | None = None,
) -> str:
    event_name = _safe_text(manifest.get("event_name"), "Unknown event")
    start_time = pd.to_datetime(manifest.get("start_time"), errors="coerce")
    if pd.isna(start_time):
        start_label = "unknown start time"
    else:
        start_label = start_time.strftime("%A, %B %d, %Y %I:%M %p %z")

    fights = manifest.get("fights", [])
    all_fighters = [
        _safe_text(fight.get(side))
        for fight in fights
        for side in ["fighter_a", "fighter_b"]
        if _safe_text(fight.get(side))
    ]
    fighter_total = len(all_fighters)
    map_lookup = _build_fighter_lookup(fighter_map_frame, alias_lookup)
    stats_lookup = _build_fighter_lookup(stats_frame, alias_lookup)
    gym_lookup = _build_fighter_lookup(gym_frame, alias_lookup)
    context_lookup = _build_fighter_lookup(context_frame, alias_lookup)

    mapped_count = sum(
        1
        for fighter in all_fighters
        if _safe_text((map_lookup.get(_fighter_lookup_key(fighter, alias_lookup)) or {}).get("espn_url"))
    )
    stats_count = sum(1 for fighter in all_fighters if _fighter_lookup_key(fighter, alias_lookup) in stats_lookup)
    gym_count = sum(1 for fighter in all_fighters if _fighter_lookup_key(fighter, alias_lookup) in gym_lookup)
    context_watch_count = sum(
        1
        for fighter in all_fighters
        if _fighter_has_context_signal(context_lookup.get(_fighter_lookup_key(fighter, alias_lookup)))
    )

    lines = [
        "Card Preview",
        "------------",
        f"Event: {event_name}",
        f"Start: {start_label}",
        f"Bouts: {len(fights)}",
        "",
        "Readiness",
        "---------",
        f"ESPN links mapped: {mapped_count}/{fighter_total} fighters",
        f"Offline stats cached: {stats_count}/{fighter_total} fighters",
        f"Cached gym profiles: {gym_count}/{fighter_total} fighters",
        f"Context watchlist entries: {context_watch_count}/{fighter_total} fighters",
        "",
        "Fight List",
        "----------",
    ]

    for index, fight in enumerate(fights, start=1):
        fighter_a = _safe_text(fight.get("fighter_a"), "TBD")
        fighter_b = _safe_text(fight.get("fighter_b"), "TBD")
        rounds = _safe_int(fight.get("scheduled_rounds"), 3)
        main_tag = " [main event]" if index == 1 else ""
        rounds_tag = f" ({rounds} rounds)" if rounds and rounds != 3 else ""
        lines.append(f"{index}. {fighter_a} vs {fighter_b}{rounds_tag}{main_tag}")

    lines.extend(["", "Matchup Snapshots", "-----------------"])
    for index, fight in enumerate(fights, start=1):
        fighter_a = _safe_text(fight.get("fighter_a"), "TBD")
        fighter_b = _safe_text(fight.get("fighter_b"), "TBD")
        rounds = _safe_int(fight.get("scheduled_rounds"), 3)
        main_tag = " [main event]" if index == 1 else ""
        rounds_tag = f" ({rounds} rounds)" if rounds and rounds != 3 else ""
        a_key = _fighter_lookup_key(fighter_a, alias_lookup)
        b_key = _fighter_lookup_key(fighter_b, alias_lookup)
        a_stats = stats_lookup.get(a_key)
        b_stats = stats_lookup.get(b_key)
        a_gym = gym_lookup.get(a_key)
        b_gym = gym_lookup.get(b_key)
        a_map = map_lookup.get(a_key)
        b_map = map_lookup.get(b_key)
        a_context = context_lookup.get(a_key)
        b_context = context_lookup.get(b_key)

        lines.append(f"{index}. {fighter_a} vs {fighter_b}{rounds_tag}{main_tag}")
        lines.append("   " + _fighter_summary_line(fighter_a, stats_row=a_stats, gym_row=a_gym, map_row=a_map))
        lines.append("   " + _fighter_summary_line(fighter_b, stats_row=b_stats, gym_row=b_gym, map_row=b_map))
        lines.append("   " + _matchup_edges_line(fighter_a, fighter_b, a_stats, b_stats))

        context_parts: list[str] = []
        a_context_summary = _fighter_context_summary(a_context)
        b_context_summary = _fighter_context_summary(b_context)
        if a_context_summary:
            context_parts.append(f"{fighter_a}: {a_context_summary}")
        if b_context_summary:
            context_parts.append(f"{fighter_b}: {b_context_summary}")
        if context_parts:
            lines.append("   Context: " + " || ".join(context_parts))
        lines.append("")

    flagged_rows: list[str] = []
    if context_frame is not None and not context_frame.empty and "fighter_name" in context_frame.columns:
        for row in context_frame.to_dict("records"):
            fighter_name = _safe_text(row.get("fighter_name"))
            if not fighter_name:
                continue
            summary = _fighter_context_summary(row)
            if summary:
                flagged_rows.append(f"- {fighter_name}: {summary}")

    lines.extend(["", "Context Watchlist", "-----------------"])
    if flagged_rows:
        lines.extend(flagged_rows)
    else:
        lines.append("No context flags or notes yet.")

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    paths = derived_paths(manifest)
    alias_lookup = _alias_override_lookup()

    fighter_list = build_fighter_list_frame(manifest)
    fighter_map_template = build_fighter_map_frame(manifest)
    context_template = build_context_frame(manifest)
    existing_context = None
    fighter_map_sources: list[pd.DataFrame] = [fighter_map_template]
    global_espn_cache = ROOT / "data" / "espn_fighter_map.csv"
    if global_espn_cache.exists():
        fighter_map_sources.append(pd.read_csv(global_espn_cache))
    if paths["fighter_map"].exists():
        fighter_map_sources.append(pd.read_csv(paths["fighter_map"]))
    if paths["context"].exists():
        existing_context = pd.read_csv(paths["context"])
    merged_fighter_map = merge_espn_url_maps(*fighter_map_sources, alias_lookup=alias_lookup)
    merged_fighter_map = merge_existing_fighter_map(fighter_map_template, merged_fighter_map)
    merged_context = merge_existing_context(context_template, existing_context)
    merged_context = apply_context_gym_overrides(merged_context, load_fighter_gym_overrides())
    preview_stats, preview_gyms = _load_preview_sources(paths, alias_lookup)

    _write_csv(fighter_list, paths["fighter_list"])
    _write_csv(merged_fighter_map, paths["fighter_map"])
    _write_csv(merged_context, paths["context"])
    _write_csv(build_odds_template_frame(manifest), paths["odds_template"])
    _write_csv(build_modeled_market_template_frame(manifest), paths["modeled_market_template"])

    if not args.quiet:
        print(f"Saved fighter list to {paths['fighter_list']}")
        print(f"Saved fighter map template to {paths['fighter_map']}")
        print(f"Saved context template to {paths['context']}")
        print(f"Saved odds template to {paths['odds_template']}")
        print(f"Saved modeled-market template to {paths['modeled_market_template']}")
        print()
        print(
            build_card_preview(
                manifest,
                merged_context,
                merged_fighter_map,
                preview_stats,
                preview_gyms,
                alias_lookup,
            ),
            end="",
        )


if __name__ == "__main__":
    main()
