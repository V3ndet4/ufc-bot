from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.odds_api import load_odds_csv
from data_sources.storage import load_snapshot_history
from models.ev import american_to_decimal, implied_probability
from normalization.odds import normalize_odds_frame


CARD_WIDTH = 1120
CARD_HEIGHT = 300
CARD_GAP = 28
PAGE_PADDING = 32
PLOT_HEIGHT = 160
PLOT_TOP = 78
PLOT_LEFT = 70
PLOT_RIGHT = 48
PLOT_BOTTOM = 52
SERIES_COLORS = ("#1565c0", "#c62828")
BACKGROUND = "#f6f3ee"
CARD_BACKGROUND = "#fffdf8"
GRID_COLOR = "#d7d0c3"
TEXT_COLOR = "#1f1f1f"
MUTED_TEXT = "#6a655d"
ACCENT = "#d97706"
HEADER_BAND = "#efe7d8"
SUBTLE_ACCENT = "#b45309"
POSITIVE_MOVE = "#15803d"
NEGATIVE_MOVE = "#b91c1c"
NEUTRAL_MOVE = "#475569"
BADGE_BG = "#f4efe4"
BADGE_STROKE = "#dccfb8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an SVG line-movement board from opening odds to current odds."
    )
    parser.add_argument("--odds", required=True, help="Path to the current odds CSV input.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output SVG path, for example reports\\ufc_327_line_movement.svg",
    )
    parser.add_argument(
        "--db",
        default="data/ufc_betting.db",
        help="Optional SQLite path containing odds snapshots. Defaults to data\\ufc_betting.db.",
    )
    parser.add_argument("--event-id", help="Optional event_id filter.")
    parser.add_argument(
        "--bookmaker",
        help="Optional bookmaker filter, for example fanduel. Applies to the odds file and SQLite snapshots when a book column exists.",
    )
    parser.add_argument(
        "--fighter",
        help="Optional fighter-name filter. Matches either side of the fight using case-insensitive substring matching.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of fights to render. Default renders all fights.",
    )
    parser.add_argument(
        "--per-fight-dir",
        help="Optional directory for one SVG file per fight.",
    )
    parser.add_argument(
        "--png-dir",
        help="Optional directory for PNG exports if a converter library is available.",
    )
    return parser.parse_args()


def _bookmaker_label(bookmaker: str | None) -> str:
    if not bookmaker:
        return "All books"
    overrides = {
        "fanduel": "FanDuel",
        "draftkings": "DraftKings",
        "betmgm": "BetMGM",
        "caesars": "Caesars",
    }
    normalized = bookmaker.strip().lower()
    if normalized in overrides:
        return overrides[normalized]
    cleaned = bookmaker.replace("_", " ").replace("-", " ").strip()
    return " ".join(part.capitalize() for part in cleaned.split()) or bookmaker


def _safe_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_datetime(value: object) -> pd.Timestamp | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed


def _american_label(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+d}"


def _decimal_label(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{american_to_decimal(value):.2f}"


def _price_label(value: int | None) -> str:
    if value is None:
        return "n/a"
    return f"{_american_label(value)} | {_decimal_label(value)}"


def _probability_label(probability: float) -> str:
    return f"{probability * 100:.1f}%"


def _step_label(value: object) -> str:
    timestamp = _safe_datetime(value)
    if timestamp is None:
        return "Snapshot"
    python_dt = timestamp.to_pydatetime()
    label = python_dt.strftime("%b %d %I:%M %p")
    return label.replace(" 0", " ")


def _fight_key(row: pd.Series) -> tuple[str, str, str]:
    return str(row["event_id"]), str(row["fighter_a"]), str(row["fighter_b"])


def load_moneyline_odds(odds_path: str | Path, bookmaker: str | None = None) -> pd.DataFrame:
    odds = normalize_odds_frame(load_odds_csv(odds_path))
    for column in ["selection_name", "open_american_odds"]:
        if column not in odds.columns:
            odds[column] = pd.NA
    odds = odds.loc[odds["market"].astype(str) == "moneyline"].copy()
    if bookmaker and "book" in odds.columns:
        odds = odds.loc[odds["book"].astype(str).str.lower() == bookmaker.strip().lower()].copy()
    odds["selection_name"] = odds.apply(
        lambda row: row["fighter_a"] if str(row["selection"]) == "fighter_a" else row["fighter_b"],
        axis=1,
    )
    return odds


def load_moneyline_snapshots(
    db_path: str | Path,
    event_id: str | None = None,
    bookmaker: str | None = None,
) -> pd.DataFrame:
    path = Path(db_path)
    if not path.exists():
        return pd.DataFrame()
    snapshots = load_snapshot_history(path, event_id=event_id)
    if snapshots.empty:
        return snapshots
    if bookmaker and "book" in snapshots.columns:
        snapshots = snapshots.loc[snapshots["book"].astype(str).str.lower() == bookmaker.strip().lower()].copy()
    if "selection_name" not in snapshots.columns:
        snapshots["selection_name"] = snapshots.apply(
            lambda row: row["fighter_a"] if str(row["selection"]) == "fighter_a" else row["fighter_b"],
            axis=1,
        )
    snapshots = snapshots.loc[snapshots["market"].astype(str) == "moneyline"].copy()
    snapshots["snapshot_time"] = pd.to_datetime(snapshots["snapshot_time"], errors="coerce")
    snapshots = snapshots.loc[snapshots["snapshot_time"].notna()].copy()
    snapshots["american_odds"] = pd.to_numeric(snapshots["american_odds"], errors="coerce")
    snapshots = snapshots.loc[snapshots["american_odds"].notna()].copy()
    return snapshots


def filter_odds_frame(
    odds: pd.DataFrame,
    *,
    event_id: str | None = None,
    fighter_filter: str | None = None,
) -> pd.DataFrame:
    filtered = odds.copy()
    if event_id:
        filtered = filtered.loc[filtered["event_id"].astype(str) == event_id].copy()
    if fighter_filter:
        token = fighter_filter.strip().lower()
        filtered = filtered.loc[
            filtered["fighter_a"].astype(str).str.lower().str.contains(token)
            | filtered["fighter_b"].astype(str).str.lower().str.contains(token)
        ].copy()
    return filtered


def _build_series_points(
    fight_row: pd.Series,
    snapshots: pd.DataFrame,
    *,
    side: str,
    timeline_labels: list[str],
) -> list[dict[str, object]]:
    selection_name = str(fight_row["fighter_a"] if side == "fighter_a" else fight_row["fighter_b"])
    open_value = _safe_int(fight_row["open_american_odds"] if side == "fighter_a" else fight_row["fighter_b_open_american_odds"])
    current_value = _safe_int(fight_row["american_odds"] if side == "fighter_a" else fight_row["fighter_b_current_american_odds"])
    points: list[dict[str, object]] = []
    if open_value is not None and "Open" in timeline_labels:
        points.append(
            {
                "label": "Open",
                "display_label": "Open",
                "american_odds": open_value,
                "kind": "open",
                "selection_name": selection_name,
            }
        )

    selection_snapshots = pd.DataFrame()
    if not snapshots.empty and "selection_name" in snapshots.columns:
        selection_snapshots = snapshots.loc[snapshots["selection_name"].astype(str) == selection_name].copy()
    if not selection_snapshots.empty:
        selection_snapshots = selection_snapshots.sort_values("snapshot_time")
        previous_odds: int | None = open_value
        for snapshot in selection_snapshots.itertuples(index=False):
            american_odds = _safe_int(snapshot.american_odds)
            if american_odds is None:
                continue
            if previous_odds is not None and american_odds == previous_odds:
                continue
            timestamp = pd.Timestamp(snapshot.snapshot_time)
            label = timestamp.isoformat()
            points.append(
                {
                    "label": label,
                    "display_label": _step_label(timestamp),
                    "american_odds": american_odds,
                    "kind": "snapshot",
                    "selection_name": selection_name,
                }
            )
            previous_odds = american_odds

    if current_value is not None:
        current_point = {
            "label": "Current",
            "display_label": "Current",
            "american_odds": current_value,
            "kind": "current",
            "selection_name": selection_name,
        }
        if points and int(points[-1]["american_odds"]) == current_value:
            points[-1] = current_point
        else:
            points.append(current_point)

    for point in points:
        point["x_key"] = point["label"]
        point["implied_prob"] = implied_probability(int(point["american_odds"]))
        point["decimal_odds"] = american_to_decimal(int(point["american_odds"]))
    return [point for point in points if point["x_key"] in timeline_labels]


def build_line_movement_panels(
    odds: pd.DataFrame,
    snapshots: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    if odds.empty:
        return []
    working_snapshots = snapshots.copy() if snapshots is not None else pd.DataFrame()
    if not working_snapshots.empty:
        working_snapshots["fight_key"] = working_snapshots.apply(_fight_key, axis=1)

    panels: list[dict[str, object]] = []
    fighter_a_rows = odds.loc[odds["selection"].astype(str) == "fighter_a"].copy()
    fighter_b_rows = (
        odds.loc[
            odds["selection"].astype(str) == "fighter_b",
            ["event_id", "fighter_a", "fighter_b", "american_odds", "open_american_odds"],
        ]
        .rename(
            columns={
                "american_odds": "fighter_b_current_american_odds",
                "open_american_odds": "fighter_b_open_american_odds",
            }
        )
        .copy()
    )
    fight_rows = fighter_a_rows.merge(fighter_b_rows, on=["event_id", "fighter_a", "fighter_b"], how="inner")

    for fight_row in fight_rows.itertuples(index=False):
        fight_series = pd.Series(fight_row._asdict())
        fight_key = _fight_key(fight_series)
        fight_snapshots = (
            working_snapshots.loc[working_snapshots["fight_key"] == fight_key].copy()
            if not working_snapshots.empty
            else pd.DataFrame()
        )
        timeline_labels = []
        if _safe_int(fight_series["open_american_odds"]) is not None or _safe_int(fight_series["fighter_b_open_american_odds"]) is not None:
            timeline_labels.append("Open")
        if not fight_snapshots.empty:
            snapshot_labels = [
                pd.Timestamp(value).isoformat()
                for value in sorted(fight_snapshots["snapshot_time"].dropna().unique().tolist())
            ]
            timeline_labels.extend(snapshot_labels)
        timeline_labels.append("Current")

        series = [
            _build_series_points(fight_series, fight_snapshots, side="fighter_a", timeline_labels=timeline_labels),
            _build_series_points(fight_series, fight_snapshots, side="fighter_b", timeline_labels=timeline_labels),
        ]
        series = [line for line in series if line]
        if not series:
            continue

        implied_values = [float(point["implied_prob"]) for line in series for point in line]
        min_prob = min(implied_values)
        max_prob = max(implied_values)
        span = max(max_prob - min_prob, 0.06)
        padding = min(max(span * 0.18, 0.03), 0.12)

        panel = {
            "event_name": str(fight_series["event_name"]),
            "event_id": str(fight_series["event_id"]),
            "fighter_a": str(fight_series["fighter_a"]),
            "fighter_b": str(fight_series["fighter_b"]),
            "start_time": str(fight_series["start_time"]),
            "timeline_labels": timeline_labels,
            "series": series,
            "y_min": max(0.0, min_prob - padding),
            "y_max": min(1.0, max_prob + padding),
        }
        panels.append(panel)
    return panels


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _describe_move(points: list[dict[str, object]]) -> str:
    if not points:
        return "No price data"
    start = int(points[0]["american_odds"])
    end = int(points[-1]["american_odds"])
    start_prob = float(points[0]["implied_prob"])
    end_prob = float(points[-1]["implied_prob"])
    delta_odds = end - start
    delta_prob = (end_prob - start_prob) * 100
    if abs(delta_prob) < 0.2:
        direction = "flat"
    elif delta_prob > 0:
        direction = "toward fighter"
    else:
        direction = "away from fighter"
    return (
        f"{points[0]['selection_name']}: {_price_label(start)} -> {_price_label(end)} "
        f"({delta_odds:+d}, {delta_prob:+.1f} pts {direction})"
    )


def _line_delta_label(start: int, end: int) -> str:
    return f"{end - start:+d}"


def _probability_delta_label(start_prob: float, end_prob: float) -> str:
    return f"{(end_prob - start_prob) * 100:+.1f} pts"


def _price_status(value: int) -> str:
    if value < 0:
        return "Favorite"
    if value > 0:
        return "Underdog"
    return "Pick'em"


def _movement_tone(start: int, end: int) -> str:
    if start == end:
        return "neutral"
    if end < 0 and start >= 0:
        return "positive"
    if end >= 0 and start < 0:
        return "negative"
    if end < 0:
        return "positive" if abs(end) > abs(start) else "negative"
    return "positive" if end < start else "negative"


def _tone_colors(tone: str) -> tuple[str, str]:
    if tone == "positive":
        return POSITIVE_MOVE, "#ecfdf3"
    if tone == "negative":
        return NEGATIVE_MOVE, "#fef2f2"
    return NEUTRAL_MOVE, "#f8fafc"


def _movement_direction(start: int, end: int) -> str:
    if start == end:
        return "Held flat"
    if end < 0 and start >= 0:
        return "Flipped to favorite"
    if end >= 0 and start < 0:
        return "Flipped to underdog"
    if end < 0:
        return "Became bigger favorite" if abs(end) > abs(start) else "Drifted toward pick'em"
    return "Shortened as underdog" if end < start else "Drifted further out"


def _summary_metrics(points: list[dict[str, object]]) -> dict[str, str]:
    if not points:
        return {
            "name": "Unknown",
            "start_label": "Start",
            "open": "n/a",
            "current": "n/a",
            "line_delta": "n/a",
            "prob_delta": "n/a",
            "direction": "No price data",
        }
    start = int(points[0]["american_odds"])
    end = int(points[-1]["american_odds"])
    start_prob = float(points[0]["implied_prob"])
    end_prob = float(points[-1]["implied_prob"])
    start_kind = str(points[0].get("kind", "current"))
    if len(points) == 1:
        start_label = "Current"
    elif start_kind == "open":
        start_label = "Open"
    elif start_kind == "snapshot":
        start_label = "First seen"
    else:
        start_label = "Start"
    return {
        "name": str(points[-1]["selection_name"]),
        "start_label": start_label,
        "open": _price_label(start),
        "current": _price_label(end),
        "line_delta": _line_delta_label(start, end),
        "prob_delta": _probability_delta_label(start_prob, end_prob),
        "direction": _movement_direction(start, end),
        "status_open": _price_status(start),
        "status_current": _price_status(end),
        "tone": _movement_tone(start, end),
    }


def _slugify(value: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _history_status(panel: dict[str, object]) -> str:
    has_open = any(str(point.get("kind")) == "open" for line in panel["series"] for point in line)
    snapshot_count = sum(1 for line in panel["series"] for point in line if str(point.get("kind")) == "snapshot")
    if has_open and snapshot_count:
        return f"History: open + {snapshot_count} snapshot{'s' if snapshot_count != 1 else ''} + current"
    if has_open:
        return "History: open + current"
    if snapshot_count:
        return f"History: first seen + {snapshot_count} snapshot{'s' if snapshot_count != 1 else ''} + current"
    return "History: current only, no FanDuel opening history captured yet"


def _single_panel_svg(
    panel: dict[str, object],
    *,
    title: str = "UFC Line Movement",
    subtitle: str = "",
    source_label: str = "All books",
) -> str:
    width = CARD_WIDTH + PAGE_PADDING * 2
    height = PAGE_PADDING * 2 + 72 + CARD_HEIGHT
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" />',
        f'<rect x="0" y="0" width="{width}" height="82" fill="{HEADER_BAND}" />',
        f'<text x="{PAGE_PADDING}" y="40" font-family="Segoe UI, Arial, sans-serif" font-size="30" font-weight="700" fill="{TEXT_COLOR}">{_escape(title)}</text>',
        f'<text x="{PAGE_PADDING}" y="64" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="{MUTED_TEXT}">{_escape(subtitle or f"{source_label} opening line to current line, with both American and decimal odds plus implied probability.")}</text>',
    ]
    card_y = PAGE_PADDING + 72
    plot_width = CARD_WIDTH - PLOT_LEFT - PLOT_RIGHT
    plot_y = card_y + PLOT_TOP
    plot_bottom = plot_y + PLOT_HEIGHT
    y_min = float(panel["y_min"])
    y_max = float(panel["y_max"])
    y_span = max(y_max - y_min, 0.01)
    timeline_labels = panel["timeline_labels"]
    step_count = max(len(timeline_labels) - 1, 1)

    def x_pos(label: str) -> float:
        return PAGE_PADDING + PLOT_LEFT + (timeline_labels.index(label) / step_count) * plot_width

    def y_pos(probability: float) -> float:
        return plot_bottom - ((probability - y_min) / y_span) * PLOT_HEIGHT

    def clamp_label_x(value: float) -> float:
        left = PAGE_PADDING + PLOT_LEFT + 8
        right = PAGE_PADDING + PLOT_LEFT + plot_width - 8
        return max(left, min(right, value))

    svg_parts.append(
        f'<rect x="{PAGE_PADDING}" y="{card_y}" rx="18" ry="18" width="{CARD_WIDTH}" height="{CARD_HEIGHT}" fill="{CARD_BACKGROUND}" stroke="{GRID_COLOR}" stroke-width="1.2" />'
    )
    svg_parts.append(
        f'<text x="{PAGE_PADDING + 24}" y="{card_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="22" font-weight="700" fill="{TEXT_COLOR}">{_escape(panel["fighter_a"])} vs {_escape(panel["fighter_b"])}</text>'
    )
    svg_parts.append(
        f'<text x="{PAGE_PADDING + 24}" y="{card_y + 55}" font-family="Segoe UI, Arial, sans-serif" font-size="13" fill="{MUTED_TEXT}">{_escape(panel["event_name"])} | {_escape(panel["start_time"])}</text>'
    )
    svg_parts.append(
        f'<text x="{PAGE_PADDING + CARD_WIDTH - 24}" y="{card_y + 32}" text-anchor="end" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="{MUTED_TEXT}">{_escape(_history_status(panel))}</text>'
    )

    for guide in range(5):
        probability = y_min + (y_span * guide / 4)
        y = y_pos(probability)
        label = f"{probability * 100:.0f}%"
        svg_parts.append(
            f'<line x1="{PAGE_PADDING + PLOT_LEFT}" y1="{y:.1f}" x2="{PAGE_PADDING + PLOT_LEFT + plot_width}" y2="{y:.1f}" stroke="{GRID_COLOR}" stroke-width="1" stroke-dasharray="4 6" />'
        )
        svg_parts.append(
            f'<text x="{PAGE_PADDING + 14}" y="{y + 4:.1f}" font-family="Segoe UI, Arial, sans-serif" font-size="11" fill="{MUTED_TEXT}">{label}</text>'
        )

    for label in timeline_labels:
        x = x_pos(label)
        display = "Open" if label == "Open" else "Current" if label == "Current" else _step_label(label)
        svg_parts.append(
            f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_bottom}" stroke="#efe8db" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 24}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">{_escape(display)}</text>'
        )

    summary_y = card_y + 60
    for line_index, points in enumerate(panel["series"]):
        color = SERIES_COLORS[line_index % len(SERIES_COLORS)]
        polyline = " ".join(
            f"{x_pos(str(point['x_key'])):.1f},{y_pos(float(point['implied_prob'])):.1f}" for point in points
        )
        svg_parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" points="{polyline}" />'
        )
        summary_x = PAGE_PADDING + 620 if line_index else PAGE_PADDING + 24
        summary = _summary_metrics(points)
        move_color, move_fill = _tone_colors(summary["tone"])
        svg_parts.append(
            f'<rect x="{summary_x}" y="{summary_y}" width="476" height="58" rx="12" ry="12" fill="{HEADER_BAND}" opacity="0.92" />'
        )
        svg_parts.append(
            f'<text x="{summary_x + 12}" y="{summary_y + 16}" font-family="Segoe UI, Arial, sans-serif" font-size="11" font-weight="700" fill="{color}">{_escape(summary["name"])}</text>'
        )
        svg_parts.append(
            f'<rect x="{summary_x + 356}" y="{summary_y + 8}" width="108" height="16" rx="8" ry="8" fill="{move_fill}" stroke="{move_color}" stroke-width="1" />'
        )
        svg_parts.append(
            f'<text x="{summary_x + 410}" y="{summary_y + 19}" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-size="9" font-weight="700" fill="{move_color}">{_escape(summary["direction"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 12}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">{_escape(summary["start_label"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 58}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{TEXT_COLOR}">{_escape(summary["open"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 118}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="{SUBTLE_ACCENT}">{_escape(summary["status_open"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 166}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">Current</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 222}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{TEXT_COLOR}">{_escape(summary["current"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 290}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="{SUBTLE_ACCENT}">{_escape(summary["status_current"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 338}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">Line</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 372}" y="{summary_y + 32}" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="700" fill="{move_color}">{_escape(summary["line_delta"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 12}" y="{summary_y + 46}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">Implied</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 58}" y="{summary_y + 46}" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="700" fill="{move_color}">{_escape(summary["prob_delta"])}</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 166}" y="{summary_y + 46}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{MUTED_TEXT}">Move</text>'
        )
        svg_parts.append(
            f'<text x="{summary_x + 204}" y="{summary_y + 46}" font-family="Segoe UI, Arial, sans-serif" font-size="10" fill="{TEXT_COLOR}">{_escape(summary["status_open"])} to {_escape(summary["status_current"])}</text>'
        )
        for point in points:
            cx = x_pos(str(point["x_key"]))
            cy = y_pos(float(point["implied_prob"]))
            radius = 6.5 if point["kind"] in {"open", "current"} else 4.5
            stroke = ACCENT if point["kind"] == "current" else CARD_BACKGROUND
            svg_parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{color}" stroke="{stroke}" stroke-width="2" />'
            )
        label_points: list[tuple[dict[str, object], str, int, int]] = []
        if len(points) == 1:
            solo_dx = 14 if line_index == 0 else -14
            solo_anchor = "700"
            solo_dy = -16 if line_index == 0 else 26
            label_points.append((points[0], solo_anchor, solo_dx, solo_dy))
        else:
            start_dx = -14 if line_index == 0 else -18
            end_dx = 14 if line_index == 0 else 18
            start_dy = -16 if line_index == 0 else 12
            end_dy = -16 if line_index == 0 else 26
            label_points.append((points[0], "600", start_dx, start_dy))
            label_points.append((points[-1], "700", end_dx, end_dy))

        seen_labels: set[tuple[str, int]] = set()
        for point, font_weight, dx, dy in label_points:
            label_key = (str(point["x_key"]), int(point["american_odds"]))
            if label_key in seen_labels:
                continue
            seen_labels.add(label_key)
            anchor = "start" if dx >= 0 else "end"
            x = clamp_label_x(x_pos(str(point["x_key"])) + dx)
            y = y_pos(float(point["implied_prob"])) + dy
            svg_parts.append(
                f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="Segoe UI, Arial, sans-serif" font-size="10" font-weight="{font_weight}" fill="{color}">{_escape(_price_label(int(point["american_odds"])))}</text>'
            )
            svg_parts.append(
                f'<text x="{x:.1f}" y="{y + 12:.1f}" text-anchor="{anchor}" font-family="Segoe UI, Arial, sans-serif" font-size="9" fill="{SUBTLE_ACCENT}">{_escape(_probability_label(float(point["implied_prob"])))}</text>'
            )

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_line_movement_svg(panels: list[dict[str, object]], *, source_label: str = "All books") -> str:
    width = CARD_WIDTH + PAGE_PADDING * 2
    height = PAGE_PADDING * 2 + 72 + max(1, len(panels)) * CARD_HEIGHT + max(0, len(panels) - 1) * CARD_GAP
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}" />',
        f'<rect x="0" y="0" width="{width}" height="82" fill="{HEADER_BAND}" />',
        f'<text x="{PAGE_PADDING}" y="40" font-family="Segoe UI, Arial, sans-serif" font-size="30" font-weight="700" fill="{TEXT_COLOR}">UFC Line Movement Board</text>',
        f'<text x="{PAGE_PADDING}" y="64" font-family="Segoe UI, Arial, sans-serif" font-size="14" fill="{MUTED_TEXT}">{_escape(source_label)} opening to current with American odds, decimal odds, and implied probability. SQLite snapshots fill in the path when available.</text>',
    ]

    if not panels:
        svg_parts.append(
            f'<text x="{PAGE_PADDING}" y="120" font-family="Segoe UI, Arial, sans-serif" font-size="16" fill="{MUTED_TEXT}">No moneyline fights matched the selected filters.</text>'
        )
        svg_parts.append("</svg>")
        return "\n".join(svg_parts)

    for index, panel in enumerate(panels):
        panel_svg = _single_panel_svg(
            panel,
            title="",
            subtitle="",
            source_label=source_label,
        ).splitlines()
        body = []
        for line in panel_svg:
            if line.startswith("<svg") or line == "</svg>" or 'height="82"' in line:
                continue
            if line.startswith('<rect width=') or line.startswith('<text x="32" y="40"') or line.startswith('<text x="32" y="64"'):
                continue
            body.append(line)
        translate_y = index * (CARD_HEIGHT + CARD_GAP)
        svg_parts.append(f'<g transform="translate(0 {translate_y})">')
        svg_parts.extend(body)
        svg_parts.append("</g>")

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def write_svg(svg: str, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def write_per_fight_svgs(
    panels: list[dict[str, object]],
    output_dir: str | Path,
    *,
    source_label: str = "All books",
) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for panel in panels:
        filename = f"{_slugify(str(panel['fighter_a']))}_vs_{_slugify(str(panel['fighter_b']))}.svg"
        path = directory / filename
        svg = _single_panel_svg(
            panel,
            title=f"{panel['fighter_a']} vs {panel['fighter_b']}",
            subtitle=f"{panel['event_name']} | {source_label} opening line to current line",
            source_label=source_label,
        )
        path.write_text(svg, encoding="utf-8")
        written.append(path)
    return written


def try_write_pngs(_panels: list[dict[str, object]], _output_dir: str | Path) -> list[Path]:
    return []


def main() -> None:
    args = parse_args()
    odds = load_moneyline_odds(args.odds, bookmaker=args.bookmaker)
    filtered_odds = filter_odds_frame(odds, event_id=args.event_id, fighter_filter=args.fighter)
    if args.limit and args.limit > 0:
        fighter_a_rows = filtered_odds.loc[filtered_odds["selection"].astype(str) == "fighter_a"].head(args.limit)
        fight_keys = {
            (str(row.event_id), str(row.fighter_a), str(row.fighter_b))
            for row in fighter_a_rows.itertuples(index=False)
        }
        filtered_odds = filtered_odds.loc[
            filtered_odds.apply(lambda row: _fight_key(row) in fight_keys, axis=1)
        ].copy()

    snapshots = load_moneyline_snapshots(args.db, event_id=args.event_id, bookmaker=args.bookmaker)
    if args.fighter and not snapshots.empty:
        token = args.fighter.strip().lower()
        snapshots = snapshots.loc[
            snapshots["fighter_a"].astype(str).str.lower().str.contains(token)
            | snapshots["fighter_b"].astype(str).str.lower().str.contains(token)
        ].copy()

    panels = build_line_movement_panels(filtered_odds, snapshots)
    source_label = _bookmaker_label(args.bookmaker)
    svg = render_line_movement_svg(panels, source_label=source_label)
    output_path = write_svg(svg, args.output)
    print(f"Wrote line movement board to {output_path}")
    if args.per_fight_dir:
        written = write_per_fight_svgs(panels, args.per_fight_dir, source_label=source_label)
        print(f"Wrote {len(written)} per-fight SVG charts to {args.per_fight_dir}")
    if args.png_dir:
        pngs = try_write_pngs(panels, args.png_dir)
        if pngs:
            print(f"Wrote {len(pngs)} PNG charts to {args.png_dir}")
        else:
            print("PNG export skipped: no SVG-to-PNG converter is installed in the current environment.")


if __name__ == "__main__":
    main()
