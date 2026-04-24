from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_sources.storage import load_snapshot_history


ARCHIVE_COLUMNS = [
    "event_id",
    "event_name",
    "start_time",
    "fighter_a",
    "fighter_b",
    "market",
    "selection",
    "book",
    "american_odds",
    "actual_result",
    "source_card",
    "odds_source",
    "odds_is_fallback",
    "result_status",
    "method",
    "scheduled_rounds",
    "is_title_fight",
]

MARKET_DEFINITIONS = (
    {
        "market": "moneyline",
        "selections": ("fighter_a", "fighter_b"),
        "closing_columns": {
            "fighter_a": "closing_fighter_a_odds",
            "fighter_b": "closing_fighter_b_odds",
        },
    },
    {
        "market": "fight_goes_to_decision",
        "selections": ("fight_goes_to_decision",),
        "closing_columns": {
            "fight_goes_to_decision": "closing_fight_goes_to_decision_odds",
        },
    },
    {
        "market": "fight_doesnt_go_to_decision",
        "selections": ("fight_doesnt_go_to_decision",),
        "closing_columns": {
            "fight_doesnt_go_to_decision": "closing_fight_doesnt_go_to_decision_odds",
        },
    },
    {
        "market": "inside_distance",
        "selections": ("fighter_a", "fighter_b"),
        "closing_columns": {
            "fighter_a": "closing_fighter_a_inside_distance_odds",
            "fighter_b": "closing_fighter_b_inside_distance_odds",
        },
    },
    {
        "market": "by_decision",
        "selections": ("fighter_a", "fighter_b"),
        "closing_columns": {
            "fighter_a": "closing_fighter_a_by_decision_odds",
            "fighter_b": "closing_fighter_b_by_decision_odds",
        },
    },
)


def build_historical_archive(
    cards_root: str | Path,
    *,
    snapshot_db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(cards_root)
    archive_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    snapshot_fallback_odds = _build_snapshot_odds_lookup(snapshot_db_path)

    for card_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        data_dir = card_dir / "data"
        results_path = data_dir / "results.csv"
        if not results_path.exists():
            continue

        results = pd.read_csv(results_path)
        if results.empty:
            summary_rows.append(
                {
                    "card": card_dir.name,
                    "event_id": "",
                    "event_name": "",
                    "rows_written": 0,
                    "fights_written": 0,
                    "push_rows": 0,
                    "fallback_fights": 0,
                    "fallback_rows": 0,
                    "skipped_missing_odds": 0,
                    "status": "results_empty",
                }
            )
            continue

        event_id = _safe_text(results.iloc[0].get("event_id"))
        event_name = _safe_text(results.iloc[0].get("event_name"))
        fight_meta = _build_fight_meta_lookup(data_dir)
        fallback_odds = _build_fallback_odds_lookup(data_dir)

        card_rows: list[dict[str, object]] = []
        skipped_missing_odds = 0
        fallback_fights = 0
        fallback_rows = 0

        for row in results.to_dict(orient="records"):
            fighter_a = _safe_text(row.get("fighter_a"))
            fighter_b = _safe_text(row.get("fighter_b"))
            if not fighter_a or not fighter_b:
                continue

            key = _fight_key(row.get("event_id", event_id), fighter_a, fighter_b)
            meta = fight_meta.get(key, {})

            for definition in MARKET_DEFINITIONS:
                market = str(definition["market"])
                resolved_rows, missing_count, used_fallback = _resolve_market_rows(
                    result_row=row,
                    market=market,
                    selections=tuple(definition["selections"]),
                    closing_columns=dict(definition["closing_columns"]),
                    fallback=_merge_fallback_sources(
                        fallback_odds.get((key, market)),
                        snapshot_fallback_odds.get((key, market)),
                    ),
                    source_card=card_dir.name,
                    event_id=event_id,
                    event_name=event_name,
                    meta=meta,
                )
                if resolved_rows:
                    fallback_fights += int(used_fallback)
                    fallback_rows += sum(int(item["odds_is_fallback"]) for item in resolved_rows)
                    card_rows.extend(resolved_rows)
                else:
                    skipped_missing_odds += missing_count

        archive_rows.extend(card_rows)
        summary_rows.append(
            {
                "card": card_dir.name,
                "event_id": event_id,
                "event_name": event_name,
                "rows_written": len(card_rows),
                "fights_written": len(results),
                "push_rows": sum(1 for item in card_rows if str(item.get("actual_result", "")).lower() == "push"),
                "fallback_fights": fallback_fights,
                "fallback_rows": fallback_rows,
                "skipped_missing_odds": skipped_missing_odds,
                "status": "ok" if card_rows else "no_exportable_fights",
            }
        )

    archive = pd.DataFrame(archive_rows)
    if archive.empty:
        archive = pd.DataFrame(columns=ARCHIVE_COLUMNS)
    else:
        archive = archive[ARCHIVE_COLUMNS].sort_values(
            ["start_time", "event_id", "fighter_a", "fighter_b", "market", "selection"],
            na_position="last",
        ).reset_index(drop=True)
        archive["american_odds"] = pd.to_numeric(archive["american_odds"], errors="coerce").astype("Int64")
    summary = pd.DataFrame(summary_rows)
    return archive, summary


def write_historical_archive(
    cards_root: str | Path,
    *,
    output_path: str | Path,
    summary_output_path: str | Path | None = None,
    snapshot_db_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    archive, summary = build_historical_archive(cards_root, snapshot_db_path=snapshot_db_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.to_csv(output, index=False)

    if summary_output_path is not None:
        summary_path = Path(summary_output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return output, archive, summary


def build_historical_moneyline_archive(
    cards_root: str | Path,
    *,
    snapshot_db_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    archive, summary = build_historical_archive(cards_root, snapshot_db_path=snapshot_db_path)
    filtered = archive.loc[archive["market"].astype(str) == "moneyline"].reset_index(drop=True)
    return filtered, summary


def write_historical_moneyline_archive(
    cards_root: str | Path,
    *,
    output_path: str | Path,
    summary_output_path: str | Path | None = None,
    snapshot_db_path: str | Path | None = None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    archive, summary = build_historical_moneyline_archive(cards_root, snapshot_db_path=snapshot_db_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive.to_csv(output, index=False)

    if summary_output_path is not None:
        summary_path = Path(summary_output_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
    return output, archive, summary


def _resolve_market_rows(
    *,
    result_row: dict[str, object],
    market: str,
    selections: tuple[str, ...],
    closing_columns: dict[str, str],
    fallback: dict[str, object] | None,
    source_card: str,
    event_id: str,
    event_name: str,
    meta: dict[str, object],
) -> tuple[list[dict[str, object]], int, bool]:
    resolved_prices: dict[str, int] = {}
    used_fallback = False

    for selection in selections:
        column = closing_columns.get(selection)
        price = _safe_int_or_none(result_row.get(column)) if column else None
        if price is not None:
            resolved_prices[selection] = price

    if len(resolved_prices) != len(selections):
        if fallback is None:
            return [], 1, False
        for selection in selections:
            if selection in resolved_prices:
                continue
            fallback_price = _safe_int_or_none(fallback.get(selection))
            if fallback_price is None:
                return [], 1, False
            resolved_prices[selection] = fallback_price
            used_fallback = True

    result_status = _safe_text(result_row.get("result_status"), "official")
    rows: list[dict[str, object]] = []
    for selection in selections:
        rows.append(
            {
                "event_id": _safe_text(result_row.get("event_id"), event_id),
                "event_name": _safe_text(result_row.get("event_name"), event_name),
                "start_time": _safe_text(meta.get("start_time")),
                "fighter_a": _safe_text(result_row.get("fighter_a")),
                "fighter_b": _safe_text(result_row.get("fighter_b")),
                "market": market,
                "selection": selection,
                "book": _resolve_book_name(fallback, used_fallback),
                "american_odds": resolved_prices[selection],
                "actual_result": _grade_market_selection(result_row, market, selection),
                "source_card": source_card,
                "odds_source": _resolve_odds_source(fallback, used_fallback),
                "odds_is_fallback": int(used_fallback),
                "result_status": result_status,
                "method": _safe_text(result_row.get("method")),
                "scheduled_rounds": _safe_float(meta.get("scheduled_rounds"), pd.NA),
                "is_title_fight": _safe_float(meta.get("is_title_fight"), pd.NA),
            }
        )
    return rows, 0, used_fallback


def _build_fight_meta_lookup(data_dir: Path) -> dict[tuple[str, str, str], dict[str, object]]:
    lookup: dict[tuple[str, str, str], dict[str, object]] = {}
    for filename in ("odds_template.csv", "oddsapi_odds.csv", "bfo_odds.csv"):
        path = data_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            fighter_a = _safe_text(row.get("fighter_a"))
            fighter_b = _safe_text(row.get("fighter_b"))
            event_id = _safe_text(row.get("event_id"))
            if not fighter_a or not fighter_b or not event_id:
                continue
            key = _fight_key(event_id, fighter_a, fighter_b)
            current = lookup.setdefault(key, {})
            for column in ("start_time", "scheduled_rounds", "is_title_fight"):
                value = row.get(column, pd.NA)
                if column not in current or _is_missing(current[column]):
                    if not _is_missing(value):
                        current[column] = value
    return lookup


def _build_fallback_odds_lookup(data_dir: Path) -> dict[tuple[tuple[str, str, str], str], dict[str, object]]:
    lookup: dict[tuple[tuple[str, str, str], str], dict[str, object]] = {}
    for filename in ("oddsapi_odds.csv", "bfo_odds.csv", "odds_template.csv"):
        path = data_dir / filename
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty or "selection" not in frame.columns or "market" not in frame.columns:
            continue
        for key, market_rows in frame.groupby(["event_id", "fighter_a", "fighter_b", "market"], dropna=False):
            working = market_rows.copy()
            working["selection"] = working["selection"].astype(str).str.strip()
            selection_prices: dict[str, int] = {}
            for selection, selection_rows in working.groupby("selection", dropna=False):
                prices = pd.to_numeric(selection_rows["american_odds"], errors="coerce").dropna()
                if prices.empty:
                    continue
                selection_prices[str(selection).strip()] = int(round(float(prices.iloc[0])))
            if not selection_prices:
                continue
            canonical_key = (_fight_key(key[0], key[1], key[2]), _safe_text(key[3]))
            if canonical_key in lookup:
                continue
            books = working.get("book", pd.Series(dtype=object)).fillna(path.stem).astype(str).str.strip()
            unique_books = {book for book in books.tolist() if book}
            lookup[canonical_key] = {
                **selection_prices,
                "book": unique_books.pop() if len(unique_books) == 1 else path.stem,
                "source": path.stem,
            }
    return lookup


def _build_snapshot_odds_lookup(
    snapshot_db_path: str | Path | None,
) -> dict[tuple[tuple[str, str, str], str], dict[str, object]]:
    if snapshot_db_path is None:
        return {}
    db_path = Path(snapshot_db_path)
    if not db_path.exists():
        return {}

    frame = load_snapshot_history(db_path)
    required_columns = {"event_id", "fighter_a", "fighter_b", "market", "selection", "american_odds"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return {}

    working = frame.copy()
    working["american_odds"] = pd.to_numeric(working["american_odds"], errors="coerce")
    working = working.loc[working["american_odds"].notna()].copy()
    if working.empty:
        return {}

    if "snapshot_time" in working.columns:
        working["snapshot_sort_key"] = pd.to_datetime(working["snapshot_time"], errors="coerce")
    else:
        working["snapshot_sort_key"] = pd.NaT
    if "snapshot_id" in working.columns:
        working["snapshot_id"] = pd.to_numeric(working["snapshot_id"], errors="coerce").fillna(-1)
    else:
        working["snapshot_id"] = pd.RangeIndex(start=0, stop=len(working), step=1)

    latest_rows = (
        working.sort_values(["snapshot_sort_key", "snapshot_id"], na_position="last")
        .groupby(["event_id", "fighter_a", "fighter_b", "market", "selection"], dropna=False, as_index=False)
        .tail(1)
    )

    lookup: dict[tuple[tuple[str, str, str], str], dict[str, object]] = {}
    for key, market_rows in latest_rows.groupby(["event_id", "fighter_a", "fighter_b", "market"], dropna=False):
        selection_prices: dict[str, int] = {}
        for row in market_rows.to_dict(orient="records"):
            selection = _safe_text(row.get("selection"))
            price = _safe_int_or_none(row.get("american_odds"))
            if selection and price is not None:
                selection_prices[selection] = price
        if not selection_prices:
            continue

        books = market_rows.get("book", pd.Series(dtype=object)).fillna("snapshot_db").astype(str).str.strip()
        unique_books = {book for book in books.tolist() if book}
        lookup[(_fight_key(key[0], key[1], key[2]), _safe_text(key[3]))] = {
            **selection_prices,
            "book": unique_books.pop() if len(unique_books) == 1 else "snapshot_db",
            "source": "snapshot_db",
        }
    return lookup


def _merge_fallback_sources(*sources: dict[str, object] | None) -> dict[str, object] | None:
    merged: dict[str, object] = {}
    source_name = ""
    book_name = ""

    for source in sources:
        if source is None:
            continue
        for key, value in source.items():
            if key in {"book", "source"}:
                continue
            if key in merged:
                continue
            price = _safe_int_or_none(value)
            if price is not None:
                merged[key] = price
        if not book_name:
            book_name = _safe_text(source.get("book"))
        if not source_name:
            source_name = _safe_text(source.get("source"))

    if not merged:
        return None

    merged["book"] = book_name or source_name or "fallback"
    merged["source"] = source_name or book_name or "fallback"
    return merged


def _grade_market_selection(result_row: dict[str, object], market: str, selection: str) -> str:
    winner_side = _safe_text(result_row.get("winner_side")).lower()
    result_status = _safe_text(result_row.get("result_status")).lower()
    went_decision = int(_safe_float(result_row.get("went_decision"), 0.0) or 0)
    ended_inside_distance = int(_safe_float(result_row.get("ended_inside_distance"), 0.0) or 0)

    if result_status in {"draw", "majority draw", "split draw", "no contest", "nc"} or winner_side in {"draw", "no_contest"}:
        return "push"
    if market == "moneyline":
        return "win" if winner_side == selection else "loss"
    if market == "fight_goes_to_decision":
        return "win" if went_decision == 1 else "loss"
    if market == "fight_doesnt_go_to_decision":
        return "win" if ended_inside_distance == 1 else "loss"
    if market == "inside_distance":
        return "win" if winner_side == selection and ended_inside_distance == 1 else "loss"
    if market == "by_decision":
        return "win" if winner_side == selection and went_decision == 1 else "loss"
    return "loss"


def _resolve_book_name(fallback: dict[str, object] | None, used_fallback: bool) -> str:
    if used_fallback and fallback is not None:
        return _safe_text(fallback.get("book"), _safe_text(fallback.get("source"), "fallback"))
    return "results_close"


def _resolve_odds_source(fallback: dict[str, object] | None, used_fallback: bool) -> str:
    if used_fallback and fallback is not None:
        return _safe_text(fallback.get("source"), _safe_text(fallback.get("book"), "fallback"))
    return "results_close"


def _fight_key(event_id: object, fighter_a: object, fighter_b: object) -> tuple[str, str, str]:
    return (_safe_text(event_id), _safe_text(fighter_a), _safe_text(fighter_b))


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _safe_float(value: object, default: object = 0.0) -> object:
    if _is_missing(value):
        return default
    return float(value)


def _safe_int_or_none(value: object) -> int | None:
    if _is_missing(value):
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return int(round(float(numeric)))


def _is_missing(value: object) -> bool:
    return bool(pd.isna(value) or str(value).strip() == "")
