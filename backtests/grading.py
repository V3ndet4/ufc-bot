from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

from models.ev import american_to_decimal, implied_probability


@dataclass(frozen=True)
class TrackedExpression:
    market_key: str
    selection_key: str


def normalize_name(value: object) -> str:
    ascii_text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold().replace("’", "'"))
    return " ".join(normalized.split())


def fight_key(fighter_a: object, fighter_b: object) -> str:
    names = sorted([normalize_name(fighter_a), normalize_name(fighter_b)])
    return "||".join(names)


def infer_tracked_expression(row: pd.Series) -> TrackedExpression:
    expression = str(row.get("chosen_value_expression") or row.get("selection_name") or "").strip()
    fighter_a = str(row.get("fighter_a", "")).strip()
    fighter_b = str(row.get("fighter_b", "")).strip()

    if expression == "Fight goes to decision":
        return TrackedExpression("fight_goes_to_decision", "fight_goes_to_decision")
    if expression == "Fight doesn't go to decision":
        return TrackedExpression("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision")
    if expression == f"{fighter_a} inside distance":
        return TrackedExpression("inside_distance", "fighter_a")
    if expression == f"{fighter_b} inside distance":
        return TrackedExpression("inside_distance", "fighter_b")
    if expression == f"{fighter_a} by submission":
        return TrackedExpression("submission", "fighter_a")
    if expression == f"{fighter_b} by submission":
        return TrackedExpression("submission", "fighter_b")
    if expression == f"{fighter_a} by KO/TKO":
        return TrackedExpression("ko_tko", "fighter_a")
    if expression == f"{fighter_b} by KO/TKO":
        return TrackedExpression("ko_tko", "fighter_b")
    if expression == "Fight ends by submission":
        return TrackedExpression("fight_ends_by_submission", "fight_ends_by_submission")
    if expression == "Fight ends by KO/TKO":
        return TrackedExpression("fight_ends_by_ko_tko", "fight_ends_by_ko_tko")
    if expression == f"{fighter_a} knockdown":
        return TrackedExpression("knockdown", "fighter_a")
    if expression == f"{fighter_b} knockdown":
        return TrackedExpression("knockdown", "fighter_b")
    if expression == f"{fighter_a} takedown":
        return TrackedExpression("takedown", "fighter_a")
    if expression == f"{fighter_b} takedown":
        return TrackedExpression("takedown", "fighter_b")
    if expression == f"{fighter_a} by decision":
        return TrackedExpression("by_decision", "fighter_a")
    if expression == f"{fighter_b} by decision":
        return TrackedExpression("by_decision", "fighter_b")
    if expression == fighter_a:
        return TrackedExpression("moneyline", "fighter_a")
    if expression == fighter_b:
        return TrackedExpression("moneyline", "fighter_b")
    return TrackedExpression(str(row.get("market", "moneyline") or "moneyline"), str(row.get("selection", "fighter_a") or "fighter_a"))


def attach_tracked_expression_columns(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    expressions = enriched.apply(infer_tracked_expression, axis=1)
    enriched["tracked_market_key"] = expressions.apply(lambda value: value.market_key)
    enriched["tracked_selection_key"] = expressions.apply(lambda value: value.selection_key)
    enriched["fight_key"] = enriched.apply(lambda row: fight_key(row["fighter_a"], row["fighter_b"]), axis=1)
    return enriched


def _lookup_closing_odds(result_row: pd.Series, market_key: str, selection_key: str) -> float | pd.NA:
    column_map = {
        ("moneyline", "fighter_a"): "closing_fighter_a_odds",
        ("moneyline", "fighter_b"): "closing_fighter_b_odds",
        ("fight_goes_to_decision", "fight_goes_to_decision"): "closing_fight_goes_to_decision_odds",
        ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision"): "closing_fight_doesnt_go_to_decision_odds",
        ("inside_distance", "fighter_a"): "closing_fighter_a_inside_distance_odds",
        ("inside_distance", "fighter_b"): "closing_fighter_b_inside_distance_odds",
        ("by_decision", "fighter_a"): "closing_fighter_a_by_decision_odds",
        ("by_decision", "fighter_b"): "closing_fighter_b_by_decision_odds",
        ("submission", "fighter_a"): "closing_fighter_a_submission_odds",
        ("submission", "fighter_b"): "closing_fighter_b_submission_odds",
        ("ko_tko", "fighter_a"): "closing_fighter_a_ko_tko_odds",
        ("ko_tko", "fighter_b"): "closing_fighter_b_ko_tko_odds",
        ("fight_ends_by_submission", "fight_ends_by_submission"): "closing_fight_ends_by_submission_odds",
        ("fight_ends_by_ko_tko", "fight_ends_by_ko_tko"): "closing_fight_ends_by_ko_tko_odds",
        ("knockdown", "fighter_a"): "closing_fighter_a_knockdown_odds",
        ("knockdown", "fighter_b"): "closing_fighter_b_knockdown_odds",
        ("takedown", "fighter_a"): "closing_fighter_a_takedown_odds",
        ("takedown", "fighter_b"): "closing_fighter_b_takedown_odds",
    }
    column = column_map.get((market_key, selection_key))
    if not column or column not in result_row.index:
        return pd.NA
    return result_row[column]


def _grade_pick(result_row: pd.Series, market_key: str, selection_key: str) -> str:
    winner_side = str(result_row.get("winner_side", "") or "").strip()
    result_status = str(result_row.get("result_status", "") or "").strip().lower()
    went_decision = int(result_row.get("went_decision", 0) or 0)
    ended_inside_distance = int(result_row.get("ended_inside_distance", 0) or 0)

    if result_status in {"draw", "majority draw", "split draw", "no contest", "nc", "replacement_opponent"} or winner_side in {"draw", "no_contest"}:
        return "push"
    if market_key == "moneyline":
        return "win" if winner_side == selection_key else "loss"
    if market_key == "fight_goes_to_decision":
        return "win" if went_decision == 1 else "loss"
    if market_key == "fight_doesnt_go_to_decision":
        return "win" if ended_inside_distance == 1 else "loss"
    if market_key == "inside_distance":
        return "win" if winner_side == selection_key and ended_inside_distance == 1 else "loss"
    if market_key == "submission":
        return "win" if winner_side == selection_key and _method_is_submission(result_row.get("method")) else "loss"
    if market_key == "ko_tko":
        return "win" if winner_side == selection_key and _method_is_ko_tko(result_row.get("method")) else "loss"
    if market_key == "fight_ends_by_submission":
        return "win" if _method_is_submission(result_row.get("method")) and result_status == "official" else "loss"
    if market_key == "fight_ends_by_ko_tko":
        return "win" if _method_is_ko_tko(result_row.get("method")) and result_status == "official" else "loss"
    if market_key == "knockdown":
        count = _side_count(result_row, selection_key, "knockdowns")
        if pd.isna(count):
            return "pending"
        return "win" if float(count) >= 1.0 else "loss"
    if market_key == "takedown":
        count = _side_count(result_row, selection_key, "takedowns")
        if pd.isna(count):
            return "pending"
        return "win" if float(count) >= 1.0 else "loss"
    if market_key == "by_decision":
        return "win" if winner_side == selection_key and went_decision == 1 else "loss"
    return "loss"


def _method_is_submission(method: object) -> bool:
    normalized = str(method or "").strip().lower()
    return "sub" in normalized


def _method_is_ko_tko(method: object) -> bool:
    normalized = str(method or "").strip().lower()
    if "sub" in normalized:
        return False
    return "ko" in normalized or "tko" in normalized or "knockout" in normalized


def _side_count(result_row: pd.Series, selection_key: str, suffix: str) -> float | pd.NA:
    column = f"{selection_key}_{suffix}"
    if column not in result_row.index:
        return pd.NA
    value = pd.to_numeric(pd.Series([result_row.get(column)]), errors="coerce").iloc[0]
    return value if pd.notna(value) else pd.NA


def grade_tracked_picks(picks: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if picks.empty:
        return picks.copy()

    graded = attach_tracked_expression_columns(picks)
    normalized_results = results.copy()
    normalized_results["fight_key"] = normalized_results.apply(
        lambda row: fight_key(row["fighter_a"], row["fighter_b"]),
        axis=1,
    )
    result_lookup = normalized_results.drop_duplicates(subset=["event_id", "fight_key"]).set_index(["event_id", "fight_key"])

    actual_results: list[str] = []
    closing_odds: list[float | pd.NA] = []
    clv_deltas: list[float | pd.NA] = []
    clv_edges: list[float | pd.NA] = []
    profits: list[float] = []

    for row in graded.to_dict("records"):
        result_row = result_lookup.loc[(row["event_id"], row["fight_key"])] if (row["event_id"], row["fight_key"]) in result_lookup.index else None
        if result_row is None:
            actual_results.append("pending")
            closing_odds.append(pd.NA)
            clv_deltas.append(pd.NA)
            clv_edges.append(pd.NA)
            profits.append(0.0)
            continue

        result_series = result_row if isinstance(result_row, pd.Series) else result_row.iloc[0]
        verdict = _grade_pick(result_series, str(row["tracked_market_key"]), str(row["tracked_selection_key"]))
        actual_results.append(verdict)

        close_odds = _lookup_closing_odds(result_series, str(row["tracked_market_key"]), str(row["tracked_selection_key"]))
        closing_odds.append(close_odds)
        if pd.notna(close_odds):
            clv_delta = float(int(close_odds) - int(row["chosen_expression_odds"]))
            clv_deltas.append(clv_delta)
            clv_edges.append(float(row.get("chosen_expression_prob", row.get("model_projected_win_prob", 0.0))) - implied_probability(int(close_odds)))
        else:
            clv_deltas.append(pd.NA)
            clv_edges.append(pd.NA)

        stake = float(row.get("chosen_expression_stake", row.get("suggested_stake", 0.0)) or 0.0)
        if verdict == "win":
            profits.append(round(stake * (american_to_decimal(int(row["chosen_expression_odds"])) - 1), 2))
        elif verdict == "push":
            profits.append(0.0)
        else:
            profits.append(round(-stake, 2))

    graded["actual_result"] = actual_results
    graded["closing_american_odds"] = closing_odds
    graded["clv_delta"] = clv_deltas
    graded["clv_edge"] = clv_edges
    graded["profit"] = profits
    graded["grade_status"] = graded["actual_result"].replace({"pending": "pending"}).where(
        graded["actual_result"] == "pending", "graded"
    )
    return graded
