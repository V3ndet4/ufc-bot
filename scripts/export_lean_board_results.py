from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtests.grading import normalize_name


PUSH_RESULT_STATUSES = {
    "draw",
    "majority draw",
    "split draw",
    "no contest",
    "nc",
    "replacement_opponent",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export post-card lean-board grading against settled results.")
    parser.add_argument("--lean-board", required=True, help="Input lean_board.csv path.")
    parser.add_argument("--results", required=True, help="Input results.csv path.")
    parser.add_argument("--output", required=True, help="Output CSV path for per-lean results.")
    parser.add_argument("--summary-output", help="Optional output CSV path for a lean-board summary.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _split_fight_label(value: object) -> tuple[str, str]:
    label = _safe_text(value)
    if " vs " not in label:
        return label, ""
    fighter_a, fighter_b = label.split(" vs ", 1)
    return fighter_a.strip(), fighter_b.strip()


def _grade_lean_result(row: pd.Series) -> str:
    result_status = _safe_text(row.get("result_status")).lower()
    winner_name = _safe_text(row.get("winner_name"))
    lean_side = _safe_text(row.get("lean_side"))

    if not result_status and not winner_name:
        return "pending"
    if result_status in PUSH_RESULT_STATUSES:
        return "push"
    if winner_name and normalize_name(winner_name) == normalize_name(lean_side):
        return "win"
    if winner_name:
        return "loss"
    return "pending"


def _lean_fighter_actual_outcome(row: pd.Series) -> str:
    lean_side = normalize_name(_safe_text(row.get("lean_side")))
    actual_fighter_a = normalize_name(_safe_text(row.get("actual_fighter_a")))
    actual_fighter_b = normalize_name(_safe_text(row.get("actual_fighter_b")))
    actual_winner_name = normalize_name(_safe_text(row.get("actual_winner_name")) or _safe_text(row.get("winner_name")))

    actual_names = {name for name in [actual_fighter_a, actual_fighter_b] if name}
    if not actual_names and not actual_winner_name:
        return "pending"
    if lean_side not in actual_names:
        return "not_on_final_card"
    if not actual_winner_name:
        return "pending"
    if lean_side == actual_winner_name:
        return "won"
    return "lost"


def _grading_note(row: pd.Series) -> str:
    match_status = _safe_text(row.get("result_match_status"))
    actual_outcome = _safe_text(row.get("lean_fighter_actual_outcome"))
    if match_status == "replacement_opponent":
        if actual_outcome == "not_on_final_card":
            return "Opponent changed after board build; the original selected fighter was not on the final card, so treat this as void."
        return "Opponent changed after board build; the original wager should be treated as void."
    if _safe_text(row.get("graded_result")) == "pending":
        return "No settled result matched this lean."
    return ""


def build_lean_board_results(lean_board: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_name",
        "fight",
        "actual_fight",
        "lean_side",
        "opponent_side",
        "lean_strength",
        "lean_action",
        "current_american_odds",
        "result_match_status",
        "result_status",
        "winner_name",
        "actual_winner_name",
        "graded_result",
        "lean_fighter_actual_outcome",
        "grading_note",
    ]
    if lean_board.empty:
        return pd.DataFrame(columns=columns)

    working = lean_board.copy()
    manifest_pairs = working.get("fight", pd.Series("", index=working.index)).apply(_split_fight_label)
    working["manifest_fighter_a"] = manifest_pairs.apply(lambda pair: pair[0])
    working["manifest_fighter_b"] = manifest_pairs.apply(lambda pair: pair[1])

    result_frame = results.copy()
    if "actual_fighter_a" not in result_frame.columns:
        result_frame["actual_fighter_a"] = result_frame.get("fighter_a", pd.Series("", index=result_frame.index))
    if "actual_fighter_b" not in result_frame.columns:
        result_frame["actual_fighter_b"] = result_frame.get("fighter_b", pd.Series("", index=result_frame.index))
    if "actual_winner_name" not in result_frame.columns:
        result_frame["actual_winner_name"] = result_frame.get("winner_name", pd.Series("", index=result_frame.index))
    if "result_match_status" not in result_frame.columns:
        result_frame["result_match_status"] = "exact"

    result_frame["fight"] = (
        result_frame.get("fighter_a", pd.Series("", index=result_frame.index)).fillna("").astype(str).str.strip()
        + " vs "
        + result_frame.get("fighter_b", pd.Series("", index=result_frame.index)).fillna("").astype(str).str.strip()
    )
    result_frame["actual_fight"] = (
        result_frame["actual_fighter_a"].fillna("").astype(str).str.strip()
        + " vs "
        + result_frame["actual_fighter_b"].fillna("").astype(str).str.strip()
    )

    merged = working.merge(
        result_frame[
            [
                "fight",
                "actual_fight",
                "actual_fighter_a",
                "actual_fighter_b",
                "winner_name",
                "actual_winner_name",
                "result_status",
                "result_match_status",
            ]
        ],
        on="fight",
        how="left",
    )
    merged["result_match_status"] = merged["result_match_status"].fillna("unmatched").astype(str)
    merged["graded_result"] = merged.apply(_grade_lean_result, axis=1)
    merged["lean_fighter_actual_outcome"] = merged.apply(_lean_fighter_actual_outcome, axis=1)
    merged["grading_note"] = merged.apply(_grading_note, axis=1)
    return merged[columns].copy()


def build_lean_postmortem_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "bucket",
        "picks",
        "wins",
        "losses",
        "pushes",
        "pending",
        "actual_fighter_wins",
        "actual_fighter_losses",
        "not_on_final_card",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    buckets = [("all", frame)] + [
        (str(bucket), bucket_rows.copy())
        for bucket, bucket_rows in frame.groupby("result_match_status", dropna=False)
    ]
    for bucket_name, bucket_rows in buckets:
        graded = bucket_rows.get("graded_result", pd.Series("", index=bucket_rows.index)).astype(str)
        actual = bucket_rows.get("lean_fighter_actual_outcome", pd.Series("", index=bucket_rows.index)).astype(str)
        rows.append(
            {
                "bucket": bucket_name,
                "picks": int(len(bucket_rows)),
                "wins": int((graded == "win").sum()),
                "losses": int((graded == "loss").sum()),
                "pushes": int((graded == "push").sum()),
                "pending": int((graded == "pending").sum()),
                "actual_fighter_wins": int((actual == "won").sum()),
                "actual_fighter_losses": int((actual == "lost").sum()),
                "not_on_final_card": int((actual == "not_on_final_card").sum()),
            }
        )
    summary = pd.DataFrame(rows)
    bucket_order = {"all": 0, "exact": 1, "replacement_opponent": 2, "unmatched": 3}
    summary["_bucket_order"] = summary["bucket"].map(bucket_order).fillna(99)
    return summary.sort_values(by=["_bucket_order", "bucket"]).drop(columns=["_bucket_order"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    lean_board = pd.read_csv(args.lean_board)
    results = pd.read_csv(args.results)
    report = build_lean_board_results(lean_board, results)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    if args.summary_output:
        summary = build_lean_postmortem_summary(report)
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False)
        if not args.quiet:
            print(f"Saved lean postmortem summary to {summary_path}")

    if not args.quiet:
        print(f"Saved lean board results to {output_path}")


if __name__ == "__main__":
    main()
