from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.storage import load_tracked_picks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a learning report from tracked picks.")
    parser.add_argument("--db", default="data/ufc_betting.db", help="SQLite database path.")
    parser.add_argument("--event-id", help="Optional event id filter.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _american_to_decimal(odds: object) -> float | pd.NA:
    if pd.isna(odds):
        return pd.NA
    value = int(float(odds))
    if value > 0:
        return round((value / 100) + 1, 2)
    return round((100 / abs(value)) + 1, 2)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _confidence_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.75:
        return "0.75_plus"
    if numeric >= 0.65:
        return "0.65_to_0.74"
    if numeric >= 0.55:
        return "0.55_to_0.64"
    return "below_0.55"


def _edge_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.10:
        return "0.10_plus"
    if numeric >= 0.05:
        return "0.05_to_0.099"
    if numeric >= 0.03:
        return "0.03_to_0.049"
    return "below_0.03"


def _data_quality_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.90:
        return "0.90_plus"
    if numeric >= 0.80:
        return "0.80_to_0.89"
    return "below_0.80"


def _favorite_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric < 0:
        return "favorite"
    if numeric > 0:
        return "underdog"
    return "pickem"


def _line_movement_bucket(value: object) -> str:
    numeric = _coerce_numeric(pd.Series([value])).iloc[0]
    if pd.isna(numeric):
        return "unknown"
    if numeric >= 0.02:
        return "toward_pick"
    if numeric <= -0.02:
        return "against_pick"
    return "flat"


def enrich_with_feedback_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    enriched = frame.copy()
    enriched["tracked_market_bucket"] = enriched.get("tracked_market_key", enriched.get("market", pd.Series("", index=enriched.index))).fillna("unknown")
    enriched["tier_bucket"] = enriched.get("recommended_tier", pd.Series("unknown", index=enriched.index)).fillna("unknown").astype(str)
    enriched["confidence_bucket"] = enriched.get("model_confidence", pd.Series(pd.NA, index=enriched.index)).apply(_confidence_bucket)
    enriched["edge_bucket"] = enriched.get("chosen_expression_edge", enriched.get("edge", pd.Series(pd.NA, index=enriched.index))).apply(_edge_bucket)
    enriched["data_quality_bucket"] = enriched.get("selection_stats_completeness", enriched.get("data_quality", pd.Series(pd.NA, index=enriched.index))).apply(_data_quality_bucket)
    enriched["price_bucket"] = enriched.get("chosen_expression_odds", enriched.get("american_odds", pd.Series(pd.NA, index=enriched.index))).apply(_favorite_bucket)
    enriched["line_movement_bucket"] = enriched.get("line_movement_toward_fighter", pd.Series(pd.NA, index=enriched.index)).apply(_line_movement_bucket)
    fallback_series = enriched.get("selection_fallback_used", pd.Series(pd.NA, index=enriched.index))
    enriched["fallback_bucket"] = fallback_series.apply(
        lambda value: "fallback_used" if pd.notna(value) and float(value or 0.0) >= 1.0 else "full_stats"
    )
    return enriched


def build_learning_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "fight",
                "bet",
                "tracked_at",
                "line_at_pick",
                "decimal_line_at_pick",
                "closing_line",
                "decimal_closing_line",
                "clv_delta",
                "model_prob",
                "implied_prob_at_pick",
                "edge_at_pick",
                "confidence_at_pick",
                "data_quality_at_pick",
                "stats_completeness_at_pick",
                "line_movement_bucket",
                "tier_at_pick",
                "stake",
                "actual_result",
                "profit",
                "roi_pct",
                "grade_status",
            ]
        )

    report = enrich_with_feedback_buckets(frame)
    report["fight"] = report["fighter_a"].astype(str) + " vs " + report["fighter_b"].astype(str)
    report["bet"] = report["chosen_value_expression"].fillna(report["selection_name"])
    report["line_at_pick"] = report["chosen_expression_odds"].fillna(report["american_odds"])
    report["decimal_line_at_pick"] = report["line_at_pick"].apply(_american_to_decimal)
    report["closing_line"] = report["closing_american_odds"]
    report["decimal_closing_line"] = report["closing_line"].apply(_american_to_decimal)
    report["model_prob"] = report["chosen_expression_prob"].fillna(report["model_projected_win_prob"])
    report["implied_prob_at_pick"] = report["chosen_expression_implied_prob"].fillna(report["implied_prob"])
    report["edge_at_pick"] = report["chosen_expression_edge"].fillna(report["edge"])
    report["confidence_at_pick"] = _coerce_numeric(report.get("model_confidence", pd.Series(pd.NA, index=report.index))).round(4)
    report["data_quality_at_pick"] = _coerce_numeric(report.get("data_quality", pd.Series(pd.NA, index=report.index))).round(4)
    report["stats_completeness_at_pick"] = _coerce_numeric(report.get("selection_stats_completeness", pd.Series(pd.NA, index=report.index))).round(4)
    report["tier_at_pick"] = report.get("recommended_tier", pd.Series("", index=report.index))
    report["stake"] = report["chosen_expression_stake"].fillna(report["suggested_stake"]).fillna(0.0)
    report["roi_pct"] = report.apply(
        lambda row: round((float(row["profit"]) / float(row["stake"])) * 100, 2)
        if float(row.get("stake", 0.0) or 0.0) > 0
        else 0.0,
        axis=1,
    )
    columns = [
        "event_id",
        "event_name",
        "fight",
        "bet",
        "tracked_at",
        "line_at_pick",
        "decimal_line_at_pick",
        "closing_line",
        "decimal_closing_line",
        "clv_delta",
        "model_prob",
        "implied_prob_at_pick",
        "edge_at_pick",
        "confidence_at_pick",
        "data_quality_at_pick",
        "stats_completeness_at_pick",
        "line_movement_bucket",
        "tier_at_pick",
        "stake",
        "actual_result",
        "profit",
        "roi_pct",
        "grade_status",
    ]
    return report[columns].sort_values(by=["tracked_at", "fight"]).reset_index(drop=True)


def build_learning_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_name",
                "bets",
                "graded_bets",
                "wins",
                "losses",
                "pending",
                "total_stake",
                "total_profit",
                "roi_pct",
                "avg_edge_at_pick",
                "avg_clv_delta",
            ]
        )

    working = build_learning_report(frame)
    grouped_rows: list[dict[str, object]] = []
    for (event_id, event_name), event_rows in working.groupby(["event_id", "event_name"], dropna=False):
        graded_rows = event_rows.loc[event_rows["grade_status"].astype(str) != "pending"].copy()
        wins = int((graded_rows["actual_result"].astype(str) == "win").sum()) if not graded_rows.empty else 0
        losses = int((graded_rows["actual_result"].astype(str) == "loss").sum()) if not graded_rows.empty else 0
        pending = int((event_rows["grade_status"].astype(str) == "pending").sum())
        total_stake = float(pd.to_numeric(graded_rows["stake"], errors="coerce").fillna(0.0).sum())
        total_profit = float(pd.to_numeric(graded_rows["profit"], errors="coerce").fillna(0.0).sum())
        avg_edge_at_pick = pd.to_numeric(event_rows["edge_at_pick"], errors="coerce").dropna().mean()
        avg_clv_delta = pd.to_numeric(graded_rows["clv_delta"], errors="coerce").dropna().mean()
        grouped_rows.append(
            {
                "event_id": event_id,
                "event_name": event_name,
                "bets": int(len(event_rows)),
                "graded_bets": int(len(graded_rows)),
                "wins": wins,
                "losses": losses,
                "pending": pending,
                "total_stake": round(total_stake, 2),
                "total_profit": round(total_profit, 2),
                "roi_pct": round((total_profit / total_stake) * 100, 2) if total_stake > 0 else 0.0,
                "avg_edge_at_pick": round(0.0 if pd.isna(avg_edge_at_pick) else float(avg_edge_at_pick), 4),
                "avg_clv_delta": round(0.0 if pd.isna(avg_clv_delta) else float(avg_clv_delta), 4),
            }
        )
    return pd.DataFrame(grouped_rows).sort_values(by=["event_name", "event_id"]).reset_index(drop=True)


def build_filter_performance_report(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "dimension",
                "bucket",
                "bets",
                "graded_bets",
                "wins",
                "losses",
                "pushes",
                "pending",
                "total_stake",
                "total_profit",
                "roi_pct",
                "avg_edge_at_pick",
                "avg_clv_delta",
                "recommendation",
            ]
        )

    working = enrich_with_feedback_buckets(frame)
    working["edge_at_pick"] = _coerce_numeric(working.get("chosen_expression_edge", working.get("edge", pd.Series(pd.NA, index=working.index))))
    working["stake_at_pick"] = _coerce_numeric(working.get("chosen_expression_stake", working.get("suggested_stake", pd.Series(0.0, index=working.index)))).fillna(0.0)
    working["profit"] = _coerce_numeric(working.get("profit", pd.Series(0.0, index=working.index))).fillna(0.0)
    working["clv_delta"] = _coerce_numeric(working.get("clv_delta", pd.Series(pd.NA, index=working.index)))
    working["is_pending"] = working.get("grade_status", pd.Series("pending", index=working.index)).fillna("pending").astype(str) == "pending"
    dimensions = [
        ("market", "tracked_market_bucket"),
        ("tier", "tier_bucket"),
        ("confidence", "confidence_bucket"),
        ("edge", "edge_bucket"),
        ("data_quality", "data_quality_bucket"),
        ("price_side", "price_bucket"),
        ("line_movement", "line_movement_bucket"),
        ("fallback", "fallback_bucket"),
    ]

    rows: list[dict[str, object]] = []
    for dimension, bucket_column in dimensions:
        for bucket, bucket_rows in working.groupby(bucket_column, dropna=False):
            graded_rows = bucket_rows.loc[~bucket_rows["is_pending"]].copy()
            wins = int((graded_rows.get("actual_result", pd.Series("", index=graded_rows.index)).astype(str) == "win").sum())
            losses = int((graded_rows.get("actual_result", pd.Series("", index=graded_rows.index)).astype(str) == "loss").sum())
            pushes = int((graded_rows.get("actual_result", pd.Series("", index=graded_rows.index)).astype(str) == "push").sum())
            total_stake = float(graded_rows["stake_at_pick"].sum())
            total_profit = float(graded_rows["profit"].sum())
            avg_edge = graded_rows["edge_at_pick"].dropna().mean()
            avg_clv = graded_rows["clv_delta"].dropna().mean()
            roi_pct = round((total_profit / total_stake) * 100, 2) if total_stake > 0 else 0.0
            recommendation = "needs_more_data"
            if len(graded_rows) >= 3:
                if roi_pct >= 5 and (pd.isna(avg_clv) or float(avg_clv) >= 0):
                    recommendation = "keep_or_expand"
                elif roi_pct < 0 or (pd.notna(avg_clv) and float(avg_clv) < 0):
                    recommendation = "tighten_filter"
                else:
                    recommendation = "monitor"
            rows.append(
                {
                    "dimension": dimension,
                    "bucket": str(bucket),
                    "bets": int(len(bucket_rows)),
                    "graded_bets": int(len(graded_rows)),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "pending": int(bucket_rows["is_pending"].sum()),
                    "total_stake": round(total_stake, 2),
                    "total_profit": round(total_profit, 2),
                    "roi_pct": roi_pct,
                    "avg_edge_at_pick": round(0.0 if pd.isna(avg_edge) else float(avg_edge), 4),
                    "avg_clv_delta": round(0.0 if pd.isna(avg_clv) else float(avg_clv), 4),
                    "recommendation": recommendation,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(by=["dimension", "bucket"])
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    tracked = load_tracked_picks(args.db, event_id=args.event_id)
    report = build_learning_report(tracked)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    if not args.quiet:
        print(f"Saved learning report to {output_path}")


if __name__ == "__main__":
    main()
