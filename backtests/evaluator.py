from __future__ import annotations

from pathlib import Path

import pandas as pd

from bankroll.sizing import suggested_stake
from models.ev import american_to_decimal, implied_probability


def evaluate_backtest(
    selections: pd.DataFrame,
    min_edge: float,
    bankroll: float,
    fractional_kelly: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    report = selections.copy()
    report["implied_prob"] = report["american_odds"].apply(implied_probability)
    report["edge"] = report["model_projected_win_prob"] - report["implied_prob"]
    report = report.loc[report["edge"] >= min_edge].copy()
    if report.empty:
        return report, {
            "picks": 0,
            "wins": 0,
            "losses": 0,
            "total_staked": 0.0,
            "total_profit": 0.0,
            "roi": 0.0,
        }

    report["stake"] = report.apply(
        lambda row: suggested_stake(
            bankroll=bankroll,
            projected_win_prob=row["model_projected_win_prob"],
            american_odds=int(row["american_odds"]),
            fraction=fractional_kelly,
        ),
        axis=1,
    )
    report["decimal_odds"] = report["american_odds"].apply(american_to_decimal)
    report["profit"] = report.apply(
        lambda row: round(row["stake"] * (row["decimal_odds"] - 1), 2)
        if row["actual_result"] == "win"
        else 0.0
        if row["actual_result"] == "push"
        else round(-row["stake"], 2),
        axis=1,
    )

    wins = int((report["actual_result"] == "win").sum())
    losses = int((report["actual_result"] == "loss").sum())
    pushes = int((report["actual_result"] == "push").sum())
    total_staked = round(float(report["stake"].sum()), 2)
    total_profit = round(float(report["profit"].sum()), 2)
    roi = round((total_profit / total_staked), 4) if total_staked else 0.0

    summary = {
        "picks": int(len(report)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "total_staked": total_staked,
        "total_profit": total_profit,
        "roi": roi,
    }
    if "recommended_tier" in report.columns:
        for tier in ["A", "B", "C"]:
            tier_rows = report.loc[report["recommended_tier"].astype(str) == tier]
            tier_staked = float(tier_rows["stake"].sum()) if not tier_rows.empty else 0.0
            tier_profit = float(tier_rows["profit"].sum()) if not tier_rows.empty else 0.0
            summary[f"{tier}_picks"] = int(len(tier_rows))
            summary[f"{tier}_wins"] = int((tier_rows["actual_result"] == "win").sum()) if not tier_rows.empty else 0
            summary[f"{tier}_roi"] = round((tier_profit / tier_staked), 4) if tier_staked else 0.0
    return report, summary


def write_summary_csv(summary: dict[str, float | int], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(path, index=False)
