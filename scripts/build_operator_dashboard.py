from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an operator dashboard with exposure and lean panels.")
    parser.add_argument("--fight-report", required=True, help="Fight-week report CSV path.")
    parser.add_argument("--lean-board", required=True, help="Lean-board CSV path.")
    parser.add_argument("--value-report", required=True, help="Value-report CSV path.")
    parser.add_argument("--betting-board", required=True, help="Betting-board CSV path.")
    parser.add_argument("--passes", required=True, help="Pass-reasons CSV path.")
    parser.add_argument("--parlays", help="Optional parlay-board CSV path.")
    parser.add_argument("--output", required=True, help="HTML output path.")
    parser.add_argument("--threshold-policy", help="Optional threshold-policy JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _safe_float(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _safe_text(value: object, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value).strip()


def _load_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def _load_threshold_policy(path: str | Path | None) -> dict[str, object] | None:
    if not path:
        return None
    policy_path = Path(path)
    if not policy_path.exists():
        return None
    return json.loads(policy_path.read_text(encoding="utf-8"))


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _badge(label: str, tone: str) -> str:
    return f'<span class="badge badge-{html.escape(tone)}">{html.escape(label)}</span>'


def _table_html(frame: pd.DataFrame, columns: list[tuple[str, str]]) -> str:
    if frame.empty:
        return '<div class="empty">No rows for this panel.</div>'
    header_html = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body_rows: list[str] = []
    for row in frame.to_dict("records"):
        cells = "".join(f"<td>{html.escape(str(row.get(column, '') or ''))}</td>" for column, _ in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_operator_dashboard_html(
    *,
    fight_report: pd.DataFrame,
    lean_board: pd.DataFrame,
    value_report: pd.DataFrame,
    betting_board: pd.DataFrame,
    passes: pd.DataFrame,
    parlays: pd.DataFrame | None = None,
    threshold_policy: dict[str, object] | None = None,
) -> str:
    event_name = ""
    for frame in [fight_report, value_report, betting_board, lean_board]:
        if not frame.empty and "event_name" in frame.columns:
            event_name = _safe_text(frame.iloc[0].get("event_name", ""))
            if event_name:
                break
    event_name = event_name or "UFC Operator Dashboard"

    working_value = value_report.copy()
    if not working_value.empty:
        working_value["fight"] = working_value["fighter_a"].astype(str) + " vs " + working_value["fighter_b"].astype(str)
        working_value["raw_stake"] = pd.to_numeric(
            working_value.get("raw_chosen_expression_stake", working_value.get("chosen_expression_stake", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        working_value["governed_stake"] = pd.to_numeric(
            working_value.get("chosen_expression_stake", 0.0),
            errors="coerce",
        ).fillna(0.0)
        working_value["trimmed_stake"] = (working_value["raw_stake"] - working_value["governed_stake"]).clip(lower=0.0)
        working_value["effective_edge_numeric"] = pd.to_numeric(
            working_value.get("effective_edge", working_value.get("chosen_expression_edge", 0.0)),
            errors="coerce",
        ).fillna(0.0)
        working_value["model_confidence_numeric"] = pd.to_numeric(
            working_value.get("model_confidence", 0.0),
            errors="coerce",
        ).fillna(0.0)
    actionable = (
        working_value.loc[working_value.get("recommended_action", pd.Series("", index=working_value.index)).isin(["Bettable now", "Watchlist"])]
        .copy()
        if not working_value.empty
        else pd.DataFrame()
    )
    raw_exposure = float(actionable.get("raw_stake", pd.Series(dtype=float)).sum()) if not actionable.empty else 0.0
    governed_exposure = float(actionable.get("governed_stake", pd.Series(dtype=float)).sum()) if not actionable.empty else 0.0
    trimmed_exposure = float(actionable.get("trimmed_stake", pd.Series(dtype=float)).sum()) if not actionable.empty else 0.0
    average_confidence = float(actionable.get("model_confidence_numeric", pd.Series(dtype=float)).mean()) if not actionable.empty else 0.0
    average_edge = float(actionable.get("effective_edge_numeric", pd.Series(dtype=float)).mean()) if not actionable.empty else 0.0
    a_count = int((actionable.get("recommended_tier", pd.Series("", index=actionable.index)).astype(str) == "A").sum()) if not actionable.empty else 0
    b_count = int((actionable.get("recommended_tier", pd.Series("", index=actionable.index)).astype(str) == "B").sum()) if not actionable.empty else 0

    exposure_panel = pd.DataFrame()
    if not actionable.empty:
        grouped = (
            actionable.groupby("fight", as_index=False)
            .agg(
                bets=("chosen_value_expression", "count"),
                raw_stake=("raw_stake", "sum"),
                governed_stake=("governed_stake", "sum"),
                trimmed_stake=("trimmed_stake", "sum"),
                top_risk=("stake_governor_reason", lambda values: ", ".join(sorted({str(value).strip() for value in values if str(value).strip()}))),
            )
            .sort_values(by=["governed_stake", "raw_stake", "fight"], ascending=[False, False, True])
            .reset_index(drop=True)
        )
        grouped["raw_stake"] = grouped["raw_stake"].apply(_format_currency)
        grouped["governed_stake"] = grouped["governed_stake"].apply(_format_currency)
        grouped["trimmed_stake"] = grouped["trimmed_stake"].apply(_format_currency)
        grouped["top_risk"] = grouped["top_risk"].replace("", "none")
        exposure_panel = grouped.head(12)

    plays_panel = pd.DataFrame()
    if not actionable.empty:
        plays_panel = actionable.sort_values(
            by=["recommended_tier", "governed_stake", "effective_edge_numeric"],
            ascending=[True, False, False],
        ).copy()
        plays_panel["bet"] = plays_panel["chosen_value_expression"].map(lambda value: _safe_text(value))
        plays_panel["line"] = plays_panel["chosen_expression_odds"].map(
            lambda value: f"{int(float(value)):+d}" if pd.notna(value) else ""
        )
        plays_panel["edge"] = plays_panel["effective_edge_numeric"].map(_format_pct)
        plays_panel["confidence"] = plays_panel["model_confidence_numeric"].map(lambda value: f"{value:.2f}")
        plays_panel["stake"] = plays_panel["governed_stake"].map(_format_currency)
        plays_panel["notes"] = plays_panel.get("stake_governor_reason", "").map(lambda value: _safe_text(value, "none"))
        plays_panel["risk"] = plays_panel.get("risk_flags", "").map(lambda value: _safe_text(value, "none"))
        plays_panel = plays_panel[["fight", "bet", "line", "edge", "confidence", "recommended_tier", "recommended_action", "stake", "notes", "risk"]].head(12)

    lean_panel = pd.DataFrame()
    if not lean_board.empty:
        lean_panel = lean_board.copy()
        lean_panel["edge"] = pd.to_numeric(lean_panel.get("edge", 0.0), errors="coerce").fillna(0.0).map(_format_pct)
        lean_panel["lean_prob"] = pd.to_numeric(lean_panel.get("lean_prob", 0.0), errors="coerce").fillna(0.0).map(_format_pct)
        lean_panel = lean_panel[["fight", "lean_side", "lean_strength", "lean_action", "edge", "lean_prob", "top_reasons"]].head(8)
        lean_panel = lean_panel.rename(columns={"top_reasons": "reasons"})

    pass_panel = pd.DataFrame()
    if not passes.empty:
        pass_panel = passes.copy().head(8)
        pass_panel["pass_reason"] = pass_panel["pass_reason"].map(lambda value: _safe_text(value))
        pass_panel["risk_flags"] = pass_panel.get("risk_flags", "").map(lambda value: _safe_text(value, "none"))
        pass_panel = pass_panel[["fight", "selection_name", "pass_reason", "risk_flags"]]

    parlay_panel = pd.DataFrame()
    if parlays is not None and not parlays.empty:
        parlay_panel = parlays.copy().head(3)
        parlay_panel = parlay_panel[
            [
                "parlay_name",
                "american_odds",
                "decimal_odds",
                "edge",
                "expected_value",
                "parlay_confidence",
                "legs",
            ]
        ]

    policy_panel = ""
    if threshold_policy:
        selected = threshold_policy.get("selected", {}) if isinstance(threshold_policy.get("selected", {}), dict) else {}
        policy_panel = f"""
        <section class="panel">
          <div class="panel-header">
            <h2>Threshold Policy</h2>
            {_badge(str(threshold_policy.get("status", "unknown")).upper(), "ink")}
          </div>
          <div class="policy-grid">
            <div><span>Min edge</span><strong>{_format_pct(float(selected.get("min_edge", 0.0) or 0.0))}</strong></div>
            <div><span>Min confidence</span><strong>{float(selected.get("min_model_confidence", 0.0) or 0.0):.2f}</strong></div>
            <div><span>Min stats</span><strong>{float(selected.get("min_stats_completeness", 0.0) or 0.0):.2f}</strong></div>
            <div><span>Fallback rows</span><strong>{"excluded" if bool(selected.get("exclude_fallback_rows", False)) else "included"}</strong></div>
            <div><span>Policy sample</span><strong>{int(selected.get("graded_bets", 0) or 0)}</strong></div>
            <div><span>Policy ROI</span><strong>{float(selected.get("roi_pct", 0.0) or 0.0):+.2f}%</strong></div>
          </div>
        </section>
        """

    summary_cards = f"""
    <section class="summary-grid">
      <div class="metric-card">
        <span class="metric-label">Governed Exposure</span>
        <strong>{_format_currency(governed_exposure)}</strong>
        <small>raw {_format_currency(raw_exposure)} | trimmed {_format_currency(trimmed_exposure)}</small>
      </div>
      <div class="metric-card">
        <span class="metric-label">Bettable / Watchlist</span>
        <strong>{len(actionable)}</strong>
        <small>A-tier {a_count} | B-tier {b_count}</small>
      </div>
      <div class="metric-card">
        <span class="metric-label">Average Edge</span>
        <strong>{_format_pct(average_edge) if actionable is not None else "0.0%"}</strong>
        <small>across active expressions</small>
      </div>
      <div class="metric-card">
        <span class="metric-label">Average Confidence</span>
        <strong>{average_confidence:.2f}</strong>
        <small>model confidence on active card</small>
      </div>
      <div class="metric-card">
        <span class="metric-label">Parlay Coverage</span>
        <strong>{0 if parlays is None or parlays.empty else len(parlays)}</strong>
        <small>best-value 3-5 leg builds</small>
      </div>
    </section>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(event_name)} Operator Dashboard</title>
  <style>
    :root {{
      --paper: #f4efe6;
      --ink: #1f1d1a;
      --muted: #6c655a;
      --panel: rgba(255, 250, 242, 0.92);
      --line: rgba(87, 74, 56, 0.16);
      --accent: #b64a2b;
      --accent-soft: #f1c9a5;
      --olive: #6c7a42;
      --sand: #d9b382;
      --shadow: 0 18px 46px rgba(61, 42, 24, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(182, 74, 43, 0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(108, 122, 66, 0.14), transparent 28%),
        linear-gradient(180deg, #f8f2e8 0%, var(--paper) 100%);
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: end;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: clamp(2rem, 5vw, 3.7rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      color: var(--muted);
      max-width: 720px;
      font-size: 1rem;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }}
    .metric-card {{
      padding: 18px 20px;
    }}
    .metric-card strong {{
      display: block;
      margin-top: 8px;
      font-size: 1.9rem;
      letter-spacing: -0.04em;
    }}
    .metric-card small {{
      color: var(--muted);
    }}
    .metric-label {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.75rem;
      color: var(--muted);
    }}
    .dashboard-grid {{
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 16px;
      align-items: start;
    }}
    .panel {{
      padding: 18px 18px 16px;
      margin-bottom: 16px;
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      margin-bottom: 14px;
    }}
    h2 {{
      margin: 0;
      font-size: 1.05rem;
      letter-spacing: 0.02em;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.94rem;
    }}
    th, td {{
      padding: 10px 8px;
      text-align: left;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 0.73rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border: 1px solid transparent;
    }}
    .badge-ink {{
      background: rgba(31, 29, 26, 0.08);
      color: var(--ink);
      border-color: rgba(31, 29, 26, 0.12);
    }}
    .policy-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .policy-grid div {{
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(217, 179, 130, 0.10);
      border: 1px solid rgba(217, 179, 130, 0.18);
    }}
    .policy-grid span {{
      display: block;
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .policy-grid strong {{
      font-size: 1.1rem;
      letter-spacing: -0.02em;
    }}
    .empty {{
      color: var(--muted);
      padding: 6px 0 2px;
    }}
    @media (max-width: 980px) {{
      .dashboard-grid {{
        grid-template-columns: 1fr;
      }}
      .hero {{
        flex-direction: column;
        align-items: start;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <header class="hero">
      <div>
        <h1>{html.escape(event_name)}</h1>
        <div class="subtitle">Operator view across side leans, actionable expressions, and governed exposure. Raw staking intent is shown beside final governed exposure so trim decisions stay visible.</div>
      </div>
      {_badge("Exposure Dashboard", "ink")}
    </header>
    {summary_cards}
    {policy_panel}
    <div class="dashboard-grid">
      <div>
        <section class="panel">
          <div class="panel-header"><h2>Exposure Panel</h2></div>
          {_table_html(exposure_panel, [("fight", "Fight"), ("bets", "Bets"), ("raw_stake", "Raw"), ("governed_stake", "Governed"), ("trimmed_stake", "Trimmed"), ("top_risk", "Stake Notes")])}
        </section>
        <section class="panel">
          <div class="panel-header"><h2>Active Plays</h2></div>
          {_table_html(plays_panel, [("fight", "Fight"), ("bet", "Bet"), ("line", "Line"), ("edge", "Edge"), ("confidence", "Confidence"), ("recommended_tier", "Tier"), ("recommended_action", "Action"), ("stake", "Stake"), ("notes", "Stake Notes"), ("risk", "Risk")])}
        </section>
      </div>
      <div>
        <section class="panel">
          <div class="panel-header"><h2>Lean Board</h2></div>
          {_table_html(lean_panel, [("fight", "Fight"), ("lean_side", "Lean"), ("lean_strength", "Strength"), ("lean_action", "Action"), ("edge", "Edge"), ("lean_prob", "Prob"), ("reasons", "Reasons")])}
        </section>
        <section class="panel">
          <div class="panel-header"><h2>Pass Monitor</h2></div>
          {_table_html(pass_panel, [("fight", "Fight"), ("selection_name", "Side"), ("pass_reason", "Pass Reason"), ("risk_flags", "Risk Flags")])}
        </section>
        <section class="panel">
          <div class="panel-header"><h2>Parlay Board</h2></div>
          {_table_html(parlay_panel, [("parlay_name", "Build"), ("american_odds", "American"), ("decimal_odds", "Decimal"), ("edge", "Edge"), ("expected_value", "EV"), ("parlay_confidence", "Confidence"), ("legs", "Legs")])}
        </section>
      </div>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    fight_report = _load_csv(args.fight_report)
    lean_board = _load_csv(args.lean_board)
    value_report = _load_csv(args.value_report)
    betting_board = _load_csv(args.betting_board)
    passes = _load_csv(args.passes)
    parlays = _load_csv(args.parlays) if args.parlays else pd.DataFrame()
    threshold_policy = _load_threshold_policy(args.threshold_policy)

    output_html = build_operator_dashboard_html(
        fight_report=fight_report,
        lean_board=lean_board,
        value_report=value_report,
        betting_board=betting_board,
        passes=passes,
        parlays=parlays,
        threshold_policy=threshold_policy,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_html, encoding="utf-8")

    if not args.quiet:
        print(f"Saved operator dashboard to {output_path}")


if __name__ == "__main__":
    main()
