from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ev import implied_probability


PROP_TYPE_TO_COLUMNS = {
    "inside_distance": {
        "fighter_a": "fighter_a_inside_distance_prob",
        "fighter_b": "fighter_b_inside_distance_prob",
    },
    "submission": {
        "fighter_a": "fighter_a_submission_prob",
        "fighter_b": "fighter_b_submission_prob",
    },
    "ko_tko": {
        "fighter_a": "fighter_a_ko_tko_prob",
        "fighter_b": "fighter_b_ko_tko_prob",
    },
    "by_decision": {
        "fighter_a": "fighter_a_by_decision_prob",
        "fighter_b": "fighter_b_by_decision_prob",
    },
    "moneyline": {
        "fighter_a": "fighter_a_model_win_prob",
        "fighter_b": "fighter_b_model_win_prob",
    },
    "fight_goes_to_decision": {
        "fight": "fight_goes_to_decision_model_prob",
    },
    "fight_doesnt_go_to_decision": {
        "fight": "fight_doesnt_go_to_decision_model_prob",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check manually-entered prop prices against the fight-week report model.")
    parser.add_argument("--fight-report", required=True, help="Fight-week report CSV path.")
    parser.add_argument("--props", required=True, help="CSV with manual prop prices to evaluate.")
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


def _probability_to_american_odds(probability: object) -> int | pd.NA:
    if pd.isna(probability):
        return pd.NA
    value = float(probability)
    if value <= 0 or value >= 1:
        return pd.NA
    if value >= 0.5:
        return int(round(-(value / (1 - value)) * 100))
    return int(round(((1 - value) / value) * 100))


def evaluate_manual_props(fight_report: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    required = {"fighter_a", "fighter_b", "prop_type", "american_odds"}
    missing = required - set(props.columns)
    if missing:
        raise ValueError(f"Missing required prop columns: {', '.join(sorted(missing))}")

    report = fight_report.copy()
    report["fight_key"] = report["fighter_a"].astype(str).str.strip() + "||" + report["fighter_b"].astype(str).str.strip()
    report_lookup = report.set_index("fight_key")

    rows: list[dict[str, object]] = []
    for prop in props.to_dict("records"):
        fighter_a = str(prop["fighter_a"]).strip()
        fighter_b = str(prop["fighter_b"]).strip()
        prop_type = str(prop["prop_type"]).strip().lower()
        fight_key = f"{fighter_a}||{fighter_b}"
        if fight_key not in report_lookup.index:
            raise ValueError(f"Fight not found in report: {fighter_a} vs {fighter_b}")
        if prop_type not in PROP_TYPE_TO_COLUMNS:
            raise ValueError(f"Unsupported prop_type: {prop_type}")

        report_row = report_lookup.loc[fight_key]
        selection_name = str(prop.get("selection_name", "") or "").strip()
        if prop_type in {"fight_goes_to_decision", "fight_doesnt_go_to_decision"}:
            model_prob = float(report_row[PROP_TYPE_TO_COLUMNS[prop_type]["fight"]])
            prop_label = "Fight goes to decision" if prop_type == "fight_goes_to_decision" else "Fight doesn't go to decision"
        else:
            if not selection_name:
                raise ValueError(f"selection_name is required for prop_type={prop_type} on {fighter_a} vs {fighter_b}")
            if selection_name == fighter_a:
                side = "fighter_a"
            elif selection_name == fighter_b:
                side = "fighter_b"
            else:
                raise ValueError(f"selection_name must match fighter_a or fighter_b for {fighter_a} vs {fighter_b}")
            model_prob = float(report_row[PROP_TYPE_TO_COLUMNS[prop_type][side]])
            suffix_map = {
                "inside_distance": "inside distance",
                "submission": "submission",
                "ko_tko": "KO/TKO",
                "by_decision": "by decision",
                "moneyline": "moneyline",
            }
            prop_label = f"{selection_name} {suffix_map[prop_type]}"

        american_odds = int(float(prop["american_odds"]))
        implied_prob = implied_probability(american_odds)
        fair_american_odds = _probability_to_american_odds(model_prob)
        fair_decimal_odds = _american_to_decimal(fair_american_odds)
        edge = round(model_prob - implied_prob, 4)
        verdict = "value" if edge >= 0.03 else "thin" if edge >= 0.0 else "no_value"

        rows.append(
            {
                "fighter_a": fighter_a,
                "fighter_b": fighter_b,
                "selection_name": selection_name,
                "prop_type": prop_type,
                "prop_label": prop_label,
                "book_american_odds": american_odds,
                "book_decimal_odds": _american_to_decimal(american_odds),
                "model_prob": round(model_prob, 4),
                "book_implied_prob": round(implied_prob, 4),
                "fair_american_odds": fair_american_odds,
                "fair_decimal_odds": fair_decimal_odds,
                "edge": edge,
                "verdict": verdict,
                "notes": str(prop.get("notes", "") or "").strip(),
            }
        )

    return pd.DataFrame(rows).sort_values(by=["verdict", "edge"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    fight_report = pd.read_csv(args.fight_report)
    props = pd.read_csv(args.props)
    evaluated = evaluate_manual_props(fight_report, props)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated.to_csv(output_path, index=False)
    if not args.quiet:
        print(f"Checked {len(evaluated)} props")
        print(f"Saved prop check to {output_path}")


if __name__ == "__main__":
    main()
