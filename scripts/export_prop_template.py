from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_manual_props import _american_to_decimal, _probability_to_american_odds


PROP_TEMPLATE_SPECS = [
    ("moneyline", "fighter_a_model_win_prob", "fighter"),
    ("moneyline", "fighter_b_model_win_prob", "fighter"),
    ("inside_distance", "fighter_a_inside_distance_prob", "fighter"),
    ("inside_distance", "fighter_b_inside_distance_prob", "fighter"),
    ("submission", "fighter_a_submission_prob", "fighter"),
    ("submission", "fighter_b_submission_prob", "fighter"),
    ("ko_tko", "fighter_a_ko_tko_prob", "fighter"),
    ("ko_tko", "fighter_b_ko_tko_prob", "fighter"),
    ("by_decision", "fighter_a_by_decision_prob", "fighter"),
    ("by_decision", "fighter_b_by_decision_prob", "fighter"),
    ("fight_goes_to_decision", "fight_goes_to_decision_model_prob", "fight"),
    ("fight_doesnt_go_to_decision", "fight_doesnt_go_to_decision_model_prob", "fight"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a manual prop entry template from a fight-week report.")
    parser.add_argument("--fight-report", required=True, help="Fight-week report CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--min-model-prob",
        type=float,
        default=0.08,
        help="Minimum model probability required to include a prop row.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export every supported prop row instead of a curated subset.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")
    return parser.parse_args()


def _include_curated_prop(report_row: dict[str, object], prop_type: str, model_prob: float, fair_american: object) -> bool:
    decision_prob = float(report_row.get("projected_decision_prob", 0.0) or 0.0)
    finish_prob = float(report_row.get("projected_finish_prob", 0.0) or 0.0)
    if prop_type == "moneyline":
        return model_prob >= 0.22
    if prop_type == "inside_distance":
        return model_prob >= 0.18 and pd.notna(fair_american) and int(fair_american) >= 125
    if prop_type == "submission":
        return model_prob >= 0.10 and pd.notna(fair_american) and int(fair_american) >= 220
    if prop_type == "ko_tko":
        return model_prob >= 0.12 and pd.notna(fair_american) and int(fair_american) >= 180
    if prop_type == "by_decision":
        return model_prob >= 0.14 and decision_prob >= 0.42 and pd.notna(fair_american) and int(fair_american) >= 180
    if prop_type == "fight_goes_to_decision":
        return decision_prob >= 0.22 and pd.notna(fair_american) and int(fair_american) >= 120
    if prop_type == "fight_doesnt_go_to_decision":
        return finish_prob >= 0.22 and pd.notna(fair_american) and int(fair_american) >= 120
    return False


def export_prop_template(
    fight_report: pd.DataFrame,
    *,
    min_model_prob: float = 0.08,
    full: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for report_row in fight_report.to_dict("records"):
        fighter_a = str(report_row["fighter_a"]).strip()
        fighter_b = str(report_row["fighter_b"]).strip()
        for prop_type, model_column, scope in PROP_TEMPLATE_SPECS:
            model_prob = report_row.get(model_column)
            if pd.isna(model_prob) or float(model_prob) < min_model_prob:
                continue

            fair_american = _probability_to_american_odds(model_prob)
            fair_decimal = _american_to_decimal(fair_american)
            if not full and not _include_curated_prop(report_row, prop_type, float(model_prob), fair_american):
                continue
            if scope == "fight":
                selection_name = ""
                prop_label = (
                    "Fight goes to decision"
                    if prop_type == "fight_goes_to_decision"
                    else "Fight doesn't go to decision"
                )
            else:
                selection_name = fighter_a if "_a_" in model_column else fighter_b
                suffix_map = {
                    "moneyline": "moneyline",
                    "inside_distance": "inside distance",
                    "submission": "submission",
                    "ko_tko": "KO/TKO",
                    "by_decision": "by decision",
                }
                prop_label = f"{selection_name} {suffix_map[prop_type]}"

            rows.append(
                {
                    "fighter_a": fighter_a,
                    "fighter_b": fighter_b,
                    "selection_name": selection_name,
                    "prop_type": prop_type,
                    "prop_label": prop_label,
                    "american_odds": "",
                    "sportsbook": "",
                    "model_prob": round(float(model_prob), 4),
                    "fair_american_odds": fair_american,
                    "fair_decimal_odds": fair_decimal,
                    "notes": "",
                }
            )

    return pd.DataFrame(rows).sort_values(by=["fighter_a", "fighter_b", "prop_type", "selection_name"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    fight_report = pd.read_csv(args.fight_report)
    template = export_prop_template(
        fight_report,
        min_model_prob=args.min_model_prob,
        full=args.full,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    if not args.quiet:
        print(f"Exported {len(template)} prop rows")
        print(f"Saved prop template to {output_path}")


if __name__ == "__main__":
    main()
