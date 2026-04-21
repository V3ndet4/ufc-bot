from __future__ import annotations


def derive_style_label(
    *,
    stance: str,
    strike_margin: float,
    grappling_rate: float,
    control_avg: float,
    ko_win_rate: float,
    submission_win_rate: float,
    decision_rate: float,
    knockdown_avg: float = 0.0,
    distance_strike_share: float = 0.0,
    clinch_strike_share: float = 0.0,
    ground_strike_share: float = 0.0,
) -> str:
    stance_label = str(stance or "").strip().title() or "Unknown stance"

    if control_avg >= 4.0 or ground_strike_share >= 0.20 or grappling_rate >= 2.2:
        base = "Control grappler"
    elif submission_win_rate >= 0.30 and (ground_strike_share >= 0.12 or grappling_rate >= 1.3):
        base = "Submission grappler"
    elif clinch_strike_share >= 0.25 and control_avg >= 1.8:
        base = "Clinch grinder"
    elif ko_win_rate >= 0.40 and (knockdown_avg >= 0.18 or strike_margin >= 1.0):
        base = "Power striker"
    elif distance_strike_share >= 0.68 and strike_margin >= 0.8:
        base = "Volume striker"
    elif decision_rate >= 0.55 and abs(strike_margin) < 0.8 and grappling_rate < 1.2:
        base = "Point fighter"
    else:
        base = "All-rounder"

    return f"{base} | {stance_label}"
