from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.gym_overrides import apply_context_gym_overrides, apply_fighter_gym_overrides, load_fighter_gym_overrides


def test_apply_fighter_gym_overrides_replaces_current_camp_and_flags_change() -> None:
    fighter_gyms = pd.DataFrame(
        [
            {
                "fighter_name": "Khamzat Chimaev",
                "gym_name": "Allstars Training Center",
                "gym_tier": "B",
                "gym_changed_flag": 0,
                "previous_gym_name": "",
            }
        ]
    )
    overrides = pd.DataFrame(
        [
            {
                "fighter_name": "Khamzat Chimaev",
                "fighter_name_normalized": "khamzat chimaev",
                "gym_name": "Santo Performance Studio",
                "gym_tier": "",
                "gym_source": "manual_current_camp",
                "previous_gym_name": "Allstars Training Center",
                "gym_changed_flag": 1,
                "verified_at": "2026-05-03T00:00:00+00:00",
            }
        ]
    )

    updated = apply_fighter_gym_overrides(
        fighter_gyms,
        overrides,
        timestamp="2026-05-03T01:00:00+00:00",
    )

    assert updated.loc[0, "gym_name"] == "Santo Performance Studio"
    assert updated.loc[0, "gym_name_normalized"] == "santo performance studio"
    assert updated.loc[0, "gym_source"] == "manual_current_camp"
    assert updated.loc[0, "gym_tier"] == ""
    assert int(updated.loc[0, "gym_changed_flag"]) == 1
    assert updated.loc[0, "previous_gym_name"] == "Allstars Training Center"
    assert updated.loc[0, "last_changed_at"] == "2026-05-03T00:00:00+00:00"


def test_apply_context_gym_overrides_merges_camp_change_context() -> None:
    context = pd.DataFrame(
        [
            {
                "fighter_name": "Khamzat Chimaev",
                "new_gym_flag": 0,
                "camp_change_flag": 0,
                "news_alert_count": 0,
                "news_radar_score": 0.2,
                "news_radar_label": "",
                "news_radar_summary": "",
                "context_notes": "Existing note.",
            }
        ]
    )
    overrides = pd.DataFrame(
        [
            {
                "fighter_name": "Khamzat Chimaev",
                "fighter_name_normalized": "khamzat chimaev",
                "new_gym_flag": 1,
                "camp_change_flag": 1,
                "news_alert_count": 2,
                "news_radar_score": 0.7,
                "news_radar_label": "camp_change",
                "news_radar_summary": "Current camp at Santo Performance Studio.",
                "context_notes": "Manual override.",
            }
        ]
    )

    updated = apply_context_gym_overrides(context, overrides)

    assert int(updated.loc[0, "new_gym_flag"]) == 1
    assert int(updated.loc[0, "camp_change_flag"]) == 1
    assert int(updated.loc[0, "news_alert_count"]) == 2
    assert float(updated.loc[0, "news_radar_score"]) == 0.7
    assert updated.loc[0, "news_radar_label"] == "camp_change"
    assert updated.loc[0, "news_radar_summary"] == "Current camp at Santo Performance Studio."
    assert updated.loc[0, "context_notes"] == "Existing note. | Manual override."


def test_default_override_file_loads_active_current_camp_rows() -> None:
    overrides = load_fighter_gym_overrides(ROOT / "data" / "fighter_gym_overrides.csv")

    by_name = {row["fighter_name"]: row for _, row in overrides.iterrows()}
    assert by_name["Khamzat Chimaev"]["gym_name"] == "Santo Performance Studio"
    assert by_name["Joel Alvarez"]["gym_name"] == "Centro Deportivo Tibet"
    assert by_name["Djorden Santos"]["gym_name"] == "American Top Team"
