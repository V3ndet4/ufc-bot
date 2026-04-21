from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "llm_export_full_bot_review_bundle.txt"

INCLUDED_FILES = [
    "README.md",
    "NEXT_STEPS.md",
    "TASKS.md",
    "requirements.txt",
    ".env.example",
    "pytest.ini",
    "events/current_event.txt",
    "models/threshold_policy.json",
    "scripts/prepare_event.py",
    "scripts/run_event_pipeline.py",
    "scripts/build_fight_week_report.py",
    "scripts/run_value_scan.py",
    "scripts/project_fight_probs.py",
    "scripts/export_learning_report.py",
    "scripts/export_learning_summary.py",
    "scripts/build_historical_database.py",
    "scripts/generate_sample_training_data.py",
    "scripts/train_side_model.py",
    "scripts/train_selective_clv_model.py",
    "scripts/train_threshold_optimizer.py",
    "features/fighter_features.py",
    "normalization/odds.py",
    "models/projection.py",
    "models/ev.py",
    "models/decision_support.py",
    "models/side.py",
    "models/selective.py",
    "models/threshold_policy.py",
    "models/trainer.py",
    "bankroll/sizing.py",
    "backtests/evaluator.py",
    "backtests/grading.py",
    "data_sources/espn.py",
    "data_sources/bestfightodds.py",
    "data_sources/odds_api.py",
    "data_sources/storage.py",
    "data_sources/ufc_stats.py",
    "data_sources/external_ufc_history.py",
    "data_sources/ufc_history.py",
    "data_sources/historical_odds.py",
    "data_sources/historical_fighter_stats.py",
    "data_sources/sherdog.py",
    "sample_odds.csv",
    "sample_fighter_stats.csv",
    "sample_historical_odds.csv",
    "sample_espn_fighter_map.csv",
]

PREVIEW_GLOBS = [
    ("Latest Betting Board Preview", "cards/*/reports/betting_board.csv", 8),
    ("Latest Shortlist Preview", "cards/*/reports/value_bets_shortlist.csv", 8),
    ("Latest Learning Summary Preview", "cards/*/reports/learning_summary.csv", 12),
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def section(title: str) -> str:
    return "\n" + "=" * 76 + f"\n{title}\n" + "=" * 76 + "\n"


def emit_file(lines: list[str], rel_path: str) -> None:
    path = ROOT / rel_path
    if not path.exists():
        return
    lines.append(section(f"FILE: {rel_path}"))
    lines.append(read_text(path))
    if not lines[-1].endswith("\n"):
        lines.append("\n")


def emit_preview(lines: list[str], title: str, path: Path, max_lines: int) -> None:
    content_lines = read_text(path).splitlines()
    preview = "\n".join(content_lines[:max_lines])
    lines.append(section(f"{title}: {path.relative_to(ROOT).as_posix()}"))
    lines.append(preview + "\n")


def current_event_manifest() -> str | None:
    current_event_path = ROOT / "events" / "current_event.txt"
    if not current_event_path.exists():
        return None
    manifest_rel = current_event_path.read_text(encoding="utf-8", errors="replace").strip()
    manifest_path = ROOT / manifest_rel
    if not manifest_rel or not manifest_path.exists():
        return None
    return manifest_rel


def latest_non_empty(glob_pattern: str) -> Path | None:
    candidates = sorted(ROOT.glob(glob_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            if len(read_text(path).splitlines()) > 1:
                return path
        except OSError:
            continue
    return candidates[0] if candidates else None


def test_inventory() -> list[str]:
    return sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_*.py"))


def included_inventory() -> list[tuple[str, int, int]]:
    inventory: list[tuple[str, int, int]] = []
    for rel_path in INCLUDED_FILES:
        path = ROOT / rel_path
        if not path.exists():
            continue
        inventory.append((rel_path, len(read_text(path).splitlines()), path.stat().st_size))
    return inventory


def threshold_summary() -> str | None:
    path = ROOT / "models" / "threshold_policy.json"
    if not path.exists():
        return None
    try:
        data = json.loads(read_text(path))
    except json.JSONDecodeError:
        return None

    baseline = data.get("baseline", {})
    selected = data.get("selected", {})
    return (
        "Threshold policy snapshot: "
        f"status={data.get('status')} | "
        f"graded_bets={data.get('graded_bets')} | "
        f"baseline_roi_pct={baseline.get('roi_pct')} | "
        f"selected_roi_pct={selected.get('roi_pct')} | "
        f"selected_min_edge={selected.get('min_edge')} | "
        f"selected_min_model_confidence={selected.get('min_model_confidence')} | "
        f"selected_min_stats_completeness={selected.get('min_stats_completeness')}"
    )


def event_summary() -> str | None:
    manifest_rel = current_event_manifest()
    if not manifest_rel:
        return None
    try:
        manifest = json.loads(read_text(ROOT / manifest_rel))
    except json.JSONDecodeError:
        return f"Active event manifest: {manifest_rel}"

    return (
        "Active event: "
        f"{manifest.get('event_name')} | "
        f"event_id={manifest.get('event_id')} | "
        f"start_time={manifest.get('start_time')} | "
        f"fights={len(manifest.get('fights', []))} | "
        f"manifest={manifest_rel}"
    )


def build_bundle() -> str:
    lines: list[str] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    inventory = included_inventory()
    tests = test_inventory()
    manifest_rel = current_event_manifest()

    lines.append("UFC Bot External Review Bundle\n")
    lines.append(f"Generated at: {timestamp}\n")
    lines.append(f"Repo root: {ROOT}\n")
    lines.append(f"Output file: {OUTPUT_PATH.name}\n")
    lines.append(f"Included full files: {len(inventory)}\n")
    lines.append(f"Test inventory count: {len(tests)}\n")
    lines.append("\n")

    lines.append("How to use this with another LLM\n")
    lines.append("- Upload this single file.\n")
    lines.append("- Suggested prompt:\n")
    lines.append(
        '  "Act as a senior Python, ML, and sports-betting systems reviewer. Critique this UFC betting bot. '
        "Find bugs, hidden assumptions, data leakage risks, bad evaluation logic, bankroll/staking issues, "
        "scraping fragility, maintainability problems, and missing tests. Cite file names in every finding. "
        'Then give a prioritized improvement plan and the top 5 code changes you would make first."\n'
    )
    lines.append("\n")

    lines.append("Review focus\n")
    lines.append("- Live event pipeline reliability and operator workflow\n")
    lines.append("- Projection quality, calibration, and leakage risk\n")
    lines.append("- EV math, thresholding, and stake sizing\n")
    lines.append("- Scraper/data-source fragility and data quality assumptions\n")
    lines.append("- Maintainability, monolithic scripts, and test coverage gaps\n")
    lines.append("\n")

    lines.append("Repo snapshot\n")
    lines.append("- Stack: Python, pandas, requests, BeautifulSoup, scikit-learn, SQLite, PowerShell wrappers\n")
    lines.append("- Canonical operator path: PowerShell scripts around a manifest-driven per-card workflow\n")
    lines.append("- Large assets omitted from inline export: pickle models, large CSV data files, generated reports, SQLite databases, virtualenvs\n")
    if event_summary():
        lines.append(f"- {event_summary()}\n")
    if threshold_summary():
        lines.append(f"- {threshold_summary()}\n")
    lines.append("\n")

    lines.append("Useful local commands\n")
    lines.append(r"- .\scripts\set_next_card.ps1 --status" + "\n")
    lines.append(r"- .\scripts\run_next_card.ps1" + "\n")
    lines.append(r"- .\scripts\capture_next_card_odds.ps1" + "\n")
    lines.append(r"- python scripts\run_event_pipeline.py --manifest events\ufc_fn_sterling_zalal.json --stats-source espn" + "\n")
    lines.append(r"- python scripts\run_value_scan.py --input sample_odds.csv --fighter-stats sample_fighter_stats.csv --output reports\value_bets.csv" + "\n")
    lines.append(r"- pytest -q" + "\n")
    lines.append("\n")

    lines.append("Included file inventory\n")
    for rel_path, line_count, size_bytes in inventory:
        lines.append(f"- {rel_path} | lines={line_count} | bytes={size_bytes}\n")
    if manifest_rel and manifest_rel not in {rel for rel, _, _ in inventory}:
        manifest_path = ROOT / manifest_rel
        lines.append(
            f"- {manifest_rel} | lines={len(read_text(manifest_path).splitlines())} | bytes={manifest_path.stat().st_size}\n"
        )
    lines.append("\n")

    lines.append("Test inventory\n")
    for rel_path in tests:
        lines.append(f"- {rel_path}\n")
    lines.append("\n")

    lines.append("Generated report previews\n")
    for title, glob_pattern, max_lines in PREVIEW_GLOBS:
        preview_path = latest_non_empty(glob_pattern)
        if preview_path is None:
            continue
        lines.append(f"- {title}: {preview_path.relative_to(ROOT).as_posix()}\n")
    lines.append("\n")

    for title, glob_pattern, max_lines in PREVIEW_GLOBS:
        preview_path = latest_non_empty(glob_pattern)
        if preview_path is not None:
            emit_preview(lines, title, preview_path, max_lines)

    for rel_path in INCLUDED_FILES:
        emit_file(lines, rel_path)

    if manifest_rel and manifest_rel not in INCLUDED_FILES:
        emit_file(lines, manifest_rel)

    return "".join(lines)


def main() -> None:
    OUTPUT_PATH.write_text(build_bundle(), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
