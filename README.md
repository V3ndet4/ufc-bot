# UFC Betting Bot

Starter Python project for UFC bet selection.

This scaffold now does five things:

- loads upcoming fight markets from a CSV or API payload
- stores odds snapshots in SQLite for repeatable analysis
- joins fighter stats into a reusable feature set
- projects win probability from fighter-level features
- normalizes fighter and market data into a stable schema
- calculates implied probability, edge, and suggested stake

The projection model already includes automatic physical-dimension edges from:
- `reach_diff`
- `height_diff`

It now also includes an early-finisher signal from:
- `first_round_finish_rate_diff`

This version does not place bets automatically. It is built for value detection and reporting first.

This version now also supports:

- separate moneyline and finish/decision projections
- tracked-pick storage in SQLite
- post-event auto-grading with CLV fields

## Project Layout

- `data_sources/` source-specific ingestion
- `normalization/` canonical event and odds transforms
- `features/` fighter-level feature engineering
- `models/` edge and pricing logic
- `bankroll/` stake sizing rules
- `backtests/` historical evaluation hooks
- `scripts/` operator-facing entrypoints
- `cards/<slug>/inputs/` per-card fighter lists, maps, and context
- `cards/<slug>/data/` per-card odds, stats, and results
- `cards/<slug>/reports/` per-card reports, boards, and charts
- `tests/` smoke tests for core math

## Quick Start

PowerShell is the primary operator path for this repo.

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python scripts\record_odds_snapshot.py --input sample_odds.csv
python scripts\project_fight_probs.py --odds sample_odds.csv --fighter-stats sample_fighter_stats.csv --output reports\projected_probs.csv
python scripts\run_value_scan.py --input sample_odds.csv --output reports\value_bets.csv
python scripts\build_fight_week_report.py --odds cards\upcoming_card\data\bfo_odds.csv --fighter-stats cards\upcoming_card\data\fighter_stats.csv --output cards\upcoming_card\reports\fight_week_report.csv
python scripts\run_backtest.py --input sample_historical_odds.csv --fighter-stats sample_fighter_stats.csv --output reports\backtest_summary.csv
python scripts\run_value_scan.py --input sample_odds.csv --fighter-stats sample_fighter_stats.csv --output reports\value_bets.csv --db data\ufc_betting.db
python scripts\grade_tracked_picks.py --results sample_results.csv --db data\ufc_betting.db --output reports\graded_picks.csv
python scripts\fetch_ufc_stats.py --output data\ufc_stats_fighter_stats.csv
python scripts\fetch_espn_stats.py --mapping sample_espn_fighter_map.csv --output data\espn_fighter_stats.csv
python scripts\fetch_bestfightodds_event_odds.py --template cards\upcoming_card\data\odds_template.csv --event-url https://www.bestfightodds.com/events/ufc-london-4081 --output cards\upcoming_card\data\bfo_odds.csv
python scripts\fetch_the_odds_api_odds.py --template cards\upcoming_card\data\odds_template.csv --bookmaker fanduel --output cards\upcoming_card\data\oddsapi_odds.csv
```

Each Odds API refresh now also saves a timestamped snapshot to `data/ufc_betting.db` by default, so repeated FanDuel pulls build the history used by the line-movement board. Add `--no-snapshot` if you explicitly want to skip that write.

For the simplest daily workflow on the active card, run:

```powershell
.\scripts\capture_next_card_odds.ps1
```

That single command will:
- refresh FanDuel odds for the active card
- save the new odds snapshot to SQLite
- rebuild the line-movement board and per-fight charts
- refresh the downstream fight report and betting outputs without re-fetching stats

Running it once or twice a day is enough to start building a useful line-movement history into the chart.

If you want it to refresh everything and immediately open the active card reports folder afterward, run:

```powershell
.\scripts\capture_and_open_next_card_reports.ps1
```

If you just want to open the active card reports folder at any time, run:

```powershell
.\scripts\open_next_card_reports.ps1
```

The main event workflow is now PowerShell-first:

```powershell
.\scripts\set_next_card.ps1 --status
.\scripts\run_next_card.ps1
```

The PowerShell entry scripts default to compact child output, so the run now ends with a cleaner generated-reports block, then the lean board, and finally the parlay board at the bottom.

The active manifest-driven pipeline is now the canonical path. Card files are no longer meant to sit loose at the repo root or directly under `data/` and `reports/`; they are organized under `cards/<slug>/...`.

Useful PowerShell shortcuts:

```powershell
.\scripts\prepare_next_card.ps1
.\scripts\refresh_next_card_stats.ps1
.\scripts\refresh_next_card_odds.ps1
.\scripts\scan_next_card.ps1
.\scripts\grade_next_card.ps1
.\scripts\run_fight_week_watch.ps1
.\scripts\install_fight_week_watch_tasks.ps1
.\scripts\remove_fight_week_watch_tasks.ps1
```

Card-specific PowerShell shortcuts are still available:

```powershell
.\scripts\run_ufc_327_card.ps1
.\scripts\refresh_ufc_327_odds.ps1
.\scripts\scan_ufc_327_card.ps1
.\scripts\grade_ufc_327_card.ps1
```

If you want to work directly with the manifest pipeline:

```powershell
.\.venv-win\Scripts\python.exe scripts\run_event_pipeline.py --manifest events\ufc_327_prochazka_ulberg.json --stats-source espn
```

`auto` is the default and prefers ESPN when `cards/<slug>/inputs/fighter_map.csv` exists for the event.
After the base stats refresh, the pipeline now also runs a best-effort weekly UFC-history enrichment from `Greco1899/scrape_ufc_stats`. Use `--skip-external-history` if you want to leave the base stats file untouched.
That enrichment currently backfills UFC-only history/profile fields such as layoff, recent damage, finish profile, control-based grappling metrics, and a lightweight opponent-strength layer that normalizes striking/grappling/control metrics against schedule quality.
When a tracked-picks database is available, the event pipeline also refreshes `models/threshold_policy.json` before the scan and writes an operator dashboard to `cards/<slug>/reports/operator_dashboard.html`. If fight-week alerts exist, that dashboard now includes the radar panel automatically.

The operator wrappers now default to `ODDS_SOURCE=oddsapi` with `ODDS_API_BOOKMAKER=fanduel`, which matches the current FanDuel-first workflow.

For the stripped-down moneyline board, use the core command:

```powershell
.\scripts\run_core_card.ps1
```

That command does not refresh fighter stats, build dashboards, build parlays, or add Sherdog/Tapology as base stat sources. It reads the active card's cached `fighter_stats.csv`, current `oddsapi_odds.csv`, and cached `lean_board.csv` when available; anchors model probabilities to the no-vig market; then writes one simple moneyline board to `cards/<slug>/reports/core_board.csv`. The board keeps the old operator context columns for gym tier, news radar, lean drivers, risks, and what to watch for. By default it caps the board at the top 3 qualifying bet candidates; override with `--max-bets`.

Console output is colorized by default in interactive terminals. Use `--color always` to force ANSI color or `--color never` for plain text.

Optional core props and parlays stay off by default:

```powershell
.\scripts\run_core_card.ps1 --include-props --include-parlays
```

Props are only scored when actual prop prices exist in `modeled_market_odds.csv`; parlays are built only from core `BET` rows with positive expected value.

To refresh the accuracy stack in one command, run:

```powershell
.\scripts\run_accuracy_upgrade.ps1
```

That refreshes external UFC-history features for the active card, rebuilds `data/prop_outcome_history.csv`, retrains the per-market prop model, appends `data/model_experiments.csv`, and regenerates the card accuracy, prop threshold, prop walk-forward, market-family, prop CLV readiness, fighter identity, odds movement CLV, and tracked CLV reports. Use `--skip-external-refresh` to use the cached history files only.

Prop bets stay blocked until their market has enough outcome sample, price archive coverage, and positive tracked CLV. When `run_event_pipeline.py` refreshes Odds API modeled markets, it archives priced non-moneyline prop rows automatically so the CLV gate can build forward history.

That flow will:

- regenerate the fighter list, context sheet, and odds template from the manifest
- fetch only the listed fighters from UFC Stats
- refresh BestFightOdds lines when available
- skip report/value generation cleanly if live odds are still missing
- write all generated assets into the card workspace under `cards/<slug>/`

Bash/WSL wrappers still exist, but they are now secondary convenience commands rather than the canonical runtime.

## Input Format

The value scan can still use a simple odds CSV with these columns:

- `event_id`
- `event_name`
- `start_time`
- `fighter_a`
- `fighter_b`
- `market`
- `selection`
- `book`
- `american_odds`
- `projected_win_prob`

Each row represents a single bettable selection, for example:

- `moneyline`, `fighter_a`
- `moneyline`, `fighter_b`
- `method_ko_tko`, `fighter_a`

The projection and backtest scripts additionally expect a fighter stats CSV with:

- `fighter_name`
- `wins`
- `losses`
- `height_in`
- `reach_in`
- `sig_strikes_landed_per_min`
- `sig_strikes_absorbed_per_min`
- `takedown_avg`
- `takedown_defense_pct`

## Real Fighter Data

You can scrape live fighter stats from UFC Stats into the same schema used by the projection and backtest scripts:

```bash
python scripts/fetch_ufc_stats.py --output data/ufc_stats_fighter_stats.csv
python scripts/project_fight_probs.py --odds sample_odds.csv --fighter-stats data/ufc_stats_fighter_stats.csv --output reports/projected_probs.csv
```

To run a smaller fetch while iterating, limit the directory scrape to a few initials:

```bash
python scripts/fetch_ufc_stats.py --letters a b c --output data/ufc_stats_subset.csv
```

This source is HTML-scraped from `ufcstats.com`, not a stable public API, so selector breakage is possible if the site layout changes.

## ESPN Fighter Data

If `ufcstats.com` is unavailable from your network, you can scrape ESPN fighter pages from a mapping CSV:

```bash
python scripts/fetch_espn_stats.py --mapping sample_espn_fighter_map.csv --output data/espn_fighter_stats.csv
```

The mapping CSV must contain:

- `fighter_name`
- `espn_url`

This ESPN path computes:

- `sig_strikes_landed_per_min` from the ESPN striking table and fight durations
- `sig_strikes_absorbed_per_min` from the ESPN striking table and fight durations
- `takedown_avg` from takedowns landed per 15 minutes using the ESPN clinch table

ESPN does not expose takedown defense directly on the public fighter pages used here, so `takedown_defense_pct` is currently filled with takedown accuracy as a proxy. Treat that field as a temporary approximation, not a true defensive metric.

The ESPN export now also includes:

- `days_since_last_fight`
- `losses_in_row`
- `first_round_finish_wins`
- `first_round_finish_rate`
- `recent_result_score`
- `recent_strike_margin_per_min`
- `ufc_fight_count`
- `ufc_debut_flag`
- `stats_completeness`
- `fallback_used`

If you already have a base fighter stats file and want to layer in the external UFC-history fields manually, run:

```powershell
python scripts\enrich_fighter_stats_with_external_ufc_history.py --input cards\upcoming_card\data\fighter_stats.csv
```

You can also annotate the fighter mapping CSV with optional manual context flags:

- `short_notice_flag`
- `short_notice_acceptance_flag`
- `short_notice_success_flag`
- `new_gym_flag`
- `new_contract_flag`
- `cardio_fade_flag`

Use `short_notice_flag=1` when the current matchup was accepted on a short turnaround.
Use `short_notice_acceptance_flag=1` for fighters who have shown they are willing to say yes to short-notice opportunities. The model only gives that trait credit when the fighter is actually in a short-notice spot.
Use `short_notice_success_flag=1` for fighters who have historically handled short-notice spots well. That trait is also only applied when the fighter is actually in a short-notice matchup.

Use `cardio_fade_flag=1` for fighters who are dangerous early but tend to slow down after round 1. The model treats that as a negative full-fight win signal.

## BestFightOdds Event Odds

To fill a card template with current BestFightOdds moneylines:

```bash
python scripts/fetch_bestfightodds_event_odds.py --template cards/upcoming_card/data/odds_template.csv --event-url https://www.bestfightodds.com/events/ufc-london-4081 --output cards/upcoming_card/data/bfo_odds.csv
```

This script writes current moneylines using a consensus price derived from the available books on the event page. It does not yet scrape historical open or close into dedicated columns.

## The Odds API FanDuel Odds

If you have an `ODDS_API_KEY`, you can pull direct FanDuel moneylines from The Odds API instead of scraping consensus odds:

```bash
python scripts/fetch_the_odds_api_odds.py --template cards/upcoming_card/data/odds_template.csv --bookmaker fanduel --output cards/upcoming_card/data/oddsapi_odds.csv
```

For the Seattle template used in this repo:

```bash
python scripts/fetch_the_odds_api_odds.py --template cards/seattle_card/data/odds_template.csv --bookmaker fanduel --output cards/seattle_card/data/oddsapi_odds.csv
```

This path currently supports `h2h` moneylines. It matches events from The Odds API to the local fight template by fighter names, then writes the selected bookmaker price into the same schema used by the projection and reporting scripts.

## Quality Controls

The projection model now shrinks probabilities toward 50/50 when fighter data is incomplete and treats longer layoffs as a prep-and-improvement upside signal. `run_value_scan.py` also supports:

- `MIN_MODEL_CONFIDENCE`
- `MIN_STATS_COMPLETENESS`
- `EXCLUDE_FALLBACK_ROWS`
- `BANKROLL`
- `FRACTIONAL_KELLY`
- `MAX_BET_STAKE_PCT`
- `MAX_CARD_EXPOSURE_PCT`
- `MAX_FIGHT_EXPOSURE_PCT`
- `WATCHLIST_STAKE_MULTIPLIER`
- `MEDIUM_FRAGILITY_STAKE_MULTIPLIER`
- `HIGH_FRAGILITY_STAKE_MULTIPLIER`

If `models/threshold_policy.json` exists, `run_value_scan.py` will automatically raise its floors to the stricter of:

- the current env settings
- the optimized policy learned from graded tracked picks

Example:

```bash
MIN_MODEL_CONFIDENCE=0.55 MIN_STATS_COMPLETENESS=0.80 EXCLUDE_FALLBACK_ROWS=true python scripts/run_value_scan.py --input cards/upcoming_card/data/bfo_odds.csv --fighter-stats cards/upcoming_card/data/fighter_stats.csv --output cards/upcoming_card/reports/bfo_value_bets.csv
```

Stake sizing is now two-stage:

- raw Kelly stake is still calculated first
- a bankroll governor then caps single-bet size, caps per-fight and per-card exposure, and trims watchlist or fragile plays before the final actionable stake is written

The report now keeps both the raw and governed stake fields so you can see what the model wanted to bet versus what the bankroll rules actually allow.

The workflow now also builds a `parlay_board.csv` from the finished value report. The parlay builder:

- keeps only the best single leg per fight, choosing between eligible moneyline and priced prop expressions
- restricts the pool to strong A/B value legs with enough projected probability
- allows `-250` and `-300` favorites normally, with elite `A`-tier profiles able to extend as far as `-400`
- ranks the top 2-leg through 5-leg combinations by combined probability, edge, EV, and confidence instead of raw payout

The event pipeline now also writes an HTML operator dashboard with:

- governed vs raw card exposure
- per-fight exposure rollups
- top active plays
- lean board snapshot
- pass-monitor panel

For a fight-level comparison of open line, current line, model probability, confidence, and context flags:

```bash
python scripts/build_fight_week_report.py --odds cards/upcoming_card/data/bfo_odds.csv --fighter-stats cards/upcoming_card/data/fighter_stats.csv --output cards/upcoming_card/reports/fight_week_report.csv
```

For a visual line-movement board from open to current, with any recorded SQLite snapshots inserted between those endpoints:

```powershell
.\.venv-win\Scripts\python.exe scripts\build_line_movement_report.py `
  --odds cards\upcoming_card\data\oddsapi_odds.csv `
  --db data\ufc_betting.db `
  --bookmaker fanduel `
  --output cards\upcoming_card\reports\line_movement.svg
```

If you want a single fight, add `--fighter "Jiri Prochazka"` or another fighter name filter.

`run_next_card.ps1` and `run_next_card.sh` now default to `--odds-source oddsapi --odds-api-bookmaker fanduel`, and the line-movement board generated by the event pipeline will use the same FanDuel filter automatically.

That fight-week report now carries both a winner model and a separate finish/decision model:

- `fighter_a_model_win_prob`
- `fighter_b_model_win_prob`
- `projected_finish_prob`
- `projected_decision_prob`
- `fighter_a_inside_distance_prob`
- `fighter_b_inside_distance_prob`
- `fighter_a_by_decision_prob`
- `fighter_b_by_decision_prob`

`run_value_scan.py` can now persist tracked picks directly to SQLite:

```bash
python scripts/run_value_scan.py --input cards/upcoming_card/data/oddsapi_odds.csv --fighter-stats cards/upcoming_card/data/fighter_stats.csv --fight-report cards/upcoming_card/reports/fight_week_report.csv --output cards/upcoming_card/reports/value_bets.csv --db data/ufc_betting.db
```

After the event, import a results CSV and auto-grade the stored picks:

```bash
python scripts/grade_tracked_picks.py --results cards/upcoming_card/data/results.csv --db data/ufc_betting.db --output cards/upcoming_card/reports/graded_picks.csv
```

The grading wrappers now also export:

- `graded_picks.csv`
- `lean_board_results.csv`
- `lean_postmortem_summary.csv`
- `learning_report.csv`
- `learning_postmortem.csv`
- `learning_postmortem_summary.csv`
- `learning_summary.csv`
- `filter_performance.csv`

To export the bucketed feedback report directly from tracked history:

```bash
python scripts/export_filter_performance.py --db data/ufc_betting.db --output reports/filter_performance.csv
```

PowerShell shortcut:

```powershell
.\scripts\export_filter_performance.ps1
```

Current active-card shortcut:

```powershell
.\scripts\export_current_filter_performance.ps1
```

Add `--event-id <event_id>` if you want the breakdown for a single event instead of the full database history.

The results CSV should contain at least:

- `event_id`
- `fighter_a`
- `fighter_b`
- `winner_name`
- `winner_side`
- `result_status`
- `went_decision`
- `ended_inside_distance`

It can also include closing prices for CLV tracking:

- `closing_fighter_a_odds`
- `closing_fighter_b_odds`
- `closing_fight_goes_to_decision_odds`
- `closing_fight_doesnt_go_to_decision_odds`
- `closing_fighter_a_inside_distance_odds`
- `closing_fighter_b_inside_distance_odds`
- `closing_fighter_a_by_decision_odds`
- `closing_fighter_b_by_decision_odds`

## Manual Prop Workflow

You can now export a prop-entry sheet from the fight report, paste sportsbook prices into it, and run a direct model check.

1. Export the template:

```powershell
.\.venv-win\Scripts\python.exe scripts\export_prop_template.py `
  --fight-report cards\ufc_327_prochazka_ulberg\reports\fight_week_report.csv `
  --output my_props.csv
```

That export is now curated by default so it only includes the most realistic props per fight.
If you want the full sheet of every supported prop row, add `--full`.

2. Fill in the `american_odds` column in `my_props.csv` for the rows you want to test.

3. Check those props against the model:

```powershell
.\.venv-win\Scripts\python.exe scripts\check_manual_props.py `
  --fight-report cards\ufc_327_prochazka_ulberg\reports\fight_week_report.csv `
  --props my_props.csv `
  --output cards\ufc_327_prochazka_ulberg\reports\manual_prop_check.csv
```

Supported prop types:

- `moneyline`
- `inside_distance`
- `submission`
- `ko_tko`
- `by_decision`
- `fight_goes_to_decision`
- `fight_doesnt_go_to_decision`

Exact `submission`, `KO/TKO`, and `by_decision` prices are still manual-entry unless the live odds source exposes those props for the fight. The current workflow does not auto-fill those exact prices from The Odds API for every fight, so you should paste the sportsbook price into `my_props.csv` before running the checker.

## Next Steps

1. Replace the CSV input with a real odds source in `data_sources/`.
2. Expand fighter stats and event metadata coverage.
3. Tune projection weights against a larger historical sample.
4. Compare current price to open and closing line.
5. Backtest against historical UFC results before risking real bankroll.
