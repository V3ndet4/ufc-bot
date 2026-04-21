# UFC Bot Handoff - 2026-04-21

## Current State
- Repo: `ufc-bot`
- Active event pointer: `events/ufc_fn_sterling_zalal.json`
- Active event: `UFC Fight Night: Sterling vs. Zalal`
- Event date: `2026-04-25`
- Test baseline: `125 passed`
- Canonical runtime: manifest-driven PowerShell workflow

## What We Confirmed
- The bot is operational and generates current card outputs, not just a scaffold.
- The active workflow already produces:
- `fight_week_report.csv`
- `lean_board.csv`
- `betting_board.csv`
- `parlay_board.csv`
- `line_movement.svg`
- `operator_dashboard.html`
- `data/ufc_betting.db` currently contains:
- `odds_snapshots`: `710`
- `tracked_picks`: `62`
- `fight_results`: `12`

## Main Conclusion
Using the GitHub UFC history from `Greco1899/scrape_ufc_stats` would make the bot materially more complete on the modeling side, but it is not enough by itself to make the bot fully complete or trustworthy.

It already plugs into:
- `data_sources/external_ufc_history.py`
- `scripts/enrich_fighter_stats_with_external_ufc_history.py`
- `data_sources/historical_training.py`
- `scripts/train_side_model.py`
- `scripts/train_confidence_model.py`

What it helps with:
- leak-safe prior-fight features
- better fighter profile and history fields
- opponent-strength adjustments
- stronger model training inputs

What it does not solve:
- tracked picks are still mostly `moneyline` in the graded sample
- the full board is not backed by enough historical graded market data
- prop recommendation trust is still thin because historical graded coverage is small
- current threshold policy is still materially negative on the graded sample

## Important Repo Facts
- Snapshot persistence now supports:
- `moneyline`
- `fight_goes_to_decision`
- `fight_doesnt_go_to_decision`
- `inside_distance`
- `by_decision`
- Tracked markets in `data/ufc_betting.db`: `moneyline` only
- Graded picks: `60`
- Pending picks: `2`
- Tracked picks currently come from one completed event: `ufc-fn-2026-04-18`
- `models/threshold_policy.json` still resolves to the baseline policy rather than an improved optimized policy
- Historical archive output now writes to:
- `data/historical_market_odds.csv`
- `reports/historical_market_archive_summary.csv`

Interpretation:
- the bot may be finding some CLV or pricing efficiency
- the recommendation and betting logic are not yet trustworthy enough

## Historical Data Status
There are two historical data paths in the repo.

### Older direct scraper path
- `scripts/build_historical_database.py`
- `scripts/collect_real_ufc_data.py`
- `data_sources/ufc_history.py`
- `data_sources/historical_odds.py`

Status:
- partially implemented
- useful later
- not the fastest path to a trustworthy bot right now

### Better current path: archive from completed card folders
- `data_sources/historical_archive.py`
- `scripts/build_historical_moneyline_archive.py`
- `scripts/grade_tracked_picks.py`

Status:
- already tested
- better suited to building training and feedback data from the bot's own workflow
- now upgraded to export a full-market archive instead of only moneyline rows

## Biggest Product Gap
The board and report layer are ahead of the market-history layer.

Right now:
- props can appear in reports
- prop logic exists in model and report code
- grading supports props in principle

But in practice:
- live snapshot capture now supports the modeled prop set
- completed results can now backfill decision-prop closes from snapshot history
- there is not enough historical graded prop coverage to trust full-board recommendations

## Chosen Direction
- Build data first
- Keep the full board as the target product

## Completed In This Window
- Expanded live Odds API snapshot persistence beyond moneyline to the modeled market set:
- `moneyline`
- `fight_goes_to_decision`
- `fight_doesnt_go_to_decision`
- `inside_distance`
- `by_decision`
- Added decision-market closing odds backfill in `scripts/fetch_event_results.py` from the latest eligible pre-event snapshots.
- Reworked `data_sources/historical_archive.py` so the completed-card archive exports full-market rows rather than just moneyline rows.
- Updated archive refresh commands to write:
- `data/historical_market_odds.csv`
- `reports/historical_market_archive_summary.csv`
- Added targeted tests for:
- decision-market closing backfill
- modeled-market snapshot persistence
- full-market historical archive export

## Best Next Implementation Slice
Use the new full-market archive to tighten recommendation honesty and training inputs:
- wire archive consumers and training scripts to the full-market historical dataset
- explicitly surface thin-sample prop rows as `Report-only`
- decide whether `prepare_event` / odds template generation should emit full-market rows directly instead of relying on snapshot-only prop capture

## Concrete Next Work
1. Make the Greco GitHub dataset the canonical historical fighter-history backbone.
2. Keep the completed-card archive as the canonical betting, closing, and result feedback layer.
3. Make downstream consumers use `data/historical_market_odds.csv` instead of assuming moneyline-only history.
4. Retrain side, confidence, and selective models on merged real datasets.
5. Revisit threshold policy training once more than one completed event contributes graded picks.
6. Make unsupported or thin-sample prop rows explicitly show `Report-only` instead of looking recommendation-ready.
7. Decide whether event prep should generate full-market template rows up front.

## Prompt For The Next Window
Read `docs/ufc_bot_handoff_2026-04-21.md` and continue from there.

Focus on:
- using the Greco GitHub historical dataset as the fighter-history backbone
- wiring consumers to the new full-market archive
- deciding whether to generate full-market template rows during event prep
- keeping recommendations honest until each market has enough closed, graded history
