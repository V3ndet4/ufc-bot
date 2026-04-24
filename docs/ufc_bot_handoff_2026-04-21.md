# UFC Bot Handoff - 2026-04-21

## 2026-04-23 Update
- Old `ufc-bot-next-card-*` scheduled tasks were removed.
- Replaced that with a nightly `ufc-bot-fight-week-watch-night` task at `10:00 PM` local time.
- Nightly watch is news/gym-only; it does not hit the Odds API path.
- Card workflow still keeps the odds wiring intact for later runs.
- Codex config was tightened for lower token use:
- `model = "gpt-5.4-mini"`
- `project_doc_max_bytes = 12000`
- `tool_output_token_limit = 1200`
- `model_reasoning_effort = "medium"`
- `plan_mode_reasoning_effort = "medium"`
- `model_verbosity = "low"`
- `personality = "pragmatic"`

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
- Event prep now writes both:
- `cards/<slug>/data/odds_template.csv` for the live moneyline workflow
- `cards/<slug>/data/modeled_market_template.csv` for full-market Odds API pulls
- The Odds API flow can now also write:
- `cards/<slug>/data/modeled_market_odds.csv`
- Tracked markets in `data/ufc_betting.db`: `moneyline` only
- Graded picks: `60`
- Pending picks: `2`
- Tracked picks currently come from one completed event: `ufc-fn-2026-04-18`
- `models/threshold_policy.json` still resolves to the baseline policy rather than an improved optimized policy
- Historical archive output now writes to:
- `data/historical_market_odds.csv`
- `reports/historical_market_archive_summary.csv`
- Thin non-moneyline markets are now explicitly downgraded to `Report-only` when the archive does not show at least `2` completed events and `8` fights for that market.
- The operator dashboard now includes a `Market Readiness` panel showing archive sample counts and the reason a prop stayed `Report-only`.

## SQLite Clarification
- `data/ufc_betting.db` is intentionally local runtime state and is not committed to GitHub.
- On a fresh clone, the SQLite DB may not exist yet. That is normal.
- The DB is created automatically by the DB-backed scripts such as:
- `scripts/fetch_the_odds_api_odds.py`
- `scripts/record_odds_snapshot.py`
- `scripts/run_value_scan.py --db data/ufc_betting.db`
- `scripts/grade_tracked_picks.py`
- A missing SQLite file is not the real bottleneck.
- The real bottleneck is missing real feedback data inside that DB:
- repeated live `odds_snapshots`
- saved `tracked_picks`
- completed `fight_results`
- graded post-event history across multiple cards
- No real DB history means no trustworthy real-world edge, even if the schema and code are in place.

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
- Wired archive-based recommendation honesty into `scripts/run_value_scan.py` / `models/decision_support.py` so thin prop rows stay visible in the main report but are forced to `Report-only` and kept out of shortlist, betting board, and tracked-pick persistence.
- Pointed `scripts/train_side_model.py` and `scripts/train_confidence_model.py` at `data/historical_market_odds.csv` by default, with `--skip-historical-odds` available for DB-only training.
- `scripts/run_event_pipeline.py` now refreshes the confidence model alongside side/selective/threshold refreshes.
- `scripts/prepare_event.py` now emits a separate modeled-market prep artifact via `scripts/event_manifest.py`:
- `cards/<slug>/data/modeled_market_template.csv`
- `scripts/fetch_the_odds_api_odds.py` can now fill that full-market template directly, and the pipeline refreshes `cards/<slug>/data/modeled_market_odds.csv` as a separate non-snapshot Odds API artifact.
- `scripts/build_operator_dashboard.py` now surfaces prop archive coverage, sample counts, and `Report-only` reasons in a dedicated readiness panel.
- Added targeted tests for:
- decision-market closing backfill
- modeled-market snapshot persistence
- full-market historical archive export
- modeled-market template generation
- modeled-market Odds API artifact filling
- operator dashboard readiness rendering
- modeled-market pipeline refresh wiring

## Best Next Implementation Slice
The next best slice is the fighter-history backbone, not another report-layer tweak:
- make the Greco GitHub UFC dataset the canonical prior-fight history source
- add unmatched-fighter / alias-override reporting so historical joins stop silently degrading training quality
- add per-market evaluation exports so prop markets can be judged by calibration, CLV, and sample size rather than by anecdotal card output

## Concrete Next Work
1. Make the Greco GitHub dataset the canonical historical fighter-history backbone.
2. Keep the completed-card archive as the canonical betting, closing, and result feedback layer.
3. Build a stable unmatched-fight report plus a manual alias-override file in the historical training path.
4. Build more real DB history by capturing live snapshots, persisting tracked picks, and grading multiple completed cards.
5. Retrain side, confidence, and selective models on merged real datasets.
6. Revisit threshold policy training once more than one completed event contributes graded picks.
7. Add per-market evaluation outputs for ROI, CLV, calibration, and sample size by market bucket.
8. Decide whether `build_fight_week_report.py` should consume the new event-level `modeled_market_odds.csv` artifact before falling back to live alternative-market fetches.

## Prompt For The Next Window
Read `docs/ufc_bot_handoff_2026-04-21.md` and continue from there.

Focus on:
- using the Greco GitHub historical dataset as the fighter-history backbone
- keeping the completed-card archive as the betting/result feedback layer
- tightening fighter matching with explicit unmatched and alias-override visibility
- treating local SQLite as generated runtime state, not committed source data
- building enough real graded DB history to support honest recommendation gating
- adding market-level evaluation so prop progress can be measured honestly
- deciding whether the new `modeled_market_odds.csv` artifact should feed the report layer directly
