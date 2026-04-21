# UFC Bot Handoff - Historical Data Build

This note describes the older archived `upcoming_card` / UFC London workflow.
Those files now live under `cards/upcoming_card/...` instead of the repo root, `data/`, and `reports/`.

## Current Workflow

This repo is currently set up to:

1. Fetch fighter stats from ESPN using a fighter mapping CSV.
2. Merge optional manual context flags into that mapping.
3. Fill current UFC card moneylines from BestFightOdds.
4. Enrich those odds with BestFightOdds open and current range fields.
5. Build a fight-week report comparing market lines, model output, confidence, and manual context.
6. Run projections and a value scan with confidence-aware downweighting plus hard filters for incomplete fighter data.

## NEW: Historical Data Pipeline

We've now added the foundation for model training:

### Files Added

1. `data_sources/ufc_history.py` - Scrapes historical UFC fight outcomes from ufcstats.com
2. `data_sources/historical_odds.py` - Scrapes closing lines from BestFightOdds
3. `data_sources/historical_fighter_stats.py` - Builds fighter stats as of fight time (no future leakage!)
4. `models/trainer.py` - Trains logistic regression with time-series cross-validation
5. `scripts/build_historical_database.py` - Master script to collect all historical data
6. `scripts/generate_sample_training_data.py` - Creates synthetic data for testing
7. `scripts/train_model_pipeline.bat` - End-to-end training pipeline

### Database Schema

`data/historical_ufc.db` contains:
- `events` - Event metadata
- `fights` - All UFC fights with outcomes
- `fight_odds` - Open and closing lines
- `fighter_histories` - Fighter stats as of each fight date (NO future data)

### Training Pipeline

```powershell
# Generate sample data (for testing)
python scripts\generate_sample_training_data.py --n-fights 2000

# Train model
python models\trainer.py --save-path models\ufc_model.pkl

# Or run full pipeline
.\scripts\train_model_pipeline.bat
```

### Key Design Principles

1. **No Data Leakage** - Fighter stats are as of fight date, not current
2. **Time-Series Split** - Validation uses only past data
3. **Market as Feature** - Closing line is used, not just predicted against
4. **Calibration** - Model outputs true probabilities, not just rankings

## Next Steps for Real Data

1. **Scrape actual UFC fight history**
   ```powershell
   python scripts\build_historical_database.py --limit 100
   ```

2. **Match with BestFightOdds historical**
   - Need to map ufcstats fighters to BestFightOdds names
   - Collect closing lines for past 5+ years

3. **Build fighter stat histories**
   - For each fighter, track stats accumulating over time
   - This is the most complex piece

4. **Train on real data**
   - Replace synthetic data with real scraped data
   - Tune hyperparameters
   - Validate on recent fights not used in training

## Current Status

- ✅ Sample data generator (testable today)
- ✅ Training pipeline (testable today)
- ✅ Model evaluation with time-series CV
- ⏳ Real UFC data scraping (next step)
- ⏳ BestFightOdds historical matching
- ⏳ Fighter stat history builder

## What Works Now

Run this to see the training pipeline in action:
```powershell
.\scripts\train_model_pipeline.bat
```

This will:
1. Generate 2000 synthetic fights
2. Train a logistic regression model
3. Show cross-validation results
4. Output feature importance

## Important Files

- `cards/upcoming_card/inputs/fighter_map.csv`
  - fighter name to ESPN URL mapping for the March 21, 2026 UFC London card
- `cards/upcoming_card/inputs/fighter_context.csv`
  - manual context inputs
  - columns:
    - `fighter_name`
    - `short_notice_flag`
    - `short_notice_acceptance_flag`
    - `short_notice_success_flag`
    - `new_gym_flag`
    - `new_contract_flag`
    - `cardio_fade_flag`
    - `context_notes`
- `cards/upcoming_card/data/fighter_stats.csv`
  - ESPN-derived fighter stats output
- `cards/upcoming_card/data/odds_template.csv`
  - base odds template for the card
- `cards/upcoming_card/data/bfo_odds.csv`
  - BestFightOdds-enriched odds file
- `cards/upcoming_card/reports/bfo_value_bets.csv`
  - latest value scan output
- `cards/upcoming_card/reports/fight_week_report.csv`
  - fight-level comparison report for open/current market, model probabilities, confidence, and context flags

## PowerShell Commands

Run from:

```powershell
(.venv-win) PS <repo-root>
```

Primary operator commands:

```powershell
.\scripts\set_next_card.ps1 --status
.\scripts\run_next_card.ps1
.\scripts\prepare_next_card.ps1
.\scripts\refresh_next_card_odds.ps1
.\scripts\scan_next_card.ps1
.\scripts\grade_next_card.ps1
```

Refresh ESPN stats with manual context:

```powershell
python scripts\fetch_espn_stats.py `
  --mapping cards\upcoming_card\inputs\fighter_map.csv `
  --context cards\upcoming_card\inputs\fighter_context.csv `
  --output cards\upcoming_card\data\fighter_stats.csv
```

Refresh BestFightOdds current odds plus open/range history:

```powershell
python scripts\fetch_bestfightodds_event_odds.py `
  --template cards\upcoming_card\data\odds_template.csv `
  --event-url https://www.bestfightodds.com/events/ufc-london-4081 `
  --include-history `
  --output cards\upcoming_card\data\bfo_odds.csv
```

Run the value scan:

```powershell
$env:MIN_MODEL_CONFIDENCE='0.55'
$env:MIN_STATS_COMPLETENESS='0.80'
$env:EXCLUDE_FALLBACK_ROWS='true'
python scripts\run_value_scan.py `
  --input cards\upcoming_card\data\bfo_odds.csv `
  --fighter-stats cards\upcoming_card\data\fighter_stats.csv `
  --output cards\upcoming_card\reports\bfo_value_bets.csv
```

Bash/WSL remains available as a secondary path:

```bash
./scripts/run_london_card.sh
```

Build the fight-week comparison report:

```powershell
python scripts\build_fight_week_report.py `
  --odds cards\upcoming_card\data\bfo_odds.csv `
  --fighter-stats cards\upcoming_card\data\fighter_stats.csv `
  --output cards\upcoming_card\reports\fight_week_report.csv
```

Run projections only:

```powershell
python scripts\project_fight_probs.py `
  --odds cards\upcoming_card\data\bfo_odds.csv `
  --fighter-stats cards\upcoming_card\data\fighter_stats.csv `
  --output cards\upcoming_card\reports\projected_probs.csv
```

## Current Model Additions

The ESPN fighter export now includes:

- `age_years`
- `days_since_last_fight`
- `losses_in_row`
- `first_round_finish_wins`
- `first_round_finish_rate`
- `recent_result_score`
- `recent_strike_margin_per_min`
- `recent_grappling_rate`
- `ufc_fight_count`
- `ufc_debut_flag`
- `stats_completeness`
- `fallback_used`

The model also uses:

- age-curve weighting by weight class
- layoff upside effect
- recent-form effect
- UFC debut penalty
- line movement input when `open_american_odds` exists
- confidence shrinkage toward 50/50 when data is incomplete

## Important Limitations

- ESPN public fighter pages do not expose true takedown defense directly.
  - `takedown_defense_pct` is currently approximated from takedown accuracy.
- Some fighters do not have full ESPN rate tables.
  - those rows get fallback zeroes for some rate features
  - `stats_completeness` and `fallback_used` indicate this
- For upcoming fights, there is no true closing line yet.
  - current file includes:
    - `american_odds`
    - `open_american_odds`
    - `current_best_range_low`
    - `current_best_range_high`

## Suggested Next Improvements

1. Add historical BestFightOdds scrape support for backtests.
2. Add opponent-strength adjustment from prior UFC-level opponents.
3. Add a cleaner manual fight-week notes workflow from `cards/upcoming_card/inputs/fighter_context.csv`.
4. Tune default thresholds for:
   - `MIN_MODEL_CONFIDENCE`
   - `MIN_STATS_COMPLETENESS`
   - `EXCLUDE_FALLBACK_ROWS`

## Current Status

- Unit tests pass.
- ESPN fetch works.
- BestFightOdds event fill works.
- BestFightOdds open/range enrichment works.
- Fight-week comparison report works.
- Value scan supports fallback exclusion and minimum stats completeness.
- The March 21, 2026 UFC London card is loaded into repo files and ready to refresh.
- **NEW**: Model training pipeline scaffolded and testable with synthetic data.
