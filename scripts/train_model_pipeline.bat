@echo off
echo UFC Model Training Pipeline
echo ============================
echo.

set PYTHON=.venv-win\Scripts\python.exe

echo Step 1: Generate sample training data
echo ----------------------------------------
%PYTHON% scripts\generate_sample_training_data.py --n-fights 2000
if errorlevel 1 goto error

echo.
echo Step 2: Train model
echo ----------------------------------------
%PYTHON% models\trainer.py --save-path models\ufc_model.pkl
if errorlevel 1 goto error

echo.
echo Step 3: Run walk-forward backtest
echo ----------------------------------------
%PYTHON% backtests\run_model_backtest.py --db-path data\historical_ufc.db --model-path models\ufc_model.pkl --evaluation-mode walk_forward --output reports\model_backtest.csv --detailed-output reports\model_backtest_picks.csv
if errorlevel 1 goto error

echo.
echo ========================================
echo PIPELINE COMPLETE
echo ========================================
echo.
echo Check reports\model_backtest.csv for results
echo.
goto end

:error
echo.
echo ERROR: Pipeline failed!
if not defined CI pause
exit /b 1

:end
if not defined CI pause
