@echo off
REM ============================================================
REM  run_app.bat
REM  One-click script to set up and launch the B2B Churn
REM  Risk Predictor dashboard on Windows.
REM
REM  Usage: double-click this file, or run "run_app.bat" from
REM  the terminal inside the churn-predictor folder.
REM ============================================================

echo.
echo === B2B Churn Risk Predictor ===
echo.

REM --- Step 1: Create virtual environment if it doesn't exist ---
if not exist "venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/5] Virtual environment already exists, skipping.
)

REM --- Step 2: Install dependencies ---
echo [2/5] Installing dependencies...
.\venv\Scripts\python.exe -m pip install -r requirements.txt -q

REM --- Step 3: Create data folder if missing ---
if not exist "data\processed" (
    echo [3/5] Creating data\processed folder...
    mkdir data\processed
) else (
    echo [3/5] data\processed folder already exists, skipping.
)

REM --- Step 4: Run the pipeline only if predictions.csv is missing ---
if not exist "data\processed\predictions.csv" (
    echo [4/5] Running data pipeline - this may take a minute...
    .\venv\Scripts\python.exe data_prep.py
    .\venv\Scripts\python.exe train_model.py
    .\venv\Scripts\python.exe explain.py
) else (
    echo [4/5] Pipeline output already exists, skipping.
    echo        (Delete files in data\processed\ if you want to regenerate them.)
)

REM --- Step 5: Launch the dashboard ---
echo [5/5] Launching dashboard...
echo.
echo Login with -- Username: stakeholder   Password: demo123
echo.
.\venv\Scripts\python.exe -m streamlit run app.py

pause
