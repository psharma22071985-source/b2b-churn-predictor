# B2B Enterprise Churn Risk Predictor

## Project Title & Purpose

**B2B Enterprise Churn Risk Predictor** is a machine learning-powered dashboard that predicts which B2B telecom accounts (modeled on a self-service platform such as AT&T's Premier) are at risk of churn — and explains *why*, using SHAP-based explainability.

The project is aimed at solving a real business problem: B2B account teams often only discover a customer is at risk *after* they've already disengaged or cancelled, because warning signs (support tickets, billing delays, low platform logins, abandoned upgrade flows) are scattered across separate systems. This project unifies those signals into a single account-level churn risk score, explains the top contributing factors per account, and surfaces the result in a stakeholder-friendly dashboard with suggested retention actions.

## Team Members

| Name | GitHub Username |
|---|---|
| Payal Sheoran | [psharma22071985-source](https://github.com/psharma22071985-source) |

*(Solo project — no additional team members.)*

## How to Run the Code

### Step 1: Clone the repository
```
git clone https://github.com/psharma22071985-source/b2b-churn-predictor.git
cd b2b-churn-predictor
```

### Step 2: Create the virtual environment
```
python3 -m venv venv
```

### Step 3: Activate the virtual environment
**Windows:**
```
venv\Scripts\activate
```
**Mac/Linux:**
```
source venv/bin/activate
```
✅ Confirm `(venv)` appears at the start of your terminal prompt before continuing.

### Step 4: Install dependencies
```
pip install -r requirements.txt
```
*(On Windows, if `pip` resolves to the wrong Python installation, use the venv's Python directly instead:)*
```
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Step 5: Create the output data folder (first-time setup only)
```
mkdir data\processed
```
*(Mac/Linux: `mkdir -p data/processed`. Skip this step if the folder already exists.)*

### Step 6: Run the data/model pipeline, in this exact order
```
python data_prep.py
python train_model.py
python explain.py
```
*(Windows, if needed: prefix each with `.\venv\Scripts\python.exe` instead of `python`.)*

Each script depends on the previous one's output, so run them in sequence. You only need to re-run these three if you change the code in `data_prep.py`, `train_model.py`, or `explain.py` — otherwise, skip straight to Step 7.

### Step 7: Launch the dashboard
```
streamlit run app.py
```
*(Windows, if needed: `.\venv\Scripts\python.exe -m streamlit run app.py`)*

### Step 8: Log in
Your browser opens automatically to `http://localhost:8501`. Log in with:
- **Username:** `stakeholder`
- **Password:** `demo123`

### Step 9: Stop the app
Back in the terminal, press `Ctrl+C`.

## Important Setup Instructions

- **Python version:** 3.10+ recommended.
- **First-time setup:** the `data/processed/` folder must exist before running `data_prep.py`. If it's missing:
  ```
  mkdir -p data/processed
  ```
- **Windows users:** if you have multiple Python installations, always invoke scripts through the virtual environment's own Python (`.\venv\Scripts\python.exe`) rather than relying on the system `python` command, to avoid packages installing to the wrong environment.
- **Data note:** the dataset (`data_prep.py`) generates a statistically realistic *synthetic* dataset — it is not pulled from a live production system, since real enterprise account data isn't publicly available. Churn labels are generated from weighted, realistic relationships between features (e.g., short tenure + high ticket volume + low satisfaction increases churn probability).
- **Re-running the pipeline:** if you change `data_prep.py` or want fresh data, delete the existing CSVs in `data/processed/` and `model.pkl`, then re-run all three pipeline scripts in order before relaunching the dashboard.

## Project Structure

```
b2b-churn-predictor/
├── project_spec.md              # Full project specification (AI context anchor)
├── data_prep.py                  # Generates the synthetic B2B account dataset
├── train_model.py                 # Trains & compares Logistic Regression and XGBoost
├── explain.py                     # SHAP explainability + suggested retention actions
├── app.py                         # Streamlit dashboard (5 stakeholder views + login)
├── requirements.txt
├── data/
│   ├── raw/                       # (place any real source CSVs here, if used)
│   └── processed/                 # generated: accounts.csv, predictions.csv, etc.
└── model.pkl                      # generated: trained XGBoost pipeline
```

## Key Features

- 5 categories of churn signals: Contract & Account Health, Billing & Payments, Support & Service, Digital Engagement, Product & Shop Usage
- Dual-model comparison: Logistic Regression (baseline) vs. XGBoost (primary)
- SHAP-based per-account explainability with plain-language risk drivers
- 5-view stakeholder dashboard: Executive Summary, Account Explorer, Account Detail, Account Manager View, Model Insights
- Demo login gate for a realistic stakeholder-tool experience
