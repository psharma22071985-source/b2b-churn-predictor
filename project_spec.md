# Project Specification Document — Telecom B2B Churn Risk Predictor
## (Modeled on a large carrier such as AT&T)

## 1. Project Overview & Context

**The Problem:**
Telecom account teams often see service-quality complaints (billing disputes, outages, slow support resolution) and churn outcomes as two separate, disconnected signals. Internally, retention analysts lack a single view connecting *why* customers are unhappy to *which* accounts are actually at risk of leaving. Externally, customers experience repeated billing errors or unresolved complaints and eventually churn — often after multiple issues that were never escalated in time.

**The Solution:**
A churn prediction system that combines structured account data (tenure, contract type, billing, service usage) with real-world, publicly available telecom complaint patterns, to predict which accounts are at risk and explain *why* — surfaced through a complete, usable dashboard built for a retention/account team.

**Target Audience/User:**
- **Primary:** Enterprise/retention account managers at a telecom provider — need a ranked list of at-risk accounts with clear reasons, not just a score.
- **Secondary:** Retention/product leadership — need portfolio-level visibility into which complaint themes are driving churn nationally.

---

## 2. The Technology Stack

**Core Language:** Python 3.11

**ML Model (explicitly specified):**
- **Primary model:** XGBoost Classifier — chosen for strong performance on structured/tabular churn data and native compatibility with SHAP for explainability.
- **Baseline comparison model:** Logistic Regression (scikit-learn) — used as an interpretable benchmark to show performance lift from XGBoost.
- **Explainability:** SHAP (TreeExplainer) — generates per-account, per-prediction feature attribution.

**Frameworks & Libraries:**
- pandas, numpy — data processing
- scikit-learn — preprocessing, Logistic Regression baseline, train/test split, evaluation metrics
- XGBoost — primary classifier
- SHAP — explainability
- Streamlit — full UI/dashboard layer
- plotly — interactive charts within the dashboard

**AI-Assisted Development Tooling (open-source):**
- Windsurf or Cursor (free tier) as the primary build environment
- This `project_spec.md` file used as the persistent context anchor fed to the AI at the start of each session

**Hardware/Deployment:** Runs entirely locally (laptop, CPU only). No cloud hosting, no external database service — local CSV + SQLite only.

---

## 3. Core Features & MVP

**Phase 1 (MVP):**
- Load and clean the Telco Customer Churn dataset (tenure, contract type, monthly charges, service subscriptions, churn label)
- Load and summarize FCC Consumer Complaints Data filtered to a telecom carrier, to build a "national complaint theme" reference table (top issue categories and their relative frequency)
- Train Logistic Regression baseline, then XGBoost; compare accuracy/F1/AUC
- Generate SHAP-based top-3 risk drivers per account
- Build the full Streamlit UI described in Section 6 below

**Phase 2 (Enhancements):**
- Cross-reference an account's SHAP drivers against the national FCC complaint-theme table (e.g., flag when an account's top driver matches a nationally common complaint category)
- Add a churn-risk trend view over time
- Add CSV export and filter/sort controls

**Out of Scope (Anti-Goals):**
- No live CRM/billing system integration — static historical data only
- No user authentication or multi-user roles
- No automated customer outreach (email/SMS) — internal decision-support only
- No real-time streaming scoring — batch-scored only

---

## 4. Data Architecture & State Management

**Inputs:**
1. **Telco Customer Churn dataset** (Kaggle/IBM) — customer_id, tenure_months, contract_type, monthly_charges, total_charges, services subscribed, churn_label
2. **FCC Consumer Complaints Data** (public, opendata.fcc.gov) — complaint date, issue category, product/service type, company, state — filtered to the modeled carrier, aggregated into complaint-theme frequency, not used per-account (no individual complaint-to-account linkage, since these are separate public datasets)

**Outputs:**
- Per-account churn probability score (0–1)
- Top 3 SHAP-driven risk factors per account
- National complaint-theme reference chart (context panel)
- Exportable CSV of the full scored account table

**Schemas:**
```
accounts table:
- account_id (string, PK)
- tenure_months (int)
- contract_type (string)
- monthly_charges (float)
- total_charges (float)
- services_subscribed (string/list)
- churn_label (int, 0/1)  -- training data only

predictions table:
- account_id (string, FK)
- churn_probability (float)
- top_driver_1 / top_driver_2 / top_driver_3 (string)
- scored_date (date)

complaint_themes table:
- issue_category (string)
- complaint_count (int)
- pct_of_total (float)
- period (string, e.g., "last 12 months")
```

---

## 5. AI System Rules & Aesthetics

**Code Style:** Clear, readable Python; docstrings on all functions; separate modules (`data_prep.py`, `train_model.py`, `explain.py`, `app.py`).

**Error Handling:** Never fail silently; log errors with timestamps; validate required columns exist before processing; explicitly handle missing values.

**UI/UX Aesthetics:** Clean, minimal, business-user-first (not data-scientist-first). Dark-mode optional, high-contrast, color-coded risk tiers (red/yellow/green). Technical detail (SHAP plots, raw model output) tucked into an expandable "details" section, not the main view.

---

## 6. Complete UI / User Experience

This is the full end-to-end experience a retention account manager would have using the app.

**Screen 1 — Portfolio Overview (landing page)**
- A sortable table: Account ID | Churn Risk Score | Risk Tier (color-coded) | Contract Type | Tenure
- Sorted by churn risk descending by default
- A summary strip at top: total accounts, % high-risk, average churn probability
- A small bar chart (plotly) showing the national FCC complaint-theme breakdown for context (e.g., "38% of complaints nationally are billing-related")

**Screen 2 — Account Detail (drill-in, opens on row click)**
- Account's churn probability displayed prominently (e.g., large percentage with color)
- Top 3 SHAP-driven risk factors, written in plain language (e.g., "Short tenure (4 months) is the strongest factor increasing this account's risk")
- A note flagging if the account's top driver matches a nationally common complaint theme (Phase 2 feature)
- A "Notes" text box (session-only, not persisted) where the account manager can jot down planned action

**Screen 3 — Model Insights (secondary tab, for leadership/analysts)**
- Logistic Regression vs. XGBoost performance comparison (accuracy, F1, AUC) in a simple table
- Overall SHAP summary plot (which features matter most across all accounts, not just one)
- National complaint-theme reference table in full

**Navigation:** Simple top tab bar — "Portfolio" | "Model Insights" — no login, no multi-page routing complexity, consistent with the MVP/anti-goals scope.

**Export:** A "Download CSV" button on Screen 1, exporting the full scored table.

---

## 7. Data Sources (Researched, Live/Public)

- **Telco Customer Churn Dataset** (Kaggle/IBM): https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **FCC Consumer Complaints Data** (live, public, updates nightly): https://www.fcc.gov/consumer-complaints-center-data
  - Direct CSV download endpoint: https://opendata.fcc.gov/api/views/3xyp-aqkj/rows.csv?accessType=DOWNLOAD

---

## 8. Build Plan — Windsurf / Cursor

**Step 1 — Environment setup**
```
mkdir churn-predictor && cd churn-predictor
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost shap streamlit plotly
```

**Step 2 — Feed the AI this spec file first**
Open the folder in Windsurf/Cursor. First prompt to the AI:
> "Read project_spec.md fully before writing any code. Build strictly to the MVP scope in Section 3, using the schema in Section 4 and the UI design in Section 6."

**Step 3 — Build in order, testing after each file**
1. `data_prep.py` — load & clean Telco churn CSV; load & aggregate FCC complaints CSV into the complaint_themes table
2. `train_model.py` — train Logistic Regression, then XGBoost; print comparison metrics
3. `explain.py` — SHAP top-3 driver extraction per account
4. `app.py` — Streamlit app implementing Screens 1–3 exactly as described in Section 6

**Step 4 — Version control**
`git init` early; commit after each working phase.

**Step 5 — Final check against this spec**
Before submission, re-read Sections 3 and 6 and confirm every listed screen/feature is actually present in the running app — this is what "complete UI experience" means for grading purposes.
