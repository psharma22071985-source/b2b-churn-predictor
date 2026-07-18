# B2B Enterprise Churn Risk Predictor

A working prototype that predicts churn risk for B2B enterprise telecom accounts,
with SHAP-based explainability and a Streamlit dashboard. See `project_spec.md`
for the full project specification.

## How to Run

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the pipeline in order (each step depends on the previous one's output):
   ```
   python data_prep.py
   python train_model.py
   python explain.py
   ```

4. Launch the dashboard:
   ```
   streamlit run app.py
   ```
   This opens the app in your browser, usually at http://localhost:8501

## Project Structure

```
churn-predictor/
├── project_spec.md              # Full project specification (AI context anchor)
├── data_prep.py                 # Generates realistic B2B account dataset
├── train_model.py               # Trains Logistic Regression + XGBoost, compares
├── explain.py                   # SHAP explainability, generates predictions.csv
├── app.py                       # Streamlit dashboard (3 screens)
├── requirements.txt
├── data/
│   ├── raw/                     # (place any real source CSVs here)
│   └── processed/               # generated: accounts.csv, predictions.csv, etc.
└── model.pkl                    # generated: trained XGBoost pipeline
```

## Data Note

The dataset is synthetically generated (`data_prep.py`) using statistically
grounded patterns modeled after real telecom churn drivers (tenure, contract type,
billing delays, support ticket volume, CSAT, usage trend), enriched with Faker for
realistic B2B company identities. The "complaint themes" reference chart is
structured to match categories tracked in the FCC's public Consumer Complaints
Data (https://www.fcc.gov/consumer-complaints-center-data) — for a live pull,
download the current CSV from that page and replace the static reference table
in `data_prep.py`.
