"""
train_model.py
Trains a Logistic Regression baseline and an XGBoost classifier on the expanded
B2B telecom account dataset (5 feature categories), compares performance, and
saves the fitted model + preprocessing pipeline for use in explain.py.

Output:
    model.pkl
    data/processed/test_split.csv
    data/processed/model_comparison.csv
"""

import pickle
from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

NUMERIC_FEATURES = [
    # Category 1: Contract & Account Health
    "tenure_months", "days_to_contract_renewal", "auto_renewal_enabled",
    "ctn_count", "device_count", "annual_revenue", "pct_lines_disconnected_90d",
    # Category 2: Billing & Payments
    "monthly_charges", "avg_invoice_delay_days", "late_payments_12mo", "autopay_enabled",
    # Category 3: Support & Service
    "support_tickets_90d", "support_calls_90d", "avg_resolution_time_hours",
    "escalations_90d", "csat_score",
    # Category 4: Digital Engagement
    "platform_logins_90d", "days_since_last_login",
    "self_service_completion_rate", "upgrade_funnel_abandon_rate", "dropoff_severity",
    # Category 5: Product & Shop Usage
    "rate_plan_changes_90d", "feature_adoption_count",
    "device_upgrades_90d", "accessory_purchases_90d",
]
CATEGORICAL_FEATURES = ["contract_type", "industry_segment", "primary_dropoff_page"]
TARGET = "churn_label"


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def main():
    log("Loading processed accounts data...")
    df = pd.read_csv("data/processed/accounts.csv")

    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in accounts.csv: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    log("Training Logistic Regression baseline...")
    log_reg_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    log_reg_pipe.fit(X_train, y_train)
    lr_preds = log_reg_pipe.predict(X_test)
    lr_probs = log_reg_pipe.predict_proba(X_test)[:, 1]

    log("Training XGBoost classifier...")
    xgb_pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9,
            random_state=42, eval_metric="logloss"
        )),
    ])
    xgb_pipe.fit(X_train, y_train)
    xgb_preds = xgb_pipe.predict(X_test)
    xgb_probs = xgb_pipe.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        "model": ["Logistic Regression", "XGBoost"],
        "accuracy": [accuracy_score(y_test, lr_preds), accuracy_score(y_test, xgb_preds)],
        "f1_score": [f1_score(y_test, lr_preds), f1_score(y_test, xgb_preds)],
        "roc_auc": [roc_auc_score(y_test, lr_probs), roc_auc_score(y_test, xgb_probs)],
    })
    log("Model comparison:\n" + results.to_string(index=False))
    results.to_csv("data/processed/model_comparison.csv", index=False)

    test_df = df.loc[idx_test].copy()
    test_df.to_csv("data/processed/test_split.csv", index=False)

    with open("model.pkl", "wb") as f:
        pickle.dump(xgb_pipe, f)

    log("Saved model.pkl and data/processed/test_split.csv")
    log("Training complete.")


if __name__ == "__main__":
    main()
