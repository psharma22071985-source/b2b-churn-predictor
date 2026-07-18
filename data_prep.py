"""
data_prep.py
Generates a realistic B2B enterprise telecom account dataset for churn prediction,
modeled on a self-service platform (e.g., AT&T Premier-style) where B2B billing
account admins manage many employee wireless lines: rate plans, features, data
plans, mobile share/family plans, device upgrades, and accessory purchases.

Data is organized into 5 business-relevant categories so churn drivers can be
explained at the category level, not just the individual-feature level:
  1. Contract & Account Health
  2. Billing & Payments
  3. Support & Service
  4. Digital Engagement (Adobe Analytics-style behavioral data)
  5. Product & Shop Usage

Output:
    data/processed/accounts.csv
    data/processed/complaint_themes.csv
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime

RANDOM_SEED = 42
N_ACCOUNTS = 6000

np.random.seed(RANDOM_SEED)
fake = Faker()
Faker.seed(RANDOM_SEED)

INDUSTRIES = [
    "Banking & Financial Services", "Pharmaceuticals", "Healthcare",
    "Small Business / Retail", "Manufacturing", "Logistics",
    "Education", "Hospitality", "Technology",
    "Government", "Energy & Utilities"
]
CONTRACT_TYPES = ["Month-to-Month", "1-Year", "2-Year"]

# Pool of account managers -- each manages a portfolio of accounts, used for the
# Account Manager Performance view (identifying reps who may need coaching support).
ACCOUNT_MANAGERS = [f"AM-{i:03d}" for i in range(1, 26)]  # 25 account managers

# Digital funnel stages, Adobe Analytics-style -- where an account's admin most
# commonly drops off during self-service tasks (upgrades, plan changes, checkout).
FUNNEL_STAGES = [
    "Login / Authentication", "Plan & Feature Selection", "Device Browse/Compare",
    "Add to Cart", "Checkout / Payment", "Order Confirmation", "Support Handoff",
]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def generate_accounts(n: int) -> pd.DataFrame:
    """Generate n synthetic B2B telecom self-service platform accounts."""

    # --- 1. Contract & Account Health ---
    tenure_months = np.random.gamma(shape=2.0, scale=14, size=n).clip(1, 84).round().astype(int)
    contract_type = np.random.choice(CONTRACT_TYPES, size=n, p=[0.45, 0.30, 0.25])
    days_to_contract_renewal = np.random.randint(1, 365, size=n)
    auto_renewal_enabled = np.random.choice([1, 0], size=n, p=[0.65, 0.35])
    ctn_count = np.random.poisson(lam=45, size=n).clip(2, None)
    pct_lines_disconnected_90d = np.random.exponential(scale=3, size=n).clip(0, 40).round(1)

    # --- 2. Billing & Payments ---
    monthly_charges = (ctn_count * np.random.normal(45, 8, size=n)).clip(200, 50000).round(2)
    avg_invoice_delay_days = np.random.exponential(scale=4, size=n).clip(0, 60).round(1)
    late_payments_12mo = np.random.poisson(lam=1.2, size=n)
    autopay_enabled = np.random.choice([1, 0], size=n, p=[0.7, 0.3])

    # --- 3. Support & Service ---
    support_tickets_90d = np.random.poisson(lam=2.5, size=n)
    support_calls_90d = np.random.poisson(lam=1.8, size=n)
    avg_resolution_time_hours = np.random.gamma(shape=2, scale=6, size=n).clip(0.5, 96).round(1)
    escalations_90d = np.random.poisson(lam=0.4, size=n)
    csat_score = np.random.normal(loc=7.5, scale=1.8, size=n).clip(1, 10).round(1)

    # --- 4. Digital Engagement (Adobe Analytics-style) ---
    platform_logins_90d = np.random.poisson(lam=12, size=n)
    days_since_last_login = np.random.exponential(scale=15, size=n).clip(0, 180).round().astype(int)
    self_service_completion_rate = np.random.beta(5, 2, size=n).round(2)  # 0-1
    upgrade_funnel_abandon_rate = np.random.beta(2, 5, size=n).round(2)   # 0-1
    # Adobe Analytics-style: which funnel stage this account's admin most often
    # abandons on, and how severe that drop-off is relative to peers.
    funnel_stage_weights = [0.10, 0.18, 0.14, 0.12, 0.28, 0.05, 0.13]  # checkout/payment is the biggest gap
    primary_dropoff_page = np.random.choice(FUNNEL_STAGES, size=n, p=funnel_stage_weights)
    dropoff_severity = np.random.beta(2, 4, size=n).round(2)  # 0-1, higher = bigger experience gap

    # --- 5. Product & Shop Usage ---
    rate_plan_changes_90d = np.random.poisson(lam=0.8, size=n)
    feature_adoption_count = np.random.poisson(lam=2.2, size=n)  # mobile share, family plan, etc.
    device_upgrades_90d = np.random.poisson(lam=0.6, size=n)
    accessory_purchases_90d = np.random.poisson(lam=0.9, size=n)

    industry = np.random.choice(INDUSTRIES, size=n)
    device_count = (ctn_count - np.random.poisson(lam=1.5, size=n)).clip(1, None)
    annual_revenue = (monthly_charges * 12 * np.random.uniform(0.95, 1.15, size=n)).round(2)
    account_manager = np.random.choice(ACCOUNT_MANAGERS, size=n)

    # --- Composite churn risk score, weighted by realistic B2B telecom churn drivers ---
    risk_score = (
        0.030 * (24 - np.clip(tenure_months, 0, 24))
        + 0.35 * pct_lines_disconnected_90d / 10
        + 0.06 * avg_invoice_delay_days
        + 0.15 * late_payments_12mo
        + 0.10 * support_tickets_90d
        + 0.12 * escalations_90d
        + 0.20 * (7.5 - csat_score)
        + 0.04 * days_since_last_login / 10
        - 0.30 * self_service_completion_rate * 10
        + 0.25 * upgrade_funnel_abandon_rate * 10
        + 0.22 * dropoff_severity * 10
        - 0.10 * feature_adoption_count
        - 0.05 * device_upgrades_90d
        + np.where(contract_type == "Month-to-Month", 1.0, 0.0)
        + np.where(days_to_contract_renewal < 30, 0.8, 0.0)  # renewal window risk
        - 0.6 * auto_renewal_enabled
        + np.random.normal(0, 0.8, size=n)
    )
    z = (risk_score - risk_score.mean()) / risk_score.std()
    churn_prob = 1 / (1 + np.exp(-(z - 1.1)))
    churn_label = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "account_id": [f"ACC-{100000+i}" for i in range(n)],
        "company_name": [fake.company() for _ in range(n)],
        "industry_segment": industry,
        "account_manager": account_manager,
        "annual_revenue": annual_revenue,
        "device_count": device_count,
        # Category 1: Contract & Account Health
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "days_to_contract_renewal": days_to_contract_renewal,
        "auto_renewal_enabled": auto_renewal_enabled,
        "ctn_count": ctn_count,
        "pct_lines_disconnected_90d": pct_lines_disconnected_90d,
        # Category 2: Billing & Payments
        "monthly_charges": monthly_charges,
        "avg_invoice_delay_days": avg_invoice_delay_days,
        "late_payments_12mo": late_payments_12mo,
        "autopay_enabled": autopay_enabled,
        # Category 3: Support & Service
        "support_tickets_90d": support_tickets_90d,
        "support_calls_90d": support_calls_90d,
        "avg_resolution_time_hours": avg_resolution_time_hours,
        "escalations_90d": escalations_90d,
        "csat_score": csat_score,
        # Category 4: Digital Engagement
        "platform_logins_90d": platform_logins_90d,
        "days_since_last_login": days_since_last_login,
        "self_service_completion_rate": self_service_completion_rate,
        "upgrade_funnel_abandon_rate": upgrade_funnel_abandon_rate,
        "primary_dropoff_page": primary_dropoff_page,
        "dropoff_severity": dropoff_severity,
        # Category 5: Product & Shop Usage
        "rate_plan_changes_90d": rate_plan_changes_90d,
        "feature_adoption_count": feature_adoption_count,
        "device_upgrades_90d": device_upgrades_90d,
        "accessory_purchases_90d": accessory_purchases_90d,
        # Target
        "churn_label": churn_label,
    })
    return df


def generate_complaint_themes() -> pd.DataFrame:
    """
    Representative national telecom complaint-theme distribution, structured to match
    categories tracked in the FCC's public Consumer Complaints Data. For a live pull:
    https://www.fcc.gov/consumer-complaints-center-data
    """
    return pd.DataFrame({
        "issue_category": [
            "Billing / Unexpected Charges", "Service Quality / Outages",
            "Unwanted Calls", "Slamming / Cramming",
            "Data Caps / Throttling", "Access for Disabled Consumers", "Other",
        ],
        "pct_of_total": [34.0, 22.5, 18.0, 9.5, 8.0, 4.0, 4.0],
        "period": ["last 12 months"] * 7,
    })


def generate_funnel_dropoff_summary(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adobe Analytics-style summary: aggregates each account's primary drop-off page
    into a national/portfolio-wide view of where customers most often abandon
    self-service journeys -- the "experience gap" signal for leadership.
    """
    summary = (
        accounts_df.groupby("primary_dropoff_page")
        .agg(
            account_count=("account_id", "count"),
            avg_dropoff_severity=("dropoff_severity", "mean"),
            churn_rate=("churn_label", "mean"),
        )
        .reset_index()
    )
    summary["pct_of_accounts"] = (summary["account_count"] / len(accounts_df) * 100).round(1)
    summary["avg_dropoff_severity"] = summary["avg_dropoff_severity"].round(2)
    summary["churn_rate"] = summary["churn_rate"].round(3)
    return summary.sort_values("pct_of_accounts", ascending=False)


def main():
    log("Starting data preparation (B2B telecom self-service platform model)...")
    accounts_df = generate_accounts(N_ACCOUNTS)

    required_cols = {"account_id", "tenure_months", "contract_type", "churn_label"}
    missing = required_cols - set(accounts_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    null_counts = accounts_df.isnull().sum()
    if null_counts.any():
        log(f"WARNING: Found nulls, filling with defaults:\n{null_counts[null_counts > 0]}")
        accounts_df = accounts_df.fillna(0)

    themes_df = generate_complaint_themes()
    funnel_df = generate_funnel_dropoff_summary(accounts_df)

    accounts_df.to_csv("data/processed/accounts.csv", index=False)
    themes_df.to_csv("data/processed/complaint_themes.csv", index=False)
    funnel_df.to_csv("data/processed/funnel_dropoff_summary.csv", index=False)

    log(f"Saved accounts.csv -> {accounts_df.shape[0]} rows, {accounts_df.shape[1]} columns")
    log(f"Churn rate: {accounts_df['churn_label'].mean():.1%}")
    log(f"Saved complaint_themes.csv -> {themes_df.shape[0]} categories")
    log(f"Saved funnel_dropoff_summary.csv -> top drop-off stage: {funnel_df.iloc[0]['primary_dropoff_page']} "
        f"({funnel_df.iloc[0]['pct_of_accounts']}% of accounts)")
    log("Data preparation complete.")


if __name__ == "__main__":
    main()
