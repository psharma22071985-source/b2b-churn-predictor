"""
explain.py
Loads the trained model, scores every account in the test split, and uses SHAP
to extract the top 3 risk drivers per account in plain business language —
grouped by category (Contract, Billing, Support, Digital Engagement, Product
Usage) — plus a suggested retention action per top-driver category, so the
output is directly actionable for account managers and leadership.

Output:
    data/processed/predictions.csv
"""

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import shap

NUMERIC_FEATURES = [
    "tenure_months", "days_to_contract_renewal", "auto_renewal_enabled",
    "ctn_count", "device_count", "annual_revenue", "pct_lines_disconnected_90d",
    "monthly_charges", "avg_invoice_delay_days", "late_payments_12mo", "autopay_enabled",
    "support_tickets_90d", "support_calls_90d", "avg_resolution_time_hours",
    "escalations_90d", "csat_score",
    "platform_logins_90d", "days_since_last_login",
    "self_service_completion_rate", "upgrade_funnel_abandon_rate", "dropoff_severity",
    "rate_plan_changes_90d", "feature_adoption_count",
    "device_upgrades_90d", "accessory_purchases_90d",
]
CATEGORICAL_FEATURES = ["contract_type", "industry_segment", "primary_dropoff_page"]

# feature -> (plain-language label, business category)
FEATURE_META = {
    "tenure_months": ("Short account tenure", "Contract & Account Health"),
    "days_to_contract_renewal": ("Approaching contract renewal window", "Contract & Account Health"),
    "auto_renewal_enabled": ("Auto-renewal not enabled", "Contract & Account Health"),
    "ctn_count": ("Large number of managed lines (CTNs)", "Contract & Account Health"),
    "device_count": ("Low device-to-line ratio", "Contract & Account Health"),
    "annual_revenue": ("High annual revenue account", "Contract & Account Health"),
    "pct_lines_disconnected_90d": ("Rising line disconnects (90d)", "Contract & Account Health"),
    "monthly_charges": ("High monthly spend", "Billing & Payments"),
    "avg_invoice_delay_days": ("History of late invoice payments", "Billing & Payments"),
    "late_payments_12mo": ("Repeated late payments (12mo)", "Billing & Payments"),
    "autopay_enabled": ("Autopay not enabled", "Billing & Payments"),
    "support_tickets_90d": ("High support ticket volume (90d)", "Support & Service"),
    "support_calls_90d": ("High support call volume (90d)", "Support & Service"),
    "avg_resolution_time_hours": ("Slow support resolution times", "Support & Service"),
    "escalations_90d": ("Recent support escalations", "Support & Service"),
    "csat_score": ("Low customer satisfaction (CSAT)", "Support & Service"),
    "platform_logins_90d": ("Low platform login frequency", "Digital Engagement"),
    "days_since_last_login": ("Long time since last platform login", "Digital Engagement"),
    "self_service_completion_rate": ("Low self-service task completion rate", "Digital Engagement"),
    "upgrade_funnel_abandon_rate": ("High upgrade funnel abandonment", "Digital Engagement"),
    "dropoff_severity": ("Severe drop-off at a specific self-service step (experience gap)", "Digital Engagement"),
    "rate_plan_changes_90d": ("Low rate plan/feature activity", "Product & Shop Usage"),
    "feature_adoption_count": ("Low feature adoption (mobile share, family plan, etc.)", "Product & Shop Usage"),
    "device_upgrades_90d": ("Low device upgrade activity", "Product & Shop Usage"),
    "accessory_purchases_90d": ("Low accessory purchase activity", "Product & Shop Usage"),
}

CATEGORY_ACTIONS = {
    "Contract & Account Health": "Proactively reach out ahead of renewal; confirm auto-renewal status; review disconnected lines.",
    "Billing & Payments": "Offer autopay enrollment incentive; review invoice accuracy; flag account to billing support.",
    "Support & Service": "Escalate to a senior account rep for a service recovery call; audit recent ticket resolution quality.",
    "Digital Engagement": "Send a guided platform walkthrough or personal onboarding session; simplify the upgrade flow for this account.",
    "Product & Shop Usage": "Recommend relevant plan/feature bundles; offer a device upgrade promotion to re-engage.",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def resolve_feature(raw_name: str):
    """Map a (possibly one-hot-expanded) transformed feature name back to label + category."""
    base = raw_name.split("__")[-1] if "__" in raw_name else raw_name
    for key, (label, category) in FEATURE_META.items():
        if base.startswith(key):
            return label, category
    if base.startswith("contract_type"):
        return f"Contract type: {base.split('_')[-1]}", "Contract & Account Health"
    if base.startswith("industry_segment"):
        return f"Industry segment: {base.split('_')[-1]}", "Product & Shop Usage"
    if base.startswith("primary_dropoff_page"):
        page = base.replace("primary_dropoff_page_", "")
        return f"Frequently drops off at: {page}", "Digital Engagement"
    return base, "Other"


def main():
    log("Loading model and test split...")
    with open("model.pkl", "rb") as f:
        pipe = pickle.load(f)

    test_df = pd.read_csv("data/processed/test_split.csv")
    required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["account_id"])
    missing = required - set(test_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in test_split.csv: {missing}")

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["clf"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    log("Scoring accounts...")
    churn_probs = pipe.predict_proba(X_test)[:, 1]

    log("Computing SHAP values (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    results = []
    for i, account_id in enumerate(test_df["account_id"].values):
        row_shap = shap_values[i]
        top_idx = np.argsort(-np.abs(row_shap))[:3]

        drivers, categories = [], []
        for j in top_idx:
            label, category = resolve_feature(feature_names[j])
            direction = "increasing" if row_shap[j] > 0 else "decreasing"
            drivers.append(f"{label} ({direction} risk)")
            categories.append(category)

        top_category = categories[0]
        row_data = test_df.iloc[i]
        needs_help = bool(
            row_data["self_service_completion_rate"] < 0.4
            or row_data["support_tickets_90d"] >= 5
            or row_data["escalations_90d"] >= 2
        )
        results.append({
            "account_id": account_id,
            "company_name": row_data["company_name"],
            "industry_segment": row_data["industry_segment"],
            "account_manager": row_data["account_manager"],
            "annual_revenue": row_data["annual_revenue"],
            "ctn_count": row_data["ctn_count"],
            "device_count": row_data["device_count"],
            "primary_dropoff_page": row_data["primary_dropoff_page"],
            "churn_probability": round(float(churn_probs[i]), 3),
            "needs_help": needs_help,
            "top_driver_1": drivers[0],
            "top_driver_2": drivers[1],
            "top_driver_3": drivers[2],
            "top_driver_category": top_category,
            "suggested_action": CATEGORY_ACTIONS.get(top_category, "Review account manually."),
        })

    predictions_df = pd.DataFrame(results)
    predictions_df["scored_date"] = datetime.now().strftime("%Y-%m-%d")
    predictions_df = predictions_df.sort_values("churn_probability", ascending=False)
    predictions_df.to_csv("data/processed/predictions.csv", index=False)

    log(f"Saved predictions.csv -> {predictions_df.shape[0]} scored accounts")
    log(f"High-risk accounts (>0.5 probability): {(predictions_df['churn_probability'] > 0.5).sum()}")
    log("Top driver category breakdown:\n" + predictions_df["top_driver_category"].value_counts().to_string())
    log("Explainability step complete.")


if __name__ == "__main__":
    main()
