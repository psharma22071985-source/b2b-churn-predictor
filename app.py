"""
app.py
B2B Telecom Self-Service Platform -- Churn Risk & Business Health Dashboard.

Built for non-technical stakeholders (leadership, sales, retention teams) to
see, at a glance, where the B2B account portfolio stands: revenue at risk,
CTNs/devices managed, platform engagement, complaint patterns, and which
accounts/account managers need attention -- with a simple login gate for
a realistic "stakeholder logs in" experience.

Views:
  1. Login gate
  2. Executive Summary  - KPIs, revenue at risk, complaint & funnel patterns
  3. Account Explorer   - filterable, searchable account table
  4. Account Detail     - drill-in with SHAP drivers + suggested action
  5. Account Manager View - portfolio health by rep, training-need flags
  6. Model Insights     - Logistic Regression vs XGBoost comparison
"""

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="B2B Telecom Churn & Health Dashboard", layout="wide")

CATEGORY_COLORS = {
    "Contract & Account Health": "#4C78A8",
    "Billing & Payments": "#F58518",
    "Support & Service": "#E45756",
    "Digital Engagement": "#72B7B2",
    "Product & Shop Usage": "#54A24B",
}

# Demo credentials only -- for a real deployment this would be SSO/OAuth.
DEMO_USERS = {"stakeholder": "demo123", "admin": "demo123"}


def login_gate():
    """Simple session-based login gate for a realistic 'stakeholder logs in' feel."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("📡 B2B Telecom Churn & Health Dashboard")
        st.caption("Stakeholder Login")
        with st.form("login_form"):
            username = st.text_input("Username", value="stakeholder")
            password = st.text_input("Password", type="password", value="")
            submitted = st.form_submit_button("Log In")
            if submitted:
                if DEMO_USERS.get(username) == password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Demo login: username 'stakeholder', password 'demo123'.")
        st.info("Demo credentials -- username: **stakeholder**, password: **demo123**")
        st.stop()


@st.cache_data
def load_data():
    predictions = pd.read_csv("data/processed/predictions.csv")
    accounts = pd.read_csv("data/processed/accounts.csv")
    themes = pd.read_csv("data/processed/complaint_themes.csv")
    comparison = pd.read_csv("data/processed/model_comparison.csv")
    funnel = pd.read_csv("data/processed/funnel_dropoff_summary.csv")
    merged = predictions.merge(
        accounts[["account_id", "contract_type", "tenure_months", "days_to_contract_renewal",
                  "support_tickets_90d", "support_calls_90d", "platform_logins_90d",
                  "rate_plan_changes_90d", "device_upgrades_90d", "accessory_purchases_90d"]],
        on="account_id", how="left"
    )
    return merged, themes, comparison, funnel


def risk_tier(prob: float) -> str:
    if prob >= 0.6:
        return "🔴 High"
    elif prob >= 0.35:
        return "🟡 Medium"
    return "🟢 Low"


def fmt_money(x: float) -> str:
    if x >= 1_000_000:
        return f"${x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"${x/1_000:.0f}K"
    return f"${x:.0f}"


def main():
    login_gate()

    st.title("📡 B2B Telecom Churn & Health Dashboard")
    st.caption("For a B2B wireless self-service platform (billing admins managing employee CTNs, "
               "devices, plans & features). Generalizable to any telecom B2B portfolio -- banking, "
               "pharma, retail, and other enterprise accounts.")
    colu, coll = st.columns([6, 1])
    with coll:
        if st.button("Log Out"):
            st.session_state.authenticated = False
            st.rerun()

    data, themes, comparison, funnel = load_data()
    data["risk_tier"] = data["churn_probability"].apply(risk_tier)
    data["revenue_at_risk"] = data["annual_revenue"] * (data["churn_probability"] >= 0.35)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧭 Executive Summary", "📊 Account Explorer", "🔍 Account Detail",
        "👤 Account Manager View", "🧠 Model Insights"
    ])

    # ============== VIEW 1: Executive Summary ==============
    with tab1:
        st.subheader("Portfolio Health at a Glance")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Accounts", f"{len(data):,}")
        c2.metric("Annual Revenue (Portfolio)", fmt_money(data["annual_revenue"].sum()))
        c3.metric("Revenue at Risk", fmt_money(data["revenue_at_risk"].sum()),
                   delta=f"{(data['revenue_at_risk'].sum()/data['annual_revenue'].sum()):.1%} of total",
                   delta_color="inverse")
        c4.metric("Total CTNs Managed", f"{int(data['ctn_count'].sum()):,}")
        c5.metric("Total Devices", f"{int(data['device_count'].sum()):,}")

        c6, c7, c8, c9 = st.columns(4)
        c6.metric("Avg. Churn Probability", f"{data['churn_probability'].mean():.1%}")
        c7.metric("Accounts Needing Help", f"{int(data['needs_help'].sum()):,}")
        c8.metric("Avg. Platform Logins (90d)", f"{data['platform_logins_90d'].mean():.1f}")
        c9.metric("Total Support Tickets (90d)", f"{int(data['support_tickets_90d'].sum()):,}")

        st.divider()
        colL, colR = st.columns(2)
        with colL:
            st.markdown("**Revenue at Risk by Industry**")
            rev_by_industry = data.groupby("industry_segment")["revenue_at_risk"].sum().sort_values(ascending=True).reset_index()
            fig_rev = px.bar(rev_by_industry, x="revenue_at_risk", y="industry_segment", orientation="h",
                              labels={"revenue_at_risk": "Revenue at Risk ($)", "industry_segment": ""},
                              color="revenue_at_risk", color_continuous_scale="Reds")
            st.plotly_chart(fig_rev, use_container_width=True)
        with colR:
            st.markdown("**What's Driving Churn (At-Risk Accounts)**")
            at_risk = data[data["churn_probability"] >= 0.35]
            cat_counts = at_risk["top_driver_category"].value_counts().reset_index()
            cat_counts.columns = ["category", "count"]
            fig_cat = px.pie(cat_counts, names="category", values="count",
                              color="category", color_discrete_map=CATEGORY_COLORS, hole=0.4)
            st.plotly_chart(fig_cat, use_container_width=True)

        st.divider()
        st.markdown("**Monthly Sales Activity by Feature (Portfolio Average, per Account)**")
        sales_activity = pd.DataFrame({
            "feature": ["Rate Plan Changes", "Device Upgrades", "Accessory Purchases"],
            "avg_per_account_90d": [
                data["rate_plan_changes_90d"].mean(),
                data["device_upgrades_90d"].mean(),
                data["accessory_purchases_90d"].mean(),
            ]
        })
        fig_sales = px.bar(sales_activity, x="feature", y="avg_per_account_90d",
                            labels={"avg_per_account_90d": "Avg. per Account (90d)", "feature": ""},
                            color="feature", color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sales.update_layout(showlegend=False)
        st.plotly_chart(fig_sales, use_container_width=True)

        st.divider()
        st.markdown("**Digital Experience Gap — Where Customers Drop Off**")
        st.caption("Adobe Analytics-style funnel: which self-service step accounts most often abandon.")
        colC, colD = st.columns([2, 1])
        with colC:
            fig_funnel = px.bar(
                funnel.sort_values("pct_of_accounts", ascending=True),
                x="pct_of_accounts", y="primary_dropoff_page", orientation="h",
                color="churn_rate", color_continuous_scale="Reds",
                labels={"pct_of_accounts": "% of Accounts", "primary_dropoff_page": "", "churn_rate": "Churn Rate"},
            )
            st.plotly_chart(fig_funnel, use_container_width=True)
        with colD:
            worst_stage = funnel.sort_values("churn_rate", ascending=False).iloc[0]
            st.warning(
                f"**Highest-risk experience gap:** {worst_stage['primary_dropoff_page']}\n\n"
                f"Churn rate here: **{worst_stage['churn_rate']:.1%}** "
                f"(vs. {data['churn_probability'].mean():.1%} portfolio avg)"
            )
            st.dataframe(funnel[["primary_dropoff_page", "pct_of_accounts", "churn_rate"]],
                         use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**National Telecom Complaint Themes (Context)**")
        st.caption("Structured to match categories tracked in the FCC's public Consumer Complaints Data: "
                   "https://www.fcc.gov/consumer-complaints-center-data")
        fig_themes = px.bar(themes.sort_values("pct_of_total", ascending=True),
                             x="pct_of_total", y="issue_category", orientation="h",
                             labels={"pct_of_total": "% of Complaints", "issue_category": ""},
                             color="pct_of_total", color_continuous_scale="Blues")
        st.plotly_chart(fig_themes, use_container_width=True)

    # ============== VIEW 2: Account Explorer ==============
    with tab2:
        st.subheader("Account Explorer")
        colf1, colf2, colf3, colf4 = st.columns(4)
        with colf1:
            tier_filter = st.multiselect("Risk tier", options=["🔴 High", "🟡 Medium", "🟢 Low"],
                                          default=["🔴 High", "🟡 Medium"])
        with colf2:
            cat_filter = st.multiselect("Top driver category", options=list(CATEGORY_COLORS.keys()),
                                         default=list(CATEGORY_COLORS.keys()))
        with colf3:
            industry_filter = st.multiselect("Industry", options=sorted(data["industry_segment"].unique()),
                                              default=sorted(data["industry_segment"].unique()))
        with colf4:
            help_filter = st.selectbox("Needs help?", options=["All", "Yes only", "No only"])

        filtered = data[
            data["risk_tier"].isin(tier_filter)
            & data["top_driver_category"].isin(cat_filter)
            & data["industry_segment"].isin(industry_filter)
        ]
        if help_filter == "Yes only":
            filtered = filtered[filtered["needs_help"]]
        elif help_filter == "No only":
            filtered = filtered[~filtered["needs_help"]]
        filtered = filtered.sort_values("churn_probability", ascending=False)

        search = st.text_input("Search by company name")
        if search:
            filtered = filtered[filtered["company_name"].str.contains(search, case=False, na=False)]

        st.dataframe(
            filtered[["account_id", "company_name", "industry_segment", "account_manager",
                      "annual_revenue", "ctn_count", "device_count", "contract_type",
                      "churn_probability", "risk_tier", "needs_help", "top_driver_category"]],
            use_container_width=True, height=450,
        )
        st.download_button("⬇️ Download Filtered Table (CSV)", data=filtered.to_csv(index=False),
                           file_name="churn_risk_table.csv", mime="text/csv")

    # ============== VIEW 3: Account Detail ==============
    with tab3:
        st.subheader("Account Detail Lookup")
        selected_account = st.selectbox(
            "Select an account:",
            options=data.sort_values("churn_probability", ascending=False)["company_name"] + " (" + data["account_id"] + ")",
        )
        acc_id = selected_account.split("(")[-1].replace(")", "")
        row = data[data["account_id"] == acc_id].iloc[0]

        colX, colY = st.columns([2, 1])
        with colX:
            st.markdown(f"### {row['company_name']}")
            risk_color = {"🔴 High": "red", "🟡 Medium": "orange", "🟢 Low": "green"}[row["risk_tier"]]
            st.markdown(
                f"**Churn Probability:** <span style='color:{risk_color}; font-size:28px; font-weight:bold'>"
                f"{row['churn_probability']:.1%}</span> — {row['risk_tier']}", unsafe_allow_html=True,
            )
            if row["needs_help"]:
                st.error("⚠️ Flagged as **Needs Help** — low self-service success, high tickets, or recent escalation.")
            st.write(f"**Industry:** {row['industry_segment']}  |  **Account Manager:** {row['account_manager']}")
            st.write(f"**Annual Revenue:** {fmt_money(row['annual_revenue'])}  |  **CTNs Managed:** {int(row['ctn_count'])}  |  **Devices:** {int(row['device_count'])}")
            st.write(f"**Contract:** {row['contract_type']}  |  **Renewal in:** {int(row['days_to_contract_renewal'])} days")
            st.write(f"**Primary drop-off point:** {row['primary_dropoff_page']}")

            st.markdown("**Top Risk Drivers (SHAP-explained):**")
            st.write(f"1. {row['top_driver_1']}")
            st.write(f"2. {row['top_driver_2']}")
            st.write(f"3. {row['top_driver_3']}")
            st.success(f"**Suggested retention action:** {row['suggested_action']}")
        with colY:
            st.markdown("**90-Day Activity**")
            st.metric("Platform Logins", int(row["platform_logins_90d"]))
            st.metric("Support Tickets", int(row["support_tickets_90d"]))
            st.metric("Support Calls", int(row["support_calls_90d"]))
            st.metric("Rate Plan Changes", int(row["rate_plan_changes_90d"]))
            st.metric("Device Upgrades", int(row["device_upgrades_90d"]))

        st.text_area("📝 Account manager notes (session only, not saved)", "")

    # ============== VIEW 4: Account Manager Performance ==============
    with tab4:
        st.subheader("Account Manager Portfolio Health")
        st.caption("Aggregated by account manager — identifies reps whose portfolios show elevated "
                   "risk, which may indicate a need for coaching, additional support, or training.")
        am_summary = data.groupby("account_manager").agg(
            portfolio_size=("account_id", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            high_risk_accounts=("risk_tier", lambda x: (x == "🔴 High").sum()),
            accounts_needing_help=("needs_help", "sum"),
            total_revenue=("annual_revenue", "sum"),
        ).reset_index()
        am_summary["avg_churn_probability"] = am_summary["avg_churn_probability"].round(3)
        portfolio_avg_risk = data["churn_probability"].mean()
        am_summary["may_benefit_from_training"] = am_summary["avg_churn_probability"] > (portfolio_avg_risk * 1.25)
        am_summary = am_summary.sort_values("avg_churn_probability", ascending=False)

        flagged = am_summary["may_benefit_from_training"].sum()
        st.metric("Account Managers Flagged for Coaching Support", int(flagged))

        st.dataframe(am_summary, use_container_width=True, height=400)

        fig_am = px.bar(am_summary.head(15), x="account_manager", y="avg_churn_probability",
                         color="may_benefit_from_training",
                         color_discrete_map={True: "#E45756", False: "#4C78A8"},
                         labels={"avg_churn_probability": "Avg. Portfolio Churn Risk", "account_manager": ""})
        st.plotly_chart(fig_am, use_container_width=True)

    # ============== VIEW 5: Model Insights ==============
    with tab5:
        st.subheader("Model Performance Comparison")
        st.dataframe(comparison, use_container_width=True)
        fig2 = px.bar(comparison.melt(id_vars="model", var_name="metric", value_name="score"),
                      x="metric", y="score", color="model", barmode="group",
                      title="Logistic Regression vs. XGBoost")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Churn Probability Distribution")
        fig3 = px.histogram(data, x="churn_probability", nbins=30,
                             title="Distribution of Predicted Churn Probabilities")
        st.plotly_chart(fig3, use_container_width=True)


if __name__ == "__main__":
    main()
