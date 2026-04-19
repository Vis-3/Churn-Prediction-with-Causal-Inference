import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Causal Dashboard", layout="wide")
st.title("Customer Churn Causal Dashboard")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")


@st.cache_data
def load_data():
    risk_path     = os.path.join(DATA_DIR, "churn_risk_results.csv")
    features_path = os.path.join(DATA_DIR, "churn_features.csv")

    if os.path.exists(risk_path):
        df = pd.read_csv(risk_path)
    elif os.path.exists(features_path):
        df = pd.read_csv(features_path)
        st.warning("Full risk results not found — run 05_business_insights.ipynb for segments and ROI.")
    else:
        st.error("No data file found. Run the notebooks first.")
        st.stop()
    return df


df_full = load_data()

HAS_RISK    = "Risk_Level" in df_full.columns
HAS_SEGMENT = "segment" in df_full.columns
HAS_UPLIFT  = "uplift_X_T_MonthToMonth" in df_full.columns
HAS_PROB    = "Churn_Probability" in df_full.columns

# ── Sidebar filters ──────────────────────────────────────────────────────────
st.sidebar.header("Filters")
tenure_range = st.sidebar.slider("Tenure (months)", 0, 72, (0, 72))

if HAS_RISK:
    risk_options = ["All"] + list(df_full["Risk_Level"].dropna().unique())
    selected_risk = st.sidebar.selectbox("Risk Level", risk_options)
else:
    selected_risk = "All"

if HAS_SEGMENT:
    seg_options = ["All"] + sorted(df_full["segment"].dropna().unique().tolist())
    selected_seg = st.sidebar.selectbox("Segment", seg_options)
else:
    selected_seg = "All"

df = df_full[
    (df_full["tenure"] >= tenure_range[0]) &
    (df_full["tenure"] <= tenure_range[1])
]
if selected_risk != "All" and HAS_RISK:
    df = df[df["Risk_Level"] == selected_risk]
if selected_seg != "All" and HAS_SEGMENT:
    df = df[df["segment"] == selected_seg]

# ── KPI row ──────────────────────────────────────────────────────────────────
st.header("Overview")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(df):,}")

churn_col = "Churn" if "Churn" in df.columns else None
if churn_col:
    k2.metric("Churn Rate", f"{df[churn_col].mean()*100:.1f}%")

if HAS_PROB:
    k3.metric("Avg Churn Probability", f"{df['Churn_Probability'].mean()*100:.1f}%")

if HAS_SEGMENT and HAS_UPLIFT:
    persuadable = df[df["segment"] == "Persuadable"]
    k4.metric("Persuadable Customers", f"{len(persuadable):,}")

# ── Causal effects ───────────────────────────────────────────────────────────
st.header("Causal Effects (Propensity Score Matching)")
st.caption("Each value is the estimated causal increase in churn rate for customers with that treatment, holding confounders constant.")

psm_df = pd.DataFrame({
    "Treatment": ["Fiber optic internet", "Month-to-month contract",
                  "High monthly charges", "Low tenure (≤6 months)"],
    "Causal effect on churn (pp)": [36.1, 30.4, 21.3, 21.2],
})

fig, ax = plt.subplots(figsize=(8, 3.5))
colors = ["#e74c3c"] * len(psm_df)
ax.barh(psm_df["Treatment"], psm_df["Causal effect on churn (pp)"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("Causal effect (percentage points)")
ax.set_title("PSM Causal Effects")
for i, v in enumerate(psm_df["Causal effect on churn (pp)"]):
    ax.text(v + 0.3, i, f"+{v}pp", va="center", fontsize=9)
plt.tight_layout()
st.pyplot(fig)

# ── Risk distribution ─────────────────────────────────────────────────────────
if HAS_RISK:
    st.header("Risk Level Distribution")
    c1, c2 = st.columns(2)

    with c1:
        risk_counts = df["Risk_Level"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(risk_counts, labels=risk_counts.index, autopct="%1.1f%%",
               colors=["#2ecc71", "#f39c12", "#e74c3c"])
        ax.set_title("Risk Level Breakdown")
        st.pyplot(fig)

    with c2:
        if HAS_PROB:
            fig, ax = plt.subplots()
            for level, colour in [("Low", "#2ecc71"), ("Medium", "#f39c12"), ("High", "#e74c3c")]:
                subset = df[df["Risk_Level"] == level]["Churn_Probability"]
                if len(subset):
                    ax.hist(subset, bins=30, alpha=0.6, label=level, color=colour)
            ax.set_xlabel("Churn Probability")
            ax.set_title("Probability Distribution by Risk Level")
            ax.legend()
            st.pyplot(fig)

# ── Segment analysis ─────────────────────────────────────────────────────────
if HAS_SEGMENT:
    st.header("Uplift Segments")

    seg_summary = df.groupby("segment").agg(
        customers=("segment", "count"),
        avg_churn_prob=("Churn_Probability", "mean") if HAS_PROB else ("segment", "count"),
        avg_monthly_charges=("MonthlyCharges", "mean"),
    ).round(3)

    if HAS_UPLIFT:
        seg_summary["avg_uplift"] = df.groupby("segment")["uplift_X_T_MonthToMonth"].mean().round(3)

    st.dataframe(seg_summary)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        df["segment"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Customers by Segment")
        ax.set_xlabel("")
        plt.xticks(rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        if HAS_UPLIFT:
            fig, ax = plt.subplots()
            df.boxplot(column="uplift_X_T_MonthToMonth", by="segment", ax=ax)
            ax.set_title("Uplift by Segment")
            ax.set_xlabel("")
            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig)

# ── ROI summary ───────────────────────────────────────────────────────────────
st.header("ROI Analysis")
roi_path = os.path.join(DATA_DIR, "roi_analysis.csv")
if os.path.exists(roi_path):
    roi_df = pd.read_csv(roi_path, index_col=0)
    st.dataframe(roi_df.style.format({
        "avg_monthly_charge": "${:.2f}",
        "intervention_cost": "${:,.0f}",
        "net_value": "${:,.0f}",
        "ROI_%": "{:.1f}%",
    }))
    if "Persuadable" in roi_df.index:
        p = roi_df.loc["Persuadable"]
        st.success(
            f"Targeting **{int(p['n_customers']):,} Persuadable** customers: "
            f"${p['net_value']:,.0f} net value at **{p['ROI_%']:.1f}% ROI**. "
            f"Intervention cost: ${p['intervention_cost']:,.0f}."
        )
        st.info("Sleeping Dogs have **negative uplift** — contacting them increases churn. Do not intervene.")
else:
    st.info("ROI analysis not available — run 05_business_insights.ipynb to generate roi_analysis.csv.")

# ── High-priority customer table ──────────────────────────────────────────────
if HAS_SEGMENT:
    st.header("Persuadable Customers (Priority Intervention List)")
    persuadable_df = df[df["segment"] == "Persuadable"].copy()

    display_cols = ["tenure", "MonthlyCharges", "ContractMonths", "NumServices",
                    "Churn_Probability", "Risk_Level"]
    if HAS_UPLIFT:
        display_cols.append("uplift_X_T_MonthToMonth")
    display_cols = [c for c in display_cols if c in persuadable_df.columns]

    st.write(f"**{len(persuadable_df):,} customers** — sorted by churn probability")
    st.dataframe(
        persuadable_df[display_cols]
        .sort_values("Churn_Probability", ascending=False)
        .head(50)
        .reset_index(drop=True)
    )
