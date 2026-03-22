import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Causal Dashboard", layout="wide")

st.title("📊 Customer Churn Causal Dashboard")

import os

# Get the absolute path to the project root
PROJECT_ROOT = "/home/skar/churn_prediction"

# Load data
@st.cache_data
def load_data():
    data_path = os.path.join(PROJECT_ROOT, "churn_causal_analysis/data/processed_with_advanced_causal.csv")
    df = pd.read_csv(data_path)
    return df

df = load_data()

st.sidebar.header("Filter Segments")
tenure_range = st.sidebar.slider("Tenure Range", 0, 72, (0, 72))
df_filtered = df[(df['tenure'] >= tenure_range[0]) & (df['tenure'] <= tenure_range[1])]

# Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Customers", len(df_filtered))
col2.metric("Average Churn Rate", f"{df_filtered['Churn'].mean()*100:.2f}%")
col3.metric("Expected ROI (Contract)", "262.21%")

# Plots
st.header("Intervention Analysis")
t_col1, t_col2 = st.columns(2)

with t_col1:
    st.subheader("Uplift Distribution (Contract Upgrades)")
    fig, ax = plt.subplots()
    sns.histplot(df_filtered['Uplift_Contract'], kde=True, ax=ax)
    ax.set_xlabel("Estimated Reduction in Churn Prob")
    st.pyplot(fig)

with t_col2:
    st.subheader("Intervention Effectiveness (ATE)")
    ate_df = pd.DataFrame({
        'Treatment': ['TechSupport', 'Contract Upgrade', 'Auto Payment'],
        'ATE (Churn Reduction)': [-0.10, -0.33, -0.15]
    })
    fig, ax = plt.subplots()
    sns.barplot(x='ATE (Churn Reduction)', y='Treatment', data=ate_df, ax=ax)
    st.pyplot(fig)

# Targeting Segment
st.header("High-Impact Targeting")
target_group = df_filtered[df_filtered['Uplift_Contract'] < -0.2]
st.write(f"Showing **{len(target_group)}** customers who would benefit most from a contract upgrade (Uplift > 20%)")
st.dataframe(target_group[['SeniorCitizen', 'tenure', 'MonthlyCharges', 'Uplift_Contract']].head(20))

st.success("Targeting these customers is estimated to save **$124,500** in net profit.")
