# Customer Churn Prediction with Causal Inference

An end-to-end churn prediction system that goes beyond correlation — using **Propensity Score Matching**, **Uplift Modeling**, and **Causal Forests** to identify interventions that *cause* retention and quantify their ROI.

## Model Performance

The final calibrated voting ensemble (Logistic Regression + Gradient Boosting + XGBoost) achieved:

| Metric | Score |
|---|---|
| ROC-AUC | **85.2%** |
| Churn Recall | 75–78% |
| Churn Precision | 54–56% |
| Best threshold | 0.31 (F1-optimised) |

> Previous baseline (Advanced Ensemble without interaction features): ROC-AUC 83.4%

## Key Findings

### Causal Effects (Propensity Score Matching)

Four treatments were evaluated using PSM to isolate causal effects from confounders:

| Treatment | Causal Effect on Churn |
|---|---|
| Fiber optic internet | **+36.1 percentage points** |
| Month-to-month contract | +30.4 pp |
| High monthly charges | +21.3 pp |
| Low tenure (≤6 months) | +21.2 pp |

Fiber optic + month-to-month is the single highest-risk combination.

### Feature Engineering

Five interaction features were added on top of base features, all confirmed important by SHAP:

- `price_stress` — monthly charges × month-to-month flag
- `price_tenure_risk` — monthly charges / (tenure + 1)
- `fiber_contract_risk` — fiber optic × month-to-month
- `early_customer` — tenure ≤ 6 months
- `fiber_new_customer` — fiber optic × early customer

### Uplift Segmentation

Customers are segmented by churn probability and individual treatment response:

| Segment | Count | Avg Churn Prob | Strategy |
|---|---|---|---|
| Sleeping Dog | 2,889 | 28.5% | Do not contact — negative uplift |
| Lost Cause | 1,757 | 27.2% | Skip — intervention won't help |
| **Persuadable** | **1,634** | **66.2%** | **Intervene — highest ROI** |
| Sure Thing | 763 | 11.0% | Monitor only |

### ROI Analysis

Targeting only the **Persuadable** segment:

| | Value |
|---|---|
| Customers targeted | 1,634 |
| Intervention cost | $207,339 |
| Net value recovered | $166,262 |
| **ROI** | **80.2%** |
| CAC avoided | $139,300 (vs. acquiring 398 replacements at $350 each) |

Targeting other segments produces negative ROI — **Sleeping Dogs in particular should not be contacted**.

## Project Structure

```
churn_causal_analysis/
├── data/
│   ├── telco_churn.csv              # Raw dataset (7,043 customers, 21 features)
│   ├── churn_features.csv           # Fully engineered dataset (35 features)
│   ├── churn_causal.csv             # + uplift scores from causal models
│   ├── churn_risk_results.csv       # + risk level, segment, recommended action
│   └── roi_analysis.csv             # ROI by segment
├── models/
│   ├── calibrated_model.pkl         # Final production model
│   ├── ensemble_model.pkl           # Uncalibrated ensemble
│   ├── threshold.pkl                # Optimal classification threshold (0.31)
│   ├── feature_columns.pkl          # Feature list (35 columns)
│   └── scaler.pkl                   # StandardScaler
├── notebooks/
│   ├── 01_eda.ipynb                 # Data cleaning, distributions, key patterns
│   ├── 02_feature_engineering.ipynb # All feature engineering + train/test split
│   ├── 03_modeling.ipynb            # 11 baselines, Optuna tuning, ensemble, SHAP
│   ├── 04_causal_inference.ipynb    # PSM causal effects + uplift modeling
│   └── 05_business_insights.ipynb  # Risk scoring, segmentation, ROI analysis
├── src/
│   ├── api.py                       # FastAPI real-time prediction service
│   └── dashboard.py                 # Streamlit business dashboard
└── requirements.txt
```

## How to Run

### Setup
```bash
pip install -r requirements.txt
```

### Run notebooks (in order)
```
01_eda → 02_feature_engineering → 03_modeling → 04_causal_inference → 05_business_insights
```

Pre-trained models are already saved in `models/`. Notebook 03 will load them automatically — skip retraining by running only the **"Load Pre-trained Model"** cell.

### Dashboard
```bash
streamlit run src/dashboard.py
```

### Prediction API
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Docs at `http://localhost:8000/docs`

## Tech Stack

- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn (SMOTE)
- **Tuning:** Optuna
- **Causal inference:** causalml, econml (CausalForestDML), DoWhy
- **Explainability:** SHAP
- **Production:** FastAPI, Streamlit
