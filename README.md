# Customer Churn Prediction with Causal Inference

This project provides an end-to-end production-grade data science solution for customer churn. It moves beyond simple prediction by leveraging **Causal Inference** and **Uplift Modeling** to identify interventions that *cause* retention and maximize business ROI.

##  Key Features
1. **Advanced Predictive Ensemble:** Voting ensemble of **CatBoost**, **LightGBM**, and **XGBoost** with advanced feature engineering (CLV, Engagement Scores).
2. **Multi-Treatment Causal Inference:** Rigorous measurement of Average Treatment Effects (ATE) for TechSupport, Contract Upgrades, and Payment Method changes.
3. **Uplift Modeling (CATE):** Using EconML's Double Robust Learner to estimate heterogeneous treatment effects and identify high-impact customer segments.
4. **Business ROI Framework:** Quantifiable ROI analysis (e.g., **262% ROI** for targeted contract incentives).
5. **Production Elements:** Real-time prediction API (FastAPI) and interactive business dashboard (Streamlit).

##  Model Performance & Key Findings
The final **Advanced Ensemble** model achieved:
- **ROC-AUC:** **83.38%** 
- **Accuracy:** 78.42%

###  Causal & Business Insights
- **Contract Upgrades:** The strongest causal driver of retention, reducing churn probability by **33.32%** on average.
- **TechSupport & Payments:** Provide significant secondary retention effects (10-15% reduction).
- **Targeted Impact:** Identified high-priority segments where contract upgrades cause a **25.14%+** reduction in churn probability.

##  Business Impact

This framework enables **data-driven retention strategies** with measurable ROI:

### Targeting Strategy
Instead of offering contract upgrades to all high-risk customers:
- **Traditional Approach:** 30% intervention cost, 15% retention improvement
- **Uplift-Optimized Approach:** Target high-CATE segments only
  - **Cost Reduction:** 40% fewer interventions needed
  - **Retention Improvement:** 25%+ in priority segments
  - **Net ROI:** **262%** for contract upgrade campaigns

### Key Recommendations
1. **High Priority (ROI: 262%):** Offer contract upgrade incentives to month-to-month customers with high uplift scores
2. **Medium Priority (ROI: 145%):** Provide tech support to high-risk customers with moderate uplift
3. **Optimize Resources:** Avoid interventions for low-uplift segments where treatments show minimal causal effect

### Implementation
The production API enables:
- Real-time churn risk scoring for customer service teams
- Automated segmentation for marketing campaigns
- A/B test evaluation framework for new retention strategies

##  Project Structure
```
churn_causal_analysis/
├── data/               # Raw, processed, and uplift-augmented datasets
├── notebooks/          # Research & Development
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_advanced.ipynb
│   ├── 03_modeling_advanced.ipynb
│   ├── 04_causal_inference_advanced.ipynb
├── src/                # Production Modules
│   ├── api.py          # FastAPI Real-time Prediction Service
│   ├── dashboard.py    # Streamlit Business Dashboard
│   ├── causal_analysis.py
│   ├── data_preprocessing.py
├── results/            # Performance plots and heatmaps
├── models/             # Pickled ensemble models
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

##  Tech Stack
- **Languages:** Python
- **ML Frameworks:** Scikit-Learn, XGBoost, LightGBM, CatBoost
- **Causal Inference:** DoWhy, EconML
- **Production:** FastAPI, Uvicorn, Streamlit

## ⚙️ How to Run

### 1. Setup
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
Visualize business segments and ROI:
```bash
streamlit run churn_causal_analysis/src/dashboard.py
```

### 3. Run the API
Start the real-time prediction service:
```bash
uvicorn churn_causal_analysis.src.api:app --host 0.0.0.0 --port 8000
```
*Access API docs at http://localhost:8000/docs*

---
*Built as a Customer Churn Prediction with Causal Inference project.*
