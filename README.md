# Customer Churn Prediction with Causal Inference

This project provides an end-to-end data science solution for customer churn that moves beyond simple prediction. It leverages **Causal Inference** to understand which interventions actually *cause* retention and **Uplift Modeling** to identify which customers should be targeted for specific interventions.

##  Key Features
1. **Predictive Modeling:** XGBoost and Random Forest models trained with SMOTE to handle class imbalance, achieving high ROC-AUC.
2. **Causal Inference (Propensity Score Matching):** Rigorously measuring the Average Treatment Effect (ATE) of interventions like 'TechSupport' on customer retention.
3. **DoWhy Framework:** Using Directed Acyclic Graphs (DAGs) to identify and estimate causal effects, followed by robustness checks (refutation tests).
4. **Uplift Modeling (CATE):** Estimating the Conditional Average Treatment Effect using EconML to identify heterogeneous treatment effects across the customer base.
5. **Business ROI Analysis:** Segmentation of customers into high-risk/high-uplift groups to maximize intervention ROI.

##  Project Structure
```
churn_causal_analysis/
├── data/               # Raw and processed datasets
├── notebooks/          # Step-by-step implementation
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_causal_inference.ipynb
│   └── 05_business_insights.ipynb
├── src/                # Reusable Python modules
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── causal_analysis.py
│   └── visualization.py
├── results/            # Saved plots and reports
├── models/             # Pickled model files
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

##  Model Performance & Key Findings
The final **Random Forest (with SMOTE)** model achieved the following metrics on the test set:
- **ROC-AUC:** **82.64%** (exceeded the 0.80 target)
- **Accuracy:** 77.57%


###  Causal Insights
- **Intervention Analyzed:** 'TechSupport' service.
- **Average Treatment Effect (ATE):** Consistently significant across PSM and DoWhy estimations.
- **Uplift Targeting:** Identified a high-priority segment where providing TechSupport causes a **25.14 percentage point reduction** in churn probability.

##  Tech Stack
- **Languages:** Python
- **ML Frameworks:** Scikit-Learn, XGBoost, LightGBM
- **Causal Inference:** DoWhy, EconML
- **Interpretability:** SHAP
- **Visualization:** Matplotlib, Seaborn

##  Business Impact
- **Targeted Interventions:** Moving from "targeting high-risk customers" to "targeting high-uplift customers" saves marketing budget and avoids "sleeping dog" effects.
- **Data-Driven Strategy:** Causal findings prove that certain services (like TechSupport) reduce churn probability, justifying investment in those service areas.
- **Quantifiable ROI:** Provides a framework to calculate the expected return on investment for retention campaigns.

##  How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Navigate through the notebooks in `notebooks/` sequentially (01 to 05).
3. Reusable functions can be imported from `src/`.

---

