import pickle
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

X_test = np.load('churn_causal_analysis/data/X_test.npy')
y_test = np.load('churn_causal_analysis/data/y_test.npy')
with open('churn_causal_analysis/models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
