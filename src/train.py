"""
train.py

Train the final HistGradientBoosting model for the
Microfinance Fraud Detection project.
"""

from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score
)


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load Data
# --------------------------------------------------

X_train = joblib.load(MODELS_DIR / "X_train.pkl")
X_test = joblib.load(MODELS_DIR / "X_test.pkl")

y_train = joblib.load(MODELS_DIR / "y_train.pkl")
y_test = joblib.load(MODELS_DIR / "y_test.pkl")


# --------------------------------------------------
# Hyperparameter Search
# --------------------------------------------------

param_dist = {

    "learning_rate": [0.01, 0.03, 0.05, 0.1],

    "max_iter": [100, 200, 300, 500],

    "max_leaf_nodes": [15, 31, 63, 127],

    "min_samples_leaf": [10, 20, 30, 50],

    "l2_regularization": [0.0, 0.1, 0.5, 1.0],

    "max_depth": [None, 5, 10, 15]

}

model = HistGradientBoostingClassifier(
    random_state=42
)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    scoring="average_precision",
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

print("Training model...")

search.fit(X_train, y_train)

best_model = search.best_estimator_

print("\nBest Parameters:")
print(search.best_params_)


# --------------------------------------------------
# Threshold Optimization
# --------------------------------------------------

y_prob = best_model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_test,
    y_prob
)

f1_scores = (
    2 * precision * recall /
    (precision + recall + 1e-8)
)

best_index = np.argmax(f1_scores[:-1])

best_threshold = thresholds[best_index]

test_pr_auc = average_precision_score(
    y_test,
    y_prob
)

print(f"\nBest Threshold : {best_threshold:.4f}")
print(f"Test PR-AUC    : {test_pr_auc:.4f}")


# --------------------------------------------------
# Save Model
# --------------------------------------------------

joblib.dump(
    best_model,
    MODELS_DIR / "best_model.pkl"
)

joblib.dump(
    {
        "name": "HistGradientBoosting (Tuned)",
        "threshold": float(best_threshold),
        "pr_auc_on_test": float(test_pr_auc),
        "best_params": search.best_params_
    },
    MODELS_DIR / "best_model_meta.pkl"
)

print("\nModel saved successfully.")
print("Training completed.")