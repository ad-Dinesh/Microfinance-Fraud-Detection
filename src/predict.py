"""
predict.py

Load the trained model and make fraud predictions.
"""

from pathlib import Path
import joblib
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------

model = joblib.load(MODELS_DIR / "best_model.pkl")
preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
meta = joblib.load(MODELS_DIR / "best_model_meta.pkl")

THRESHOLD = meta["threshold"]


# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_fraud(input_df: pd.DataFrame):

    # Apply preprocessing
    X = preprocessor.transform(input_df)

    # Predict probability
    probability = model.predict_proba(X)[:, 1][0]

    # Apply threshold
    prediction = int(probability >= THRESHOLD)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "threshold": THRESHOLD
    }