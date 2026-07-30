"""
preprocessing.py

Preprocessing pipeline for the Microfinance Fraud Detection project.
"""

from pathlib import Path
import joblib
import pandas as pd


class DataPreprocessor:
    """
    Loads the trained preprocessing pipeline and transforms new data.
    """

    def __init__(self, models_dir=None):

        if models_dir is None:
            root = Path(__file__).resolve().parent.parent
            models_dir = root / "models"

        self.preprocessor = joblib.load(
            models_dir / "preprocessor.pkl"
        )

    def transform(self, df: pd.DataFrame):
        """
        Transform raw input data using the saved preprocessing pipeline.
        """

        return self.preprocessor.transform(df)

    def get_feature_names(self):
        """
        Return feature names after preprocessing (if available).
        """

        try:
            return self.preprocessor.get_feature_names_out()

        except Exception:
            return None