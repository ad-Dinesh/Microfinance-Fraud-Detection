"""
Shared custom transformers for the Microfinance Fraud Detection project.

IMPORTANT: This class MUST live in its own importable module (not be
defined inline inside a notebook). joblib/pickle stores a *reference*
(module path + class name) to custom classes, not their code. If the
class is defined inside a notebook's __main__ namespace, any other
notebook/process trying to unpickle it later will fail with:

    AttributeError: Can't get attribute 'FrequencyEncoder' on <module '__main__'>

By keeping it here and importing it the same way everywhere
(`from src.transformers import FrequencyEncoder`), every notebook/script
that loads preprocessor.pkl will resolve it correctly.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with how often it appeared in the training data
    (as a proportion, 0-1). Unseen categories at transform time get 0.
    Keeps high-cardinality columns to a single numeric column instead of
    exploding into hundreds of one-hot columns."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True) for col in X.columns
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0.0)
        return X.values.astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)
