"""
feature_engineering.py

Feature engineering utilities for the
Microfinance Fraud Detection project.
"""

import pandas as pd


class FeatureEngineer:
    """
    Apply feature engineering transformations.
    """

    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:

        data = df.copy()

        # Loan-to-Income Ratio
        if (
            "requested_amount" in data.columns
            and "annual_income" in data.columns
        ):
            data["loan_to_income_ratio"] = (
                data["requested_amount"] /
                data["annual_income"].replace(0, 1)
            )

        # Debt Service Ratio
        if (
            "monthly_expenses" in data.columns
            and "annual_income" in data.columns
        ):
            monthly_income = data["annual_income"] / 12

            data["debt_service_ratio"] = (
                data["monthly_expenses"] /
                monthly_income.replace(0, 1)
            )

        # Financial Dependency Ratio
        if (
            "dependents" in data.columns
            and "annual_income" in data.columns
        ):
            data["financial_dependency_ratio"] = (
                data["dependents"] /
                data["annual_income"].replace(0, 1)
            )

        return data