"""
utils.py

Utility functions for the Microfinance Fraud Detection project.
"""

def get_risk_level(probability: float) -> str:
    """
    Return a risk level based on fraud probability.
    """

    if probability < 0.20:
        return "Low Risk"

    elif probability < 0.50:
        return "Medium Risk"

    else:
        return "High Risk"


def format_probability(probability: float) -> str:
    """
    Format probability as a percentage.
    """

    return f"{probability * 100:.2f}%"


def prediction_label(prediction: int) -> str:
    """
    Return prediction label.
    """

    if prediction == 1:
        return "Fraud"

    return "Genuine"