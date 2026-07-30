import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path
def load_css():
    with open("dashboard/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ----------------------------
# Project Paths
# ----------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = ROOT_DIR / "models"

sys.path.append(str(ROOT_DIR))

from src.predict import predict_fraud

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="Microfinance Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Microfinance Fraud Detection System")

st.markdown(
"""
Detect fraudulent microfinance loan applications using a trained
HistGradientBoosting Machine Learning model.
"""
)
with st.sidebar:

    st.header("📊 Model Information")

    st.success("HistGradientBoosting (Tuned)")

    meta = joblib.load(MODELS_DIR / "best_model_meta.pkl")

    st.metric(
        "Threshold",
        f"{meta['threshold']:.3f}"
    )

    st.metric(
        "Test PR-AUC",
        f"{meta['pr_auc_on_test']:.3f}"
    )

    st.markdown("---")

    st.info(
"""
Dataset

• 50,000 Applications

• Fraud Rate ≈ 3%

• Hyperparameter Tuned

• Threshold Optimized
"""
)
tab1, tab2 = st.tabs(
[
    "📝 Prediction",
    "ℹ About Model"
]
)
with tab1:

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:

        customer_age = st.number_input(
            "Customer Age",
            18,
            80,
            30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        annual_income = st.number_input(
            "Annual Income",
            0.0,
            10000000.0,
            500000.0
        )

        credit_score = st.slider(
            "Credit Score",
            300,
            900,
            700
        )

    with col2:

        requested_amount = st.number_input(
            "Requested Loan Amount",
            0.0,
            5000000.0,
            100000.0
        )

        loan_term = st.slider(
            "Loan Term (Months)",
            3,
            60,
            12
        )

        interest_rate = st.slider(
            "Interest Rate (%)",
            5.0,
            30.0,
            12.0
        )

        previous_loans = st.slider(
            "Previous Loan Count",
            0,
            20,
            1
        )

        # ----------------------------
        # Prediction Button
        # ----------------------------

    if st.button("🔍 Predict Fraud", use_container_width=True):

        input_data = {
            "branch_id": "BR001",
            "customer_age": customer_age,
            "gender": gender,
            "marital_status": "Single",
            "education_level": "Graduate",
            "occupation": "Salaried",
            "annual_income": annual_income,
            "monthly_expenses": annual_income * 0.40,
            "credit_score": credit_score,
            "years_with_institution": 3,
            "previous_loan_count": previous_loans,
            "dependents": 2,
            "home_ownership": "Owned",
            "mobile_registered": 1,
            "region": "Urban",
            "state": "Telangana",
            "distance_to_branch_km": 5,
            "branch_transaction_volume": 500,
            "branch_fraud_rate_history": 0.03,
            "loan_purpose": "Business",
            "requested_amount": requested_amount,
            "loan_term_months": loan_term,
            "interest_rate": interest_rate,
            "loan_to_income_ratio": requested_amount / max(annual_income, 1),
            "collateral_provided": 1,
            "guarantors_count": 1,
            "group_lending": 0,
            "application_channel": "Mobile App",
            "processing_time_days": 2,
            "device_type": "Android",
            "browser_type": "Chrome",
            "network_type": "4G",
            "debt_service_ratio": 0.30,
            "avg_monthly_transactions": 25,
            "avg_transaction_amount": 3500,
            "total_transaction_volume": 85000,
            "transaction_velocity": 1.2,
            "cash_out_percentage": 0.25,
            "suspicious_transaction_count": 0,
            "high_value_transaction_flag": 0,
            "unusual_transaction_ratio": 0.05,
            "app_login_frequency": 12,
            "app_login_anomaly_score": 0.10,
            "device_changes": 0,
            "ip_changes": 0,
            "sim_card_changes": 0,
            "account_age_days": 720,
            "failed_login_attempts": 0,
            "biometric_verification": 1,
            "blacklist_match": 0,
            "duplicate_application": 0,
            "identity_verification_failed": 0,
            "kyc_completeness": 100,
            "document_verification_score": 95,
            "sanctions_screening": "Clear",
            "pep_flag": 0,
            "aml_alert_count": 0,
            "reference_check_result": "Pass",
            "income_discrepancy": 0,
            "social_connections": 25,
            "fraudster_connections": 0,
            "same_address_users": 1,
            "network_risk_score": 0.10,
            "typing_speed_anomaly": 0.05,
            "click_pattern_anomaly": 0.08,
            "bureau_score": 720,
            "application_date": "2026-07-29",
            "application_day_of_week": "Tuesday",
            "application_hour": 10,
            "income_stability_score": 0.90,
            "employment_tenure_years": 5,
            "credit_utilization_ratio": 0.35,
            "active_loans_count": 1,
            "total_debt_obligation": 50000,
            "financial_dependency_ratio": 0.20,
            "savings_balance": 150000
        }

        input_df = pd.DataFrame([input_data])

        result = predict_fraud(input_df)

        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:

            if result["prediction"] == 1:
                st.error("🔴 Fraud")
            else:
                st.success("🟢 Genuine")

        with col2:

            st.metric(
                "Fraud Probability",
                f"{result['probability']*100:.2f}%"
            )

        with col3:

            if result["probability"] < 0.20:
                risk = "🟢 Low"

            elif result["probability"] < 0.50:
                risk = "🟡 Medium"

            else:
                risk = "🔴 High"

            st.metric(
                "Risk Level",
                risk
            )



with tab2:

    st.header("About")

    st.write("""
This project predicts fraudulent microfinance loan applications
using a tuned HistGradientBoostingClassifier.

Pipeline includes:

- Data Understanding
- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Preprocessing
- Model Comparison
- Hyperparameter Tuning
- Threshold Optimization
- Model Evaluation
""")

    st.success("Final Model: HistGradientBoosting (Tuned)")