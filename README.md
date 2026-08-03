<div align="center">

#  Microfinance Fraud Detection System

### Intelligent Fraud Detection using Machine Learning & Predictive Analytics

<p align="center">
Detect fraudulent loan applications using advanced Machine Learning techniques and an interactive Streamlit dashboard.
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikitlearn)

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)

![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)

![NumPy](https://img.shields.io/badge/NumPy-Numerical-blue?style=for-the-badge&logo=numpy)

![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

### 🚀 Built for Real-World Financial Fraud Detection

Machine Learning • Data Science • Streamlit • Fraud Analytics • Predictive Modeling

</div>

---

# 📖 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Project Features](#project-features)
- [Technology Stack](#technology-stack)
- [Dataset Overview](#dataset-overview)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Project Architecture](#project-architecture)
- [Folder Structure](#folder-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Development](#model-development)
- [Results](#results)
- [Dashboard](#dashboard)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

# 🚀 Project Overview

Financial fraud has become one of the biggest challenges for banks, microfinance institutions, and digital lending platforms. Detecting fraudulent loan applications at an early stage helps financial organizations reduce monetary losses, improve customer trust, and make smarter lending decisions.

This project presents an end-to-end Machine Learning solution capable of identifying potentially fraudulent loan applications using customer information, financial history, behavioral patterns, and device-related attributes.

Unlike traditional rule-based systems, this solution learns complex relationships from historical data and predicts fraudulent applications with a probability score rather than relying on fixed rules.

The project demonstrates the complete lifecycle of a production-oriented Machine Learning project, including:

- Data Understanding
- Exploratory Data Analysis
- Data Cleaning
- Feature Engineering
- Data Preprocessing
- Model Training
- Hyperparameter Optimization
- Model Evaluation
- Threshold Optimization
- Interactive Streamlit Dashboard

The final application enables users to enter loan application details through an intuitive web interface and receive an instant fraud prediction with its confidence score and risk level.

---

# 🎯 Business Problem

Fraudulent loan applications create significant financial losses for microfinance institutions every year.

Traditional fraud detection systems often rely on manually written rules, making them difficult to scale and ineffective against evolving fraud strategies.

The objective of this project is to build an intelligent Machine Learning model capable of automatically identifying suspicious loan applications before loan approval.

Such a system can help organizations:

- Reduce financial losses
- Improve loan approval quality
- Detect suspicious customer behavior
- Minimize manual verification effort
- Increase operational efficiency
- Support better decision-making

---

# ⭐ Project Features

## Machine Learning

- End-to-End ML Pipeline
- Data Cleaning
- Feature Engineering
- Feature Preprocessing
- Hyperparameter Tuning
- Threshold Optimization
- Model Evaluation
- Model Persistence using Joblib

---

## Dashboard

- Interactive Streamlit Interface
- Real-Time Fraud Prediction
- Fraud Probability Score
- Risk Level Indicator
- User-Friendly Input Forms
- Professional Banking-Style UI

---

## Data Processing

- Missing Value Handling
- Categorical Encoding
- Numerical Scaling
- Automated Feature Transformation
- Production-Ready Prediction Pipeline

---

## Model Evaluation

Multiple Machine Learning algorithms were evaluated and compared before selecting the best-performing model.

Algorithms explored include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

The final model was selected based on predictive performance and generalization capability after hyperparameter optimization.

---

# 🛠 Technology Stack

| Category | Technologies |
|----------|--------------|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Gradient Boosting | XGBoost, LightGBM, CatBoost |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Model Serialization | Joblib |
| Development | VS Code |
| Version Control | Git |
| Repository Hosting | GitHub |



---

# 📊 Dataset Overview

The dataset used in this project represents historical loan application records collected from a microfinance environment. It contains demographic, financial, behavioral, and security-related attributes that help distinguish genuine applications from fraudulent ones.

## Dataset Summary

| Property | Value |
|----------|-------|
| Total Records | 50,000 |
| Total Features | 77 |
| Target Variable | `is_fraud` |
| Fraud Distribution | ~3% |
| Problem Type | Binary Classification |

---

## Feature Categories

### 👤 Customer Information

- Customer Age
- Gender
- Marital Status
- Education Level
- Occupation
- Dependents
- Home Ownership

---

### 💰 Financial Information

- Annual Income
- Monthly Expenses
- Credit Score
- Bureau Score
- Savings Balance
- Total Debt Obligation
- Credit Utilization Ratio
- Income Stability Score

---

### 🏦 Loan Information

- Loan Purpose
- Requested Amount
- Loan Term
- Interest Rate
- Guarantors
- Collateral
- Group Lending

---

### 📱 Behavioral Features

- Login Frequency
- Device Changes
- IP Changes
- SIM Card Changes
- Failed Login Attempts
- Typing Speed Anomaly
- Click Pattern Anomaly

---

### 🛡 Security & Verification

- KYC Completeness
- AML Alert Count
- Sanctions Screening
- Identity Verification
- Blacklist Match
- PEP Flag
- Duplicate Application

---

# 🧹 Data Preprocessing

Before training the Machine Learning models, several preprocessing techniques were applied to improve data quality and model performance.

## Data Cleaning

✔ Removed duplicate records

✔ Handled missing values

✔ Corrected inconsistent values

✔ Removed unnecessary columns

✔ Eliminated data leakage features

---

## Feature Engineering

Several derived features were generated to improve predictive capability.

Examples include:

- Loan-to-Income Ratio
- Debt Service Ratio
- Financial Dependency Ratio

These engineered features provide additional insights into applicant financial behavior.

---

## Data Transformation

The preprocessing pipeline includes:

- One-Hot Encoding for categorical variables
- Standard Scaling for numerical variables
- ColumnTransformer Pipeline
- Production-ready preprocessing using Joblib

---

# 🤖 Machine Learning Pipeline

The project follows an end-to-end Machine Learning workflow.

```text
Raw Dataset
      │
      ▼
Data Understanding
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Data Preprocessing
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Optimization
      │
      ▼
Threshold Optimization
      │
      ▼
Model Evaluation
      │
      ▼
Best Model Selection
      │
      ▼
Model Serialization
      │
      ▼
Streamlit Dashboard
```

---

# 🏗 Project Architecture

```text
                         User
                           │
                           ▼
                  Streamlit Dashboard
                           │
                           ▼
                   Input Validation
                           │
                           ▼
                Feature Engineering
                           │
                           ▼
                Data Preprocessing
                           │
                           ▼
                 Trained ML Model
                           │
                           ▼
              Fraud Probability Score
                           │
                           ▼
          Fraud Prediction & Risk Level
```

---

# 📂 Project Structure

```text
Microfinance-Fraud-Detection/
│
├── dashboard/
│   ├── app.py
│   └── style.css
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│
├── models/
│   ├── best_model.pkl
│   ├── best_model_meta.pkl
│   ├── preprocessor.pkl
│   ├── X_train.pkl
│   ├── X_test.pkl
│   ├── y_train.pkl
│   └── y_test.pkl
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_EDA.ipynb
│   ├── 03_Data_Cleaning.ipynb
│   ├── 04_Preprocessing.ipynb
│   ├── 05_Model_Training.ipynb
│   └── 06_Model_Evaluation.ipynb
│
├── reports/
│
├── src/
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── transformers.py
│   └── utils.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# ⚙ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/Microfinance-Fraud-Detection.git

cd Microfinance-Fraud-Detection
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Launch the Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

---

#  Quick Start

Once the application starts:

1. Open the local Streamlit URL in your browser.
2. Enter the applicant's information.
3. Click **Predict Fraud**.
4. View the predicted fraud status.
5. Review the fraud probability score.
6. Interpret the applicant's risk level.

---

# 💻 Usage Workflow

```text
Enter Applicant Details
          │
          ▼
Validate Inputs
          │
          ▼
Generate Engineered Features
          │
          ▼
Apply Preprocessing Pipeline
          │
          ▼
Predict Fraud Probability
          │
          ▼
Display Prediction
          │
          ▼
Risk Assessment
```

---

# 🔄 Model Development Workflow

The development process followed a structured Machine Learning pipeline:

### Phase 1

- Data Understanding

---

### Phase 2

- Exploratory Data Analysis

---

### Phase 3

- Data Cleaning

---

### Phase 4

- Feature Engineering

---

### Phase 5

- Data Preprocessing

---

### Phase 6

- Model Training

---

### Phase 7

- Hyperparameter Tuning

---

### Phase 8

- Model Evaluation

---

### Phase 9

- Model Selection

---

### Phase 10

- Streamlit Deployment


---

# 📊 Model Comparison

Multiple Machine Learning algorithms were trained and evaluated to identify the best-performing model for fraud detection.

| Model | Status |
|---------|--------|
| Logistic Regression | Evaluated |
| Decision Tree | Evaluated |
| Random Forest | Evaluated |
| Gradient Boosting | Evaluated |
| HistGradientBoosting | ⭐ Selected |
| XGBoost | Evaluated |
| LightGBM | Evaluated |
| CatBoost | Evaluated |

The final model was selected after comparing performance across multiple evaluation metrics and performing hyperparameter optimization.

---

# 🏆 Final Model

**Selected Model**

> **HistGradientBoostingClassifier**

### Why this model?

- Excellent generalization performance
- Handles complex feature interactions
- Fast inference time
- Robust against overfitting
- Suitable for tabular financial datasets
- Performs well on imbalanced classification problems after threshold optimization

---

# 📈 Model Performance

| Metric | Score |
|---------|--------|
| PR-AUC | **0.0945** |
| Precision (Fraud) | **0.16** |
| Recall (Fraud) | **0.17** |
| F1 Score | **0.16** |

---

# 🎯 Threshold Optimization

Instead of relying on the default prediction threshold (0.50), the model's threshold was optimized to improve fraud detection performance.

Benefits:

- Better fraud recall
- Improved business usability
- Reduced false negatives
- Balanced precision and recall

---

# 📉 Evaluation Techniques

The model was evaluated using multiple performance metrics.

✔ Classification Report

✔ Confusion Matrix

✔ Precision-Recall Curve

✔ ROC Curve

✔ PR-AUC

✔ Threshold Optimization

✔ Cross Validation

---

# 💼 Business Impact

Deploying a Machine Learning fraud detection system provides several operational and financial advantages.

### Financial Benefits

- Reduce fraudulent loan approvals
- Minimize financial losses
- Improve portfolio quality
- Increase operational efficiency

---

### Customer Benefits

- Faster loan approval
- Fair risk assessment
- Better customer experience
- Reduced manual verification delays

---

### Organizational Benefits

- Data-driven lending decisions
- Scalable fraud monitoring
- Automated risk assessment
- Improved compliance support

---

# 🖥 Dashboard

The Streamlit dashboard provides an intuitive interface for real-time fraud prediction.

## Dashboard Features

- Professional banking-style interface
- Applicant information form
- Fraud prediction
- Fraud probability score
- Risk level indicator
- Model information
- Responsive layout

---

## Dashboard Preview

Create an **images** folder inside the project and place your screenshots.

```text
images/

dashboard_home.png

prediction.png

risk_analysis.png
```

Then display them inside README.

```markdown
## Dashboard

### Home Page

![Home](images/dashboard_home.png)

---

### Prediction

![Prediction](images/prediction.png)

---

### Risk Analysis

![Risk](images/risk_analysis.png)
```

---

# 🔬 Future Improvements

This project can be further enhanced by integrating advanced production-level capabilities.

### Explainable AI

- SHAP
- LIME

---

### Deep Learning

- Neural Networks
- Autoencoders

---

### Cloud Deployment

- AWS
- Azure
- Google Cloud Platform

---

### API Development

- FastAPI
- Flask REST API

---

### Containerization

- Docker
- Docker Compose

---

### MLOps

- MLflow
- DVC
- Model Monitoring
- Data Drift Detection
- CI/CD Pipelines

---

### Real-Time Fraud Detection

- Apache Kafka
- Streaming Prediction
- Event-driven Processing

---

# 🧪 Testing

The project has been designed with modular components to simplify testing.

Recommended testing includes:

- Unit testing for preprocessing
- Prediction pipeline validation
- Input validation
- Dashboard functionality
- Model loading tests

---

# 🤝 Contributing

Contributions are welcome.

If you would like to improve this project:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push to your branch.
5. Open a Pull Request.

---

# 📚 Learning Outcomes

This project demonstrates practical knowledge of:

- Machine Learning
- Data Cleaning
- Feature Engineering
- Feature Preprocessing
- Hyperparameter Optimization
- Model Evaluation
- Threshold Optimization
- Streamlit Dashboard Development
- Git & GitHub
- Production-ready Project Structure

---

# 👨‍💻 Author

## Dharavath Dinesh

Computer Science & Engineering Student

Passionate about

- Machine Learning
- Artificial Intelligence
- Data Science
- Full Stack Development

### GitHub

https://github.com/ad-Dinesh

---

# 📄 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project for educational and research purposes.

---

# ⭐ Support

If you found this project useful,

⭐ Star the repository

🍴 Fork the repository

🛠 Contribute to improvements

---

<div align="center">

## Thank You for Visiting!

If you enjoyed exploring this project, consider giving it a ⭐ on GitHub.

**Happy Coding! 🚀**

</div>
