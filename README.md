# Customer Churn Prediction 📉

A machine learning project that predicts customer churn for a telecom company using **Logistic Regression**, with a full pipeline covering exploratory data analysis, preprocessing, class-imbalance handling, threshold tuning, and model interpretability.

---

## 🎯 Problem Statement

Customer churn — when a subscriber stops using a service — is one of the most expensive problems in the telecom industry. This project builds a classification model to **predict which customers are likely to churn**, so the business can proactively target them with retention offers.

**Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (7,043 customers, 21 features)

---

## 🤔 Why Logistic Regression?

Logistic Regression was chosen as the primary model because it is highly interpretable — each feature's coefficient directly shows its impact on churn probability (see Feature Importance section below). This matters in a business setting, where stakeholders need to understand *why* a customer is predicted to churn, not just *that* they will. More complex models (Random Forest, XGBoost) could potentially push raw accuracy higher, but they trade away this transparency — a key consideration for customer-retention decisions.

---

## 🔍 Project Workflow

| Stage | What was done |
|---|---|
| **Data Cleaning** | Removed `customerID`, fixed `TotalCharges` type issues, dropped missing rows |
| **EDA** | Analyzed churn distribution vs. contract type, tenure, monthly charges, payment method, internet service, and demographics |
| **Preprocessing** | `StandardScaler` for numeric features, `OneHotEncoder` for categorical features via `ColumnTransformer` |
| **Modeling** | Logistic Regression — baseline vs. `class_weight="balanced"` to address class imbalance |
| **Evaluation** | 5-fold `StratifiedKFold` cross-validation on Accuracy, Precision, Recall, F1, ROC-AUC |
| **Threshold Tuning** | Swept decision thresholds (0.30–0.70) to find the F1-optimal cutoff instead of the default 0.5 |
| **Interpretability** | Extracted and visualized top positive/negative churn-driving features from model coefficients |
| **Deployment-ready** | Wrapped prediction logic in a reusable `predict_churn()` function and saved the final model with `joblib` |

---

## 📊 Key Results

Final model performance on the held-out test set (Balanced Logistic Regression):

| Metric | Churn Class |
|---|---|
| Accuracy | `0.75` |
| Precision | `0.52` |
| Recall | `0.76` |
| F1 Score | `0.61` |

**Decision threshold** was tuned (rather than using the default 0.5) via F1-maximization on cross-validated predictions, deliberately favoring recall over precision — since missing a customer who will actually churn (a false negative) is more costly to the business than flagging a customer who won't (a false alarm).

**Top churn drivers identified:**
- Month-to-month contracts
- Fiber optic internet service
- Electronic check payment method
- Low tenure

*(See `visuals/` for the confusion matrix and feature coefficient charts.)*

---

## 🗂️ Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── models/
│   └── customer_churn_logistic_model.pkl
│
├── visuals/
│   ├── churn_distribution.png
│   ├── contract_churn.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Nahidhassn/customer-churn-prediction.git
cd customer-churn-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook notebooks/churn_analysis.ipynb
```

---

## 🧠 Using the Trained Model

```python
import joblib

artifact = joblib.load("models/customer_churn_logistic_model.pkl")
model = artifact["model"]
threshold = artifact["threshold"]

probabilities = model.predict_proba(new_customer_data)[:, 1]
predictions = (probabilities >= threshold).astype(int)
```

---

## 🛠️ Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Machine Learning** — scikit-learn (Logistic Regression, Pipelines, ColumnTransformer)
- **Model Persistence** — joblib

---

## 📌 Future Improvements

- Try ensemble models (Random Forest, XGBoost) and compare against the Logistic Regression baseline
- Add SHAP values for deeper model explainability
- Deploy as a REST API or simple Streamlit app for interactive predictions
- Experiment with SMOTE instead of `class_weight="balanced"`

---

## 👤 Author

**Nahid Hasan**
[Email](mailto:nahidhassan.ctc@gmail.com)

*This project was built as part of my data science portfolio to demonstrate end-to-end ML workflow: from raw data to a deployable, interpretable churn prediction model.*
