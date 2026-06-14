# Customer Churn Prediction

## Goal
Predict which telecom customers are likely to stop using the service (churn), using exploratory data analysis and classification models.

## Dataset
A telecom customer dataset (5,000 records) with demographic info, service subscriptions, contract details, billing information, and churn label. Structured similarly to the popular IBM Telco Customer Churn dataset (`telecom_churn.csv`).

## Workflow
1. **EDA** — Explored churn distribution, churn rate by contract type, tenure patterns, and monthly charges (`eda_overview.png`). Key finding: month-to-month contracts and short-tenure customers churn far more often than long-term/contract customers.
2. **Preprocessing** — Label-encoded categorical features, scaled numeric features (tenure, MonthlyCharges, TotalCharges) for logistic regression.
3. **Modeling** — Trained and compared three classifiers:
   - Logistic Regression
   - Random Forest
   - XGBoost
4. **Evaluation** — Compared models using Accuracy, Recall, and ROC-AUC (`model_results.csv`, `roc_curves.png`).
5. **Feature Importance** — Identified top drivers of churn from Random Forest and XGBoost (`feature_importance.png`).
6. **Confusion Matrix** — Visualized prediction errors for the best-performing model (`confusion_matrix.png`).

## Results
| Model | Accuracy | Recall | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~0.72 | ~0.22 | ~0.71 |
| Random Forest | ~0.71 | ~0.20 | ~0.69 |
| XGBoost | ~0.71 | ~0.30 | ~0.70 |

(See `model_results.csv` for exact figures from your run.)

## Key Insights
- Customers on **month-to-month contracts** have the highest churn rate.
- **Tenure** is a strong predictor — customers in their first few months are at the highest risk.
- **Electronic check** payment and **fiber optic** internet service are associated with higher churn.
- Lack of online security / tech support add-ons correlates with higher churn.

## Tech Stack
Python, pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn

## How to Run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
python generate_data.py     # creates telecom_churn.csv
python churn_analysis.py    # runs full pipeline, saves charts and results
```

## Files
- `generate_data.py` — generates the dataset
- `telecom_churn.csv` — dataset
- `churn_analysis.py` — full EDA + modeling pipeline
- `eda_overview.png`, `roc_curves.png`, `feature_importance.png`, `confusion_matrix.png` — visual outputs
- `model_results.csv` — model comparison table
