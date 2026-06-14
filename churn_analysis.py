"""
Customer Churn Prediction
==========================
Goal: Predict which customers are likely to stop using a telecom service.

Pipeline:
1. Load & explore data (EDA)
2. Preprocess (encode categoricals, scale numerics)
3. Train classification models: Logistic Regression, Random Forest, XGBoost
4. Evaluate with accuracy, recall, ROC-AUC
5. Visualize results & feature importance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, roc_auc_score, roc_curve,
    classification_report, confusion_matrix
)
from xgboost import XGBClassifier

sns.set_style('whitegrid')
OUT = '.'

# ---------------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------------
df = pd.read_csv(f'{OUT}/telecom_churn.csv')
print("Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

sns.countplot(data=df, x='Churn', ax=axes[0, 0], palette='Set2')
axes[0, 0].set_title('Overall Churn Distribution')

sns.countplot(data=df, x='Contract', hue='Churn', ax=axes[0, 1], palette='Set2')
axes[0, 1].set_title('Churn by Contract Type')
axes[0, 1].tick_params(axis='x', rotation=15)

sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=20, ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Tenure Distribution by Churn')

sns.boxplot(data=df, x='Churn', y='MonthlyCharges', ax=axes[1, 1], palette='Set2')
axes[1, 1].set_title('Monthly Charges by Churn')

plt.tight_layout()
plt.savefig(f'{OUT}/eda_overview.png', dpi=120)
plt.close()
print("Saved eda_overview.png")

churn_by_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
print("\nChurn rate by contract type:\n", churn_by_contract)

# ---------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------
data = df.drop(columns=['customerID']).copy()
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

categorical_cols = data.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

X = data.drop(columns=['Churn'])
y = data['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------------
# 4. Model Training
# ---------------------------------------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric='logloss', random_state=42
    )
}

results = []
roc_data = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({'Model': name, 'Accuracy': acc, 'Recall': rec, 'ROC-AUC': auc})
    roc_data[name] = roc_curve(y_test, y_proba)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Recall: {rec:.4f} | ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUT}/model_results.csv', index=False)
print("\nModel comparison:\n", results_df)

# ---------------------------------------------------------------
# 5. ROC Curve Comparison
# ---------------------------------------------------------------
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in roc_data.items():
    X_eval = X_test_scaled if name == 'Logistic Regression' else X_test
    auc = roc_auc_score(y_test, models[name].predict_proba(X_eval)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/roc_curves.png', dpi=120)
plt.close()
print("Saved roc_curves.png")

# ---------------------------------------------------------------
# 6. Feature Importance (Random Forest & XGBoost)
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

rf_importance = pd.Series(models['Random Forest'].feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
sns.barplot(x=rf_importance.values, y=rf_importance.index, ax=axes[0], palette='viridis')
axes[0].set_title('Top 10 Features - Random Forest')

xgb_importance = pd.Series(models['XGBoost'].feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
sns.barplot(x=xgb_importance.values, y=xgb_importance.index, ax=axes[1], palette='magma')
axes[1].set_title('Top 10 Features - XGBoost')

plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance.png', dpi=120)
plt.close()
print("Saved feature_importance.png")

# ---------------------------------------------------------------
# 7. Confusion Matrix for best model (by ROC-AUC)
# ---------------------------------------------------------------
best_model_name = results_df.sort_values('ROC-AUC', ascending=False).iloc[0]['Model']
best_model = models[best_model_name]
X_eval = X_test_scaled if best_model_name == 'Logistic Regression' else X_test
cm = confusion_matrix(y_test, best_model.predict(X_eval))

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
plt.title(f'Confusion Matrix - {best_model_name} (Best Model)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(f'{OUT}/confusion_matrix.png', dpi=120)
plt.close()
print(f"Saved confusion_matrix.png (Best model: {best_model_name})")

print("\nAll outputs generated successfully.")
