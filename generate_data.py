"""
Generates a synthetic telecom customer churn dataset
(structured similarly to the popular IBM Telco Customer Churn dataset).
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000

genders = np.random.choice(['Male', 'Female'], n)
senior = np.random.choice([0, 1], n, p=[0.84, 0.16])
partner = np.random.choice(['Yes', 'No'], n)
dependents = np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7])
tenure = np.random.randint(0, 73, n)
phone_service = np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1])
multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], n, p=[0.42, 0.48, 0.1])
internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
online_security = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22])
online_backup = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22])
device_protection = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.34, 0.44, 0.22])
tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22])
streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22])
streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.38, 0.40, 0.22])
contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24])
paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.59, 0.41])
payment_method = np.random.choice(
    ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
    n, p=[0.34, 0.23, 0.22, 0.21]
)

monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
total_charges = np.clip(total_charges, 0, None)

# Build churn probability based on realistic patterns
churn_prob = (
    0.05
    + 0.25 * (contract == 'Month-to-month')
    + 0.10 * (internet_service == 'Fiber optic')
    + 0.10 * (payment_method == 'Electronic check')
    + 0.15 * (tenure < 6)
    - 0.15 * (tenure > 48)
    - 0.05 * (contract == 'Two year')
    + 0.05 * (online_security == 'No')
    + 0.05 * (tech_support == 'No')
    + 0.0015 * (monthly_charges - 65)
)
churn_prob = np.clip(churn_prob, 0.02, 0.95)
churn = np.where(np.random.rand(n) < churn_prob, 'Yes', 'No')

df = pd.DataFrame({
    'customerID': [f'C{1000+i}' for i in range(n)],
    'gender': genders,
    'SeniorCitizen': senior,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phone_service,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn,
})

df.to_csv('telecom_churn.csv', index=False)
print("Dataset saved. Shape:", df.shape)
print(df['Churn'].value_counts(normalize=True))
