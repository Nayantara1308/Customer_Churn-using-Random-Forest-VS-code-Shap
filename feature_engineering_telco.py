# feature_engineering_telco.py
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Clean data
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].str.strip()

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)

# Convert binary Yes/No
binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']
for c in binary_cols:
    if c in df.columns:
        df[c] = df[c].map({'Yes':1,'No':0})

# Replace service variants
replace_map = {'No internet service':'No','No phone service':'No'}
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].replace(replace_map)

# One-hot encode categorical variables
df = pd.get_dummies(df, drop_first=True)

# Scale numeric columns
scaler = StandardScaler()
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

# Save processed file
df.to_csv('data/telco_churn_fe.csv', index=False)
print('✅ Saved feature-engineered file: data/telco_churn_fe.csv')
