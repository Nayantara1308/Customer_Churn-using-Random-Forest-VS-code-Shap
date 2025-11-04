
import os, joblib, numpy as np, pandas as pd, shap, matplotlib.pyplot as plt
from tqdm import tqdm

# Safe batch settings
BATCH_SIZE = 50
EXPLAIN_SAMPLE_SIZE = 100

MODEL_PATH = 'outputs/churn_model.pkl'
DATA_PATH = 'data/telco_churn_fe.csv'
OUT_DIR = 'outputs/shap_safe'
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model and data...")
model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Use only the first N samples for explanation
df_sample = df.sample(n=min(EXPLAIN_SAMPLE_SIZE, len(df)), random_state=42)
X = df_sample.values

print("Initializing SHAP explainer...")
explainer = explainer = shap.TreeExplainer(model, feature_perturbation='interventional')

shap_values = []
print("Computing SHAP values...")
for i in tqdm(range(0, len(X), BATCH_SIZE)):
    batch = X[i:i+BATCH_SIZE]
    vals = explainer.shap_values(batch)
    shap_values.append(vals)

# Handle multi-class shap_values safely
if isinstance(shap_values[0], list):
    # Average over classes for simplicity
    shap_values = np.mean([np.concatenate(v, axis=0) for v in zip(*shap_values)], axis=0)
else:
    shap_values = np.concatenate(shap_values, axis=0)

# Ensure it's 2D
if shap_values.ndim > 2:
    shap_values = np.mean(shap_values, axis=1)


# Handle feature importance alignment safely
print("Preparing SHAP feature importance DataFrame...")

# If shap_values is a list (multi-class), average across classes
if isinstance(shap_values, list):
    shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

# If still mismatched, truncate or pad
num_features = len(df_sample.columns)
if shap_values.shape[1] != num_features:
    print(f"⚠️ Mismatch detected: shap_values has {shap_values.shape[1]} features, dataset has {num_features}. Adjusting...")
    min_len = min(num_features, shap_values.shape[1])
    shap_values = shap_values[:, :min_len]
    df_sample = df_sample.iloc[:, :min_len]

# Compute mean absolute importance
feature_importance = np.abs(shap_values).mean(axis=0)

# Build DataFrame safely
imp_df = pd.DataFrame({
    'feature': df_sample.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Save outputs
imp_df.to_csv(os.path.join(OUT_DIR, 'shap_feature_importance.csv'), index=False)

plt.figure(figsize=(8,6))
imp_df.head(20).plot.barh(x='feature', y='importance', legend=False)
plt.title('Top 20 Feature Importances (SHAP)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_top_features.png'))
plt.close()

print("✅ Done! Results saved in", OUT_DIR)

