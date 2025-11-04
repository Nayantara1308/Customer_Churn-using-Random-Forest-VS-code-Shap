
import os, pandas as pd, matplotlib.pyplot as plt

OUT_DIR = 'outputs/shap_safe'
imp = pd.read_csv(os.path.join(OUT_DIR, 'shap_feature_importance.csv'))

plt.figure(figsize=(8,6))
imp.head(20).plot.barh(x='feature', y='importance', legend=False)
plt.title('Top 20 Feature Importances (SHAP)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_top_features_pretty.png'))
plt.close()

print("✅ Plots generated successfully in", OUT_DIR)
