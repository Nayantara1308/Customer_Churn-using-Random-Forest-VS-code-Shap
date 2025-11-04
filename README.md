Customer Churn Prediction — Explainable AI Project (SHAP Analysis)

This project predicts customer churn (whether a customer will leave or stay) using a Random Forest Classifier and explains the model’s decisions using SHAP (SHapley Additive exPlanations). 
It demonstrates both strong predictive modeling and explainability, showing not just what the model predicts, but why.

Objectives

- Predict customer churn from behavioral and demographic data.
- Identify key drivers of churn using SHAP values.
- Build a transparent and interpretable ML pipeline.
- Generate actionable business insights from model explanations.

Key Insights

- Tenure and MonthlyCharges are the most influential factors driving churn.
- Short-tenure customers with high monthly charges show the highest churn probability.
- Demographic features like Partner and SeniorCitizen have minimal predictive contribution (SHAP ≈ 7×10⁻⁶).
- The model’s predictions can be trusted and audited using explainable AI methods.

Project Architecture

shap_safe_project/
│
├── data/
│   └── telco_churn_fe.csv                # Preprocessed customer data
│
├── outputs/
│   ├── churn_model.pkl                   # Trained Random Forest model
│   └── shap_safe/
│       ├── shap_feature_importance.csv   # SHAP summary data
│       ├── shap_top_features.png         # Feature importance plot
│       └── shap_beeswarm.png (optional)
│
├── compute_shap_safe.py                  # SHAP computation (lightweight)
├── plot_shap_from_saved.py               # Visualization script
└── streamlit_shap_viewer.py              # Interactive SHAP dashboard

How to Run the Project

1️⃣ Create and activate a virtual environment
    python -m venv venv
    venv\Scripts\activate

2️⃣ Install dependencies
    pip install --upgrade pip
    pip install joblib pandas numpy scikit-learn shap matplotlib seaborn tqdm streamlit

3️⃣ Run the SHAP computation
    python compute_shap_safe.py

4️⃣ Generate visualizations
    python plot_shap_from_saved.py

5️⃣ Launch the interactive SHAP dashboard
    streamlit run streamlit_shap_viewer.py

Outputs

- shap_feature_importance.csv : Ranked list of features with average absolute SHAP values
- shap_top_features.png : Bar chart of top 20 important features
- shap_beeswarm.png : (Optional) Beeswarm plot showing per-customer impact
- Streamlit dashboard : Interactive bar chart and data table for explainability

Technical Stack

Programming: Python
Libraries: pandas, numpy, scikit-learn, shap, matplotlib, streamlit
Model: RandomForestClassifier
Explainability: SHAP (TreeExplainer)
Visualization: Matplotlib, Streamlit

Model Summary

- Model trained using RandomForestClassifier.
- Handles both categorical and numerical variables.
- Tuned for balanced accuracy and interpretability.
- Evaluated using precision, recall, and ROC-AUC.

Explainability Summary

- SHAP Value: How much each feature pushed a prediction toward churn or retention.
- Positive SHAP Value: Increases churn likelihood.
- Negative SHAP Value: Reduces churn likelihood.
- Mean Absolute SHAP Value: Average influence of a feature across all customers.

Business Impact

- Identifies high-risk customers based on key churn drivers.
- Provides actionable insights:
  • Offer discounts or loyalty programs to short-tenure, high-charge customers.
  • Focus retention efforts on customers with month-to-month contracts.
- Builds trust by explaining every prediction transparently.

Future Enhancements

- Add LIME or Permutation Importance for comparison.
- Deploy the model with a Streamlit interface for real-time churn prediction.
- Integrate with CRM dashboards for automatic churn risk alerts.

Author

Nayantara Singh
Data Science & AI Professional


