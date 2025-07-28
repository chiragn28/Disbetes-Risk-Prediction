import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt  # Required for waterfall

# Load model
APP_DIR = Path(__file__).parent
MODEL = pickle.load(open(APP_DIR / "diabetes_model.pkl", "rb"))

st.title("🩺 Diabetes Risk Prediction")
st.markdown("Enter patient details below to estimate the probability of diabetes.")

# Define features
features = [
    ("Pregnancies", 0, 20, 2),
    ("Glucose", 0, 200, 110),
    ("BloodPressure", 0, 140, 70),
    ("SkinThickness", 0, 100, 20),
    ("Insulin", 0, 900, 80),
    ("BMI", 0.0, 70.0, 30.0),
    ("DiabetesPedigreeFunction", 0.0, 2.5, 0.5),
    ("Age", 21, 100, 33)
]

# Collect input
user_input = []
for label, minv, maxv, default in features:
    if isinstance(default, float):
        val = st.slider(label, min_value=minv, max_value=maxv, value=default, step=0.1)
    else:
        val = st.slider(label, min_value=minv, max_value=maxv, value=default)
    user_input.append(val)

feature_names = [f[0] for f in features]
input_df = pd.DataFrame([user_input], columns=feature_names)

if st.button("🔍 Predict"):
    # Predict
    proba = MODEL.predict_proba(input_df)[0][1]
    pred = int(proba >= 0.5)
    st.metric("🧮 Predicted Diabetes Risk", f"{proba:.2f}")
    if pred:
        st.success("⚠️ Likely Diabetic")
    else:
        st.info("✅ Not Likely Diabetic")

    # SHAP Explanation - Waterfall Plot (Clearer & More Professional)
    try:
        st.subheader("📊 How This Prediction Was Made")

        # Extract model parts
        preprocessor = MODEL.named_steps['pre']
        model = MODEL.named_steps['rf']

        # Transform input
        input_transformed = preprocessor.transform(input_df)
        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()

        # Compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values_full = explainer.shap_values(input_transformed)

        # Handle base value
        expected_value = explainer.expected_value
        if isinstance(expected_value, (np.ndarray, list)) and len(expected_value) > 1:
            base_value = expected_value[1]
        else:
            base_value = expected_value if np.isscalar(expected_value) else expected_value[0]

        # Handle shap_values shape: (1, 8, 2)
        if isinstance(shap_values_full, np.ndarray) and shap_values_full.ndim == 3:
            shap_val = shap_values_full[0, :, 1]  # [sample=0, features, class=1]
        elif isinstance(shap_values_full, list):
            shap_val = shap_values_full[1][0] if len(shap_values_full) > 1 else shap_values_full[0][0]
        else:
            shap_val = shap_values_full[0]

        # Create SHAP Explanation object
        shap_explanation = shap.Explanation(
            values=shap_val,
            base_values=base_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        )

        # Instead of waterfall
        fig, ax = plt.subplots(figsize=(6, 5))
        shap.plots.bar(shap_explanation, show=False)

        # Improve layout
        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)
        plt.close(fig)

        # Add interpretation help
        st.caption("""
        🔹 **Red bars**: Increase diabetes risk  
        🔹 **Blue bars**: Decrease diabetes risk  
        🔹 **Length**: Magnitude of impact on prediction
        """)

    except Exception as e:
        st.warning(f"SHAP explanation not available: {str(e)}")