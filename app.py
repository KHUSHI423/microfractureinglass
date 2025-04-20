import streamlit as st
import joblib
import numpy as np
import os

# Safe model loader
def load_model(path, name):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
        model = joblib.load(path)
        st.success(f"{name} loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")
        return None

# Define model file paths
clf_path = 'C:\Users\khush\Downloads\microfracture\fracture_detection_model.pkl'
scaler_path = 'C:\Users\khush\Downloads\microfracture\scaler.pkl'
reg_path = 'C:\Users\khush\Downloads\microfracture\lifespan_model.pkl



# Load models safely
clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")
reg_model = load_model(reg_path, "Regression Model")

# Proceed only if all models are loaded
if clf_model and scaler and reg_model:
    st.title("🔍 Microfracture Risk and Lifespan Estimator")

    # User inputs
    voltage = st.slider("📟 Piezo Voltage (V)", 0.0, 3.3, 0.1)
    thickness = st.number_input("🔍 Glass Thickness (cm)", min_value=0.1, max_value=2.0, value=0.4)

    input_data = np.array([[voltage, thickness]])

    try:
        input_scaled = scaler.transform(input_data)
        risk = clf_model.predict(input_scaled)[0]
        lifespan = reg_model.predict([[thickness]])[0]

        # Output results
        st.subheader("🔮 Prediction Results")
        st.write(f"📈 **Voltage**: `{voltage:.2f} V`")
        st.write(f"📏 **Glass Thickness**: `{thickness:.2f} cm`")
        st.write(f"⚠️ **Microfracture Risk**: {'High' if risk == 1 else 'Low'}")
        st.write(f"📅 **Estimated Remaining Lifespan**: `{lifespan:.2f} days`")

        # Visual feedback
        st.progress(min(voltage / 3.3, 1.0))

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.error("❌ Unable to load one or more model files. Please check your file paths and try again.")
