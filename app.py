import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import time
import random

# Safe model loader
def load_model(path, name):
    try:
        st.write(f"Attempting to load {name} from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
        model = joblib.load(path)
        st.success(f"{name} loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")
        return None

# Lifespan estimation logic
def estimate_lifespan(thickness_mm, voltage):
    if 2 <= thickness_mm < 3:
        lifespan = np.random.uniform(10, 20)
    elif 3 <= thickness_mm < 6.38:
        lifespan = np.random.uniform(20, 30)
    elif 6.38 <= thickness_mm < 20:
        lifespan = np.random.uniform(30, 50)
    elif 20 <= thickness_mm < 40:
        lifespan = np.random.uniform(15, 25)
    elif 40 <= thickness_mm <= 75:
        lifespan = np.random.uniform(20, 25)
    elif round(thickness_mm, 1) == 6.8:
        lifespan = np.random.uniform(15, 20)
    else:
        lifespan = 15  # default

    voltage_impact = lifespan * (voltage / 3.3) * 0.05
    return max(lifespan - voltage_impact, 0)

# File paths
clf_path = 'fracture_detection_model.pkl'
scaler_path = 'scaler.pkl'

# Load models
clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")

# Main UI
if clf_model and scaler:
    st.title("🔍 Microfracture Risk & Lifespan Estimator (Simulated)")

    placeholder = st.empty()
    data_list = []

    if st.button("Start Monitoring"):
        for _ in range(20):  # Simulate 20 readings
            timestamp = int(time.time())
            voltage = round(random.uniform(0.5, 3.3), 2)
            thickness_cm = round(random.uniform(0.2, 0.9), 2)
            thickness_mm = thickness_cm * 10

            input_data = np.array([[voltage, thickness_cm]])

            try:
                input_scaled = scaler.transform(input_data)
                risk = clf_model.predict(input_scaled)[0]
                lifespan_prediction = estimate_lifespan(thickness_mm, voltage)

                row = {
                    "Timestamp": timestamp,
                    "Voltage": voltage,
                    "Thickness (cm)": thickness_cm,
                    "Risk": 'High' if risk == 1 else 'Low',
                    "Lifespan (years)": round(lifespan_prediction, 2)
                }
                data_list.append(row)

                st.subheader("🔮 Prediction Results")
                st.write(f"📈 **Voltage**: `{voltage:.2f} V`")
                st.write(f
