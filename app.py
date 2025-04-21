import streamlit as st
import joblib
import numpy as np
import os
import serial
import pandas as pd
import time

# ---------- SETTINGS ----------
USE_MANUAL_RISK_LOGIC = True  # Toggle fallback logic if model always returns Low

# ---------- Safe model loader ----------
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

# ---------- Lifespan estimation ----------
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
        lifespan = 15

    voltage_impact = lifespan * (voltage / 3.3) * 0.05
    return max(lifespan - voltage_impact, 0)

# ---------- Risk Prediction Fallback ----------
def manual_risk_estimation(voltage, thickness_cm):
    if voltage > 2.5 or thickness_cm < 0.3:
        return 1  # High risk
    return 0  # Low risk

# ---------- File paths ----------
clf_path = 'fracture_detection_model.pkl'
scaler_path = 'scaler.pkl'

clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")

# ---------- Serial Setup ----------
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    st.success(f"Connected to {SERIAL_PORT}")
except Exception as e:
    st.error(f"Failed to connect to {SERIAL_PORT}: {e}")
    st.stop()

# ---------- Streamlit App ----------
if clf_model and scaler and ser:
    st.title("🔍 Microfracture Risk & Lifespan Estimator")

    placeholder = st.empty()
    data_list = []

    if st.button("Start Monitoring"):
        while True:
            line = ser.readline().decode("utf-8").strip()
            if line and not line.startswith("timestamp"):
                parts = line.split(",")
                if len(parts) == 6:
                    timestamp = int(parts[0])
                    voltage = float(parts[2])
                    thickness_cm = float(parts[3])
                    thickness_mm = thickness_cm * 10

                    input_data = np.array([[voltage, thickness_cm]])
                    st.write(f"🔍 Raw input data: {input_data}")

                    try:
                        input_scaled = scaler.transform(input_data)
                        model_output = clf_model.predict(input_scaled)[0]
                        st.write(f"📊 Model predicted: {model_output}")

                        if USE_MANUAL_RISK_LOGIC:
                            risk = manual_risk_estimation(voltage, thickness_cm)
                            st.write(f"⚠️ Using manual logic — Risk: {'High' if risk == 1 else 'Low'}")
                        else:
                            risk = model_output

                        lifespan = estimate_lifespan(thickness_mm, voltage)

                        st.subheader("🔮 Prediction Results")
                        st.write(f"📈 **Voltage**: `{voltage:.2f} V`")
                        st.write(f"📏 **Glass Thickness**: `{thickness_cm:.2f} cm`")
                        st.write(f"⚠️ **Microfracture Risk**: {'High' if risk == 1 else 'Low'}`")
                        st.write(f"📅 **Estimated Lifespan**: `{lifespan:.2f} years`")

                        row = {
                            "Timestamp": timestamp,
                            "Voltage": voltage,
                            "Thickness (cm)": thickness_cm,
                            "Risk": 'High' if risk == 1 else 'Low',
                            "Lifespan (years)": lifespan
                        }

                        data_list.append(row)
                        df = pd.DataFrame(data_list).tail(10)
                        placeholder.dataframe(df, use_container_width=True)
                        st.progress(min(voltage / 3.3, 1.0))

                    except Exception as e:
                        st.error(f"Prediction error: {e}")
            time.sleep(0.1)
else:
    st.error("❌ Unable to load one or more model files or connect to the ESP32.")
