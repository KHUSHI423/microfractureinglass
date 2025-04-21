import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
import time

# Attempt to import serial (for ESP32 communication)
try:
    import serial
    SERIAL_AVAILABLE = True
except ModuleNotFoundError:
    SERIAL_AVAILABLE = False
    st.warning("⚠️ pyserial not found. Running in simulation mode.")

# --------- Safe model loader ---------
def load_model(path, name):
    try:
        st.write(f"Attempting to load {name} from `{path}`")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
        model = joblib.load(path)
        st.success(f"{name} loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Failed to load {name}: {e}")
        return None

# --------- Lifespan Estimator Logic ---------
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

# --------- File paths ---------
clf_path = 'fracture_detection_model.pkl'
scaler_path = 'scaler.pkl'

# --------- Load models ---------
clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")

# --------- Serial Port Setup (Local Only) ---------
SERIAL_PORT = 'COM4'
BAUD_RATE = 115200
ser = None

if SERIAL_AVAILABLE:
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        st.success(f"✅ Connected to {SERIAL_PORT}")
    except Exception as e:
        st.error(f"❌ Failed to connect to {SERIAL_PORT}: {e}")
        ser = None

# --------- Main App ---------
if clf_model and scaler:
    st.title("🔍 Microfracture Risk & Lifespan Estimator")

    placeholder = st.empty()
    data_list = []

    if st.button("Start Monitoring"):
        while True:
            try:
                if ser:
                    line = ser.readline().decode("utf-8").strip()
                else:
                    # Simulated data for Streamlit Cloud
                    time.sleep(1)
                    timestamp = int(time.time())
                    voltage = np.random.uniform(0.5, 3.2)
                    thickness_cm = np.random.choice([0.3, 0.4, 0.5, 0.6])
                    line = f"{timestamp},0,{voltage:.2f},{thickness_cm},0,0"

                if line and not line.startswith("timestamp"):
                    parts = line.split(",")
                    if len(parts) == 6:
                        timestamp = int(parts[0])
                        voltage = float(parts[2])
                        thickness_cm = float(parts[3])
                        thickness_mm = thickness_cm * 10

                        input_data = np.array([[voltage, thickness_cm]])
                        input_scaled = scaler.transform(input_data)
                        risk = clf_model.predict(input_scaled)[0]
                        lifespan = estimate_lifespan(thickness_mm, voltage)

                        st.subheader("🔮 Prediction Results")
                        st.write(f"📈 **Voltage**: `{voltage:.2f} V`")
                        st.write(f"📏 **Glass Thickness**: `{thickness_cm:.2f} cm`")
                        st.write(f"⚠️ **Microfracture Risk**: {'High' if risk == 1 else 'Low'}")
                        st.write(f"📅 **Estimated Remaining Lifespan**: `{lifespan:.2f} years`")

                        data_list.append({
                            "Timestamp": timestamp,
                            "Voltage": voltage,
                            "Thickness (cm)": thickness_cm,
                            "Risk": 'High' if risk == 1 else 'Low',
                            "Lifespan (years)": lifespan
                        })

                        df = pd.DataFrame(data_list).tail(10)
                        placeholder.dataframe(df, use_container_width=True)

                        st.progress(min(voltage / 3.3, 1.0))

                time.sleep(0.1)
            except Exception as e:
                st.error(f"Error during monitoring: {e}")
                break
else:
    st.error("❌ Unable to load one or more model files. Please check your setup.")
