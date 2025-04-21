import streamlit as st
import joblib
import numpy as np
import os
import serial
import pandas as pd
import time

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

# Function to estimate lifespan based on thickness and voltage
def estimate_lifespan(thickness_mm, voltage):
    lifespan = 0

    # Lifespan based on thickness (in mm)
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
        lifespan = 15  # default fallback

    # Voltage impact: reduce lifespan slightly based on voltage
    voltage_impact = lifespan * (voltage / 3.3) * 0.05  # 5% impact per full voltage
    adjusted_lifespan = lifespan - voltage_impact
    return max(adjusted_lifespan, 0)

# File paths
clf_path = 'fracture_detection_model.pkl'
scaler_path = 'scaler.pkl'
reg_path = 'lifespan_model.pkl'
# Load models
clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")

# ---------- Real-Time Sensor Data Integration ----------
# Define your serial port and baud rate (use correct COM port for your ESP32)
SERIAL_PORT = 'COM4'  # ⚠️ Change to the correct port (Windows: COM4, Linux: /dev/ttyUSB0)
BAUD_RATE = 115200

try:
    # Try to connect to the ESP32 device using pyserial
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait a little for ESP32 to establish connection
    st.success(f"Connected to {SERIAL_PORT}")
except Exception as e:
    st.error(f"Failed to connect to {SERIAL_PORT}: {e}")
    st.stop()  # Stop the app if serial connection fails

# Main app
if clf_model and scaler and ser:
    st.title("🔍 Microfracture Risk & Lifespan Estimator")

    # Create an empty placeholder for showing live data
    placeholder = st.empty()
    data_list = []

    # Add button to start monitoring
    if st.button("Start Monitoring"):
        while True:
            line = ser.readline().decode("utf-8").strip()  # Read the data line from the ESP32
            if line and not line.startswith("timestamp"):  # Ignore header
                parts = line.split(",")  # Split CSV format
                if len(parts) == 6:
                    # Extract individual sensor data
                    timestamp = int(parts[0])
                    voltage = float(parts[2])
                    thickness_cm = float(parts[3])  # Assume this is the thickness
                    thickness_mm = thickness_cm * 10  # Convert to mm

                    # Create input data array for model prediction
                    input_data = np.array([[voltage, thickness_cm]])

                    try:
                        input_scaled = scaler.transform(input_data)
                        risk = clf_model.predict(input_scaled)[0]
                        lifespan_prediction = estimate_lifespan(thickness_mm, voltage)

                        # Display the prediction results
                        st.subheader("🔮 Prediction Results")
                        st.write(f"📈 **Voltage**: {voltage:.2f} V")
                        st.write(f"📏 **Glass Thickness**: {thickness_cm:.2f} cm")
                        st.write(f"⚠️ **Microfracture Risk**: {'High' if risk == 1 else 'Low'}")
                        st.write(f"📅 **Estimated Remaining Lifespan**: {lifespan_prediction:.2f} years")

                        # Display data in table
                        row = {
                            "Timestamp": timestamp,
                            "Voltage": voltage,
                            "Thickness (cm)": thickness_cm,
                            "Risk": 'High' if risk == 1 else 'Low',
                            "Lifespan (years)": lifespan_prediction
                        }
                        data_list.append(row)
                        df = pd.DataFrame(data_list)
                        df = df.tail(10)  # Limit to latest 10 rows
                        placeholder.dataframe(df, use_container_width=True)

                        st.progress(min(voltage / 3.3, 1.0))

                    except Exception as e:
                        st.error(f"Error during prediction: {e}")

            time.sleep(0.1)  # Small delay between reads to avoid flooding the UI
else:
    st.error("❌ Unable to load one or more model files or connect to the ESP32. Please check your setup.") 
