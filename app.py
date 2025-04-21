import streamlit as st # type: ignore
import joblib
import numpy as np
import os

# Debugging: Show current working directory and files
st.write("📁 Current directory:", os.getcwd())
st.write("📄 Files in directory:", os.listdir())

# Safe model loader
def load_model(path, name):
    try:
        st.write(f"🔄 Attempting to load `{name}` from `{path}`...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
        model = joblib.load(path)
        st.success(f"✅ {name} loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load {name}: {e}")
        return None

# Function to estimate lifespan based on thickness and voltage
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
        lifespan = 15  # default fallback

    voltage_impact = lifespan * (voltage / 3.3) * 0.05  # 5% impact per full voltage
    adjusted_lifespan = lifespan - voltage_impact
    return max(adjusted_lifespan, 0)

# File paths (make sure these files exist in the same folder as app.py)
clf_path = r'C:\Users\khush\Downloads\microfracture\fracture_detection_model.pkl'
scaler_path = r'C:\Users\khush\Downloads\microfracture\scaler.pkl'


# Load models
clf_model = load_model(clf_path, "Classifier Model")
scaler = load_model(scaler_path, "Scaler")

# Main app UI and logic
if clf_model and scaler:
    st.title("🔍 Microfracture Risk & Lifespan Estimator")

    voltage = st.slider("📟 Piezo Voltage (V)", 0.0, 3.3, 0.1)
    thickness_cm = st.number_input("🔍 Glass Thickness (cm)", min_value=0.1, max_value=7.5, value=0.4)
    thickness_mm = thickness_cm * 10  # Convert to mm

    input_data = np.array([[voltage, thickness_cm]])

    try:
        input_scaled = scaler.transform(input_data)
        risk = clf_model.predict(input_scaled)[0]
        lifespan_prediction = estimate_lifespan(thickness_mm, voltage)

        st.subheader("🔮 Prediction Results")
        st.write(f"📈 **Voltage**: `{voltage:.2f} V`")
        st.write(f"📏 **Glass Thickness**: `{thickness_cm:.2f} cm`")
        st.write(f"⚠️ **Microfracture Risk**: {'High' if risk == 1 else 'Low'}")
        st.write(f"📅 **Estimated Remaining Lifespan**: `{lifespan_prediction:.2f} years`")
        st.progress(min(voltage / 3.3, 1.0))

    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")
else:
    st.error("❌ Unable to load one or more model files. Please check your file paths and try again.")
