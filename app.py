import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_ecg_model():
    return load_model("models/cnn_ecg_final.h5")

model = load_ecg_model()

# Define your class names
CLASS_NAMES = [
    "Normal Sinus Rhythm",
    "Atrial Fibrillation",
    "Ventricular Tachycardia",
    "Premature Ventricular Contraction"
]

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("⚡ ECG Dashboard")
st.sidebar.write("Upload an ECG signal file (CSV) to classify arrhythmias.")

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.info("🧠 Model: CNN trained on ECG waveforms\n📊 Visualization: Streamlit Dashboard")

# ---------------------------
# Main App Layout
# ---------------------------
st.title("ECG Arrhythmia Classification")
st.write("This dashboard visualizes ECG signals and predicts arrhythmia types using a deep learning model.")

if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_csv(uploaded_file)

        # Let user choose which column is the ECG
        selected_col = st.sidebar.selectbox("Select ECG Column", data.columns, index=0)
        ecg_signal = data[selected_col].dropna().values

        # ---------------------------
        # ECG Plot
        # ---------------------------
        st.subheader("ECG Signal Visualization")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(ecg_signal[:1000], color="red", linewidth=1)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Uploaded ECG Segment (first 1000 samples)")
        ax.grid(True, linestyle="--", alpha=0.6)
        st.pyplot(fig)

        # ---------------------------
        # Preprocess & Predict
        # ---------------------------
        signal = np.array(ecg_signal)

        # Pad / truncate
        if len(signal) < 500:
            signal = np.pad(signal, (0, 500 - len(signal)), mode='constant')
        elif len(signal) > 500:
            signal = signal[:500]

        # Normalize (z-score)
        signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
        signal = signal.reshape(1, 500, 1).astype("float32")

        # Predict
        prediction = model.predict(signal)
        pred_class = CLASS_NAMES[np.argmax(prediction)]

        # ---------------------------
        # Prediction Results
        # ---------------------------
        st.subheader("Prediction Results")
        st.success(f"**Predicted Class:** {pred_class}")

        st.subheader("Class Probabilities")
        probs = {CLASS_NAMES[i]: float(prediction[0][i]) for i in range(len(CLASS_NAMES))}
        st.bar_chart(probs)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.warning("⬅️ Please upload an ECG CSV file from the sidebar to start.")
