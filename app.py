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
    return load_model("cnn_ecg.h5")


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
st.sidebar.title("ECG Dashboard")
st.sidebar.write("Upload an ECG signal file and classify arrhythmias.")
uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info("Model: CNN trained on ECG waveforms\nVisualization: Streamlit Dashboard")

# ---------------------------
# Main App Layout
# ---------------------------
st.title("ECG Arrhythmia Classification")
st.write("This dashboard visualizes ECG signals and predicts arrhythmia types using a deep learning model.")

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)

    # Assume ECG is in first column
    ecg_signal = data.iloc[:, 0].values

    # ---------------------------
    # ECG Plot
    # ---------------------------
    st.subheader("ECG Signal Visualization")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(ecg_signal[:1000], color="red")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Uploaded ECG Segment")
    st.pyplot(fig)

    # ---------------------------
    # Preprocess & Predict
    # ---------------------------
    signal = ecg_signal[:500]  # adjust window size as per training
    signal = signal.reshape(1, 500, 1)
    signal = signal.astype("float32") / np.max(np.abs(signal))

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

else:
    st.warning("Please upload an ECG CSV file from the sidebar to start.")
