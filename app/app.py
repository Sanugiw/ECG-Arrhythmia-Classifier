import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap

# Suppress TF deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --------------------------------------------------
# Config & Clinical Constants
# --------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cnn_ecg_best.keras")
INPUT_LENGTH = 216
SAMPLING_RATE = 360  # MIT-BIH standard frequency
CLASS_NAMES = ["F", "N", "Q", "S", "V"]
LABEL_FULL = {
    "N": "Normal Beat",
    "S": "Supraventricular Beat",
    "V": "Ventricular Beat",
    "F": "Fusion Beat",
    "Q": "Unknown / Paced Beat",
}

# --------------------------------------------------
# Load Model & Explainer
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    # Single input model — pass background as plain array, no list wrapping
    background = np.zeros((1, INPUT_LENGTH, 1))
    explainer = shap.GradientExplainer(model, background)
    return model, explainer

model, explainer = load_resources()

# --------------------------------------------------
# Logic Engines
# --------------------------------------------------
def calculate_bpm(signal_len, fs=SAMPLING_RATE):
    """Calculates instantaneous BPM based on the segmented window."""
    duration_sec = signal_len / fs
    bpm = 60 / duration_sec
    return int(bpm)

def preprocess_signal(signal: np.ndarray) -> np.ndarray:
    """Z-score normalise and reshape to (1, INPUT_LENGTH, 1)."""
    signal = np.asarray(signal, dtype=np.float32).flatten()
    if len(signal) < INPUT_LENGTH:
        signal = np.pad(signal, (0, INPUT_LENGTH - len(signal)), mode="constant")
    else:
        signal = signal[:INPUT_LENGTH]
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.reshape(1, INPUT_LENGTH, 1)

def get_shap_attribution(processed_input: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    Returns the SHAP attribution vector for the predicted class.
    shap_values shape from GradientExplainer: (1, 216, 1, n_classes)
    i.e. (batch, timesteps, channels, n_classes) — class axis is last.
    """
    shap_values = explainer.shap_values(processed_input)   # plain array, no list
    return shap_values[0, :, 0, pred_idx]

# --------------------------------------------------
# UI / UX Components
# --------------------------------------------------
st.set_page_config(page_title="CardioExplain AI", layout="wide", page_icon="🫀")

# Sidebar
st.sidebar.title("🫀 CardioExplain AI")
uploaded_file = st.sidebar.file_uploader("Upload Patient ECG (CSV)", type=["csv"])
use_shap = st.sidebar.checkbox("Compute SHAP Explanations", value=False)
st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses a 1D-CNN with SHAP to provide explainable arrhythmia detection. "
    "Upload a CSV file containing an ECG signal column to begin."
)

# Main Dashboard
st.title("Explainable ECG Arrhythmia Dashboard")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        selected_col = st.sidebar.selectbox("Select Signal Column", data.columns)
        raw_signal = data[selected_col].dropna().values.astype(np.float32)[:INPUT_LENGTH]

        # 1. Prediction Pipeline
        processed_input = preprocess_signal(raw_signal)
        preds = model.predict(processed_input, verbose=0)[0]  # shape: (n_classes,)
        pred_idx = int(np.argmax(preds))
        pred_label = CLASS_NAMES[pred_idx]
        conf = float(preds[pred_idx])

        # 2. Top Metrics (Clinical View)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Diagnosis", LABEL_FULL[pred_label])
        m2.metric("Confidence", f"{conf:.1%}")
        m3.metric("BPM (Inst.)", calculate_bpm(len(raw_signal)))
        m4.metric("Risk Level", "🔴 High" if pred_label in ["V", "S"] else "🟢 Low")

        # 3. Interactive Waveform
        st.subheader("Interactive Signal Analysis")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=raw_signal,
            mode='lines',
            name='Raw ECG',
            line=dict(color='#00ffcc', width=2)
        ))
        fig.update_layout(
            template="plotly_dark",
            height=400,
            xaxis_title="Sample Index",
            yaxis_title="Amplitude",
            margin=dict(l=40, r=20, t=30, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 4. Explainability Section
        col_a, col_b = st.columns(2)

        with col_a:
            st.write("### Probability Distribution")
            prob_df = pd.DataFrame({
                "Class": [LABEL_FULL[c] for c in CLASS_NAMES],
                "Probability": preds
            })
            prob_fig = go.Figure(go.Bar(
                x=prob_df["Class"],
                y=prob_df["Probability"],
                marker_color=['#00ffcc' if c == pred_label else '#888'
                              for c in CLASS_NAMES]
            ))
            prob_fig.update_layout(
                template="plotly_dark",
                height=300,
                yaxis_range=[0, 1],
                margin=dict(l=40, r=20, t=30, b=80)
            )
            st.plotly_chart(prob_fig, use_container_width=True)

        with col_b:
            if use_shap:
                st.write("### SHAP Feature Importance")
                with st.spinner("Computing SHAP attributions…"):
                    sv = get_shap_attribution(processed_input, pred_idx)

                colors = ['crimson' if v >= 0 else 'steelblue' for v in sv]
                shap_fig = go.Figure(go.Bar(
                    x=list(range(len(sv))),
                    y=sv.tolist(),
                    marker_color=colors,
                    name='SHAP Attribution'
                ))
                shap_fig.update_layout(
                    template="plotly_dark",
                    height=300,
                    title="Attribution per Sample (red = towards prediction, blue = against)",
                    xaxis_title="Sample Index",
                    yaxis_title="SHAP Value",
                    margin=dict(l=40, r=20, t=50, b=40)
                )
                st.plotly_chart(shap_fig, use_container_width=True)
            else:
                st.info("Enable SHAP in the sidebar for sample-level feature attribution.")

        # 5. Automated Clinical Summary
        st.markdown("---")
        st.subheader("📋 Automated Clinical Summary")
        if pred_label == "V":
            st.warning(
                f"**Clinical Alert:** Morphology consistent with a **Ventricular Beat** detected. "
                f"Characterised by absent P-wave and widened QRS complex. "
                f"High confidence ({conf:.1%}) — immediate clinical review recommended."
            )
        elif pred_label == "S":
            st.warning(
                f"**Clinical Alert:** **Supraventricular Beat** detected ({conf:.1%} confidence). "
                "Originates above the ventricles. Correlate with patient history."
            )
        elif pred_label == "N":
            st.success(
                "**Status:** Normal sinus rhythm detected. Signal morphology is consistent "
                "with regular SA-node activation."
            )
        elif pred_label == "F":
            st.info(
                f"**Observation:** **Fusion Beat** detected ({conf:.1%} confidence). "
                "Simultaneous normal and ectopic activation. Review SHAP map for dominant driver."
            )
        else:  # Q
            st.info(
                f"**Observation:** **Unknown / Paced Beat** detected ({conf:.1%} confidence). "
                "Signal may be unclassifiable or pacemaker-induced. Manual review advised."
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)  # shows full traceback in the UI for debugging

else:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/9/9e/ECG_Principle_fast.gif",
        caption="Waiting for signal input…"
    )
    st.info("Please upload a patient ECG record in CSV format to begin analysis.")