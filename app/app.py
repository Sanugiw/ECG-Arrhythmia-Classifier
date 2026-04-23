import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.signal import butter, filtfilt, find_peaks
import shap

# Suppress TF deprecation warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# --------------------------------------------------
# Config & Clinical Constants
# --------------------------------------------------
MODEL_PATH    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "cnn_ecg_best.keras")
INPUT_LENGTH  = 216
SAMPLING_RATE = 360
PRE_SAMPLES   = int(0.2 * SAMPLING_RATE)   # 72 samples before R-peak
POST_SAMPLES  = INPUT_LENGTH - PRE_SAMPLES  # 144 samples after R-peak
CLASS_NAMES   = ["F", "N", "Q", "S", "V"]
LABEL_FULL    = {
    "N": "Normal Beat",
    "S": "Supraventricular Beat",
    "V": "Ventricular Beat",
    "F": "Fusion Beat",
    "Q": "Unknown / Paced Beat",
}
RISK_CLASS    = {"V", "S", "F"}
CLASS_COLOR   = {
    "N": "#4CAF50",
    "S": "#FF9800",
    "V": "#F44336",
    "F": "#9C27B0",
    "Q": "#2196F3",
}

# --------------------------------------------------
# Load Model & Explainer
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model      = load_model(MODEL_PATH, compile=False)
    background = np.zeros((1, INPUT_LENGTH, 1))
    explainer  = shap.GradientExplainer(model, background)
    return model, explainer

model, explainer = load_resources()

# --------------------------------------------------
# Signal Processing
# --------------------------------------------------
def bandpass_filter(signal: np.ndarray, fs: int = SAMPLING_RATE,
                    lowcut: float = 0.5, highcut: float = 40.0) -> np.ndarray:
    """Butterworth bandpass filter (0.5–40 Hz), matching training preprocessing."""
    nyq  = fs / 2
    b, a = butter(4, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)

def detect_rpeaks(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Robust R-peak detector implementing the core Pan-Tompkins steps.
    Assumes the signal is already bandpass filtered (Step 1).
    """
    # Step 2: Derivative (Highlights steep QRS slopes)
    derivative = np.gradient(signal)
    
    # Step 3: Squaring (Amplifies QRS, suppresses P/T waves)
    squared = derivative ** 2
    
    # Step 4: Moving Average Integration (Creates smooth 'lumps' over QRS)
    # Window size is typically 150ms
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Step 5: Adaptive Thresholding & Peak Finding
    # We find peaks on the smooth 'integrated' signal
    min_distance = int(0.3 * fs)  # minimum 300 ms between beats
    # Dynamic threshold: mean + 0.5 * std of the integrated signal
    threshold = np.mean(integrated) + 0.5 * np.std(integrated)
    
    peaks, _ = find_peaks(integrated, distance=min_distance, height=threshold)
    
    # Note: 'peaks' currently point to the center of the integrated lump.
    # To find the exact peak on the original filtered signal, we search a small
    # window around each detected lump.
    exact_peaks = []
    search_radius = int(0.05 * fs) # 50ms search radius
    for p in peaks:
        start = max(0, p - search_radius)
        end = min(len(signal), p + search_radius)
        # Find the max index within this local window
        exact_peak = start + np.argmax(signal[start:end])
        exact_peaks.append(exact_peak)
        
    return np.array(exact_peaks)

def segment_beats(signal: np.ndarray, r_peaks: np.ndarray) -> tuple:
    """
    Slice fixed-length windows around each R-peak.
    Returns:
        beats      — list of (INPUT_LENGTH,) arrays
        valid_peaks — R-peak indices that produced a full window
    """
    beats, valid_peaks = [], []
    for peak in r_peaks:
        start = peak - PRE_SAMPLES
        end   = peak + POST_SAMPLES
        if start < 0 or end > len(signal):
            continue                       # skip incomplete edge beats
        beat = signal[start:end].astype(np.float32)
        # Z-score normalise each beat independently (matches training)
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
        beats.append(beat)
        valid_peaks.append(peak)
    return beats, np.array(valid_peaks)

def classify_beats(beats: list) -> tuple:
    """
    Run model inference on a batch of beats.
    Returns:
        labels   — predicted class string per beat  e.g. ["N", "V", "N"]
        probs    — softmax probability array (n_beats, n_classes)
        conf     — confidence (max prob) per beat
    """
    if not beats:
        return [], np.array([]), np.array([])
    batch  = np.stack(beats)[:, :, np.newaxis]   # (n_beats, 216, 1)
    probs  = model.predict(batch, verbose=0)      # (n_beats, 5)
    idxs   = np.argmax(probs, axis=1)
    labels = [CLASS_NAMES[i] for i in idxs]
    conf   = probs[np.arange(len(probs)), idxs]
    return labels, probs, conf

def majority_vote(labels: list, probs: np.ndarray) -> tuple:
    """
    Aggregate per-beat predictions into a single signal-level decision.
    Confidence = mean softmax probability of the winning class across all beats.
    Returns: (winning_label, confidence, vote_counts_dict, mean_probs)
    """
    from collections import Counter
    counts     = Counter(labels)
    winner     = counts.most_common(1)[0][0]
    winner_idx = CLASS_NAMES.index(winner)
    confidence = float(np.mean(probs[:, winner_idx]))
    mean_probs = np.mean(probs, axis=0)
    return winner, confidence, dict(counts), mean_probs

def get_shap_attribution(beat: np.ndarray, pred_idx: int) -> np.ndarray:
    """
    SHAP attribution for a single beat.
    shap_values shape: (1, 216, 1, 5) — class axis is last.
    """
    inp        = beat.reshape(1, INPUT_LENGTH, 1)
    shap_vals  = explainer.shap_values(inp)        # (1, 216, 1, 5)
    return shap_vals[0, :, 0, pred_idx]            # (216,)

# --------------------------------------------------
# UI Helpers
# --------------------------------------------------
def beat_color_sequence(labels):
    """Map each beat label to its display colour."""
    return [CLASS_COLOR[l] for l in labels]

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="CardioExplain AI", layout="wide", page_icon="🫀")

# Sidebar
st.sidebar.title("🫀 CardioExplain AI")
uploaded_file = st.sidebar.file_uploader("Upload Patient ECG (CSV)", type=["csv"])
use_shap      = st.sidebar.checkbox("Compute SHAP Explanations", value=False)
st.sidebar.markdown("---")
st.sidebar.info(
    "Upload a raw ECG signal CSV. The app will automatically detect beats, "
    "classify each one, and aggregate a signal-level diagnosis via majority voting."
)

# --------------------------------------------------
# Main Dashboard
# --------------------------------------------------
st.title("Explainable ECG Arrhythmia Dashboard")

if uploaded_file is not None:
    try:
        data         = pd.read_csv(uploaded_file)
        selected_col = st.sidebar.selectbox("Select Signal Column", data.columns)
        raw_signal   = data[selected_col].dropna().values.astype(np.float32)

        # ── Step 1: Filter ────────────────────────────────────────────────
        filtered_signal = bandpass_filter(raw_signal)

        # ── Step 2: R-peak Detection ──────────────────────────────────────
        r_peaks = detect_rpeaks(filtered_signal)

        if len(r_peaks) == 0:
            st.error(
                "No R-peaks detected. Check that your CSV contains a valid single-lead ECG "
                "signal sampled at 360 Hz, or try a different column."
            )
            st.stop()

        # ── Step 3: Segmentation ──────────────────────────────────────────
        beats, valid_peaks = segment_beats(filtered_signal, r_peaks)

        if len(beats) == 0:
            st.error("No complete beats could be extracted. Signal may be too short.")
            st.stop()

        # ── Step 4: Beat Classification ───────────────────────────────────
        beat_labels, beat_probs, beat_conf = classify_beats(beats)

        # ── Step 5: Majority Voting ───────────────────────────────────────
        winner, win_conf, vote_counts, mean_probs = majority_vote(beat_labels, beat_probs)

        # ── Top Metrics ───────────────────────────────────────────────────
        avg_rr_samples = np.mean(np.diff(valid_peaks)) if len(valid_peaks) > 1 else INPUT_LENGTH
        bpm            = int(60 / (avg_rr_samples / SAMPLING_RATE))

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Signal Diagnosis",  LABEL_FULL[winner])
        m2.metric("Voting Confidence", f"{win_conf:.1%}")
        m3.metric("Beats Detected",    len(beats))
        m4.metric("Avg Heart Rate",    f"{bpm} BPM")
        m5.metric("Risk Level",        "🔴 High" if winner in RISK_CLASS else "🟢 Low")

        st.markdown("---")

        # ── Full Signal Waveform with Beat Markers ────────────────────────
        st.subheader("📈 Full Signal with Beat Annotations")
        fig_signal = go.Figure()

        # Raw waveform
        fig_signal.add_trace(go.Scatter(
            y=filtered_signal,
            mode="lines",
            name="Filtered ECG",
            line=dict(color="#00ffcc", width=1.2),
        ))

        # Scatter one marker per beat, coloured by class
        for peak, label in zip(valid_peaks, beat_labels):
            fig_signal.add_trace(go.Scatter(
                x=[peak],
                y=[filtered_signal[peak]],
                mode="markers+text",
                marker=dict(color=CLASS_COLOR[label], size=10, symbol="triangle-down"),
                text=[label],
                textposition="top center",
                textfont=dict(size=9),
                showlegend=False,
            ))

        # Legend entries (one per class present)
        for cls in sorted(set(beat_labels)):
            fig_signal.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color=CLASS_COLOR[cls], size=10),
                name=f"{cls} — {LABEL_FULL[cls]}",
            ))

        fig_signal.update_layout(
            template="plotly_dark",
            height=420,
            xaxis_title="Sample Index",
            yaxis_title="Amplitude (normalised)",
            margin=dict(l=40, r=20, t=30, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_signal, use_container_width=True)

        st.markdown("---")

        # ── Per-Beat Table + Vote Summary ─────────────────────────────────
        col_table, col_vote = st.columns([2, 1])

        with col_table:
            st.subheader("🔍 Per-Beat Classification")
            beat_df = pd.DataFrame({
                "Beat #":     range(1, len(beats) + 1),
                "R-peak (sample)": valid_peaks,
                "Prediction": beat_labels,
                "Description": [LABEL_FULL[l] for l in beat_labels],
                "Confidence": [f"{c:.1%}" for c in beat_conf],
            })
            # Colour the Prediction column
            def highlight_pred(row):
                color = CLASS_COLOR.get(row["Prediction"], "#888")
                return [""] * 2 + [f"color: {color}; font-weight: bold"] + [""] * 2

            st.dataframe(
                beat_df.style.apply(highlight_pred, axis=1),
                use_container_width=True,
                height=280,
            )

        with col_vote:
            st.subheader("🗳️ Majority Vote")
            vote_fig = go.Figure(go.Bar(
                x=list(vote_counts.keys()),
                y=list(vote_counts.values()),
                marker_color=[CLASS_COLOR[c] for c in vote_counts.keys()],
                text=list(vote_counts.values()),
                textposition="outside",
            ))
            vote_fig.update_layout(
                template="plotly_dark",
                height=300,
                xaxis_title="Class",
                yaxis_title="Beat Count",
                margin=dict(l=30, r=20, t=20, b=40),
            )
            st.plotly_chart(vote_fig, use_container_width=True)
            st.success(
                f"**Winner:** {winner} — {LABEL_FULL[winner]}  \n"
                f"**Confidence:** {win_conf:.1%}  \n"
                f"**Votes:** {vote_counts[winner]} / {len(beats)} beats"
            )

        st.markdown("---")

        # ── Mean Probability Distribution ─────────────────────────────────
        col_prob, col_shap = st.columns(2)

        with col_prob:
            st.subheader("📊 Mean Probability Distribution")
            prob_fig = go.Figure(go.Bar(
                x=[LABEL_FULL[c] for c in CLASS_NAMES],
                y=mean_probs.tolist(),
                marker_color=['#00ffcc' if c == winner else '#555' for c in CLASS_NAMES],
                text=[f"{p:.1%}" for p in mean_probs],
                textposition="outside",
            ))
            prob_fig.update_layout(
                template="plotly_dark",
                height=320,
                yaxis_range=[0, 1],
                xaxis_title="Class",
                yaxis_title="Mean Softmax Probability",
                margin=dict(l=40, r=20, t=20, b=80),
            )
            st.plotly_chart(prob_fig, use_container_width=True)

        with col_shap:
            if use_shap:
                st.subheader("🧠 SHAP — Representative Beat")
                # Use the first beat matching the winning class for SHAP
                winner_beat_idx = next(
                    (i for i, l in enumerate(beat_labels) if l == winner), 0
                )
                with st.spinner("Computing SHAP attributions…"):
                    sv = get_shap_attribution(beats[winner_beat_idx], CLASS_NAMES.index(winner))

                shap_colors = ['crimson' if v >= 0 else 'steelblue' for v in sv]
                shap_fig = go.Figure(go.Bar(
                    x=list(range(len(sv))),
                    y=sv.tolist(),
                    marker_color=shap_colors,
                    name="SHAP Attribution",
                ))
                shap_fig.update_layout(
                    template="plotly_dark",
                    height=320,
                    title=f"Attribution for '{winner}' beat (red = towards, blue = against)",
                    xaxis_title="Sample Index",
                    yaxis_title="SHAP Value",
                    margin=dict(l=40, r=20, t=50, b=40),
                )
                st.plotly_chart(shap_fig, use_container_width=True)
            else:
                st.info("Enable **SHAP Explanations** in the sidebar to see sample-level "
                        "feature attribution for the representative beat.")

        st.markdown("---")

        # ── Automated Clinical Summary ────────────────────────────────────
        st.subheader("📋 Automated Clinical Summary")

        # Composition breakdown
        composition = ", ".join(
            f"{v} × {k} ({LABEL_FULL[k]})" for k, v in sorted(vote_counts.items())
        )
        st.markdown(
            f"**Signal composition:** {len(beats)} beats detected — {composition}."
        )

        # Dominant class alert
        if winner == "V":
            st.error(
                f"🔴 **Clinical Alert — Ventricular Ectopy Dominant**  \n"
                f"The majority of beats ({vote_counts[winner]}/{len(beats)}) show morphology "
                f"consistent with **Ventricular Beats**: absent P-wave, widened QRS complex.  \n"
                f"Mean confidence {win_conf:.1%}. **Immediate clinical review recommended.**"
            )
        elif winner == "S":
            st.warning(
                f"🟠 **Clinical Alert — Supraventricular Ectopy Dominant**  \n"
                f"{vote_counts[winner]}/{len(beats)} beats classified as **Supraventricular**.  \n"
                f"Originates above the ventricles; correlate with patient history and 12-lead ECG."
            )
        elif winner == "F":
            st.warning(
                f"🟠 **Observation — Fusion Beats Dominant**  \n"
                f"{vote_counts[winner]}/{len(beats)} beats show **Fusion** morphology "
                f"(simultaneous normal + ectopic activation).  \n"
                f"Review SHAP map to identify dominant morphological driver."
            )
        elif winner == "N":
            st.success(
                f"🟢 **Status — Normal Sinus Rhythm**  \n"
                f"{vote_counts['N']}/{len(beats)} beats classified as Normal.  \n"
                f"Signal morphology is consistent with regular SA-node activation."
            )
        else:  # Q
            st.info(
                f"🔵 **Observation — Unknown / Paced Beats Dominant**  \n"
                f"{vote_counts[winner]}/{len(beats)} beats are unclassifiable or "
                f"pacemaker-induced. Manual review advised."
            )

        # Secondary warning if any high-risk beats exist in an otherwise normal signal
        if winner == "N":
            risky = {c: vote_counts[c] for c in RISK_CLASS if c in vote_counts}
            if risky:
                risky_str = ", ".join(f"{v} × {LABEL_FULL[k]}" for k, v in risky.items())
                st.warning(
                    f"⚠️ **Incidental finding:** Despite a Normal majority, "
                    f"the signal contains {risky_str}. Clinical correlation advised."
                )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.exception(e)

else:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/9/9e/ECG_Principle_fast.gif",
        caption="Waiting for signal input…"
    )
    st.info("Please upload a raw ECG CSV file (single signal column, 360 Hz) to begin analysis.")