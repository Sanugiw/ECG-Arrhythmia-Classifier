import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_PATH = "models/cnn_ecg_best.keras"
CLASS_NAMES = ["F", "N", "Q", "S", "V"]   # replace with encoder.classes_.tolist() order if different
LABEL_FULL = {
    "N": "Normal Beat",
    "S": "Supraventricular Beat",
    "V": "Ventricular Beat",
    "F": "Fusion Beat",
    "Q": "Unknown / Paced Beat",
}
INPUT_LENGTH = 216

# --------------------------------------------------
# Load model
# --------------------------------------------------
@st.cache_resource
def load_ecg_model():
    return load_model(MODEL_PATH, compile=False)

model = load_ecg_model()

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def preprocess_signal(signal: np.ndarray, target_len: int = INPUT_LENGTH) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32).flatten()

    if len(signal) < target_len:
        signal = np.pad(signal, (0, target_len - len(signal)), mode="constant")
    elif len(signal) > target_len:
        signal = signal[:target_len]

    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.reshape(1, target_len, 1).astype(np.float32)


def get_last_conv_layer_name(model):
    return next(
        (
            layer.name
            for layer in reversed(model.layers)
            if isinstance(layer, tf.keras.layers.Conv1D)
        ),
        None,
    )


def grad_cam_1d(model, signal, class_index=None, layer_name=None):
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, np.newaxis]

    inp = signal[np.newaxis, ...]

    if layer_name is None:
        layer_name = get_last_conv_layer_name(model)

    target_layer = model.get_layer(layer_name)

    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=target_layer.output
    )

    classifier_input = tf.keras.Input(shape=target_layer.output.shape[1:])
    x = classifier_input

    passed_target = False
    for layer in model.layers:
        if layer.name == layer_name:
            passed_target = True
            continue
        if passed_target:
            x = layer(x, training=False)

    classifier_model = tf.keras.Model(classifier_input, x)

    inp_tf = tf.convert_to_tensor(inp, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_out = conv_model(inp_tf, training=False)
        tape.watch(conv_out)
        preds = classifier_model(conv_out, training=False)

        if class_index is None:
            class_index = tf.argmax(preds[0])

        class_score = preds[:, class_index]

    grads = tape.gradient(class_score, conv_out)
    if grads is None:
        return None

    weights = tf.reduce_mean(grads, axis=1)
    cam = tf.reduce_sum(conv_out[0] * weights[0], axis=-1)
    cam = tf.nn.relu(cam)

    heatmap = cam.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    if len(heatmap) != signal.shape[0]:
        x_old = np.linspace(0, 1, len(heatmap))
        x_new = np.linspace(0, 1, signal.shape[0])
        heatmap = np.interp(x_new, x_old, heatmap)

    return heatmap


def plot_ecg(signal, title="ECG Beat"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(signal, linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig)


def plot_gradcam_ecg(signal, heatmap, title="Grad-CAM on ECG Beat"):
    signal = np.squeeze(signal)
    t = np.arange(len(signal))
    norm = Normalize(vmin=0, vmax=1)

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    points = np.array([t, signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="jet", norm=norm, linewidth=2)
    lc.set_array(heatmap[:-1])

    axes[0].add_collection(lc)
    axes[0].set_xlim(t[0], t[-1])
    axes[0].set_ylim(signal.min() - 0.05, signal.max() + 0.05)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(title)
    axes[0].spines[["top", "right"]].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes[0], label="Grad-CAM score", shrink=0.8)

    axes[1].imshow(
        heatmap[np.newaxis, :],
        aspect="auto",
        cmap="jet",
        extent=[0, len(signal), 0, 1],
    )
    axes[1].set_yticks([])
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Importance Map")

    plt.tight_layout()
    st.pyplot(fig)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("🫀 ECG Beat Classifier")
st.sidebar.write("Upload a CSV containing one ECG beat or a single ECG column.")

uploaded_file = st.sidebar.file_uploader("Upload ECG CSV", type=["csv"])
show_gradcam = st.sidebar.checkbox("Show Grad-CAM", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "Model: 1D CNN\n"
    "Input length: 216 samples\n"
    "Classes: N, S, V, F, Q"
)

# --------------------------------------------------
# Main
# --------------------------------------------------
st.title("ECG Arrhythmia Classification Dashboard")
st.write(
    "This dashboard classifies a single ECG beat using a trained 1D CNN and optionally "
    "shows Grad-CAM to explain which regions influenced the prediction."
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        selected_col = st.sidebar.selectbox("Select ECG Column", data.columns, index=0)
        ecg_signal = data[selected_col].dropna().values.astype(np.float32)

        st.subheader("Uploaded ECG Signal")
        plot_ecg(ecg_signal[:INPUT_LENGTH], title="Uploaded ECG Segment")

        model_input = preprocess_signal(ecg_signal, target_len=INPUT_LENGTH)
        pred = model.predict(model_input, verbose=0)[0]

        pred_idx = int(np.argmax(pred))
        pred_label = CLASS_NAMES[pred_idx]
        pred_name = LABEL_FULL.get(pred_label, pred_label)
        conf = float(pred[pred_idx])

        st.subheader("Prediction Results")
        st.success(f"**Predicted Class:** {pred_label} — {pred_name}")
        st.write(f"**Confidence:** {conf:.2%}")

        st.subheader("Class Probabilities")
        probs_df = pd.DataFrame({
            "Class": [f"{c} — {LABEL_FULL.get(c, c)}" for c in CLASS_NAMES],
            "Probability": pred
        }).set_index("Class")
        st.bar_chart(probs_df)

        if show_gradcam:
            st.subheader("Grad-CAM Explanation")
            heatmap = grad_cam_1d(
                model,
                np.squeeze(model_input[0]),
                class_index=pred_idx,
                layer_name=get_last_conv_layer_name(model),
            )

            if heatmap is not None:
                plot_gradcam_ecg(
                    np.squeeze(model_input[0]),
                    heatmap,
                    title=f"Grad-CAM | Predicted: {pred_label} — {pred_name} ({conf:.1%})"
                )
            else:
                st.warning("Grad-CAM could not be generated for this sample.")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.warning("⬅️ Upload a CSV file from the sidebar to begin.")