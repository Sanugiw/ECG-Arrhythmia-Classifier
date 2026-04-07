# 🫀 ECG Arrhythmia Classification with Explainable AI Dashboard

An end-to-end **ECG Arrhythmia Classification system** combining **deep learning** with **Explainable AI (Grad-CAM)** and an interactive **Streamlit dashboard**.

This system not only predicts arrhythmia classes but also **explains model decisions by highlighting important regions of the ECG signal**, improving transparency and clinical relevance.

---

## 🚀 Key Features

* Multi-class ECG classification using **AAMI-standard grouped classes**
* 1D CNN optimized for **temporal signal learning**
* **Grad-CAM visualization** for interpretability
* Interactive **Streamlit dashboard**
* Real-time ECG upload, prediction, and explanation
* Deployment-ready pipeline

---

## 🧠 Project Workflow

### 1. Data Preprocessing

* Dataset: MIT-BIH Arrhythmia Database
* ECG beats extracted using R-peak annotations
* Bandpass filtering applied (0.5–40 Hz) to remove noise
* Each beat segmented using:

  * **0.2 s before R-peak**
  * **0.4 s after R-peak**
* Resulting input size: **216 samples per beat**
* Signals normalized to zero mean and unit variance

---

### 2. Dataset Labeling

Raw MIT-BIH annotations (23 classes) are grouped into **5 clinically meaningful categories (AAMI standard)**:

| Class | Description                |
| ----- | -------------------------- |
| **N** | Normal beat                |
| **S** | Supraventricular           |
| **V** | Ventricular                |
| **F** | Fusion beat                |
| **Q** | Unknown / Paced / Artifact |

This reduces class sparsity and improves model generalization.

---

### 3. Model Architecture (1D CNN)

A deep learning model designed to learn **ECG morphology and temporal patterns**.

```
Conv1D → BatchNorm → MaxPooling  
Conv1D → BatchNorm → MaxPooling  
Conv1D → BatchNorm → GlobalAvgPooling  
Dense → Dropout → Softmax
```

#### Training Details

* Optimizer: **Adam**
* Learning rate: **1e-3**
* Loss: **Categorical Crossentropy**
* Batch size: **64**
* Epochs: **30 (Early Stopping applied)**
* Class imbalance handled using **class weights**
* Train/Validation/Test split: **80 / 10 / 10 (stratified)**

---

### 4. Explainability (Grad-CAM)

Grad-CAM computes gradients of the target class score with respect to feature maps in the final convolutional layer.
These gradients are used to generate a **temporal importance map**, highlighting regions that most influenced the prediction.

#### Key Interpretations

* **Normal (N):** Focus on sharp QRS complex
* **Ventricular (V):** Broad QRS morphology
* **Supraventricular (S):** Attention includes pre-QRS region (P-wave)
* **Fusion (F):** Mixed waveform attention
* **Unknown/Paced (Q):** Spike-like features

👉 The model learns **physiological patterns, not noise**

---

### 5. Streamlit Deployment

Users can:

* Upload ECG CSV files (single column)
* Visualize waveform
* View predicted class and confidence
* See Grad-CAM explanation overlay

---

## 📊 Results & Performance

### Overall Performance

| Metric                | Value      |
| --------------------- | ---------- |
| **Accuracy**          | **94.28%** |
| **Macro F1-score**    | 0.884      |
| **Weighted F1-score** | 0.943      |

---

### Class-wise Performance

| Class | Precision | Recall | F1-score | Support |
| ----- | --------- | ------ | -------- | ------- |
| **F** | 0.83      | 0.75   | 0.79     | 40      |
| **N** | 0.96      | 0.94   | 0.95     | 1000    |
| **Q** | 0.97      | 0.96   | 0.96     | 773     |
| **S** | 0.73      | 0.85   | 0.79     | 93      |
| **V** | 0.91      | 0.95   | 0.93     | 278     |

---

### 🔍 Key Observations

* High overall accuracy with strong generalization
* Excellent performance on **N** and **Q** classes
* Strong detection of **ventricular beats (V)**
* Slight confusion between **S and N** due to morphological similarity
* **Fusion (F)** remains challenging due to limited samples

---

### ⚖️ Class Imbalance Insight

* Dataset is inherently imbalanced
* Majority classes: **N, Q**
* Minority classes: **F, S**

Despite this, the model achieves:

* Stable macro performance (**F1 ≈ 0.88**)
* Good recall for minority classes

---

## 📈 Visualization

![Grad-CAM Visualization](images/gradcam_results.png)

---

## 📁 Repository Structure

```
├── model_training/
│   ├── train_model.py
│   ├── preprocessing.py
│   └── cnn_ecg_best.keras
│
├── explainability/
│   ├── grad_cam.py
│   └── visualization.py
│
├── app/
│   ├── app.py
│   └── sample_ecg.csv
│
├── images/
│   └── gradcam_results.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/ecg-arrhythmia-dashboard.git
cd ecg-arrhythmia-dashboard

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🏋️ Train the Model

```bash
python model_training/train_model.py
```

---

## ▶️ Run the App

```bash
streamlit run app/app.py
```

---

## 🧪 Usage

1. Upload ECG CSV (single column)
2. View:

   * ECG waveform
   * Predicted class
   * Confidence score
   * Grad-CAM explanation

---

## 🛠 Tech Stack

* Python
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib
* Streamlit

---

## 🔮 Future Improvements

* Multi-lead ECG support
* Transformer-based temporal models
* Attention mechanisms for improved interpretability
* Patient-wise data splitting to avoid leakage
* Real-time ECG streaming
* Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

---

## 📌 Summary

This project integrates:

* Deep learning for ECG classification
* Explainable AI (Grad-CAM)
* Interactive deployment

to create a system that is both **accurate and interpretable**, suitable for:

* clinical decision support
* biomedical research
* educational tools

---
