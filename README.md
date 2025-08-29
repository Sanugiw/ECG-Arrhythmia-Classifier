# 🫀 ECG Arrhythmia Classification with Streamlit Dashboard

This project presents an **end-to-end ECG Arrhythmia Classification system**, combining **deep learning** with an interactive **Streamlit web dashboard**.  
A **1D Convolutional Neural Network (CNN)** is trained on ECG signals to detect abnormal rhythms, and users can interactively upload signals, visualize them, and view predictions in real time.  

---

## 🚀 Project Workflow

1. **Data Preprocessing**  
   - ECG signals (from MIT-BIH dataset or CSV files) are normalized.  
   - Noise reduction and segmentation are applied to ensure high-quality signals.  

2. **Model Training (1D CNN)**  
   - A **1D Convolutional Neural Network** is implemented in TensorFlow/Keras.  
   - Layers: Conv1D → BatchNorm → MaxPooling → Dropout → Dense → Softmax.  
   - Loss: `categorical_crossentropy`, Optimizer: `Adam`.  
   - Model trained on arrhythmia classes.  

3. **Model Saving**  
   - Trained model is stored as `cnn_ecg.h5` for deployment.  

4. **Streamlit Deployment**  
   - Users upload ECG CSV files.  
   - ECG waveform is plotted.  
   - The model predicts arrhythmia type and shows class probabilities.  
   - Provides an easy-to-use interface for clinical and educational purposes.  

---

## 🧪 Arrhythmia Classes

The model can classify the following rhythms:
- **Normal Sinus Rhythm (NSR)**  
- **Atrial Fibrillation (AFib)**  
- **Ventricular Tachycardia (VTach)**  
- **Premature Ventricular Contraction (PVC)**  

---

## 📂 Repository Structure

```

├── model\_training/
│   ├── train\_model.py       # Script to train the CNN model
│   ├── ecg\_dataset.csv      # Training data (or link to MIT-BIH)
│   └── cnn\_ecg.h5           # Saved trained model
│
├── app/
│   ├── app.py               # Streamlit application
│   └── sample\_ecg.csv       # Example ECG input
│
├── requirements.txt         # Python dependencies
└── README.md                # Documentation

````

---

## 📊 Model Architecture (1D CNN)

```python
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(input_length, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),

    Conv1D(filters=64, kernel_size=5, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_classes, activation='softmax')
])
````

* Optimizer: **Adam**
* Loss: **Categorical Crossentropy**
* Metrics: **Accuracy**

---

## ⚙️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ecg-arrhythmia-dashboard.git
   cd ecg-arrhythmia-dashboard
   ```

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Train the model (if you want to retrain):

   ```bash
   python model_training/train_model.py
   ```

5. Run the Streamlit app:

   ```bash
   streamlit run app/app.py
   ```

---

## 📊 Usage

* Upload your **ECG CSV file** in the sidebar (single column of ECG values).
* The app will:

  1. Plot the ECG waveform.
  2. Predict the arrhythmia type.
  3. Display class probabilities with a bar chart.

---

## 📈 Example Output

* **Input ECG Plot:**
  A waveform visualization of the uploaded ECG signal.

* **Prediction:**

  ```
  Predicted Class: Atrial Fibrillation (AFib)
  Confidence: 92.5%
  ```

* **Class Probabilities (Bar Chart):**

  * NSR: 5%
  * AFib: 92.5%
  * VTach: 1.5%
  * PVC: 1%

---

## 🧰 Tools & Technologies

* Python
* TensorFlow / Keras
* NumPy and Pandas
* Matplotlib
* Streamlit

---

## 🔮 Future Improvements

* 🔍 Integrate **Grad-CAM** for interpretability (highlighting important ECG regions).
* 📊 Support **multi-lead ECG inputs**.
* 🌐 Deploy app on **Streamlit Cloud / Hugging Face Spaces**.
* 📂 Extend arrhythmia coverage using **MIT-BIH Arrhythmia Database**.

---

Would you like me to also **write the `requirements.txt`** for you (TensorFlow, Streamlit, etc.) so anyone cloning your repo can run it instantly?
```
