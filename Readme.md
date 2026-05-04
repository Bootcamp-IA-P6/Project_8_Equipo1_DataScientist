# 🧠 Stroke Prediction AI

### End-to-End Machine Learning & Deep Learning Project

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![DL](https://img.shields.io/badge/Deep%20Learning-TensorFlow-red)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-blueviolet)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

## 🚀 Overview

An **end-to-end AI system** to predict stroke risk using:

* 🧠 Structured clinical data (ML models)
* 🖼️ Brain imaging (CNNs)

Built with a **production-ready architecture**, including:

* Modular pipelines
* Experiment tracking
* CI/CD
* Docker deployment

---

## 🎯 Key Results

| Model               | AUC      | F1 Score | Recall   |
| ------------------- | -------- | -------- | -------- |
| Logistic Regression | 0.84     | 0.72     | 0.78     |
| Random Forest       | 0.86     | 0.75     | 0.80     |
| 🏆 XGBoost (Optuna) | **0.89** | **0.78** | **0.83** |

> ✅ Optimized for **recall**, critical in medical diagnosis.

---

## 🖼️ Model Insights

### 📊 Confusion Matrix (Best Model)

<p align="center">
  <img src="assets/v3/cm_XGBoost_optuna_full_smote=False.png" width="400"/>
</p>

### 🔥 CNN Interpretability (Grad-CAM)

<p align="center">
  <img src="assets/cnn/gradcam_grad-cam_—_stroke_cases.png" width="400"/>
</p>

---

## 🏗️ Architecture

```id="arch01"
Stroker_project/
├── app/                # API (deployment ready)
├── src/                # ML pipeline (modular)
├── cnn/                # Deep learning pipeline
├── models/             # Trained models
├── data/               # Datasets
├── notebooks/          # Experiments & EDA
├── assets/             # Visual results
├── test/               # Unit tests
├── mlruns/             # MLflow tracking
```

---

## ⚙️ Tech Stack

* **ML:** Scikit-learn, XGBoost, Optuna
* **DL:** TensorFlow / Keras (CNN, EfficientNet)
* **MLOps:** MLflow, Docker, GitHub Actions
* **Data:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 🧪 How to Run

```bash id="run01"
# Clone repo
git clone https://github.com/your-username/stroke-prediction-ai.git
cd stroke-prediction-ai

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python main.py

# Run API
python app/app.py

# Run tests
pytest
```

---

## 🐳 Docker

```bash id="docker01"
docker-compose up --build
```

---

## 📊 Experiment Tracking

Using **MLflow**:

* Track metrics (AUC, F1, Recall)
* Compare experiments
* Store models & artifacts

---

## 🔬 Features

✔ End-to-end ML pipeline
✔ Hyperparameter tuning (Optuna)
✔ Class imbalance handling (SMOTE)
✔ Model explainability (Grad-CAM)
✔ CI/CD integration
✔ Production-ready structure

---

## 📌 Future Improvements

* 🌐 Deploy API to cloud (AWS / GCP)
* 📊 Add real-time monitoring
* 🧠 Ensemble ML + CNN models
* 📱 Build frontend dashboard

---

## 👨‍💻 Author

**AI & Data Science Project**
Focused on real-world, production-ready machine learning systems.

---

## ⭐ If you like this project...

Give it a star ⭐ and feel free to contribute!
