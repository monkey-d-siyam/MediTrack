<div align="center">
  <img src="https://via.placeholder.com/150x150?text=MediTrack+Logo" alt="MediTrack Logo">
  <h1>MediTrack</h1>
  <h3>AI-Powered Health Monitoring Web App</h3>
</div>

---

## 📖 Overview

**MediTrack** is an intelligent, full-stack healthcare application designed to bridge the gap between daily biometric tracking and predictive machine learning. Built for modern health monitoring, the system allows users to securely log their daily vitals and lifestyle metrics (Heart Rate, Sleep, Steps, and BMI) and leverages a trained supervised learning model to evaluate overall health risks.

This project was built to demonstrate the seamless integration of a **dynamic web backend (Django)** with **advanced data science pipelines (scikit-learn, Pandas)**. By turning raw metrics into actionable predictions and visual insights, it provides a comprehensive end-to-end architecture tailored for healthcare technology.

---

## ✨ Features

- **Daily Health Logging:** Clean interface for users to record biometric checkpoints effortlessly.
- **ML-Powered Risk Prediction:** Utilizes a custom-trained Gradient Boosting Classifier to assign risk levels (Low, Medium, High).
- **Interactive Dashboard Visualization:** Beautiful, responsive UI with historical tracking powered by Chart.js.
- **Model Insights & Explainability:** A dedicated developer/clinician view to explore the inner workings of the ML model, displaying evaluation metrics, confusion matrices, and feature importance.
- **Graceful Degradation:** Robust architecture that defaults back to rule-based scoring systems if the ML microservice fails to load.

---

## 🛠 Tech Stack

**Backend & Framework:**
- [Django](https://www.djangoproject.com/) (Python framework)
- SQLite (Local) / PostgreSQL (Production)
- Gunicorn & Whitenoise (Production serving)

**Machine Learning & Data Processing:**
- [Scikit-Learn](https://scikit-learn.org/) (Model training & evaluation)
- [Pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) (Data manipulation & imputation)

**Frontend & Visuals:**
- HTML5 / CSS3 / Vanilla JavaScript
- [Chart.js](https://www.chartjs.org/) (Dynamic visual analytics)

---

## 🧠 Machine Learning Engine

MediTrack’s predictive capability operates on a rigorous ML pipeline integrated directly into the Django backend.

### Dataset & Preprocessing
The model was trained on the **Pima Indians Diabetes Dataset**. Because the web app requires a specific biometric schema, we implemented intelligent **Feature Mapping**:
- **Heart Rate Proxy:** Systemically mapped from `BloodPressure` as a cardiovascular surrogate.
- **BMI:** Direct feature continuation.
- **Synthetic Behavioral Metrics:** Since the original dataset lacks lifestyle parameters, *Sleep Hours* and *Daily Steps* were synthetically generated using Gaussian distributions gently correlated with the clinical target outcomes.

### Risk Classification
The raw binary labels (`Outcome`) were expanded into a refined 3-tier classification system:
- **High Risk:** Verified clinical diabetic cases (Outcome 1).
- **Medium Risk:** Non-diabetic cases featuring borderline clinical indicators (e.g., Glucose > 105 or BMI > 30).
- **Low Risk:** Strictly healthy individual benchmarks.

### Evaluation Metrics
The pipeline utilizes a `GradientBoostingClassifier` evaluated via 5-Fold Stratified Cross-Validation. Key characteristics:
- High performance on non-linear tabular data without manual one-hot encoding.
- Extracts native, localized **Feature Importance** (Heart Rate & BMI generally stand as the highest globally weighted predictors).

> **Limitations:** The ML model maps distinct clinical markers to simpler lifestyle metrics to fit the UI input constraints. It is strictly an *academic approximation* to validate system architecture. Real-world implementation requires retraining on authenticated longitudinal wellness datasets.

---

## 📸 Screenshots

| Dashboard | Log Health | Model Insights |
| :---: | :---: | :---: |
| c:\Users\jubor\AppData\Local\Packages\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TempState\ScreenClip\{4D283F99-7C92-4916-96A9-78B498FA5769}.png | <img src="https://via.placeholder.com/350x200?text=Data+Entry" alt="Log Health View"> | <img src="https://via.placeholder.com/350x200?text=Insights+View" alt="Model Insights View"> |

*(Replace placeholders with actual application screenshots before publishing)*

---

## 🚀 Installation & Local Setup

Follow these steps to run MediTrack locally:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/MediTrack.git
cd MediTrack
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Model Training (Optional)
To regenerate the dataset and freshly construct the `.pkl` artifact from scratch:
```bash
python health/ml/train_model.py
```

### 4. Database Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Run the Application
```bash
python manage.py runserver
```
Visit `http://127.0.0.1:8000/` in your browser.

---

## ☁️ Production Deployment

MediTrack is configured for fully managed deployment on **Render**.

1. **Connect Repository:** Link your GitHub repo to a new **Web Service** on Render.
2. **Build Command:** 
   ```bash
   pip install -r requirements.txt && python manage.py collectstatic --noinput && python manage.py migrate
   ```
3. **Start Command:** 
   ```bash
   gunicorn meditrack.wsgi
   ```
4. **Environment Variables Required:**
   - `PYTHON_VERSION`: `3.12.3`
   - `DEBUG`: `False`
   - `SECRET_KEY`: `<your_production_key>`
   - `ALLOWED_HOSTS`: `<your_render_app_name>.onrender.com`
   - `DATABASE_URL`: PostgreSQL connection string (provisioned via Render).

---

## 🔮 Future Work

Our long-term architectural roadmap includes:
1. **Real Wearable Integrations:** Connect native Apple HealthKit and Google Fit APIs to pull continuous time-series data seamlessly.
2. **Federated Learning:** Train the predictive model on distributed, localized edge nodes (user devices) to enhance privacy without pooling raw data.
3. **Improved Datasets:** Expand from the Pima approximation to globally-sourced, anonymized multi-metric cardiovascular datasets.

---

## ⚠️ Disclaimer

**Educational Purposes Only.**  
MediTrack is an academic project designed to demonstrate the integration of machine learning within a full-stack web application framework. It is **not** a certified medical tool, nor is it intended to diagnose, treat, cure, or prevent any physical or mental condition. Always consult a qualified healthcare provider for professional medical advice.
