# 🌾 KsetrikahGPT  

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white) 
![Google Cloud](https://img.shields.io/badge/Google_Cloud-Run-4285F4?style=for-the-badge&logo=google-cloud&logoColor=white) 
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white) 
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)


### An End-to-End, Automated MLOps System for Agricultural Market Forecasting & AI Agronomy

KsetrikahGPT is a **production-oriented machine learning system** designed to support data-driven agricultural decision-making.  
The project combines **classical machine learning**, **multimodal AI**, and a **fully automated retraining pipeline** to ensure market price predictions remain accurate as new government data becomes available.

This repository focuses on the **ML and MLOps backbone** of the system — data ingestion, feature engineering, model training, experiment tracking, and automated retraining.

---

## 🎯 Project Goals

- Build a **reliable next-day commodity price prediction system**
- Automate **daily data ingestion and retraining**
- Track experiments and model versions systematically
- Maintain **reproducibility and explainability**
- Demonstrate **real-world MLOps practices**, not just model training

---

## 🧠 Design Philosophy

This project intentionally prioritizes:

- ✅ **Classical ML over deep learning** for tabular market data  
- ✅ **Automation over manual retraining**
- ✅ **Clarity and explainability over overengineering**

### Why no deep learning for price prediction?
Market price data is **structured, tabular, and noisy**.  
Gradient boosting models (LightGBM) are better suited than neural networks for this data type and retrain significantly faster in daily pipelines.

---

## 🧰 Technology Stack (Open-Source Core)

### Machine Learning & Data
- **Python 3.10**
- **LightGBM (via scikit-learn API)** – price prediction
- **Pandas / NumPy** – data processing
- **Joblib** – model persistence
- **MLflow** – experiment tracking

### Automation & MLOps
- **GitHub Actions** – CI/CD & scheduled retraining
- **Docker** – reproducible runtime
- **Cron-based workflows** – daily data refresh

### Multimodal AI (Agronomy Module)
- **Whisper** – speech-to-text
- **Qwen-VL** – vision + language reasoning
- **ElevenLabs** – text-to-speech

> 🔹 Note: Deep learning is used **only** for the AI Agronomist module, not for market price prediction.

---

## 📁 Project Structure

```text
KsetrikahGPT/
│
├── .github/workflows/
│   ├── daily_update.yml        # Daily data fetch + retraining pipeline
│   └── deploy.yml              # CI/CD deployment workflow
│
├── data/
│   ├── raw/                    # Immutable daily government market data
│   │   ├── agmarknet_2026-02-01.csv
│   │   └── agmarknet_2026-02-03.csv
│   │
│   ├── processed/              # Cleaned and standardized dataset
│   │   └── clean_data.csv
│   │
│   └── features/               # ML-ready feature table
│       └── training_data.csv
│
├── mlruns/                     # MLflow experiment tracking artifacts
│
├── models/                     # Trained model artifacts
│   ├── best_model.joblib
│   ├── encoders.joblib
│   └── base_model.joblib
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   └── 02_feature_engineering.ipynb
│
├── src/
│   ├── api/                    # Application entry (ML inference usage)
│   │   └── app.py
│   │
│   ├── config/
│   │   └── config.yaml         # Centralized configuration
│   │
│   ├── data/
│   │   ├── fetch_data.py       # Government data ingestion
│   │   └── preprocess.py       # Data cleaning & normalization
│   │
│   ├── features/
│   │   └── build_features.py   # Feature engineering logic
│   │
│   ├── models/
│   │   └── train.py            # LightGBM training script
│   │
│   ├── pipeline/
│   │   └── run_pipeline.py     # End-to-end automated pipeline
│   │
│   └── utils/
│       └── __init__.py
│
├── Dockerfile
├── requirements.txt
└── README.md
````

---

## 🔄 End-to-End ML Pipeline

```
    data/raw/        ← daily CSVs (Agmarknet)
    data/processed/  ← cleaned data
    data/features/   ← training_data.csv
    
    src/data/
      ├─ fetch_data.py
      ├─ preprocess.py
    
    src/features/
      └─ build_features.py
    
    src/models/
      └─ train.py   (LightGBM, sklearn)
    
    src/pipeline/
      └─ run_pipeline.py  ← orchestrator
    
    mlruns/          ← MLflow logs
    models/          ← saved .joblib models


```

---

## Project Picture Output
<img width="1875" height="666" alt="image" src="https://github.com/user-attachments/assets/c4c2e85f-a720-42af-a3c9-cd20521c4fb3" />

<img width="788" height="701" alt="image" src="https://github.com/user-attachments/assets/62065a32-72bf-4edf-b727-d65010ba91a2" />



## 📊 Market Price Prediction (Core ML Module)

### Objective

Predict **next-day mandi prices** for agricultural commodities.

### Input Signals

* Recent prices (yesterday, 2 days ago)
* Market arrivals (supply indicator)
* Commodity & market identifiers
* Temporal features (day/week context)

### Model

* **LightGBM Regressor**
* Optimized for tabular, structured data
* Fast retraining for daily updates

### Evaluation

* **SMAPE / MAE**
* Rolling retraining ensures robustness against drift

---

## 🤖 AI Agronomist (Multimodal Intelligence)

This module provides **expert-like crop diagnosis** using:

1. Crop image analysis (vision)
2. Farmer queries via voice or text
3. Multilingual natural language responses

While this repository focuses on ML infrastructure, the AI Agronomist is integrated as a downstream consumer of the system.

---

## ⚙️ Automated Retraining & CI/CD

### Daily Automation

* GitHub Actions runs on a **24-hour cron**
* Fetches fresh government data
* Retrains the ML model
* Commits updated model artifacts
* Triggers redeployment automatically

### Why this matters

Most ML models fail in production due to **data drift**.
KsetrikahGPT addresses this by design.

---

## 🧪 Experiment Tracking (MLflow)

All training runs are logged with:

* Model parameters
* Evaluation metrics
* Artifacts
* Version history

This ensures:

* Reproducibility
* Comparability
* Safe rollbacks

---

## 🚀 Deployment

* Dockerized application
* Consistent runtime across environments
* Safe build-time checks ensure models exist before deployment

---
## ☁️ Google Cloud Deployment Guide

This project features a fully automated CI/CD pipeline that deploys the application to **Google Cloud Run** whenever changes are pushed to the main branch.

### 1. GCP Setup
1. **Create a Google Cloud Project:** Note your `PROJECT_ID`.
2. **Enable APIs:**
   - Cloud Run Admin API
   - Cloud Build API
   - Artifact Registry API
3. **Create Artifact Registry:**
   - Go to **Artifact Registry** -> Create Repository.
   - **Name:** `brain-tumor-classification` (or your preferred name)
   - **Format:** Docker
   - **Region:** `us-central1` (or your preferred region).

### 2. Service Account Setup
Create a Service Account to verify identity from GitHub Actions:

```bash
# Create Service Account
gcloud iam service-accounts create github-deploy-sa --display-name="GitHub Actions Deployer"

# Grant Permissions (Cloud Run Admin, Storage Admin, Service Account User, Artifact Registry Writer)
gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:github-deploy-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:github-deploy-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

gcloud projects add-iam-policy-binding <PROJECT_ID> \
    --member="serviceAccount:github-deploy-sa@<PROJECT_ID>.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"
```

### 3. GitHub Secrets Configuration
Go to your **GitHub Repository** -> **Settings** -> **Secrets and Variables** -> **Actions** -> **New Repository Secret**.

| Secret Name | Value |
|-------------|-------|
| `GCP_SA_KEY` | The JSON Key content of the Service Account created above. |

### 4. Handling Large Models
Since our trained model (`trained_model.h5`) is ~57MB (or larger), it usually fits within GitHub's file limits. We force-add it to the repository to ensure it's available in the Docker container if it's not generated during build:

```bash
git add -f artifacts/training/trained_model.h5
git commit -m "Add model file"
git push origin main
```

**For Larger Models (>100MB):**
1. Upload the model to **Google Cloud Storage (GCS)** manually.
2. Update `app.py` to download the model from GCS on startup using the `google-cloud-storage` library.
3. Grant **Storage Object Viewer** role to your Cloud Run service account.

### 5. Deployment
Push your code to the `main` branch. The GitHub Action in `.github/workflows/main.yaml` will:
1. Authenticate with Google Cloud.
2. Build the Docker image (installing system dependencies like `libgl1` for OpenCV).
3. Push the image to Google Artifact Registry.
4. Deploy the service to Cloud Run with optimized memory (2Gi) and timeout (300s) settings.

---

## 🎓 Key Takeaways

* Built a **real ML system**, not just a model
* Applied **production MLOps practices**
* Learned tradeoffs between ML and DL
* Designed for automation, explainability, and scale

---

## 📄 License

This project is released for educational and research purposes.

---

## 🙌 Final Note

KsetrikahGPT demonstrates how **practical machine learning systems** are built in the real world — with automation, monitoring, and clear engineering boundaries.

This repository intentionally focuses on **depth over hype**.

```

---

If you want next, I can:
- 🔥 Rewrite this into **resume bullets**
- 🎤 Create **interview explanation scripts**
- 📊 Draw a **system architecture diagram**
- 📈 Add a **“Results & Metrics” section**

Just tell me what you want next.
```
