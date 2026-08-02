# NeuroGuard
### Explainable Stroke Risk Prediction System

<p align="center">

<img src="assets/logo.png" width="140"/>

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-Production-green.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)]()
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)]()
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-red.svg)]()
[![MLflow](https://img.shields.io/badge/Experiment-Tracking-blueviolet.svg)]()

</p>

---

## Overview

NeuroGuard is an end-to-end Machine Learning application that predicts stroke risk using clinical patient information while providing interpretable explanations through SHAP.

The project demonstrates a production-oriented ML workflow rather than simply training a machine learning model.

It includes:

- Data validation
- Feature engineering
- Multiple model benchmarking
- Class imbalance experiments
- Cross validation
- Threshold optimization
- Explainable AI (SHAP)
- REST API using FastAPI
- Docker deployment
- MLflow experiment tracking
- Interactive prediction dashboard

---

# Project Architecture

```
                     Patient Data
                           │
                           ▼
                  Data Validation
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                  Data Preprocessing
                           │
                           ▼
                 Model Training Pipeline
                           │
          ┌────────────────┴─────────────────┐
          ▼                                  ▼
    Model Evaluation                 SHAP Explainability
          │                                  │
          └──────────────┬───────────────────┘
                         ▼
                  Saved Artifacts
                         │
          ┌──────────────┴──────────────┐
          ▼                             ▼
     FastAPI API                 Consumer UI
          │                             │
          └──────────────┬──────────────┘
                         ▼
                  Real-time Predictions
```

---

# Features

## Machine Learning

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting

---

## Data Pipeline

- Dataset loading
- Missing value handling
- Train/Test split
- Standard Scaling
- One-Hot Encoding
- Feature persistence
- Artifact versioning

---

## Evaluation

Automatically computes

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- PR AUC
- Confusion Matrix
- Calibration Curve
- ROC Curve
- Precision Recall Curve

---

## Class Imbalance Experiments

Supports

- Baseline
- Random Oversampling
- SMOTE
- Borderline SMOTE
- ADASYN
- SMOTEENN
- SMOTETomek

---

## Threshold Optimization

Searches multiple probability thresholds and selects the best threshold using

- F1 Score
- Precision
- Recall
- Specificity
- Balanced Accuracy
- Youden Index

---

## Explainable AI

Implemented using SHAP.

Generated automatically:

- Global Feature Importance
- SHAP Summary Plot
- Beeswarm Plot
- Waterfall Plot
- Dependence Plots
- Local Prediction Explanation

---

## REST API

FastAPI endpoints

```
GET  /
GET  /health
POST /predict
```

Swagger documentation

```
http://localhost:8000/docs
```

---

## Docker Support

Run the complete application anywhere using Docker.

```
docker compose up --build
```

---

## MLflow Integration

Tracks

- Parameters
- Metrics
- Models
- Artifacts
- Experiments

Launch

```
mlflow ui
```

---

# Project Structure

```
NeuroGuard/

apps/
│
├── api/
│
ml/
│
├── data/
├── features/
├── models/
├── evaluation/
├── explainability/
├── inference/
├── visualization/
├── mlflow/
└── monitoring/

artifacts/

datasets/
models/
preprocessors/
metadata/
figures/
experiments/
explainability/

configs/

docker/

README.md
requirements.txt
Dockerfile
docker-compose.yml
```

---

# Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | 74.56% |
| Precision | 13.79% |
| Recall | 80.00% |
| F1 Score | 23.53% |
| ROC AUC | 84.37% |
| PR AUC | 26.85% |

> The dataset is highly imbalanced. Therefore ROC AUC, Recall, PR AUC and threshold tuning were prioritized over raw accuracy.

---

# Tech Stack

### Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- SciPy
- Imbalanced-Learn

### Explainability

- SHAP

### Backend

- FastAPI
- Uvicorn
- Pydantic

### Deployment

- Docker
- Docker Compose

### Experiment Tracking

- MLflow

### Visualization

- Matplotlib

---

# Installation

Clone repository

```
git clone https://github.com/yourusername/NeuroGuard.git

cd NeuroGuard
```

Create environment

```
python -m venv .venv
```

Windows

```
.venv\Scripts\activate
```

Linux

```
source .venv/bin/activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Run Training Pipeline

```
python -m ml.features.pipeline
```

The pipeline performs

- preprocessing
- model training
- evaluation
- visualization
- explainability
- artifact generation

---

# Run API

```
uvicorn apps.api.main:app --reload
```

Open

```
http://localhost:8000/docs
```

---

# Run Docker

Build

```
docker compose build
```

Run

```
docker compose up
```

---

# Example Prediction

Request

```json
{
  "gender":"Male",
  "age":67,
  "hypertension":1,
  "heart_disease":1,
  "ever_married":"Yes",
  "work_type":"Private",
  "Residence_type":"Urban",
  "avg_glucose_level":228.69,
  "bmi":36.6,
  "smoking_status":"formerly smoked"
}
```

Response

```json
{
    "prediction":1,
    "probability":0.896,
    "risk":"Very High",
    "model":"logistic_regression",
    "version":"v1.0"
}
```

---

# Generated Artifacts

```
artifacts/

datasets/
models/
preprocessors/
metadata/
figures/
experiments/
explainability/
```

Artifacts include

- trained models
- preprocessing pipeline
- threshold configuration
- evaluation metrics
- ROC curve
- confusion matrix
- SHAP plots
- feature importance
- calibration curve

---

# Future Improvements

- Continuous model retraining
- Data drift detection
- Prediction drift monitoring
- Authentication and authorization
- Database-backed prediction logging
- User feedback collection
- CI/CD pipeline
- Cloud deployment (AWS/GCP/Azure)
- Kubernetes deployment
- Model registry

---

# Disclaimer

NeuroGuard is intended for educational and research purposes only.

Predictions generated by this application must **not** be used as a substitute for professional medical diagnosis or treatment decisions.

---

# Author

**Garvit Joshi**

Machine Learning Engineer

GitHub:
https://github.com/Garvitjoshi1

LinkedIn:
https://linkedin.com/in/your-linkedin

Portfolio:
https://yourportfolio.com

---

If you found this project useful, consider giving it a star.