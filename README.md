# Fraud Detection System

An end-to-end fraud detection system built using **logistic regression**, **FastAPI**, and **React**, demonstrating the full machine learning lifecycle from training to deployment.

## Overview
This project tackles a highly imbalanced fraud detection problem using a manually implemented logistic regression model. The system exposes predictions via a FastAPI backend and provides a simple React dashboard for interaction.

## Architecture
**Creation Flow**
```
Dataset → Model Training → Evaluation → API → Frontend
```

**Runtime Flow**
```
User Input → API Validation → Feature Normalization → Model Inference → Response
```

## Tech Stack
- **ML:** Python, NumPy, Pandas
- **Backend:** FastAPI
- **Frontend:** React, Tailwind CSS
- **Deployment:** Docker

## Machine Learning
- Logistic Regression implemented from scratch
- Binary cross-entropy loss with gradient descent
- Manual feature standardization
- Threshold tuned to **0.25** to improve fraud recall
- Evaluation focuses on precision, recall, and F1-score due to class imbalance

## API Endpoints
- `GET /health` – Health check  
- `POST /predict` – Returns fraud probability and verdict  
- `GET /metrics` – Offline model evaluation metrics  

## Dataset & Reproducibility
- Dataset and trained models are not committed
- Models are generated locally using `ml/train.py`

## Key Takeaways
- Demonstrates end-to-end ML system design
- Separates training, evaluation, and inference
- Emphasizes production-style architecture over toy examples
