import joblib
import numpy as np
import pandas as pd
import os

MODEL_PATH = "model/fraud_model_manual.joblib"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model file not found. Train the model first.")

model = joblib.load(MODEL_PATH)

weights = model["weights"]
mean = model["mean"]
std = model["std"]

FEATURE_ORDER = [
    "Time",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict_fraud(data: dict) -> float:
    # 1. Build feature vector
    X = pd.DataFrame([data], columns=FEATURE_ORDER).values

    # 2. Standardize using TRAIN statistics
    X_scaled = (X - mean) / std

    # 3. Add bias term
    X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]

    # 4. Predict probability
    prob = sigmoid(X_scaled @ weights)[0]

    return float(prob)
