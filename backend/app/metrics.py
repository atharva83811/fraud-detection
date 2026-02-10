import joblib
import os

METRICS_PATH = "model/metrics.joblib"

if not os.path.exists(METRICS_PATH):
    raise RuntimeError("Metrics file not found. Train the model first.")

metrics = joblib.load(METRICS_PATH)
