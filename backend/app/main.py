from fastapi import FastAPI
from app.schemas import Transaction, PredictionResponse
from app.metrics import metrics
from app.model import predict_fraud
from app.logger import logger
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fraud Detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


from app.logger import logger


@app.post("/predict")
def predict(tx: Transaction):
    logger.info(f"Prediction requested | Amount={tx.Amount}")

    data = tx.dict()
    data["Time"] = 0.0
    for i in range(1, 29):
        data[f"V{i}"] = 0.0

    prob = predict_fraud(data)
    verdict = "FRAUD" if prob >= 0.25 else "LEGIT"

    logger.info(f"Prediction completed | Probability={prob:.4f} | Verdict={verdict}")

    return {"fraud_probability": round(prob, 4), "verdict": verdict}


@app.get("/metrics")
def get_metrics():
    return metrics
