from pydantic import BaseModel, Field


from pydantic import BaseModel


class Transaction(BaseModel):
    Amount: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    verdict: str
