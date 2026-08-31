from typing import Dict, List, Optional

from pydantic import BaseModel


class TransactionRequest(BaseModel):
    description: str
    amount: Optional[float] = None


class BatchTransactionRequest(BaseModel):
    transactions: List[str]


class PredictionResponse(BaseModel):
    description: str
    predicted_vat_code: str
    confidence: float
    probabilities: Dict[str, float]
    explanation: str


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
