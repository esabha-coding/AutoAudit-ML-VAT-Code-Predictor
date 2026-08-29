from pydantic import BaseModel
from typing import List, Optional, Dict

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