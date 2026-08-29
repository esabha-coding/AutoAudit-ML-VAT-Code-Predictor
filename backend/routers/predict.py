from fastapi import APIRouter, HTTPException
from backend.models.schemas import TransactionRequest, BatchTransactionRequest, PredictionResponse, BatchPredictionResponse
from backend.models.predictor import predictor_service

router = APIRouter(prefix="/api/v1", tags=["VAT Prediction"])

@router.post("/predict", response_model=PredictionResponse)
def predict_vat(payload: TransactionRequest):
    if not payload.description.strip():
        raise HTTPException(status_code=400, detail="Description cannot be empty.")
    return predictor_service.predict_single(payload.description)

@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_vat_batch(payload: BatchTransactionRequest):
    if not payload.transactions:
        raise HTTPException(status_code=400, detail="Transaction list cannot be empty.")
    results = predictor_service.predict_batch(payload.transactions)
    return {"results": results}