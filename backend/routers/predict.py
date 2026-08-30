from fastapi import APIRouter, HTTPException
from ..models.schemas import TransactionRequest, BatchTransactionRequest, PredictionResponse, BatchPredictionResponse
from ..models.predictor import predictor_service

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
def predict_single(request: TransactionRequest):
    try:
        result = predictor_service.predict_single(request.description)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchTransactionRequest):
    try:
        results = []
        for txn in request.transactions:
            result = predictor_service.predict_single(txn.get("description", "") if isinstance(txn, dict) else txn.description)
            results.append(result)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_model": "XGBoost + TF-IDF"
    }