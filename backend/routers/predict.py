from fastapi import APIRouter, HTTPException

try:
    from backend.models.schemas import (
        TransactionRequest,
        BatchTransactionRequest,
        PredictionResponse,
        BatchPredictionResponse,
    )
    from backend.models.predictor import predictor_service
except ModuleNotFoundError:
    from models.schemas import (
        TransactionRequest,
        BatchTransactionRequest,
        PredictionResponse,
        BatchPredictionResponse,
    )
    from models.predictor import predictor_service

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
        results = predictor_service.predict_batch(request.transactions)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor_service.loaded,
        "active_model": "XGBoost + TF-IDF",
    }
