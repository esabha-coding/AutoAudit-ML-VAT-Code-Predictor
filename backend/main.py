from fastapi import FastAPI
from backend.routers import predict

app = FastAPI(title="MTD-Ready VAT Predictor API", version="1.0.0")
app.include_router(predict.router)

@app.get("/health")
def health_check():
    return {"status": "online", "model": "XGBoost + TF-IDF"}