from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.routers import predict
except ModuleNotFoundError:
    from routers import predict

app = FastAPI(title="AutoAudit-ML VAT Code Predictor", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router, prefix="/api/v1", tags=["VAT Prediction"])


@app.get("/")
def root():
    return {"message": "AutoAudit-ML VAT Predictor API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}
