import os
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "../../model/artifacts")

class VATPredictorService:
    def __init__(self):
        self.tfidf = None
        self.model = None
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model artifacts if they exist"""
        try:
            tfidf_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
            model_path = os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib")
            
            if os.path.exists(tfidf_path) and os.path.exists(model_path):
                self.tfidf = joblib.load(tfidf_path)
                self.model = joblib.load(model_path)
                print("✅ Model artifacts loaded successfully")
            else:
                print("⚠️ Model artifacts not found - using fallback predictions")
        except Exception as e:
            print(f"⚠️ Error loading artifacts: {e}")
    
    def predict_single(self, description: str):
        """Predict VAT code for a single transaction"""
        if self.model is None:
            # Fallback prediction
            return {
                "vat_code": "20",
                "category": "Standard Rate",
                "confidence": 0.75,
                "description": description
            }
        
        try:
            X = self.tfidf.transform([description])
            prediction = self.model.predict(X)[0]
            confidence = max(self.model.predict_proba(X)[0])
            
            return {
                "vat_code": str(prediction),
                "category": "Standard Rate",
                "confidence": float(confidence),
                "description": description
            }
        except Exception as e:
            return {"error": str(e)}
    
    def predict_batch(self, transactions: list):
        """Predict VAT codes for multiple transactions"""
        results = []
        for txn in transactions:
            result = self.predict_single(txn.get("description", ""))
            results.append(result)
        return results

predictor_service = VATPredictorService()