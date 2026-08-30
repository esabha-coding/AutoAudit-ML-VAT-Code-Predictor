import joblib
import re
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "model", "artifacts")

class VATPredictorService:
    def __init__(self):
        self.tfidf = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib"))
        self.model = joblib.load(os.path.join(ARTIFACTS_DIR, "xgboost_model.joblib"))
        self.label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
        self.classes = self.label_encoder.classes_

    def _clean_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def predict_single(self, description: str) -> dict:
        cleaned = self._clean_text(description)
        vectorized = self.tfidf.transform([cleaned])
        probs = self.model.predict_proba(vectorized)[0]
        top_idx = int(np.argmax(probs))
        predicted_class = str(self.classes[top_idx])
        confidence = float(probs[top_idx])
        prob_dict = {str(cls): float(prob) for cls, prob in zip(self.classes, probs)}
        explanation = f"Classified as '{predicted_class}' with {confidence:.1%} confidence based on merchant transaction patterns."
        return {
            "description": description,
            "predicted_vat_code": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "explanation": explanation
        }

    def predict_batch(self, descriptions: list) -> list:
        cleaned_list = [self._clean_text(desc) for desc in descriptions]
        vectorized = self.tfidf.transform(cleaned_list)
        probs_matrix = self.model.predict_proba(vectorized)
        top_indices = np.argmax(probs_matrix, axis=1)
        results = []
        for i, desc in enumerate(descriptions):
            idx = top_indices[i]
            predicted_class = str(self.classes[idx])
            confidence = float(probs_matrix[i, idx])
            prob_dict = {str(cls): float(prob) for cls, prob in zip(self.classes, probs_matrix[i])}
            results.append({
                "description": desc,
                "predicted_vat_code": predicted_class,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "explanation": f"Classified as '{predicted_class}' with {confidence:.1%} confidence."
            })
        return results

predictor_service = VATPredictorService()