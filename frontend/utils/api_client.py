import streamlit as st
import requests

BASE_URL = st.secrets.get("BACKEND_URL", "https://autoaudit-ml-vat-code-predictor.onrender.com")

def predict_transaction(description: str, amount: float = None):
    """Call single prediction endpoint"""
    try:
        payload = {"description": description}
        if amount:
            payload["amount"] = amount
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {str(e)}")
        return None

def predict_batch(file):
    """Call batch prediction endpoint"""
    try:
        files = {"file": file}
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/batch",
            files=files,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {str(e)}")
        return None