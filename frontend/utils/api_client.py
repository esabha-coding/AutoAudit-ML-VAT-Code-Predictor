import requests
import streamlit as st

BASE_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")

def predict_single(description: str, amount: float = None):
    try:
        payload = {"description": description}
        if amount:
            payload["amount"] = amount
        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.ConnectionError:
        return None, f"Cannot reach backend at {BASE_URL}. Wait 60 seconds and retry."
    except requests.exceptions.Timeout:
        return None, "Request timed out. Backend waking up - retry in 30 seconds."
    except Exception as e:
        return None, str(e)

def health_check():
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=15)
        return response.json()
    except:
        return {"status": "offline", "model_loaded": False}