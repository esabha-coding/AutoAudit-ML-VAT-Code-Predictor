import streamlit as st
import requests
import json

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
        # Read CSV and convert to list of dicts
        import pandas as pd
        df = pd.read_csv(file)
        
        # Prepare transactions list
        transactions = []
        for _, row in df.iterrows():
            transactions.append({
                "description": row.get("description", ""),
                "amount": row.get("amount", None)
            })
        
        payload = {"transactions": transactions}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/predict/batch",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {str(e)}")
        return None