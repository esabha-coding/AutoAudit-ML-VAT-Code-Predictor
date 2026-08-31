import streamlit as st
import requests

BASE_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")


def _normalize_single_result(data):
    if not isinstance(data, dict):
        return data

    vat_code = data.get("predicted_vat_code") or data.get("vat_code") or "N/A"
    return {
        "vat_code": vat_code,
        "category": data.get("category") or vat_code,
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "description": data.get("description"),
        "probabilities": data.get("probabilities", {}),
        "explanation": data.get("explanation", ""),
    }


def predict_transaction(description: str, amount: float = None):
    """Call single prediction endpoint."""
    try:
        payload = {"description": description}
        if amount is not None:
            payload["amount"] = float(amount)

        response = requests.post(
            f"{BASE_URL}/api/v1/predict",
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return _normalize_single_result(response.json())
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {str(e)}")
        return None


def predict_batch(file):
    """Call batch prediction endpoint using description strings."""
    try:
        import pandas as pd

        df = pd.read_csv(file)
        if df.empty:
            st.warning("The uploaded CSV is empty.")
            return None

        description_column = None
        for candidate in ["description", "Description", "transaction_description", "Transaction Description"]:
            if candidate in df.columns:
                description_column = candidate
                break

        if description_column is None:
            st.error("CSV must contain a description column named 'description'.")
            return None

        transactions = [str(value) for value in df[description_column].fillna("").tolist()]
        payload = {"transactions": transactions}

        response = requests.post(
            f"{BASE_URL}/api/v1/predict/batch",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        response_data = response.json()
        results = response_data.get("results", [])
        return {"results": [_normalize_single_result(item) for item in results]}
    except requests.exceptions.RequestException as e:
        st.error(f"Backend error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"CSV processing error: {str(e)}")
        return None