import requests

API_URL = "http://127.0.0.1:8000/api/v1"

def get_single_prediction(description: str):
    try:
        response = requests.post(f"{API_URL}/predict", json={"description": description}, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_batch_prediction(descriptions: list[str]):
    try:
        # Increased timeout to 120 seconds for large dataset uploads
        response = requests.post(f"{API_URL}/predict/batch", json={"transactions": descriptions}, timeout=120)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}