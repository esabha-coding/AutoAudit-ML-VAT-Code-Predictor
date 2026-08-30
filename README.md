# 🇬🇧 AutoAudit-ML: MTD-Ready VAT Code Predictor

AutoAudit-ML is an automated Making Tax Digital (MTD) compliance engine that classifies UK bank statement transactions into HMRC VAT categories using Machine Learning.

It provides both a **FastAPI REST API** for high-throughput batch classification and an interactive **Streamlit Executive Dashboard** featuring multi-statement aggregation, date-wise analytics, and automated tax reporting.

---

## 🔑 Key Features

* **ML-Powered VAT Categorization:** Classifies transaction descriptions into standard, zero-rated, reduced, exempt, or outside-scope VAT codes with high accuracy.
* **Vectorized Batch Processing:** Utilizes matrix vectorization to process 10,000+ transaction lines in under 2 seconds.
* **Multi-CSV Statement Aggregation:** Upload and consolidate multiple bank statement files simultaneously into a single audit dataset.
* **Date-Wise & Distribution Analytics:** Interactive Plotly visual displays highlighting tax distribution ratios and volume trends over time.
* **FastAPI Backend Architecture:** Pydantic schema-validated REST endpoints preloaded with ML pipeline artifacts for real-time inference.

---

## 🛠️ Technology Stack

* **Machine Learning:** XGBoost, Scikit-learn (TF-IDF Vectorizer), Pandas, NumPy, Joblib
* **Backend API:** FastAPI, Uvicorn, Pydantic V2
* **Frontend Dashboard:** Streamlit, Plotly Express
* **PDF Statement Parsing:** `pdfplumber`

---

## 📁 Repository Structure

```text
AutoAudit-ML/
├── backend/                  # FastAPI Application
│   ├── models/               # Pydantic schemas & vector predictor service
│   ├── routers/              # API endpoints (/predict, /predict/batch)
│   └── main.py               # API entry point & CORS configuration
├── frontend/                 # Streamlit UI
│   ├── utils/                # API client HTTP wrappers
│   └── app.py                # Dashboard & analytics UI
├── model/
│   └── artifacts/            # Exported TF-IDF & XGBoost joblib files
├── notebooks/                # Jupyter notebook development pipeline
│   ├── 01_eda.ipynb          # PDF extraction & synthetic data generation
│   ├── 02_data_processing.ipynb # Text normalization & stratified splits
│   └── 03_baseline_model.ipynb  # Model training & metrics evaluation
├── .streamlit/               # Streamlit server & theme config
├── requirements.txt          # Python project dependencies
└── README.md