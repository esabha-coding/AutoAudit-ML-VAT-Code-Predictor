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

# 🇬🇧 AutoAudit-ML — MTD VAT Code Predictor for UK Bank Transactions

> **Live Demo:** [Streamlit Cloud Deployment](https://autoaudit-ml-vat-predictor.streamlit.app/) &nbsp;|&nbsp; **Target Compliance:** MTD ITSA | UK Accounting Automation | April 2026 Deadline

An end-to-end machine learning system that automatically predicts the correct UK VAT code for bank transaction descriptions — addressing the cold-start accuracy gap (60–70% on first run) common in legacy accounting software rules.

Trained on extracted UK bank statement descriptions and synthetic transaction variations grounded in HMRC VAT Notice 700, this system classifies transactions into five HMRC VAT categories with vectorized high-speed inference.

---

## 🛑 The Problem This Solves

From April 2026, Making Tax Digital (MTD) for Income Tax Self Assessment (ITSA) becomes mandatory across the UK. Every transaction must be digitally recorded with the correct VAT treatment before quarterly submission to HMRC.

While standard software handles general category assignments, VAT code assignment often remains a manual step requiring monthly correction. Static bank rules often achieve low accuracy on new accounts with no historical General Ledger mapping. **AutoAudit-ML directly automates this workflow.**

---

## 🏷️ HMRC VAT Classes

| Code | Description | Examples |
|:---|:---|:---|
| **Standard Rate (20%)** | Most taxable goods and services | Electronics, SaaS software, Retail, Takeaway food |
| **Reduced Rate (5%)** | Domestic energy & specific goods | British Gas, E.ON Next, Domestic power |
| **Zero Rated (0%)** | Essential cold food, books, public transport | Tesco groceries, Amazon Kindle, Trainline, TfL |
| **Exempt** | Financial services, postal, insurance | Bank charges, Royal Mail postage, Insurance premiums |
| **Outside Scope** | Non-VAT statutory & internal transfers | HMRC tax payments, Payroll transfers, Director drawings |

---

## 📊 Model Performance Evaluation

| Model Architecture | Macro F1 | Standard Rate F1 | Zero Rate F1 | Reduced F1 | Exempt F1 | Outside Scope F1 |
|:---|:---|:---|:---|:---|:---|:---|
| **TF-IDF + XGBoost Classifier** | **0.9994** | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology |
|:---|:---|
| **ML Pipeline** | XGBoost Classifier, Scikit-learn (TF-IDF Vectorizer), Pandas, NumPy, Joblib |
| **Backend REST API** | FastAPI, Uvicorn, Pydantic V2 |
| **Frontend Interface** | Streamlit, Plotly Express |
| **PDF Ingestion** | pdfplumber |
| **Serialization** | joblib |
| **Deployment** | Streamlit Community Cloud + Render |

---

## 🔌 API Reference

### `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "active_model": "XGBoost + TF-IDF"
}
```

### `POST /predict`
**Request:**
```json
{
  "description": "TESCO STORES 3421",
  "amount": 4.50
}
```
**Response:**
```json
{
  "vat_code": "zero_rated",
  "confidence": 0.94,
  "label": "Zero Rated (0%)",
  "explanation": "Transaction classified as Zero Rated because TESCO is associated with grocery retail. Cold food purchases are zero-rated under HMRC VAT Notice 700/14.",
  "all_probabilities": {
    "standard_rate": 0.04,
    "reduced_rate": 0.01,
    "zero_rated": 0.94,
    "exempt": 0.00,
    "outside_scope": 0.01
  }
}
```

### `POST /predict/batch`
Accepts CSV upload. Returns original CSV with `predicted_vat_code` and `confidence` columns appended — MTD-ready output for direct import into Xero or QuickBooks.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.11
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/esabha-coding/AutoAudit-ML-VAT-Code-Predictor.git
cd AutoAudit-ML-VAT-Code-Predictor

# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python src/train.py
```

Reads `data/transactions.csv`, trains TF-IDF + XGBoost classifier, and saves artifacts to `model/artifacts/`.

### Start Backend API

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Interactive docs available at: `http://localhost:8000/docs`

### Launch Streamlit Frontend

```bash
cd frontend
streamlit run app.py
```

Dashboard available at: `http://localhost:8501`

---

## ☁️ Deployment

| Service | Platform | Configuration |
|:---|:---|:---|
| **Frontend** | Streamlit Community Cloud | Main file: `frontend/app.py` |
| **Backend** | Render | Root: `backend/`, Start: `uvicorn main:app --host 0.0.0.0 --port $PORT` |

**Environment variable on Streamlit Cloud:**

| Key | Value |
|:---|:---|
| `BACKEND_URL` | `https://your-render-url.onrender.com` |

Every `git push origin main` automatically redeploys both services.

---

## 📚 Research Background

This project addresses a documented gap in UK accounting automation:

- Static bank rules in Xero/QuickBooks achieve **60–70% accuracy on first run** for new clients, improving to 95%+ only after 2–3 months of accumulated GL history *(Source: CodeIQ, 2026)*
- QuickBooks published research in 2025 acknowledging that *"unique formatting of transaction descriptions, wide variety of transaction categories, and vast scale of data"* remain unsolved challenges in their production model
- From **6 April 2026**, MTD ITSA becomes mandatory for 4+ million UK sole traders and landlords — creating urgent demand for automated VAT categorisation
- Under **UK GDPR Article 22**, automated financial decisions must be explainable — this system surfaces confidence scores and reasoning for every prediction

---

## ⚠️ Known Limitations

- Ambiguous merchants (e.g. AMAZON — could be zero-rated books or standard-rated electronics) reduce confidence scores — these are automatically flagged for human review
- Reduced Rate (5%) classification has the smallest training sample and highest error rate in minority-class scenarios
- Model was trained primarily on UK bank statement formats — non-UK transaction descriptions may perform below benchmark
- New unseen merchants fall back to the highest-frequency class with a low confidence score — surfaced clearly in the UI

---

## 🔮 Future Improvements

- DistilBERT fine-tuned model for semantic merchant understanding beyond keyword patterns
- Amount-aware prediction (£3.50 at TESCO → zero-rated; £35.00 → likely mixed rate)
- **Xero API write-back** — categorised transactions posted directly to chart of accounts via Xero API
- Multi-bank description normalisation (Barclays, Monzo, Starling, HSBC all format differently)
- SHAP token-level explainability layer for GDPR-compliant decision audit trail

---

## 👤 Author

**Saba Ijaz** — AI Automation Builder & Executive Accountant

4 years of remote UK e-commerce accounting experience (Amazon UK/US, TikTok Shop, dropshipping/online arbitrage). Builds LLM agents, RAG pipelines, and ML systems that automate accounting and financial workflows.

**Certifications:** Xero Advisor Certified &nbsp;|&nbsp; UK GAAP &nbsp;|&nbsp; Making Tax Digital &nbsp;|&nbsp; Amazon Seller Central

[![GitHub](https://img.shields.io/badge/GitHub-esabha--coding-black?logo=github)](https://github.com/esabha-coding)

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.
