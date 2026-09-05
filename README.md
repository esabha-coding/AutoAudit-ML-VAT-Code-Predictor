# AutoAudit-ML VAT Code Predictor

AutoAudit-ML is a machine-learning application that classifies UK bank transaction descriptions into five VAT treatment categories. It combines a Streamlit dashboard for interactive use with a FastAPI service for programmatic predictions.

## Live Demo

[Open the Streamlit dashboard](https://autoaudit-ml-vat-predictor.streamlit.app/)

The dashboard is publicly accessible and supports:

- Single transaction VAT classification
- One CSV bank statement upload at a time
- Confidence scores and probability breakdowns
- VAT distribution and confidence charts
- Downloadable prediction results
- HMRC VAT category reference information

## VAT Categories

| Category | Rate | Examples |
| --- | ---: | --- |
| Standard Rate | 20% | Electronics, software, restaurants, mobile contracts |
| Reduced Rate | 5% | Domestic energy and selected utilities |
| Zero Rated | 0% | Groceries, books, and public transport |
| Exempt | N/A | Bank charges, insurance, and postage |
| Outside Scope | N/A | HMRC payments, payroll, and internal transfers |

## How It Works

The model cleans transaction text, transforms it with a TF-IDF vectorizer, and uses an XGBoost classifier to predict the most likely VAT category. The application also returns the model confidence and probability for every category.

The saved model artifacts are stored in `model/artifacts/`:

- `tfidf_vectorizer.joblib`
- `xgboost_model.joblib`
- `label_encoder.joblib`

## Technology Stack

- Python 3.11
- Streamlit and Plotly for the dashboard
- FastAPI, Uvicorn, and Pydantic for the REST API
- Scikit-learn TF-IDF vectorization
- XGBoost classification
- Pandas and NumPy for data processing
- Joblib for model serialization

## Repository Structure

```text
AutoAudit-ML/
|-- backend/
|   |-- main.py                 # FastAPI application
|   |-- models/
|   |   |-- predictor.py        # Model inference service
|   |   `-- schemas.py          # Request and response models
|   `-- routers/
|       `-- predict.py          # Prediction routes
|-- frontend/
|   `-- app.py                  # Streamlit dashboard
|-- model/
|   `-- artifacts/              # Saved model files
|-- notebooks/                  # Exploratory and training notebooks
|-- src/
|   |-- model.py                # Model definitions
|   `-- train.py                # Training pipeline
|-- .streamlit/                 # Streamlit configuration
|-- render.yaml                 # Render backend deployment configuration
|-- requirements.txt
`-- README.md
```

## Installation

### Requirements

- Python 3.11 or newer
- Git

### Setup

```bash
git clone https://github.com/esabha-coding/AutoAudit-ML-VAT-Code-Predictor.git
cd AutoAudit-ML-VAT-Code-Predictor

python -m venv venv
```

Windows:

```powershell
.\venv\Scripts\activate
```

macOS or Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Dashboard

From the project root:

```bash
streamlit run frontend/app.py
```

The dashboard opens at `http://localhost:8501`.

## Run the FastAPI Service

From the project root:

```bash
uvicorn backend.main:app --reload --port 8000
```

The API opens at `http://localhost:8000`.

Interactive API documentation is available at `http://localhost:8000/docs`.

## API Reference

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

The detailed router health endpoint is also available at:

```http
GET /api/v1/health
```

### Single Prediction

```http
POST /api/v1/predict
Content-Type: application/json
```

Request:

```json
{
  "description": "TESCO STORES 3421",
  "amount": 4.50
}
```

Response:

```json
{
  "description": "TESCO STORES 3421",
  "predicted_vat_code": "zero_rated",
  "confidence": 0.94,
  "probabilities": {
    "standard_rate": 0.04,
    "reduced_rate": 0.01,
    "zero_rated": 0.94,
    "exempt": 0.00,
    "outside_scope": 0.01
  },
  "explanation": "Transaction classified as Zero Rated."
}
```

### Batch Prediction

```http
POST /api/v1/predict/batch
Content-Type: application/json
```

Request:

```json
{
  "transactions": [
    "TESCO STORES 3421",
    "BRITISH GAS",
    "BARCLAYS BANK CHARGE"
  ]
}
```

The response contains a `results` array with one prediction object per transaction.

## Model Performance

The current model evaluation reports a macro F1 score of `0.9994` on the project evaluation data. Performance can vary for ambiguous descriptions, unseen merchants, and transaction formats outside the UK bank statement data used for training.

These results should be treated as a model benchmark, not a guarantee of tax correctness. Review low-confidence or ambiguous predictions before submitting VAT records.

## Deployment

### Streamlit Community Cloud

- Repository: `esabha-coding/AutoAudit-ML-VAT-Code-Predictor`
- Branch: `main`
- Main file: `frontend/app.py`
- Live URL: https://autoaudit-ml-vat-predictor.streamlit.app/

Pushes to `main` trigger a Streamlit redeployment when the repository is connected to Streamlit Community Cloud.

### Render Backend

The backend deployment configuration is in `render.yaml`:

```yaml
startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

The backend can be deployed independently on Render. Set any public backend URL in the deployment environment when integrating an external client.

## Known Limitations

- The dashboard processes one CSV statement at a time.
- The model predicts from transaction text and does not currently use the transaction amount as a model feature.
- Ambiguous merchants may require manual review.
- The model is trained primarily on UK bank statement descriptions.
- The dashboard currently performs local inference from the saved artifacts; it does not require the FastAPI service for its main prediction flow.

## Future Improvements

- Amount-aware VAT classification
- More bank-specific description normalization
- Expanded validation data for minority VAT classes
- Human review workflows for low-confidence predictions
- Optional accounting-platform integrations

## Author

**Saba Ijaz** - AI Automation Builder and Executive Accountant

## License

MIT License
