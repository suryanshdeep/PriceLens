# Backend Documentation

This document tracks the FastAPI service, API contract, backend setup, and development rules.

## Purpose

The backend exposes prediction APIs for the React app and later hosts the trained model artifacts.
Right now it returns mock predictions so the full-stack flow can be tested before ML artifacts are
ready.

## Technology

- Python 3.11+
- FastAPI
- Uvicorn
- Pydantic and pydantic-settings
- scikit-learn, pandas, numpy, joblib for the ML pipeline
- pytest for API contract tests
- ruff and mypy for stricter project quality

## Directory Map

```text
backend/
  pyproject.toml
  requirements.txt
  README.md
  src/
    config.py
    api/
      main.py
      schemas.py
      services.py
    data/
    features/
    models/
    retrieval/
    utils/
      metrics.py
  tests/
    test_api_contract.py
```

## Local Setup

Use a virtual environment inside `backend/` so packages install on the D: drive.

```powershell
cd D:\PriceLens\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r requirements.txt
```

If Conda base is active, it is cleaner to deactivate it before activating `.venv`:

```powershell
conda deactivate
.\.venv\Scripts\Activate.ps1
```

Confirm the active Python:

```powershell
python -c "import sys; print(sys.executable)"
pip -V
```

Expected paths should include:

```text
D:\PriceLens\backend\.venv\
```

## Run Backend

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --reload --port 8000
```

Open:

```text
http://localhost:8000/docs
```

## Current Endpoints

### `GET /health`

Confirms the API is running.

Example response:

```json
{
  "status": "ok",
  "service": "PriceLens",
  "version": "0.1.0"
}
```

### `GET /model-info`

Returns current model metadata. It currently reports a mock baseline.

### `POST /predict`

JSON endpoint for text-based predictions.

Request:

```json
{
  "catalog_content": "Organic almond butter 16 oz jar",
  "brand": "Example Brand",
  "category": "Grocery"
}
```

Response:

```json
{
  "predicted_price": 12.49,
  "confidence_band": {
    "low": 10.8,
    "high": 14.3
  },
  "model_version": "mock_baseline_v0",
  "features_used": ["catalog_content", "brand", "category"],
  "explanation_tokens": ["organic", "almond", "butter"],
  "image_received": false
}
```

### `POST /predict-with-image`

Multipart endpoint for predictions with an optional product image.

Form fields:

- `catalog_content`
- `brand`
- `category`
- `image`

The first version accepts the image and records that it was received. The model does not use image
features yet.

## Configuration

Backend settings live in `backend/src/config.py`.

Environment variables use the `PRICELENS_` prefix. Example:

```text
PRICELENS_MODEL_VERSION=baseline_v1
```

## Testing

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
pytest
```

Current tests validate the API contract for `/health` and `/predict`.

## Development Rules

- Keep API schemas in `src/api/schemas.py`.
- Keep route handlers thin in `src/api/main.py`.
- Put prediction logic behind service classes in `src/api/services.py`.
- Use pathlib for filesystem paths.
- Do not hardcode absolute local paths.
- Keep datasets and artifacts out of Git.
- Add or update API contract tests when endpoint behavior changes.

## Next Backend Milestones

- Load trained model artifacts from `models/`.
- Replace mock prediction with real TF-IDF + Ridge inference.
- Add a metrics endpoint after baseline metrics are generated.
- Add structured logging for explanation tokens and prediction metadata.
