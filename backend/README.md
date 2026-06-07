# PriceLens Backend

FastAPI service and ML pipeline modules for PriceLens.

Full backend documentation lives in [../docs/backend.md](../docs/backend.md).

## Run API

```powershell
cd D:\PriceLens\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

## Current API Contract

`POST /predict`

```json
{
  "catalog_content": "Organic almond butter 16 oz jar",
  "brand": "Example Brand",
  "category": "Grocery"
}
```

`POST /predict-with-image` accepts multipart form fields:

- `catalog_content`
- `brand`
- `category`
- `image`

Both routes return:

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

`explanation_tokens` are included in the response for development visibility. The frontend
does not render them yet.
