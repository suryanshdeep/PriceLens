# PriceLens

PriceLens is a multimodal product price intelligence project. The first milestone is a
text-based baseline that predicts e-commerce product prices from catalog content, with a
FastAPI inference backend and a React + TypeScript frontend.

External marketplace prices are not used for model training. Live market comparison can be
added later as a separate feature.

## Project Layout

```text
backend/    FastAPI app, ML modules, tests, backend pyproject
docs/       Detailed documentation by project area
frontend/   React + TypeScript + Vite app
data/       Local datasets and downloaded images, ignored by Git
models/     Trained artifacts, ignored by Git
reports/    Metrics and experiment reports, ignored by Git
notebooks/  Notebook-first ML exploration before scripts are promoted
```

## Documentation Index

- [Backend documentation](docs/backend.md)
- [Frontend documentation](docs/frontend.md)
- [Model and ML workflow documentation](docs/model.md)

## Data Layout

Place local challenge data here:

```text
data/raw/train.csv
data/raw/test.csv
data/images/
```

Expected `train.csv` columns:

- `sample_id`
- `catalog_content`
- `image_link`
- `price`

Expected `test.csv` columns:

- `sample_id`
- `catalog_content`
- `image_link`

## Backend

```powershell
cd D:\PriceLens\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip --no-cache-dir
pip install --no-cache-dir -r requirements.txt
uvicorn src.api.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /model-info`
- `POST /predict` for JSON requests
- `POST /predict-with-image` for multipart requests with an optional image

The backend currently returns mock predictions until trained model artifacts are available.

## Frontend

```powershell
cd D:\PriceLens\frontend
npm install --cache D:\PriceLens\.npm-cache
Copy-Item .env.example .env.local
npm run dev
```

Optional environment file:

```text
VITE_API_BASE_URL=http://localhost:8000
```

## ML Workflow

Start in `notebooks/01_text_baseline_demo.ipynb`. Once the notebook workflow is stable,
promote it into scripts under `backend/src/data`, `backend/src/features`, and
`backend/src/models`.

Target command style after promotion:

```powershell
cd backend
python -m src.models.train_baseline
python -m src.models.evaluate
```

For details, see [docs/model.md](docs/model.md).
