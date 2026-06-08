# Model And ML Workflow Documentation

This document tracks the dataset contract, notebook-first workflow, baseline model plan, metrics,
and artifact strategy.

## Goal

Build a resume-grade ML workflow for product price prediction using the Amazon ML Challenge data.
The first production model is text-based. Image modeling is planned later, but the backend already
accepts images so the API contract is ready.

## Dataset Contract

Local files:

```text
data/raw/train.csv
data/raw/test.csv
data/images/
```

Expected `train.csv` columns:

- `sample_id`: unique product identifier
- `catalog_content`: unclean product text with title, features, and attributes
- `image_link`: relative or absolute path to product image
- `price`: target variable

Expected `test.csv` columns:

- `sample_id`
- `catalog_content`
- `image_link`

The dataset is unclean, so cleaning and validation are required before training.

## Data And Artifact Git Policy

These are local-only and ignored by Git:

- `data/raw/*`
- `data/interim/*`
- `data/processed/*`
- `data/images/*`
- `models/*`
- `reports/*`

`.gitkeep` files preserve the directory structure.

## Notebook-First Workflow

Start with:

```text
notebooks/01_text_baseline_demo.ipynb
```

Purpose of this notebook:

- load `data/raw/train.csv`
- validate required columns
- clean `catalog_content`
- clean and filter `price`
- create train/validation split
- train TF-IDF + Ridge regression
- train on `log1p(price)`
- convert predictions with `expm1`
- compute SMAPE, MAE, RMSE, and R2
- inspect top positive and negative TF-IDF price-signal tokens
- write `reports/baseline_metrics.md`

This notebook should be used to prove the approach before code is promoted into reusable scripts.

## Baseline Model

Initial baseline stages:

1. Median-price baseline.
2. TF-IDF text features from `catalog_content`.
3. Ridge regression on `log1p(price)`.
4. Validation metrics on held-out split.
5. Save artifacts with joblib.

Planned artifacts:

```text
models/text_baseline_pipeline.joblib
models/baseline_metadata.json
reports/baseline_metrics.md
```

## Metrics

Implemented utility functions are in:

```text
backend/src/utils/metrics.py
```

Tracked metrics:

- SMAPE
- MAE
- RMSE
- R2

For the challenge-style objective, SMAPE is the most important reporting metric.

## Explainability

For the text baseline, explainability starts with TF-IDF token inspection:

- global top positive price-signal tokens
- global top negative price-signal tokens
- later, per-prediction influential tokens

The backend currently returns `explanation_tokens` from mock logic. Once the real model is wired,
we should log model-derived tokens for developer inspection. The frontend should not display these
until the explanation quality is good enough.

## Script Promotion Plan

Phase 2 promotes the notebook flow into:

```text
backend/src/data/load_data.py
backend/src/data/clean_data.py
backend/src/data/split_data.py
backend/src/features/text_features.py
backend/src/models/train_baseline.py
backend/src/models/predict.py
backend/src/models/evaluate.py
```

## Production Commands

Train on the full local training set:

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
python -m src.models.train_baseline
```

Run a faster development smoke train:

```powershell
python -m src.models.train_baseline --sample-size 20000
```

Evaluate the saved artifact:

```powershell
python -m src.models.evaluate
```

Run a one-off command-line prediction:

```powershell
python -m src.models.predict "Organic almond butter 16 oz jar"
```

## Backend Artifact Behavior

The FastAPI backend tries to load:

```text
models/text_baseline_pipeline.joblib
models/baseline_metadata.json
```

If both files exist and can be loaded, `/predict` uses the trained TF-IDF + Ridge model. If either
file is missing or unloadable, the API falls back to mock prediction so the frontend remains usable.

## Image Modeling Boundary

Current version:

- frontend accepts image upload
- backend accepts image upload
- model does not use image features yet

Future version:

- validate `image_link` paths
- build image feature extraction
- combine image and text features
- train multimodal model

## External Marketplace Boundary

External marketplace prices must not be used for training.

Later, a separate market-comparison feature can be added with live APIs or scraping if legally and
technically appropriate. That feature should remain clearly separate from the supervised training
dataset.

## Next Model Milestones

- Run `train_baseline` against local `train.csv`.
- Inspect `reports/baseline_metrics.md`.
- Compare full-data metrics against smoke-run metrics.
- Add structured quantity/unit extraction features.
- Add backend tests for real inference fallback behavior.
