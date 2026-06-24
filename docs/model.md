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

- Run `train_enhanced_baseline` against local `train.csv`.
- Inspect `reports/enhanced_baseline_metrics.md`.
- Compare TF-IDF Ridge against TF-IDF + structured feature Ridge.
- Use saved validation/test prediction CSVs as the first ensemble-ready outputs.
- Add backend tests for real inference fallback behavior.

## Phase 3A: Structured Text Features

The Phase 3A notebook is:

```text
notebooks/02_structured_text_features.ipynb
```

It explores signals parsed from mixed `catalog_content`, including:

- pack count
- size value
- unit/size presence
- text length
- word count
- number count
- premium keyword count
- unique word ratio

Run a smoke train:

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
python -m src.models.train_enhanced_baseline --sample-size 10000 --max-features 20000
```

Run the full enhanced baseline:

```powershell
python -m src.models.train_enhanced_baseline
```

Generated local artifacts:

```text
models/text_structured_ridge_pipeline.joblib
models/text_structured_ridge_metadata.json
models/predictions/validation_text_structured_ridge.csv
models/predictions/test_text_structured_ridge.csv
reports/enhanced_baseline_metrics.md
```

## Phase 3B: Lightweight Image Metadata Features

Before deep image embeddings, Phase 3B uses cheap image metadata:

- image exists flag
- width
- height
- aspect ratio
- file size
- grayscale brightness mean
- grayscale brightness standard deviation

Build the image feature cache:

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
python -m src.data.build_image_feature_cache --split train
```

Run a smoke image-aware train:

```powershell
python -m src.models.train_image_baseline --sample-size 10000 --max-features 20000
```

Run the full image-aware baseline:

```powershell
python -m src.models.train_image_baseline
```

Generated local artifacts:

```text
data/processed/train_image_features.csv
models/text_structured_image_ridge_pipeline.joblib
models/text_structured_image_ridge_metadata.json
models/predictions/validation_text_structured_image_ridge.csv
models/predictions/test_text_structured_image_ridge.csv
reports/image_baseline_metrics.md
```

This phase does not use neural image embeddings yet. It is a low-cost signal check before CLIP/ViT
features.

## Phase 4A: Model Comparison And Ensemble Scaffold

Phase 4A creates a reusable structure for comparing model metrics and blending prediction files.
This is intentionally model-agnostic so later transformer and image-embedding predictions can use
the same CSV format.

Prediction CSV contract:

```text
sample_id,actual_price,predicted_price
```

Generate the model comparison report:

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
python -m src.models.compare_models
```

This writes:

```text
reports/model_comparison.md
```

Run equal-weight blending over the existing structured and image-metadata baselines:

```powershell
python -m src.models.ensemble --split validation
python -m src.models.ensemble --split test
```

Optimize validation weights:

```powershell
python -m src.models.optimize_ensemble --split validation
```

Generated local outputs:

```text
models/predictions/validation_ensemble_equal_weight.csv
models/predictions/test_ensemble_equal_weight.csv
reports/ensemble_equal_weight_validation_metrics.md
reports/ensemble_equal_weight_test_metrics.md
models/ensemble_optimized_validation_weights.json
models/predictions/validation_ensemble_optimized.csv
reports/ensemble_optimized_validation_metrics.md
```

Custom weights can be passed in model-file order:

```powershell
python -m src.models.ensemble --split validation --weights 0.7 0.3
```

For the current two very similar baseline models, optimized weights will likely favor the structured
text model almost entirely. The optimizer becomes more useful once transformer prediction files are
added in Phase 4B.

When transformer models are added, their validation/test prediction files should be dropped into
`models/predictions/` using the same CSV schema, then passed with repeated `--prediction-file`
arguments.

## Phase 4B: Transformer Text Embeddings

Phase 4B adds a reusable transformer embedding path. It starts with SentenceTransformers because
that gives fixed-size product text embeddings that can be cached and reused by Ridge/CatBoost-style
models.

Install optional transformer dependencies:

```powershell
cd D:\PriceLens\backend
.\.venv\Scripts\Activate.ps1
pip install --no-cache-dir -e .[transformers]
```

Build a training embedding cache:

```powershell
python -m src.data.build_text_embedding_cache --split train --model-name sentence-transformers/all-MiniLM-L6-v2 --batch-size 64
```

Train Ridge on cached embeddings:

```powershell
python -m src.models.train_transformer_baseline `
  --embedding-cache-path D:\PriceLens\models\embeddings\train_sentence-transformers__all-MiniLM-L6-v2.npz `
  --model-label minilm_text_ridge
```

Generated local artifacts:

```text
models/embeddings/train_sentence-transformers__all-MiniLM-L6-v2.npz
models/minilm_text_ridge.joblib
models/minilm_text_ridge_metadata.json
models/predictions/validation_minilm_text_ridge.csv
models/predictions/test_minilm_text_ridge.csv
reports/minilm_text_ridge_metrics.md
```

After generating transformer prediction files, rerun:

```powershell
python -m src.models.compare_models
python -m src.models.optimize_ensemble `
  --split validation `
  --prediction-file D:\PriceLens\models\predictions\validation_text_structured_ridge.csv `
  --prediction-file D:\PriceLens\models\predictions\validation_minilm_text_ridge.csv
```

Heavier models such as BERT/DeBERTa/RoBERTa can use the same cache/train pattern by changing
`--model-name` and `--model-label`.
