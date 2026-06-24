import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from tqdm.auto import tqdm

from src.data.clean_data import clean_train_data
from src.data.load_data import load_train_data
from src.data.split_data import create_train_validation_test_split
from src.features.transformer_embeddings import load_embedding_cache
from src.models.artifacts import models_dir, reports_dir
from src.models.reporting import write_json, write_metrics_report
from src.utils.metrics import mae, r2, rmse, smape

logger = logging.getLogger(__name__)


def train_transformer_baseline(
    *,
    embedding_cache_path: Path,
    model_label: str,
    alpha: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    started_at = perf_counter()
    progress = tqdm(total=7, desc="Training transformer baseline")

    raw = load_train_data()
    clean = clean_train_data(raw)
    progress.set_postfix(step="loaded data", rows=len(clean))
    progress.update(1)

    cache_sample_ids, embeddings, embedding_model_name = load_embedding_cache(embedding_cache_path)
    progress.set_postfix(step="loaded embeddings", rows=len(cache_sample_ids))
    progress.update(1)

    frame = _attach_embeddings(clean, cache_sample_ids, embeddings)
    split = create_train_validation_test_split(frame, random_state=random_state)
    progress.set_postfix(step="split data", rows=len(frame))
    progress.update(1)

    model = Ridge(alpha=alpha)
    train_started_at = perf_counter()
    model.fit(np.vstack(split.train["embedding"].to_numpy()), np.log1p(split.train["price"].to_numpy()))
    logger.info("Fitted Ridge model in %.2fs", perf_counter() - train_started_at)
    progress.set_postfix(step="trained ridge")
    progress.update(1)

    prediction_cap = float(split.train["price"].quantile(0.995) * 1.25)
    validation_pred = _predict_frame(model, split.validation, prediction_cap=prediction_cap)
    progress.set_postfix(step="predicted validation")
    progress.update(1)

    test_pred = _predict_frame(model, split.test, prediction_cap=prediction_cap)
    progress.set_postfix(step="predicted test")
    progress.update(1)
    metrics = {
        "validation": _metric_dict(split.validation["price"].to_numpy(), validation_pred),
        "test": _metric_dict(split.test["price"].to_numpy(), test_pred),
    }

    safe_label = model_label.replace("/", "_").replace("\\", "_")
    model_path = models_dir() / f"{safe_label}_ridge.joblib"
    metadata_file = models_dir() / f"{safe_label}_metadata.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    prediction_dir = models_dir() / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    validation_prediction_path = prediction_dir / f"validation_{safe_label}.csv"
    test_prediction_path = prediction_dir / f"test_{safe_label}.csv"
    _write_predictions(validation_prediction_path, split.validation, validation_pred)
    _write_predictions(test_prediction_path, split.test, test_pred)

    metadata = {
        "model_version": f"{safe_label}_v1",
        "model_type": "transformer_embedding_ridge_log_price",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(model_path),
        "embedding_cache_path": str(embedding_cache_path),
        "embedding_model_name": embedding_model_name,
        "features": ["catalog_content_transformer_embedding"],
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1",
        "prediction_cap": prediction_cap,
        "row_counts": {
            "raw": int(len(raw)),
            "clean": int(len(clean)),
            "modeled": int(len(frame)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "params": {
            "alpha": alpha,
            "random_state": random_state,
        },
        "metrics": metrics,
        "prediction_files": {
            "validation": str(validation_prediction_path),
            "test": str(test_prediction_path),
        },
    }
    write_json(metadata_file, metadata)
    write_metrics_report(reports_dir() / f"{safe_label}_metrics.md", f"{model_label} Metrics", metadata)
    progress.set_postfix(step="wrote artifacts")
    progress.update(1)
    progress.close()
    logger.info("Saved transformer baseline artifact to %s", model_path)
    logger.info("Total transformer baseline train time: %.2fs", perf_counter() - started_at)
    return metadata


def _attach_embeddings(
    frame: pd.DataFrame,
    cache_sample_ids: np.ndarray,
    embeddings: np.ndarray,
) -> pd.DataFrame:
    if len(cache_sample_ids) != len(embeddings):
        raise ValueError("Embedding cache sample_id and embedding lengths do not match")
    embedding_frame = pd.DataFrame(
        {
            "sample_id": cache_sample_ids.astype(str),
            "embedding": list(embeddings.astype(np.float32)),
        }
    )
    output = frame.copy()
    output["sample_id"] = output["sample_id"].astype(str)
    output = output.merge(embedding_frame, on="sample_id", how="inner")
    if output.empty:
        raise ValueError("No training rows matched the embedding cache")
    return output.reset_index(drop=True)


def _predict_frame(model: Ridge, frame: pd.DataFrame, *, prediction_cap: float) -> np.ndarray:
    y_pred = np.expm1(model.predict(np.vstack(frame["embedding"].to_numpy())))
    return np.clip(y_pred, a_min=0, a_max=prediction_cap)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def _write_predictions(path: Path, frame: pd.DataFrame, y_pred: np.ndarray) -> None:
    pd.DataFrame(
        {
            "sample_id": frame["sample_id"].to_numpy(),
            "actual_price": frame["price"].to_numpy(),
            "predicted_price": y_pred,
        }
    ).to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Ridge on cached transformer text embeddings.")
    parser.add_argument("--embedding-cache-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    metadata = train_transformer_baseline(
        embedding_cache_path=args.embedding_cache_path,
        model_label=args.model_label,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    print(
        "Saved "
        f"{metadata['model_version']} with validation SMAPE "
        f"{metadata['metrics']['validation']['smape']:.4f}"
    )


if __name__ == "__main__":
    main()
