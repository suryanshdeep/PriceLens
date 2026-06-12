import argparse
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.data.clean_data import clean_train_data
from src.data.load_data import load_train_data
from src.data.split_data import create_train_validation_test_split
from src.config import get_settings
from src.features.image_features import IMAGE_FEATURE_COLUMNS, add_image_features
from src.features.text_image_features import create_text_structured_image_ridge_pipeline
from src.models.artifacts import models_dir, reports_dir
from src.models.reporting import write_json, write_metrics_report
from src.utils.metrics import mae, r2, rmse, smape

logger = logging.getLogger(__name__)


def train_image_baseline(
    *,
    sample_size: int | None = None,
    max_features: int = 100_000,
    min_df: int = 2,
    ngram_max: int = 2,
    alpha: float = 1.0,
    random_state: int = 42,
    use_cache: bool = True,
) -> dict[str, Any]:
    raw = load_train_data()
    clean = clean_train_data(raw)
    clean_row_count = len(clean)
    if sample_size is not None and sample_size < len(clean):
        clean = clean.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    clean = _attach_image_features(clean, use_cache=use_cache)

    split = create_train_validation_test_split(clean, random_state=random_state)
    pipeline = create_text_structured_image_ridge_pipeline(
        max_features=max_features,
        min_df=min_df,
        ngram_max=ngram_max,
        alpha=alpha,
    )
    training_columns = ["catalog_content", *IMAGE_FEATURE_COLUMNS]
    pipeline.fit(split.train[training_columns], np.log1p(split.train["price"].to_numpy()))

    prediction_cap = float(split.train["price"].quantile(0.995) * 1.25)
    validation_pred = _predict_frame(
        pipeline,
        split.validation,
        training_columns=training_columns,
        prediction_cap=prediction_cap,
    )
    test_pred = _predict_frame(
        pipeline,
        split.test,
        training_columns=training_columns,
        prediction_cap=prediction_cap,
    )
    metrics = {
        "validation": _metric_dict(split.validation["price"].to_numpy(), validation_pred),
        "test": _metric_dict(split.test["price"].to_numpy(), test_pred),
    }

    model_path = models_dir() / "text_structured_image_ridge_pipeline.joblib"
    metadata_file = models_dir() / "text_structured_image_ridge_metadata.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    prediction_dir = models_dir() / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    _write_predictions(
        prediction_dir / "validation_text_structured_image_ridge.csv",
        split.validation,
        validation_pred,
    )
    _write_predictions(
        prediction_dir / "test_text_structured_image_ridge.csv",
        split.test,
        test_pred,
    )

    metadata = {
        "model_version": "text_structured_image_ridge_v1",
        "model_type": "tfidf_structured_image_ridge_log_price",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(model_path),
        "features": ["catalog_content", "catalog_structured_features", *IMAGE_FEATURE_COLUMNS],
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1",
        "prediction_cap": prediction_cap,
        "row_counts": {
            "raw": int(len(raw)),
            "clean": int(clean_row_count),
            "modeled": int(len(clean)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "image_feature_coverage": {
            "rows_with_image": int(clean["image_exists"].sum()),
            "coverage": float(clean["image_exists"].mean()),
        },
        "params": {
            "sample_size": sample_size,
            "max_features": max_features,
            "min_df": min_df,
            "ngram_max": ngram_max,
            "alpha": alpha,
            "random_state": random_state,
            "use_cache": use_cache,
        },
        "metrics": metrics,
        "prediction_files": {
            "validation": str(prediction_dir / "validation_text_structured_image_ridge.csv"),
            "test": str(prediction_dir / "test_text_structured_image_ridge.csv"),
        },
    }
    write_json(metadata_file, metadata)
    write_metrics_report(reports_dir() / "image_baseline_metrics.md", "Image Baseline Metrics", metadata)
    logger.info("Saved image-aware model artifact to %s", model_path)
    return metadata


def _attach_image_features(frame: pd.DataFrame, *, use_cache: bool) -> pd.DataFrame:
    cache_path = get_settings().root_dir / "data" / "processed" / "train_image_features.csv"
    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype={"sample_id": str})
        return _merge_cached_image_features(frame, cached)
    return add_image_features(frame)


def _merge_cached_image_features(frame: pd.DataFrame, cached: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    cached = cached.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    cached["sample_id"] = cached["sample_id"].astype(str)
    merged = frame.merge(cached, on="sample_id", how="left")
    for column in IMAGE_FEATURE_COLUMNS:
        merged[column] = merged[column].fillna(0.0)
    return merged


def _predict_frame(
    pipeline: Any,
    frame: pd.DataFrame,
    *,
    training_columns: list[str],
    prediction_cap: float,
) -> np.ndarray:
    y_pred = np.expm1(pipeline.predict(frame[training_columns]))
    return np.clip(y_pred, a_min=0, a_max=prediction_cap)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def _write_predictions(path: Any, frame: pd.DataFrame, y_pred: np.ndarray) -> None:
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].to_numpy(),
            "actual_price": frame["price"].to_numpy(),
            "predicted_price": y_pred,
        }
    )
    output.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train text + structured + image metadata baseline.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    metadata = train_image_baseline(
        sample_size=args.sample_size,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
        alpha=args.alpha,
        random_state=args.random_state,
        use_cache=not args.no_cache,
    )
    print(
        "Saved "
        f"{metadata['model_version']} with validation SMAPE "
        f"{metadata['metrics']['validation']['smape']:.4f}"
    )


if __name__ == "__main__":
    main()
