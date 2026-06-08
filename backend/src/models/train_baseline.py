import argparse
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np

from src.data.clean_data import clean_train_data
from src.data.load_data import load_train_data
from src.data.split_data import create_train_validation_test_split
from src.features.text_features import create_tfidf_ridge_pipeline
from src.models.artifacts import metadata_path, pipeline_path, reports_dir
from src.models.reporting import write_json, write_metrics_report
from src.utils.metrics import mae, r2, rmse, smape

logger = logging.getLogger(__name__)


def train_baseline(
    *,
    sample_size: int | None = None,
    max_features: int = 80_000,
    min_df: int = 2,
    ngram_max: int = 2,
    alpha: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    raw = load_train_data()
    clean = clean_train_data(raw)

    if sample_size is not None and sample_size < len(clean):
        clean = clean.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    split = create_train_validation_test_split(clean, random_state=random_state)

    median_price = float(split.train["price"].median())
    pipeline = create_tfidf_ridge_pipeline(
        max_features=max_features,
        min_df=min_df,
        ngram_max=ngram_max,
        alpha=alpha,
    )
    pipeline.fit(split.train["catalog_content"], np.log1p(split.train["price"].to_numpy()))

    metrics = {
        "median_validation": _evaluate_constant(split.validation["price"].to_numpy(), median_price),
        "validation": _evaluate_frame(pipeline, split.validation),
        "test": _evaluate_frame(pipeline, split.test),
    }

    model_path = pipeline_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_version": "text_tfidf_ridge_v1",
        "model_type": "tfidf_ridge_log_price",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact_path": str(model_path),
        "features": ["catalog_content"],
        "target_transform": "log1p",
        "prediction_inverse_transform": "expm1",
        "median_price": median_price,
        "row_counts": {
            "raw": int(len(raw)),
            "clean": int(len(clean)),
            "train": int(len(split.train)),
            "validation": int(len(split.validation)),
            "test": int(len(split.test)),
        },
        "params": {
            "sample_size": sample_size,
            "max_features": max_features,
            "min_df": min_df,
            "ngram_max": ngram_max,
            "alpha": alpha,
            "random_state": random_state,
        },
        "metrics": metrics,
        "top_positive_tokens": _top_tokens(pipeline, largest=True),
        "top_negative_tokens": _top_tokens(pipeline, largest=False),
    }

    write_json(metadata_path(), metadata)
    write_metrics_report(reports_dir() / "baseline_metrics.md", "Baseline Metrics", metadata)
    logger.info("Saved model artifact to %s", model_path)
    return metadata


def _evaluate_frame(pipeline: Any, frame: Any) -> dict[str, float]:
    y_true = frame["price"].to_numpy()
    y_pred = np.expm1(pipeline.predict(frame["catalog_content"]))
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return _metric_dict(y_true, y_pred)


def _evaluate_constant(y_true: np.ndarray, value: float) -> dict[str, float]:
    y_pred = np.full_like(y_true, fill_value=value, dtype=float)
    return _metric_dict(y_true, y_pred)


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def _top_tokens(pipeline: Any, *, largest: bool, limit: int = 50) -> list[str]:
    vectorizer = pipeline.named_steps["tfidf"]
    model = pipeline.named_steps["model"]
    feature_names = np.asarray(vectorizer.get_feature_names_out())
    coefficients = np.asarray(model.coef_)
    order = np.argsort(coefficients)
    selected = order[-limit:][::-1] if largest else order[:limit]
    return [str(token) for token in feature_names[selected]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PriceLens text baseline model.")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=80_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    metadata = train_baseline(
        sample_size=args.sample_size,
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
        alpha=args.alpha,
        random_state=args.random_state,
    )
    print(f"Saved {metadata['model_version']} with validation SMAPE {metadata['metrics']['validation']['smape']:.4f}")


if __name__ == "__main__":
    main()
