import argparse
import logging
from typing import Any

import numpy as np

from src.data.clean_data import clean_train_data
from src.data.load_data import load_train_data
from src.data.split_data import create_train_validation_test_split
from src.models.artifacts import metadata_path, pipeline_path, reports_dir
from src.models.reporting import read_json, write_metrics_report
from src.utils.metrics import mae, r2, rmse, smape

logger = logging.getLogger(__name__)


def evaluate_baseline(*, random_state: int | None = None) -> dict[str, Any]:
    metadata = read_json(metadata_path())
    pipeline = _load_pipeline()
    params = metadata.get("params", {})
    split_random_state = random_state if random_state is not None else int(params.get("random_state", 42))

    clean = clean_train_data(load_train_data())
    sample_size = params.get("sample_size")
    if sample_size is not None and sample_size < len(clean):
        clean = clean.sample(n=int(sample_size), random_state=split_random_state).reset_index(drop=True)

    split = create_train_validation_test_split(clean, random_state=split_random_state)
    metadata["metrics"] = {
        "validation": _evaluate_frame(pipeline, split.validation),
        "test": _evaluate_frame(pipeline, split.test),
    }
    write_metrics_report(reports_dir() / "evaluation_metrics.md", "Evaluation Metrics", metadata)
    return metadata


def _load_pipeline() -> Any:
    import joblib

    path = pipeline_path()
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def _evaluate_frame(pipeline: Any, frame: Any) -> dict[str, float]:
    y_true = frame["price"].to_numpy()
    y_pred = np.expm1(pipeline.predict(frame["catalog_content"]))
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved PriceLens baseline model.")
    parser.add_argument("--random-state", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    metadata = evaluate_baseline(random_state=args.random_state)
    print(f"Evaluation SMAPE: {metadata['metrics']['test']['smape']:.4f}")


if __name__ == "__main__":
    main()
