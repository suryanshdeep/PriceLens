import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_test_data


def blend_submission_predictions(
    *,
    prediction_files: list[Path],
    weights: list[float],
    output_path: Path,
    prediction_column: str = "price",
) -> Path:
    if len(prediction_files) < 2:
        raise ValueError("At least two prediction files are required")
    if len(prediction_files) != len(weights):
        raise ValueError("Number of prediction files must match number of weights")

    test_ids = load_test_data()[["sample_id"]].copy()
    test_ids["sample_id"] = test_ids["sample_id"].astype(str)
    merged = test_ids

    for index, path in enumerate(prediction_files):
        predictions = pd.read_csv(path, dtype={"sample_id": str})
        if prediction_column not in predictions.columns:
            raise ValueError(f"{path} does not contain prediction column '{prediction_column}'")
        predictions = predictions[["sample_id", prediction_column]].rename(
            columns={prediction_column: f"prediction_{index}"}
        )
        merged = merged.merge(predictions, on="sample_id", how="left")

    prediction_columns = [f"prediction_{index}" for index in range(len(prediction_files))]
    if merged[prediction_columns].isna().any().any():
        missing_count = int(merged[prediction_columns].isna().any(axis=1).sum())
        raise ValueError(f"{missing_count} test rows are missing at least one model prediction")

    normalized_weights = _normalize_weights(weights)
    blended = merged[prediction_columns].to_numpy(dtype=float) @ np.asarray(normalized_weights)
    output = pd.DataFrame(
        {
            "sample_id": merged["sample_id"],
            "price": np.clip(blended, a_min=0, a_max=None),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path


def _normalize_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [float(weight) / total for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend test predictions into a submission CSV.")
    parser.add_argument("--prediction-file", action="append", type=Path, required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prediction-column", default="price")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = blend_submission_predictions(
        prediction_files=args.prediction_file,
        weights=args.weights,
        output_path=args.output,
        prediction_column=args.prediction_column,
    )
    print(f"Wrote blended submission to {path}")


if __name__ == "__main__":
    main()
