import argparse
from pathlib import Path

import pandas as pd

from src.data.load_data import load_train_data
from src.models.artifacts import models_dir


def import_train_predictions(
    *,
    prediction_path: Path,
    model_label: str,
    prediction_column: str = "price",
    output_path: Path | None = None,
) -> Path:
    actuals = load_train_data()[["sample_id", "price"]].rename(columns={"price": "actual_price"})
    actuals["sample_id"] = actuals["sample_id"].astype(str)

    predictions = pd.read_csv(prediction_path, dtype={"sample_id": str})
    if prediction_column not in predictions.columns:
        raise ValueError(f"{prediction_path} does not contain prediction column '{prediction_column}'")
    predictions = predictions[["sample_id", prediction_column]].rename(
        columns={prediction_column: "predicted_price"}
    )

    merged = actuals.merge(predictions, on="sample_id", how="inner")
    if merged.empty:
        raise ValueError(f"No train rows matched predictions from {prediction_path}")

    path = output_path or models_dir() / "predictions" / f"train_{model_label}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external train predictions for ensembling.")
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prediction-column", default="price")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = import_train_predictions(
        prediction_path=args.prediction_path,
        model_label=args.model_label,
        prediction_column=args.prediction_column,
        output_path=args.output_path,
    )
    print(f"Wrote standardized predictions to {path}")


if __name__ == "__main__":
    main()
