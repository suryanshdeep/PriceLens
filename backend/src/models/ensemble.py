import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.artifacts import models_dir, reports_dir
from src.utils.metrics import mae, r2, rmse, smape


@dataclass(frozen=True)
class EnsembleResult:
    weights: dict[str, float]
    metrics: dict[str, float]
    predictions: pd.DataFrame


def blend_prediction_files(
    prediction_files: list[Path],
    *,
    weights: list[float] | None = None,
) -> EnsembleResult:
    if len(prediction_files) < 2:
        raise ValueError("At least two prediction files are required for an ensemble")

    resolved_weights = _normalize_weights(weights or [1.0] * len(prediction_files))
    merged = _load_and_merge_predictions(prediction_files)
    prediction_columns = [f"predicted_price_{index}" for index in range(len(prediction_files))]
    prediction_matrix = merged[prediction_columns].to_numpy(dtype=float)
    blended = prediction_matrix @ np.asarray(resolved_weights)
    merged["predicted_price"] = np.clip(blended, a_min=0, a_max=None)

    y_true = merged["actual_price"].to_numpy(dtype=float)
    y_pred = merged["predicted_price"].to_numpy(dtype=float)
    metrics = {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }
    return EnsembleResult(
        weights={path.stem: weight for path, weight in zip(prediction_files, resolved_weights, strict=True)},
        metrics=metrics,
        predictions=merged[["sample_id", "actual_price", "predicted_price"]],
    )


def write_ensemble_outputs(
    result: EnsembleResult,
    *,
    name: str,
    split: str,
    output_dir: Path | None = None,
    report_dir: Path | None = None,
) -> tuple[Path, Path]:
    prediction_dir = output_dir or models_dir() / "predictions"
    resolved_report_dir = report_dir or reports_dir()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    resolved_report_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{split}_{name}.csv"
    report_path = resolved_report_dir / f"{name}_{split}_metrics.md"

    result.predictions.to_csv(prediction_path, index=False)
    lines = [
        f"# {name} {split.title()} Metrics",
        "",
        "## Weights",
        "",
        *[f"- `{model}`: {weight:.4f}" for model, weight in result.weights.items()],
        "",
        "## Metrics",
        "",
        *[f"- {metric.upper()}: {value:.4f}" for metric, value in result.metrics.items()],
        "",
        f"Prediction file: `{prediction_path}`",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return prediction_path, report_path


def default_prediction_files(split: str) -> list[Path]:
    prediction_dir = models_dir() / "predictions"
    return [
        prediction_dir / f"{split}_text_structured_ridge.csv",
        prediction_dir / f"{split}_text_structured_image_ridge.csv",
    ]


def _load_and_merge_predictions(prediction_files: list[Path]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for index, path in enumerate(prediction_files):
        frame = pd.read_csv(path, dtype={"sample_id": str})
        required_columns = {"sample_id", "actual_price", "predicted_price"}
        missing = required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        frame = frame[["sample_id", "actual_price", "predicted_price"]].rename(
            columns={"predicted_price": f"predicted_price_{index}"}
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["sample_id", "actual_price"], how="inner")

    if merged is None or merged.empty:
        raise ValueError("No overlapping prediction rows found")
    return merged


def _normalize_weights(weights: list[float]) -> list[float]:
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    return [float(weight) / total for weight in weights]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blend PriceLens prediction CSV files.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--name", default="ensemble_equal_weight")
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    parser.add_argument("--prediction-file", action="append", type=Path, dest="prediction_files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.prediction_files or default_prediction_files(args.split)
    result = blend_prediction_files(files, weights=args.weights)
    prediction_path, report_path = write_ensemble_outputs(
        result,
        name=args.name,
        split=args.split,
    )
    print(result.metrics)
    print(f"Wrote ensemble predictions to {prediction_path}")
    print(f"Wrote ensemble report to {report_path}")


if __name__ == "__main__":
    main()
