import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from src.models.artifacts import models_dir, reports_dir
from src.models.ensemble import (
    EnsembleResult,
    default_prediction_files,
    _load_and_merge_predictions,
    write_ensemble_outputs,
)
from src.models.reporting import write_json
from src.utils.metrics import mae, r2, rmse, smape


def optimize_ensemble_weights(
    prediction_files: list[Path],
    *,
    step: float = 0.01,
    random_trials: int = 5000,
    random_state: int = 42,
) -> EnsembleResult:
    merged = _load_and_merge_predictions(prediction_files)
    prediction_columns = [f"predicted_price_{index}" for index in range(len(prediction_files))]
    prediction_matrix = merged[prediction_columns].to_numpy(dtype=float)
    y_true = merged["actual_price"].to_numpy(dtype=float)
    candidates = (
        _simplex_grid(model_count=len(prediction_files), step=step)
        if len(prediction_files) <= 3
        else _random_weight_candidates(
            model_count=len(prediction_files),
            trials=random_trials,
            random_state=random_state,
        )
    )

    best_weights: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None
    for weights in candidates:
        y_pred = np.clip(prediction_matrix @ np.asarray(weights), a_min=0, a_max=None)
        metrics = _metric_dict(y_true, y_pred)
        if best_metrics is None or metrics["smape"] < best_metrics["smape"]:
            best_metrics = metrics
            best_weights = np.asarray(weights)

    if best_weights is None or best_metrics is None:
        raise ValueError("No ensemble candidates were evaluated")

    merged["predicted_price"] = np.clip(prediction_matrix @ best_weights, a_min=0, a_max=None)
    return EnsembleResult(
        weights={
            path.stem: float(weight)
            for path, weight in zip(prediction_files, best_weights, strict=True)
        },
        metrics=best_metrics,
        predictions=merged[["sample_id", "actual_price", "predicted_price"]],
    )


def save_optimized_weights(
    result: EnsembleResult,
    *,
    name: str,
    split: str,
) -> Path:
    path = models_dir() / f"{name}_{split}_weights.json"
    payload = {
        "name": name,
        "split": split,
        "weights": result.weights,
        "metrics": result.metrics,
    }
    write_json(path, payload)
    return path


def _two_model_grid(step: float) -> Iterable[tuple[float, float]]:
    if step <= 0 or step > 1:
        raise ValueError("step must be in the range (0, 1]")
    count = int(round(1 / step))
    for index in range(count + 1):
        first = min(index * step, 1.0)
        yield (first, 1.0 - first)


def _simplex_grid(*, model_count: int, step: float) -> Iterable[np.ndarray]:
    if model_count == 2:
        yield from (np.asarray(weights) for weights in _two_model_grid(step))
        return
    if model_count != 3:
        raise ValueError("Grid search only supports two or three models")
    if step <= 0 or step > 1:
        raise ValueError("step must be in the range (0, 1]")
    count = int(round(1 / step))
    for first_index in range(count + 1):
        first = min(first_index * step, 1.0)
        for second_index in range(count + 1 - first_index):
            second = min(second_index * step, 1.0 - first)
            third = 1.0 - first - second
            yield np.asarray([first, second, third])


def _random_weight_candidates(
    *,
    model_count: int,
    trials: int,
    random_state: int,
) -> Iterable[np.ndarray]:
    if model_count < 2:
        raise ValueError("At least two models are required")
    rng = np.random.default_rng(random_state)
    yield np.full(model_count, 1 / model_count)
    for _ in range(trials):
        yield rng.dirichlet(np.ones(model_count))


def _metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "smape": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize ensemble weights on prediction CSVs.")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--name", default="ensemble_optimized")
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--random-trials", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--prediction-file", action="append", type=Path, dest="prediction_files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = args.prediction_files or default_prediction_files(args.split)
    result = optimize_ensemble_weights(
        files,
        step=args.step,
        random_trials=args.random_trials,
        random_state=args.random_state,
    )
    weights_path = save_optimized_weights(result, name=args.name, split=args.split)
    prediction_path, report_path = write_ensemble_outputs(
        result,
        name=args.name,
        split=args.split,
    )
    print(result.weights)
    print(result.metrics)
    print(f"Wrote optimized weights to {weights_path}")
    print(f"Wrote ensemble predictions to {prediction_path}")
    print(f"Wrote ensemble report to {report_path}")


if __name__ == "__main__":
    main()
