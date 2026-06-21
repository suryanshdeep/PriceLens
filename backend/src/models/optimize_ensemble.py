import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from src.models.artifacts import models_dir, reports_dir
from src.models.ensemble import (
    EnsembleResult,
    blend_prediction_files,
    default_prediction_files,
    write_ensemble_outputs,
)
from src.models.reporting import write_json


def optimize_ensemble_weights(
    prediction_files: list[Path],
    *,
    step: float = 0.01,
    random_trials: int = 5000,
    random_state: int = 42,
) -> EnsembleResult:
    if len(prediction_files) == 2:
        candidates = _two_model_grid(step)
    else:
        candidates = _random_weight_candidates(
            model_count=len(prediction_files),
            trials=random_trials,
            random_state=random_state,
        )

    best_result: EnsembleResult | None = None
    for weights in candidates:
        result = blend_prediction_files(prediction_files, weights=list(weights))
        if best_result is None or result.metrics["smape"] < best_result.metrics["smape"]:
            best_result = result

    if best_result is None:
        raise ValueError("No ensemble candidates were evaluated")
    return best_result


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize ensemble weights on prediction CSVs.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
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
