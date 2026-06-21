from pathlib import Path

import pandas as pd

from src.models.optimize_ensemble import optimize_ensemble_weights


def test_optimize_ensemble_weights_prefers_better_model(tmp_path: Path) -> None:
    better = tmp_path / "better.csv"
    worse = tmp_path / "worse.csv"
    pd.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "actual_price": [10.0, 20.0, 30.0],
            "predicted_price": [10.0, 20.0, 30.0],
        }
    ).to_csv(better, index=False)
    pd.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "actual_price": [10.0, 20.0, 30.0],
            "predicted_price": [20.0, 30.0, 40.0],
        }
    ).to_csv(worse, index=False)

    result = optimize_ensemble_weights([better, worse], step=0.25)

    assert result.weights["better"] == 1.0
    assert result.weights["worse"] == 0.0
    assert result.metrics["smape"] == 0.0
