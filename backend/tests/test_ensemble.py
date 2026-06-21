from pathlib import Path

import pandas as pd

from src.models.ensemble import blend_prediction_files, write_ensemble_outputs


def test_blend_prediction_files_equal_weight(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "actual_price": [10.0, 20.0],
            "predicted_price": [8.0, 18.0],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "actual_price": [10.0, 20.0],
            "predicted_price": [12.0, 22.0],
        }
    ).to_csv(second, index=False)

    result = blend_prediction_files([first, second])

    assert result.predictions["predicted_price"].tolist() == [10.0, 20.0]
    assert result.metrics["smape"] == 0.0


def test_write_ensemble_outputs(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "actual_price": [10.0, 20.0],
            "predicted_price": [8.0, 18.0],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "sample_id": ["1", "2"],
            "actual_price": [10.0, 20.0],
            "predicted_price": [12.0, 22.0],
        }
    ).to_csv(second, index=False)
    result = blend_prediction_files([first, second], weights=[0.25, 0.75])

    prediction_path, report_path = write_ensemble_outputs(
        result,
        name="test_ensemble",
        split="validation",
        output_dir=tmp_path,
        report_dir=tmp_path,
    )

    assert prediction_path.exists()
    assert report_path.exists()
    assert "SMAPE" in report_path.read_text(encoding="utf-8")
