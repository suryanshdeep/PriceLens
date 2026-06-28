from pathlib import Path

import pandas as pd

from src.models.blend_submission import blend_submission_predictions


def test_blend_submission_predictions_validates_and_writes_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({"sample_id": ["1", "2"], "price": [10.0, 20.0]}).to_csv(first, index=False)
    pd.DataFrame({"sample_id": ["1", "2"], "price": [20.0, 40.0]}).to_csv(second, index=False)

    def fake_load_test_data() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["1", "2"],
                "catalog_content": ["a", "b"],
                "image_link": ["a.jpg", "b.jpg"],
            }
        )

    monkeypatch.setattr("src.models.blend_submission.load_test_data", fake_load_test_data)

    output_path = blend_submission_predictions(
        prediction_files=[first, second],
        weights=[0.25, 0.75],
        output_path=tmp_path / "submission.csv",
    )

    output = pd.read_csv(output_path)
    assert output.columns.tolist() == ["sample_id", "price"]
    assert output["price"].tolist() == [17.5, 35.0]
