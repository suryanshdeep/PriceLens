from pathlib import Path

import pandas as pd

from src.models.import_predictions import import_train_predictions


def test_import_train_predictions_joins_actual_prices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prediction_path = tmp_path / "external.csv"
    pd.DataFrame({"sample_id": ["1", "2"], "price": [9.0, 21.0]}).to_csv(
        prediction_path,
        index=False,
    )

    def fake_load_train_data() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": ["1", "2", "3"],
                "catalog_content": ["a", "b", "c"],
                "image_link": ["a.jpg", "b.jpg", "c.jpg"],
                "price": [10.0, 20.0, 30.0],
            }
        )

    monkeypatch.setattr("src.models.import_predictions.load_train_data", fake_load_train_data)

    output_path = import_train_predictions(
        prediction_path=prediction_path,
        model_label="external_model",
        output_path=tmp_path / "standard.csv",
    )

    output = pd.read_csv(output_path)
    assert output.columns.tolist() == ["sample_id", "actual_price", "predicted_price"]
    assert output["actual_price"].tolist() == [10.0, 20.0]
    assert output["predicted_price"].tolist() == [9.0, 21.0]
