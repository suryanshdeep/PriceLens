import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.data.clean_data import clean_catalog_content
from src.models.artifacts import metadata_path, pipeline_path
from src.models.reporting import read_json


@dataclass(frozen=True)
class ModelPrediction:
    predicted_price: float
    low: float
    high: float
    explanation_tokens: list[str]


class BaselinePricePredictor:
    def __init__(self, pipeline: Any, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata

    @classmethod
    def from_artifacts(
        cls,
        *,
        model_path: Path | None = None,
        model_metadata_path: Path | None = None,
    ) -> "BaselinePricePredictor":
        resolved_model_path = model_path or pipeline_path()
        resolved_metadata_path = model_metadata_path or metadata_path()
        if not resolved_model_path.exists() or not resolved_metadata_path.exists():
            raise FileNotFoundError("Baseline model artifacts are not available")
        return cls(
            pipeline=joblib.load(resolved_model_path),
            metadata=read_json(resolved_metadata_path),
        )

    def predict(self, catalog_content: str) -> ModelPrediction:
        cleaned_text = clean_catalog_content(catalog_content)
        predicted_log_price = float(self.pipeline.predict([cleaned_text])[0])
        predicted_price = max(float(np.expm1(predicted_log_price)), 0.0)

        validation_smape = (
            self.metadata.get("metrics", {}).get("validation", {}).get("smape", 20.0)
        )
        spread = max(1.0, predicted_price * float(validation_smape) / 100.0)

        return ModelPrediction(
            predicted_price=round(predicted_price, 2),
            low=round(max(predicted_price - spread, 0.0), 2),
            high=round(predicted_price + spread, 2),
            explanation_tokens=self.explain(cleaned_text),
        )

    def explain(self, cleaned_text: str, *, limit: int = 8) -> list[str]:
        vectorizer = self.pipeline.named_steps["tfidf"]
        model = self.pipeline.named_steps["model"]
        matrix = vectorizer.transform([cleaned_text])
        if matrix.nnz == 0:
            return []

        indices = matrix.indices
        values = matrix.data
        coefficients = np.asarray(model.coef_)[indices]
        contributions = values * coefficients
        order = np.argsort(np.abs(contributions))[::-1][:limit]
        feature_names = np.asarray(vectorizer.get_feature_names_out())
        return [str(feature_names[indices[position]]) for position in order]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a price with the trained baseline model.")
    parser.add_argument("catalog_content", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = BaselinePricePredictor.from_artifacts()
    prediction = predictor.predict(args.catalog_content)
    print(
        {
            "predicted_price": prediction.predicted_price,
            "low": prediction.low,
            "high": prediction.high,
            "explanation_tokens": prediction.explanation_tokens,
        }
    )


if __name__ == "__main__":
    main()
