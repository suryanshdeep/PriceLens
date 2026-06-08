import logging
import re
from dataclasses import dataclass
from typing import Any

from src.config import get_settings
from src.models.predict import BaselinePricePredictor

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+-]{2,}")


@dataclass(frozen=True)
class PredictionResult:
    predicted_price: float
    low: float
    high: float
    features_used: list[str]
    explanation_tokens: list[str]
    model_version: str
    model_type: str
    is_mock: bool


class PricePredictionService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.predictor = self._load_predictor()

    def model_info(self) -> dict[str, Any]:
        if self.predictor is None:
            return {
                "model_version": self.settings.model_version,
                "model_type": "mock_text_baseline",
                "status": "mock_until_artifacts_are_trained",
                "supports_images": True,
                "features": ["catalog_content", "brand", "category", "image"],
                "artifact_available": False,
                "metrics": None,
            }

        metadata = self.predictor.metadata
        return {
            "model_version": str(metadata.get("model_version", "text_tfidf_ridge_v1")),
            "model_type": str(metadata.get("model_type", "tfidf_ridge_log_price")),
            "status": "trained_artifact_loaded",
            "supports_images": True,
            "features": list(metadata.get("features", ["catalog_content"])),
            "artifact_available": True,
            "metrics": metadata.get("metrics", {}).get("validation"),
        }

    def predict(
        self,
        *,
        catalog_content: str,
        brand: str | None = None,
        category: str | None = None,
        image_filename: str | None = None,
    ) -> PredictionResult:
        if self.predictor is not None:
            prediction = self.predictor.predict(catalog_content)
            logger.info(
                "Model prediction generated",
                extra={
                    "model_version": self.predictor.metadata.get("model_version"),
                    "features_used": self.predictor.metadata.get("features", ["catalog_content"]),
                    "explanation_tokens": prediction.explanation_tokens,
                    "image_filename": image_filename,
                },
            )
            features_used = list(self.predictor.metadata.get("features", ["catalog_content"]))
            if image_filename:
                features_used.append("image_received_not_modeled")
            return PredictionResult(
                predicted_price=prediction.predicted_price,
                low=prediction.low,
                high=prediction.high,
                features_used=features_used,
                explanation_tokens=prediction.explanation_tokens,
                model_version=str(self.predictor.metadata.get("model_version", "text_tfidf_ridge_v1")),
                model_type=str(self.predictor.metadata.get("model_type", "tfidf_ridge_log_price")),
                is_mock=False,
            )

        tokens = self._explanation_tokens(catalog_content)
        token_signal = min(len(tokens), 18)
        brand_signal = 2.0 if brand else 0.0
        category_signal = 1.5 if category else 0.0
        image_signal = 1.25 if image_filename else 0.0

        predicted_price = round(7.5 + token_signal * 0.85 + brand_signal + category_signal + image_signal, 2)
        spread = max(2.0, predicted_price * 0.16)

        features_used = ["catalog_content"]
        if brand:
            features_used.append("brand")
        if category:
            features_used.append("category")
        if image_filename:
            features_used.append("image")

        logger.info(
            "Mock prediction generated",
            extra={
                "model_version": self.settings.model_version,
                "features_used": features_used,
                "explanation_tokens": tokens,
                "image_filename": image_filename,
            },
        )

        return PredictionResult(
            predicted_price=predicted_price,
            low=round(predicted_price - spread, 2),
            high=round(predicted_price + spread, 2),
            features_used=features_used,
            explanation_tokens=tokens,
            model_version=self.settings.model_version,
            model_type="mock_text_baseline",
            is_mock=True,
        )

    @staticmethod
    def _explanation_tokens(catalog_content: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(catalog_content.lower())
        ignored = {"the", "and", "for", "with", "from", "pack", "product"}
        unique_tokens = []
        for token in tokens:
            if token in ignored or token in unique_tokens:
                continue
            unique_tokens.append(token)
        return unique_tokens[:8]

    @staticmethod
    def _load_predictor() -> BaselinePricePredictor | None:
        try:
            return BaselinePricePredictor.from_artifacts()
        except FileNotFoundError:
            logger.info("Baseline model artifacts not found; using mock prediction fallback")
            return None
        except Exception:
            logger.exception("Could not load baseline model artifacts; using mock prediction fallback")
            return None
