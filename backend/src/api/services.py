import logging
import re
from dataclasses import dataclass

from src.config import get_settings

logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9+-]{2,}")


@dataclass(frozen=True)
class PredictionResult:
    predicted_price: float
    low: float
    high: float
    features_used: list[str]
    explanation_tokens: list[str]


class PricePredictionService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def predict(
        self,
        *,
        catalog_content: str,
        brand: str | None = None,
        category: str | None = None,
        image_filename: str | None = None,
    ) -> PredictionResult:
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
