from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    catalog_content: str = Field(..., min_length=1)
    brand: str | None = None
    category: str | None = None


class ConfidenceBand(BaseModel):
    low: float
    high: float


class PredictionResponse(BaseModel):
    predicted_price: float
    confidence_band: ConfidenceBand
    model_version: str
    features_used: list[str]
    explanation_tokens: list[str] = []
    image_received: bool = False


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    status: str
    supports_images: bool
    features: list[str]
    artifact_available: bool = False
    metrics: dict[str, float] | None = None


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
