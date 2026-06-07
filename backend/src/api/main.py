from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    ConfidenceBand,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.services import PricePredictionService
from src.config import get_settings

settings = get_settings()
app = FastAPI(title=settings.project_name, version=settings.api_version)
service = PricePredictionService()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.project_name, version=settings.api_version)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_version=settings.model_version,
        model_type="mock_text_baseline",
        status="mock_until_artifacts_are_trained",
        supports_images=True,
        features=["catalog_content", "brand", "category", "image"],
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_json(payload: PredictionRequest) -> PredictionResponse:
    result = service.predict(
        catalog_content=payload.catalog_content,
        brand=payload.brand,
        category=payload.category,
    )
    return PredictionResponse(
        predicted_price=result.predicted_price,
        confidence_band=ConfidenceBand(low=result.low, high=result.high),
        model_version=settings.model_version,
        features_used=result.features_used,
        explanation_tokens=result.explanation_tokens,
        image_received=False,
    )


@app.post("/predict-with-image", response_model=PredictionResponse)
async def predict_with_image(
    catalog_content: Annotated[str, Form(min_length=1)],
    brand: Annotated[str | None, Form()] = None,
    category: Annotated[str | None, Form()] = None,
    image: Annotated[UploadFile | None, File()] = None,
) -> PredictionResponse:
    result = service.predict(
        catalog_content=catalog_content,
        brand=brand,
        category=category,
        image_filename=image.filename if image else None,
    )
    return PredictionResponse(
        predicted_price=result.predicted_price,
        confidence_band=ConfidenceBand(low=result.low, high=result.high),
        model_version=settings.model_version,
        features_used=result.features_used,
        explanation_tokens=result.explanation_tokens,
        image_received=image is not None,
    )
