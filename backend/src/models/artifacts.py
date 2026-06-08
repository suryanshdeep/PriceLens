from pathlib import Path

from src.config import get_settings

PIPELINE_FILENAME = "text_baseline_pipeline.joblib"
METADATA_FILENAME = "baseline_metadata.json"


def models_dir() -> Path:
    return get_settings().root_dir / "models"


def reports_dir() -> Path:
    return get_settings().root_dir / "reports"


def pipeline_path() -> Path:
    return models_dir() / PIPELINE_FILENAME


def metadata_path() -> Path:
    return models_dir() / METADATA_FILENAME
