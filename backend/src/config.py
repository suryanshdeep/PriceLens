from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "PriceLens"
    api_version: str = "0.1.0"
    model_version: str = "mock_baseline_v0"
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    root_dir: Path = Path(__file__).resolve().parents[2]
    artifacts_dir: Path = root_dir / "models"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PRICELENS_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
