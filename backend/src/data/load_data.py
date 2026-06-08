from pathlib import Path

import pandas as pd

from src.config import get_settings

TRAIN_COLUMNS = {"sample_id", "catalog_content", "image_link", "price"}
TEST_COLUMNS = {"sample_id", "catalog_content", "image_link"}


def data_raw_dir() -> Path:
    return get_settings().root_dir / "data" / "raw"


def load_train_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or data_raw_dir() / "train.csv"
    frame = _load_csv(csv_path)
    _validate_columns(frame, TRAIN_COLUMNS, csv_path)
    return frame


def load_test_data(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or data_raw_dir() / "test.csv"
    frame = _load_csv(csv_path)
    _validate_columns(frame, TEST_COLUMNS, csv_path)
    return frame


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def _validate_columns(frame: pd.DataFrame, required_columns: set[str], path: Path) -> None:
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")
