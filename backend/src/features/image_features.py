from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from PIL import Image, ImageStat

from src.config import get_settings

IMAGE_FEATURE_COLUMNS = [
    "image_exists",
    "image_width",
    "image_height",
    "image_aspect_ratio",
    "image_file_size_kb",
    "image_brightness_mean",
    "image_brightness_std",
]


def images_dir() -> Path:
    return get_settings().root_dir / "data" / "images"


def resolve_image_path(image_link: object, *, base_dir: Path | None = None) -> Path | None:
    if pd.isna(image_link):
        return None
    text = str(image_link).strip()
    if not text:
        return None

    parsed = urlparse(text)
    filename = Path(parsed.path).name if parsed.scheme else Path(text).name
    candidates = []
    raw_path = Path(text)
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.append((base_dir or images_dir()) / filename)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def extract_image_features(image_link: object, *, base_dir: Path | None = None) -> dict[str, float]:
    path = resolve_image_path(image_link, base_dir=base_dir)
    if path is None:
        return _empty_features()

    try:
        with Image.open(path) as image:
            width, height = image.size
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            brightness_mean = float(stat.mean[0])
            brightness_std = float(stat.stddev[0])
    except Exception:
        return _empty_features()

    file_size_kb = path.stat().st_size / 1024
    return {
        "image_exists": 1.0,
        "image_width": float(width),
        "image_height": float(height),
        "image_aspect_ratio": float(width / height) if height else 0.0,
        "image_file_size_kb": float(file_size_kb),
        "image_brightness_mean": brightness_mean,
        "image_brightness_std": brightness_std,
    }


def add_image_features(frame: pd.DataFrame, *, base_dir: Path | None = None) -> pd.DataFrame:
    output = frame.copy()
    features = output["image_link"].map(lambda value: extract_image_features(value, base_dir=base_dir))
    feature_frame = pd.DataFrame(features.tolist(), index=output.index)
    return pd.concat([output, feature_frame], axis=1)


def _empty_features() -> dict[str, float]:
    return {column: 0.0 for column in IMAGE_FEATURE_COLUMNS}
