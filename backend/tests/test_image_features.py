from pathlib import Path

import pandas as pd
from PIL import Image

from src.features.image_features import add_image_features, extract_image_features, resolve_image_path


def test_resolve_image_path_uses_filename_from_url(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (12, 6), color=(128, 128, 128)).save(image_path)

    resolved = resolve_image_path("https://example.com/images/sample.jpg", base_dir=tmp_path)

    assert resolved == image_path


def test_extract_image_features_returns_dimensions_and_brightness(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (12, 6), color=(128, 128, 128)).save(image_path)

    features = extract_image_features("sample.jpg", base_dir=tmp_path)

    assert features["image_exists"] == 1.0
    assert features["image_width"] == 12.0
    assert features["image_height"] == 6.0
    assert features["image_aspect_ratio"] == 2.0
    assert features["image_brightness_mean"] > 0


def test_add_image_features_adds_columns(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (10, 10), color=(255, 255, 255)).save(image_path)
    frame = pd.DataFrame({"image_link": ["sample.jpg", "missing.jpg"]})

    output = add_image_features(frame, base_dir=tmp_path)

    assert output.loc[0, "image_exists"] == 1.0
    assert output.loc[1, "image_exists"] == 0.0
