import pandas as pd

from src.features.image_features import IMAGE_FEATURE_COLUMNS
from src.models.train_image_baseline import _merge_cached_image_features


def test_merge_cached_image_features_handles_string_and_integer_ids() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["101", "102"],
            "catalog_content": ["product one", "product two"],
            "image_link": ["one.jpg", "two.jpg"],
            "price": [10.0, 20.0],
        }
    )
    cached = pd.DataFrame(
        {
            "sample_id": [101, 102],
            **{column: [1.0, 2.0] for column in IMAGE_FEATURE_COLUMNS},
        }
    )

    merged = _merge_cached_image_features(frame, cached)

    assert merged.loc[0, "image_exists"] == 1.0
    assert merged.loc[1, "image_exists"] == 2.0
