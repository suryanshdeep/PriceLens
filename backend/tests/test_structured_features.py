import pandas as pd

from src.features.structured_features import CatalogStructuredFeatures, extract_catalog_features


def test_extract_catalog_features_detects_pack_and_size() -> None:
    features = extract_catalog_features("Organic almond butter pack of 12 16 oz jars")

    assert features[4] == 12.0
    assert features[5] == 16.0
    assert features[6] == 1.0
    assert features[7] == 1.0
    assert features[8] >= 1.0


def test_catalog_structured_features_transform_shape() -> None:
    transformer = CatalogStructuredFeatures()
    transformed = transformer.fit_transform(pd.Series(["16 oz jar", "pack of 6 bottles"]))

    assert transformed.shape == (2, 10)
