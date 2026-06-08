import pandas as pd

from src.data.clean_data import clean_catalog_content, clean_price, clean_train_data


def test_clean_catalog_content_normalizes_text() -> None:
    assert clean_catalog_content("  Organic\u00a0Butter\n16 OZ  ") == "organic butter 16 oz"


def test_clean_price_parses_positive_values() -> None:
    assert clean_price("$1,299.50") == 1299.50
    assert clean_price("0") is None
    assert clean_price("not available") is None


def test_clean_train_data_filters_invalid_rows_and_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "1",
                "catalog_content": "Product A",
                "image_link": "a.jpg",
                "price": "10.50",
            },
            {
                "sample_id": "1",
                "catalog_content": "Duplicate Product A",
                "image_link": "a2.jpg",
                "price": "11.00",
            },
            {
                "sample_id": "2",
                "catalog_content": "",
                "image_link": "b.jpg",
                "price": "9.00",
            },
            {
                "sample_id": "3",
                "catalog_content": "Invalid price",
                "image_link": "c.jpg",
                "price": "-1",
            },
        ]
    )

    clean = clean_train_data(frame)

    assert len(clean) == 1
    assert clean.loc[0, "sample_id"] == "1"
    assert clean.loc[0, "catalog_content"] == "product a"
    assert clean.loc[0, "price"] == 10.50
