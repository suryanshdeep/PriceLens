import pandas as pd

from src.data.split_data import create_train_validation_test_split


def test_create_train_validation_test_split_sizes() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": [str(index) for index in range(100)],
            "catalog_content": [f"product {index}" for index in range(100)],
            "image_link": [f"{index}.jpg" for index in range(100)],
            "price": [float(index + 1) for index in range(100)],
        }
    )

    split = create_train_validation_test_split(frame, validation_size=0.1, test_size=0.1)

    assert len(split.train) == 80
    assert len(split.validation) == 10
    assert len(split.test) == 10
