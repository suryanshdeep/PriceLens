from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def create_train_validation_test_split(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> DatasetSplit:
    if validation_size <= 0 or test_size <= 0:
        raise ValueError("validation_size and test_size must be positive")
    if validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size must be less than 1")

    train_frame, holdout = train_test_split(
        frame,
        test_size=validation_size + test_size,
        random_state=random_state,
    )
    validation_fraction_of_holdout = validation_size / (validation_size + test_size)
    validation_frame, test_frame = train_test_split(
        holdout,
        test_size=1 - validation_fraction_of_holdout,
        random_state=random_state,
    )

    return DatasetSplit(
        train=train_frame.reset_index(drop=True),
        validation=validation_frame.reset_index(drop=True),
        test=test_frame.reset_index(drop=True),
    )
