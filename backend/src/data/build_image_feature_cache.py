import argparse
import logging
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from src.data.clean_data import clean_test_data, clean_train_data
from src.data.load_data import load_test_data, load_train_data
from src.features.image_features import IMAGE_FEATURE_COLUMNS, extract_image_features
from src.models.artifacts import reports_dir

logger = logging.getLogger(__name__)


def processed_dir() -> Path:
    return reports_dir().parent / "data" / "processed"


def build_image_feature_cache(*, split: str) -> Path:
    if split == "train":
        frame = clean_train_data(load_train_data())
    elif split == "test":
        frame = clean_test_data(load_test_data())
    else:
        raise ValueError("split must be 'train' or 'test'")

    output_path = processed_dir() / f"{split}_image_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    rows_iter = frame[["sample_id", "image_link"]].itertuples(index=False)
    progress = tqdm(rows_iter, total=len(frame), desc=f"Extracting {split} image features")
    for row in progress:
        features = extract_image_features(row.image_link)
        rows.append({"sample_id": str(row.sample_id), **features})

    output = pd.DataFrame(rows, columns=["sample_id", *IMAGE_FEATURE_COLUMNS])
    output.to_csv(output_path, index=False)
    logger.info("Wrote %s image feature rows to %s", len(output), output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached lightweight image features.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    path = build_image_feature_cache(split=parse_args().split)
    print(f"Wrote image feature cache to {path}")


if __name__ == "__main__":
    main()
