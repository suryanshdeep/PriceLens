import re

import pandas as pd

TEXT_NOISE_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+")
WHITESPACE_PATTERN = re.compile(r"\s+")
PRICE_PATTERN = re.compile(r"[-+]?\d*\.?\d+")


def clean_catalog_content(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = TEXT_NOISE_PATTERN.sub(" ", text)
    text = text.replace("\u00a0", " ")
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip().lower()


def clean_price(value: object) -> float | None:
    if pd.isna(value):
        return None
    if isinstance(value, int | float):
        price = float(value)
    else:
        text = str(value).replace(",", "")
        match = PRICE_PATTERN.search(text)
        if match is None:
            return None
        price = float(match.group(0))
    return price if price > 0 else None


def clean_train_data(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean["sample_id"] = clean["sample_id"].astype(str).str.strip()
    clean["catalog_content"] = clean["catalog_content"].map(clean_catalog_content)
    clean["image_link"] = clean["image_link"].fillna("").astype(str).str.strip()
    clean["price"] = clean["price"].map(clean_price)

    clean = clean.dropna(subset=["price"])
    clean = clean[(clean["sample_id"] != "") & (clean["catalog_content"] != "")]
    clean = clean.drop_duplicates(subset=["sample_id"], keep="first")
    clean["price"] = clean["price"].astype(float)
    return clean.reset_index(drop=True)


def clean_test_data(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean["sample_id"] = clean["sample_id"].astype(str).str.strip()
    clean["catalog_content"] = clean["catalog_content"].map(clean_catalog_content)
    clean["image_link"] = clean["image_link"].fillna("").astype(str).str.strip()
    clean = clean[(clean["sample_id"] != "") & (clean["catalog_content"] != "")]
    clean = clean.drop_duplicates(subset=["sample_id"], keep="first")
    return clean.reset_index(drop=True)
