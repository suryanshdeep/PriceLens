import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")
PACK_PATTERNS = [
    re.compile(r"(?:pack|case|set|box|bundle)\s+of\s+(\d+(?:\.\d+)?)"),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:pack|packs|count|ct|pcs|pieces|piece|pk)\b"),
]
SIZE_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(oz|ounce|ounces|fl oz|ml|l|liter|litre|g|gram|grams|kg|lb|lbs|pound|pounds)\b"
)
PREMIUM_TERMS = {
    "organic",
    "natural",
    "premium",
    "gourmet",
    "stainless",
    "wireless",
    "refill",
    "professional",
    "heavy duty",
    "gluten free",
}


class CatalogStructuredFeatures(BaseEstimator, TransformerMixin):
    feature_names_: list[str]

    def fit(self, x: pd.Series, y: object = None) -> "CatalogStructuredFeatures":
        self.feature_names_ = [
            "char_count",
            "word_count",
            "number_count",
            "max_number",
            "pack_count",
            "size_value",
            "has_size",
            "has_pack",
            "premium_term_count",
            "unique_word_ratio",
        ]
        return self

    def transform(self, x: pd.Series) -> np.ndarray:
        values = [extract_catalog_features(text) for text in x]
        return np.asarray(values, dtype=float)

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)


def extract_catalog_features(text: object) -> list[float]:
    content = "" if pd.isna(text) else str(text).lower()
    words = re.findall(r"[a-z0-9]+", content)
    numbers = [float(value) for value in NUMBER_PATTERN.findall(content)]
    pack_count = _extract_pack_count(content)
    size_value = _extract_size_value(content)
    premium_term_count = sum(1 for term in PREMIUM_TERMS if term in content)
    unique_word_ratio = len(set(words)) / len(words) if words else 0.0

    return [
        float(len(content)),
        float(len(words)),
        float(len(numbers)),
        max(numbers) if numbers else 0.0,
        pack_count,
        size_value,
        1.0 if size_value > 0 else 0.0,
        1.0 if pack_count > 1 else 0.0,
        float(premium_term_count),
        float(unique_word_ratio),
    ]


def _extract_pack_count(content: str) -> float:
    for pattern in PACK_PATTERNS:
        match = pattern.search(content)
        if match:
            return float(match.group(1))
    return 1.0


def _extract_size_value(content: str) -> float:
    match = SIZE_PATTERN.search(content)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    if unit in {"kg"}:
        return value * 1000
    if unit in {"lb", "lbs", "pound", "pounds"}:
        return value * 16
    if unit in {"l", "liter", "litre"}:
        return value * 1000
    return value
