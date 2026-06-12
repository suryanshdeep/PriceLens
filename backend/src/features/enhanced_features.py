from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.structured_features import CatalogStructuredFeatures


def create_text_structured_ridge_pipeline(
    *,
    max_features: int = 100_000,
    min_df: int = 2,
    ngram_max: int = 2,
    alpha: float = 1.0,
) -> Pipeline:
    features = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=(1, ngram_max),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
                "catalog_content",
            ),
            (
                "structured",
                Pipeline(
                    steps=[
                        ("extract", CatalogStructuredFeatures()),
                        ("scale", StandardScaler()),
                    ]
                ),
                "catalog_content",
            ),
        ],
        sparse_threshold=0.3,
    )

    return Pipeline(
        steps=[
            ("features", features),
            ("model", Ridge(alpha=alpha)),
        ]
    )
