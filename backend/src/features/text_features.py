from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def create_tfidf_ridge_pipeline(
    *,
    max_features: int = 80_000,
    min_df: int = 2,
    ngram_max: int = 2,
    alpha: float = 1.0,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    min_df=min_df,
                    ngram_range=(1, ngram_max),
                    strip_accents="unicode",
                    sublinear_tf=True,
                ),
            ),
            ("model", Ridge(alpha=alpha)),
        ]
    )
