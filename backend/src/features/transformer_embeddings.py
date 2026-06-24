from collections.abc import Iterable
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


def encode_texts(
    texts: Iterable[str],
    *,
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    SentenceTransformer = _load_sentence_transformer_class()
    model = SentenceTransformer(model_name, device=device)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize_embeddings,
        convert_to_numpy=True,
    )


def save_embedding_cache(
    *,
    path: Path,
    sample_ids: list[str],
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_id=np.asarray(sample_ids, dtype=str),
        embedding=embeddings.astype(np.float32),
        model_name=np.asarray([model_name], dtype=str),
    )


def load_embedding_cache(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    if not path.exists():
        raise FileNotFoundError(f"Embedding cache not found: {path}")
    cache = np.load(path, allow_pickle=False)
    return cache["sample_id"].astype(str), cache["embedding"], str(cache["model_name"][0])


def batch_iter(values: list[str], *, batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def encode_texts_batched(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int = 64,
    device: str | None = None,
    normalize_embeddings: bool = True,
) -> np.ndarray:
    SentenceTransformer = _load_sentence_transformer_class()
    model = SentenceTransformer(model_name, device=device)
    batches = []
    progress = tqdm(
        batch_iter(texts, batch_size=batch_size),
        total=_batch_count(len(texts), batch_size),
        desc="Encoding text",
    )
    for batch in progress:
        embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
        )
        batches.append(embeddings)
    return np.vstack(batches).astype(np.float32)


def _batch_count(total: int, batch_size: int) -> int:
    return (total + batch_size - 1) // batch_size


def _load_sentence_transformer_class() -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for transformer embedding features. "
            "Install it with: pip install -e .[transformers]"
        ) from exc
    return SentenceTransformer
