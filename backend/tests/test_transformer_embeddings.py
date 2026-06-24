from pathlib import Path

import numpy as np

from src.features.transformer_embeddings import load_embedding_cache, save_embedding_cache


def test_save_and_load_embedding_cache(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.npz"
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    save_embedding_cache(
        path=path,
        sample_ids=["101", "102"],
        embeddings=embeddings,
        model_name="test-model",
    )
    sample_ids, loaded_embeddings, model_name = load_embedding_cache(path)

    assert sample_ids.tolist() == ["101", "102"]
    assert np.allclose(loaded_embeddings, embeddings)
    assert model_name == "test-model"
