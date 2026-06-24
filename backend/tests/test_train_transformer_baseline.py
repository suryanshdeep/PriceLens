import numpy as np
import pandas as pd

from src.models.train_transformer_baseline import _attach_embeddings


def test_attach_embeddings_inner_joins_by_sample_id() -> None:
    frame = pd.DataFrame(
        {
            "sample_id": ["1", "2", "3"],
            "catalog_content": ["a", "b", "c"],
            "image_link": ["a.jpg", "b.jpg", "c.jpg"],
            "price": [1.0, 2.0, 3.0],
        }
    )
    sample_ids = np.asarray(["1", "3"])
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    output = _attach_embeddings(frame, sample_ids, embeddings)

    assert output["sample_id"].tolist() == ["1", "3"]
    assert len(output) == 2
    assert output.loc[0, "embedding"].shape == (2,)
