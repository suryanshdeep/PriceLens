import argparse
import logging
from time import perf_counter
from pathlib import Path

from src.data.clean_data import clean_test_data, clean_train_data
from src.data.load_data import load_test_data, load_train_data
from src.features.transformer_embeddings import encode_texts_batched, save_embedding_cache
from src.models.artifacts import models_dir

logger = logging.getLogger(__name__)


def build_text_embedding_cache(
    *,
    split: str,
    model_name: str,
    output_path: Path | None = None,
    sample_size: int | None = None,
    batch_size: int = 64,
    device: str | None = None,
) -> Path:
    started_at = perf_counter()
    if split == "train":
        frame = clean_train_data(load_train_data())
    elif split == "test":
        frame = clean_test_data(load_test_data())
    else:
        raise ValueError("split must be 'train' or 'test'")

    if sample_size is not None and sample_size < len(frame):
        frame = frame.sample(n=sample_size, random_state=42).reset_index(drop=True)

    logger.info("Prepared %s rows for %s embedding cache in %.2fs", len(frame), split, perf_counter() - started_at)
    encode_started_at = perf_counter()
    embeddings = encode_texts_batched(
        frame["catalog_content"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    logger.info("Encoded %s rows in %.2fs", len(frame), perf_counter() - encode_started_at)
    path = output_path or _default_cache_path(split=split, model_name=model_name)
    write_started_at = perf_counter()
    save_embedding_cache(
        path=path,
        sample_ids=frame["sample_id"].astype(str).tolist(),
        embeddings=embeddings,
        model_name=model_name,
    )
    logger.info("Wrote %s embeddings to %s in %.2fs", len(frame), path, perf_counter() - write_started_at)
    logger.info("Total embedding cache build time: %.2fs", perf_counter() - started_at)
    return path


def _default_cache_path(*, split: str, model_name: str) -> Path:
    safe_model_name = model_name.replace("/", "__").replace("\\", "__")
    return models_dir() / "embeddings" / f"{split}_{safe_model_name}.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transformer text embedding cache.")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s - %(message)s")
    args = parse_args()
    path = build_text_embedding_cache(
        split=args.split,
        model_name=args.model_name,
        output_path=args.output_path,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Wrote text embedding cache to {path}")


if __name__ == "__main__":
    main()
