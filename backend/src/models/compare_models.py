import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.artifacts import models_dir, reports_dir
from src.models.reporting import read_json


def collect_model_metadata(metadata_paths: list[Path] | None = None) -> pd.DataFrame:
    paths = metadata_paths or sorted(models_dir().glob("*metadata.json"))
    rows = []
    for path in paths:
        metadata = read_json(path)
        rows.extend(_rows_from_metadata(path, metadata))
    return pd.DataFrame(rows).sort_values(["split", "smape", "mae"]).reset_index(drop=True)


def write_model_comparison_report(frame: pd.DataFrame, output_path: Path | None = None) -> Path:
    path = output_path or reports_dir() / "model_comparison.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Model Comparison", ""]

    for split in sorted(frame["split"].unique()):
        split_frame = frame[frame["split"] == split].copy()
        lines.extend([f"## {split.title()}", ""])
        lines.append("| Model | Type | SMAPE | MAE | RMSE | R2 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for row in split_frame.itertuples(index=False):
            lines.append(
                f"| `{row.model_version}` | `{row.model_type}` | "
                f"{row.smape:.4f} | {row.mae:.4f} | {row.rmse:.4f} | {row.r2:.4f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _rows_from_metadata(path: Path, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in metadata.get("metrics", {}).items():
        if not isinstance(metrics, dict) or "smape" not in metrics:
            continue
        rows.append(
            {
                "metadata_file": str(path),
                "model_version": metadata.get("model_version", path.stem),
                "model_type": metadata.get("model_type", "unknown"),
                "split": split,
                "smape": float(metrics.get("smape", float("nan"))),
                "mae": float(metrics.get("mae", float("nan"))),
                "rmse": float(metrics.get("rmse", float("nan"))),
                "r2": float(metrics.get("r2", float("nan"))),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare saved PriceLens model metadata.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = collect_model_metadata()
    output_path = write_model_comparison_report(frame, args.output)
    print(frame.to_string(index=False))
    print(f"Wrote model comparison report to {output_path}")


if __name__ == "__main__":
    main()
