import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_metrics_report(path: Path, title: str, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]

    lines.extend(
        [
            f"- Model version: `{metadata['model_version']}`",
            f"- Model type: `{metadata['model_type']}`",
            f"- Training rows: {metadata['row_counts']['train']}",
            f"- Validation rows: {metadata['row_counts']['validation']}",
            f"- Test rows: {metadata['row_counts']['test']}",
            "",
            "## Metrics",
            "",
        ]
    )

    for split_name, split_metrics in metadata["metrics"].items():
        lines.append(f"### {split_name.title()}")
        lines.append("")
        for metric_name, metric_value in split_metrics.items():
            lines.append(f"- {metric_name.upper()}: {metric_value:.4f}")
        lines.append("")

    top_tokens = metadata.get("top_positive_tokens", [])
    if top_tokens:
        lines.extend(["## Top Positive Price-Signal Tokens", ""])
        lines.extend(f"- `{token}`" for token in top_tokens[:30])
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
