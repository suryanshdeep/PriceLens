import json
from pathlib import Path

from src.models.compare_models import collect_model_metadata, write_model_comparison_report


def test_collect_model_metadata_reads_metric_rows(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "model_a",
                "model_type": "ridge",
                "metrics": {
                    "validation": {"smape": 10.0, "mae": 2.0, "rmse": 3.0, "r2": 0.5},
                    "test": {"smape": 11.0, "mae": 2.5, "rmse": 3.5, "r2": 0.4},
                },
            }
        ),
        encoding="utf-8",
    )

    frame = collect_model_metadata([metadata_path])

    assert set(frame["split"]) == {"validation", "test"}
    assert frame.loc[frame["split"] == "validation", "model_version"].iloc[0] == "model_a"


def test_write_model_comparison_report(tmp_path: Path) -> None:
    metadata_path = tmp_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_version": "model_a",
                "model_type": "ridge",
                "metrics": {"validation": {"smape": 10.0, "mae": 2.0, "rmse": 3.0, "r2": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    frame = collect_model_metadata([metadata_path])

    report_path = write_model_comparison_report(frame, tmp_path / "comparison.md")

    assert "model_a" in report_path.read_text(encoding="utf-8")
