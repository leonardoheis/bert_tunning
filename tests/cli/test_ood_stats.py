from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from click.testing import CliRunner

from src.cli.ood_stats import compute_ood_stats_cmd


def test_compute_ood_stats_cmd_help() -> None:
    result = CliRunner().invoke(compute_ood_stats_cmd, ["--help"])
    assert result.exit_code == 0
    assert "ood_stats" in result.output.lower() or "retraining" in result.output.lower()


def test_compute_ood_stats_cmd_fails_on_class_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": ["decreto uno", "decreto dos", "ordenanza uno", "ordenanza dos"],
            "label": ["decreto", "decreto", "ordenanza", "ordenanza"],
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()

    mock_model = MagicMock()
    # Model was trained on different classes than the cache reflects.
    mock_model.config.id2label = {0: "resolucion", 1: "boletines"}

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch("src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained") as mock_mdl,
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_mdl.return_value = mock_model
        result = CliRunner().invoke(
            compute_ood_stats_cmd,
            [
                "--model-path",
                str(model_path),
                "--model",
                "beto",
                "--cache-path",
                str(cache_path),
            ],
        )

    assert result.exit_code != 0
    assert "do not match" in str(result.output).lower()
