from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from click.testing import CliRunner

from src.cli.ood_calibration import evaluate_ood_calibration_cmd


def test_evaluate_ood_calibration_cmd_help() -> None:
    result = CliRunner().invoke(evaluate_ood_calibration_cmd, ["--help"])
    assert result.exit_code == 0
    assert "calibrat" in result.output.lower() or "false-positive" in result.output.lower()


def test_evaluate_ood_calibration_cmd_fails_when_stats_missing(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": ["decreto uno", "decreto dos", "ordenanza uno", "ordenanza dos"],
            "label": ["decreto", "decreto", "ordenanza", "ordenanza"],
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()

    result = CliRunner().invoke(
        evaluate_ood_calibration_cmd,
        ["--model-path", str(model_path), "--model", "beto", "--cache-path", str(cache_path)],
    )

    assert result.exit_code != 0
    assert "ood_stats.npz" in str(result.output)


def test_evaluate_ood_calibration_cmd_fails_on_class_mismatch(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": ["decreto uno", "decreto dos", "ordenanza uno", "ordenanza dos"],
            "label": ["decreto", "decreto", "ordenanza", "ordenanza"],
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    (model_path / "ood_stats.npz").touch()

    mock_model = MagicMock()
    # Model was trained on different classes than the cache reflects.
    mock_model.config.id2label = {0: "resolucion", 1: "boletines"}

    with (
        patch("src.cli.ood_calibration.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli.ood_calibration.AutoModelForSequenceClassification.from_pretrained"
        ) as mock_mdl,
        patch("src.cli.ood_calibration.load_stats"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        mock_mdl.return_value = mock_model
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
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
