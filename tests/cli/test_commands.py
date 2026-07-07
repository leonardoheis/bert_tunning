from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd
from src.schema import PredictResult


def test_train_cmd_help() -> None:
    result = CliRunner().invoke(train_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Fine-tune" in result.output


def test_predict_cmd_help() -> None:
    result = CliRunner().invoke(predict_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Classify" in result.output


def test_predict_folder_cmd_help() -> None:
    result = CliRunner().invoke(predict_folder_cmd, ["--help"])
    assert result.exit_code == 0
    assert "folder" in result.output.lower()


def test_predict_folder_cmd_logs_to_wandb_when_flag_set(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    output = tmp_path / "results.csv"
    fake_results = [PredictResult(filename="a.pdf", label="decreto", confidence=0.9, certain=True)]

    with (
        patch("src.cli.predict.predict_folder", return_value=fake_results),
        patch("src.cli.predict.log_predict_folder_results") as mock_log,
    ):
        result = CliRunner().invoke(
            predict_folder_cmd,
            ["--log-wandb", "--output", str(output), str(folder)],
        )

    assert result.exit_code == 0
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["folder_path"] == str(folder)


def test_predict_folder_cmd_skips_wandb_by_default(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    output = tmp_path / "results.csv"
    fake_results = [PredictResult(filename="a.pdf", label="decreto", confidence=0.9, certain=True)]

    with (
        patch("src.cli.predict.predict_folder", return_value=fake_results),
        patch("src.cli.predict.log_predict_folder_results") as mock_log,
    ):
        result = CliRunner().invoke(predict_folder_cmd, ["--output", str(output), str(folder)])

    assert result.exit_code == 0
    mock_log.assert_not_called()


def test_clean_cmd_help() -> None:
    result = CliRunner().invoke(clean_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Wipe" in result.output
