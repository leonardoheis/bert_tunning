from click.testing import CliRunner

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd


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


def test_clean_cmd_help() -> None:
    result = CliRunner().invoke(clean_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Wipe" in result.output
