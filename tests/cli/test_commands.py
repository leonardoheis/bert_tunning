from pathlib import Path
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from src.cli.clean import clean_cmd
from src.cli.predict import predict_cmd, predict_folder_cmd
from src.cli.train import train_cmd
from src.schema import OodMetrics, PredictResult


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


def test_predict_cmd_prints_ood_metrics_when_present(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.005,
            mahalanobis_p_value_theoretical=0.017,
            cosine_z=0.1,
            knn_distance=1.0,
            in_distribution=False,
        ),
    )

    with patch("src.cli.predict.predict_pdf", return_value=fake_result):
        result = CliRunner().invoke(predict_cmd, [str(pdf_path)])

    assert result.exit_code == 0
    assert "Mahalanobis p (chi2, theoretical): 0.017000" in result.output
    assert "In-Dist.     : False" in result.output


def test_clean_cmd_help() -> None:
    result = CliRunner().invoke(clean_cmd, ["--help"])
    assert result.exit_code == 0
    assert "Wipe" in result.output


def test_predict_folder_cmd_writes_flat_ood_columns_to_csv(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    output = tmp_path / "results.csv"
    fake_results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            ood_metrics=OodMetrics(
                mahalanobis_p_value=0.5,
                mahalanobis_p_value_theoretical=0.6,
                cosine_z=1.0,
                knn_distance=2.0,
                in_distribution=True,
            ),
        ),
    ]

    with patch("src.cli.predict.predict_folder", return_value=fake_results):
        result = CliRunner().invoke(predict_folder_cmd, ["--output", str(output), str(folder)])

    assert result.exit_code == 0
    df = pd.read_csv(output)
    assert "ood_metrics" not in df.columns
    assert df["knn_distance"].iloc[0] == 2.0  # noqa: PLR2004
