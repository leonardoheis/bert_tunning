from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from src.cli.ood_calibration import build_calibration_report, evaluate_ood_calibration_cmd
from src.settings import Settings


def test_build_calibration_report_percentile_direction() -> None:
    # Mahalanobis: LOW p-value = anomalous, so the suggested threshold for a 25% target
    # false-positive rate is the value below which 25% of in-distribution p-values fall —
    # the 25th percentile, NOT the 75th.
    p_values = np.array([0.1, 0.2, 0.3, 0.4])
    # Cosine: HIGH z-score = anomalous, so the suggested threshold for a 25% target rate
    # is the value above which 25% of in-distribution z-scores fall — the 75th percentile.
    z_scores = np.array([1.0, 2.0, 3.0, 4.0])

    report = build_calibration_report(p_values, z_scores, target_fp_rate=0.25)

    assert report.suggested_maha_threshold == pytest.approx(np.percentile(p_values, 25))
    assert report.suggested_cosine_threshold == pytest.approx(np.percentile(z_scores, 75))
    # Sanity check the two percentiles land on opposite ends, catching a swapped formula.
    assert report.suggested_maha_threshold < np.median(p_values)
    assert report.suggested_cosine_threshold > np.median(z_scores)


def test_build_calibration_report_fp_rates() -> None:
    # Values are picked relative to the configured thresholds directly, so the test
    # doesn't hardcode a threshold value that could drift from Settings.
    below = Settings.OOD_MAHALANOBIS_P_THRESHOLD / 2
    above = Settings.OOD_MAHALANOBIS_P_THRESHOLD * 2
    p_values = np.array([below, below, above, above])

    z_below = Settings.OOD_COSINE_THRESHOLD - 1.0
    z_above = Settings.OOD_COSINE_THRESHOLD + 1.0
    z_scores = np.array([z_below, z_above, z_above, z_below])

    report = build_calibration_report(p_values, z_scores, target_fp_rate=0.01)

    assert report.fp_rate_maha == pytest.approx(0.5)
    assert report.fp_rate_cosine == pytest.approx(0.5)


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
