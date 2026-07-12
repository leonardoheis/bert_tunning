from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from click.testing import CliRunner, Result

from src.cli.ood_calibration import build_calibration_report, evaluate_ood_calibration_cmd
from src.ood import LoadedModel
from src.schema import ClassEmbeddingStats
from src.settings import Settings


def _make_stats() -> ClassEmbeddingStats:
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.array([[0.0] * 8] * 5 + [[5.0] * 8] * 5),
        knn_train_labels=[0] * 5 + [1] * 5,
    )


def test_build_calibration_report_percentile_direction() -> None:
    # Mahalanobis: LOW p-value = anomalous, so the suggested threshold for a 25% target
    # false-positive rate is the value below which 25% of in-distribution p-values fall —
    # the 25th percentile, NOT the 75th.
    p_values = np.array([0.1, 0.2, 0.3, 0.4])
    # Cosine: HIGH z-score = anomalous, so the suggested threshold for a 25% target rate
    # is the value above which 25% of in-distribution z-scores fall — the 75th percentile.
    z_scores = np.array([1.0, 2.0, 3.0, 4.0])
    knn_distances = np.array([1.0, 2.0, 3.0, 4.0])

    report = build_calibration_report(p_values, z_scores, knn_distances, target_fp_rate=0.25)

    assert report.suggested_maha_threshold == pytest.approx(np.percentile(p_values, 25))
    assert report.suggested_cosine_threshold == pytest.approx(np.percentile(z_scores, 75))
    # Sanity check the two percentiles land on opposite ends, catching a swapped formula.
    assert report.suggested_maha_threshold < np.median(p_values)
    assert report.suggested_cosine_threshold > np.median(z_scores)


def test_build_calibration_report_knn_percentile_direction() -> None:
    # k-NN mean distance: HIGH distance = anomalous, same direction as cosine, so the
    # suggested threshold for a 25% target false-positive rate is the value above which
    # 25% of in-distribution knn distances fall — the 75th percentile, NOT the 25th.
    p_values = np.array([0.1, 0.2, 0.3, 0.4])
    z_scores = np.array([1.0, 2.0, 3.0, 4.0])
    knn_distances = np.array([1.0, 2.0, 3.0, 4.0])

    report = build_calibration_report(p_values, z_scores, knn_distances, target_fp_rate=0.25)

    assert report.suggested_knn_threshold == pytest.approx(np.percentile(knn_distances, 75))
    assert report.suggested_knn_threshold > np.median(knn_distances)


def test_build_calibration_report_fp_rates() -> None:
    # Values are picked relative to the configured thresholds directly, so the test
    # doesn't hardcode a threshold value that could drift from Settings.
    below = Settings.OOD_MAHALANOBIS_P_THRESHOLD / 2
    above = Settings.OOD_MAHALANOBIS_P_THRESHOLD * 2
    p_values = np.array([below, below, above, above])

    z_below = Settings.OOD_COSINE_THRESHOLD - 1.0
    z_above = Settings.OOD_COSINE_THRESHOLD + 1.0
    z_scores = np.array([z_below, z_above, z_above, z_below])

    knn_below = Settings.OOD_KNN_DISTANCE_THRESHOLD - 1.0
    knn_above = Settings.OOD_KNN_DISTANCE_THRESHOLD + 1.0
    knn_distances = np.array([knn_below, knn_above, knn_above, knn_below])

    report = build_calibration_report(p_values, z_scores, knn_distances, target_fp_rate=0.01)

    assert report.fp_rate_maha == pytest.approx(0.5)
    assert report.fp_rate_cosine == pytest.approx(0.5)
    assert report.fp_rate_knn == pytest.approx(0.5)


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
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch("src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained") as mock_mdl,
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


def _run_successful_calibration(
    tmp_path: Path, *, extra_args: list[str]
) -> tuple[Result, MagicMock]:
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
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    def _fake_extract_embeddings(
        _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
    ) -> npt.NDArray[np.float64]:
        return np.zeros((len(texts), 8), dtype=np.float64)

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=_make_stats()),
        patch("src.cli._ood_common.extract_embeddings", side_effect=_fake_extract_embeddings),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results") as mock_log,
    ):
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
            [
                "--model-path",
                str(model_path),
                "--model",
                "beto",
                "--cache-path",
                str(cache_path),
                *extra_args,
            ],
        )
    return result, mock_log


def test_evaluate_ood_calibration_cmd_logs_to_wandb_when_flag_set(tmp_path: Path) -> None:
    result, mock_log = _run_successful_calibration(tmp_path, extra_args=["--log-wandb"])

    assert result.exit_code == 0
    mock_log.assert_called_once()
    assert mock_log.call_args.kwargs["target_fp_rate"] == Settings.TARGET_FP_RATE


def test_evaluate_ood_calibration_cmd_skips_wandb_by_default(tmp_path: Path) -> None:
    result, mock_log = _run_successful_calibration(tmp_path, extra_args=[])

    assert result.exit_code == 0
    mock_log.assert_not_called()


def _make_stats_missing_class(missing_label_id: int) -> ClassEmbeddingStats:
    """Same shape as _make_stats(), but one class has zero stored k-NN training points —
    knn_mean_distance returns NaN for any test document whose true label is that class."""
    present_label_id = 1 - missing_label_id
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.array([[float(present_label_id) * 5.0] * 8] * 5),
        knn_train_labels=[present_label_id] * 5,
    )


def _run_calibration_with_stats(
    tmp_path: Path, stats: ClassEmbeddingStats
) -> tuple[Result, MagicMock]:
    # 20 docs/class so the stratified test split reliably contains both classes.
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": [f"decreto {i}" for i in range(20)] + [f"ordenanza {i}" for i in range(20)],
            "label": ["decreto"] * 20 + ["ordenanza"] * 20,
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    (model_path / "ood_stats.npz").touch()

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    def _fake_extract_embeddings(
        _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
    ) -> npt.NDArray[np.float64]:
        return np.zeros((len(texts), 8), dtype=np.float64)

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=stats),
        patch("src.cli._ood_common.extract_embeddings", side_effect=_fake_extract_embeddings),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results") as mock_log,
    ):
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
            ["--model-path", str(model_path), "--model", "beto", "--cache-path", str(cache_path)],
        )
    return result, mock_log


def test_evaluate_ood_calibration_cmd_skips_docs_with_nan_knn_distance(tmp_path: Path) -> None:
    # "decreto" (label 0) has no stored k-NN training points, so every decreto test doc
    # gets a NaN knn_distance — the command should skip those and still succeed using the
    # remaining (ordenanza) docs, rather than letting NaN corrupt the whole report.
    result, _ = _run_calibration_with_stats(tmp_path, _make_stats_missing_class(0))

    assert result.exit_code == 0
    assert "Skipping" in result.output or "skipping" in result.output.lower()


def test_evaluate_ood_calibration_cmd_fails_when_every_doc_has_nan_knn_distance(
    tmp_path: Path,
) -> None:
    # Both classes missing k-NN training data — every test doc is NaN, nothing left to
    # calibrate against.
    stats = ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((0, 8)),
        knn_train_labels=[],
    )

    result, _ = _run_calibration_with_stats(tmp_path, stats)

    assert result.exit_code != 0
    assert "cannot calibrate" in str(result.output).lower()


def test_evaluate_ood_calibration_cmd_uses_empirical_not_chi2_p_value(tmp_path: Path) -> None:
    with patch(
        "src.cli.ood_calibration.mahalanobis_empirical_p_value", return_value=0.5
    ) as mock_empirical:
        result, _ = _run_successful_calibration(tmp_path, extra_args=[])
    assert result.exit_code == 0
    mock_empirical.assert_called()
