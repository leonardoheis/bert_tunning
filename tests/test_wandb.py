from unittest.mock import MagicMock, patch

from src.ood import OodThresholds
from src.schema import CalibrationReport, PredictResult
from src.wandb import log_ood_calibration_results, log_predict_folder_results


def _logged_row(mock_table_cls: MagicMock, mock_table: MagicMock) -> dict[str, object]:
    """Pair the wandb.Table(columns=...) call with an add_data(...) call by position,
    keyed by column name — a plain `x in columns` / `x in add_data.args` membership check
    can't catch the column list and the add_data positional values drifting out of sync."""
    columns = mock_table_cls.call_args.kwargs["columns"]
    args = mock_table.add_data.call_args.args
    return dict(zip(columns, args, strict=True))


def test_log_predict_folder_results_logs_a_table_per_result() -> None:
    results = [
        PredictResult(filename="a.pdf", label="decreto", confidence=0.9, certain=True),
        PredictResult(filename="b.pdf", label="ordenanza", confidence=0.5, certain=False),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init") as mock_init,
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log") as mock_log,
        patch("src.wandb.wandb.finish") as mock_finish,
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["job_type"] == "predict-folder"
    mock_table_cls.assert_called_once()
    assert mock_table.add_data.call_count == len(results)
    mock_log.assert_called_once_with({"predictions": mock_table})
    mock_finish.assert_called_once()


def test_log_predict_folder_results_table_includes_knn_distance_column() -> None:
    expected_knn_distance = 4.2
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            knn_distance=expected_knn_distance,
        ),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init"),
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log"),
        patch("src.wandb.wandb.finish"),
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    assert _logged_row(mock_table_cls, mock_table)["knn_distance"] == expected_knn_distance


def test_log_predict_folder_results_table_includes_review_route_column() -> None:
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            review_route="accept",
        ),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init"),
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log"),
        patch("src.wandb.wandb.finish"),
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    assert _logged_row(mock_table_cls, mock_table)["review_route"] == "accept"


def test_log_predict_folder_results_table_includes_theoretical_mahalanobis_column() -> None:
    expected_theoretical_p = 0.1708
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            mahalanobis_p_value_theoretical=expected_theoretical_p,
        ),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init"),
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log"),
        patch("src.wandb.wandb.finish"),
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    row = _logged_row(mock_table_cls, mock_table)
    assert row["mahalanobis_p_value_theoretical"] == expected_theoretical_p


def test_log_ood_calibration_results_logs_summary_metrics() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.2951,
        fp_rate_cosine=0.0104,
        fp_rate_knn=0.0087,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=13.7186,
        suggested_knn_threshold=4.2,
        fp_rate_tfidf=0.0093,
        suggested_tfidf_threshold=2.71,
    )
    thresholds = OodThresholds(
        mahalanobis_p=0.001, cosine_z=13.7366, knn_distance=16.7908, tfidf_cosine_z=2.5
    )
    with (
        patch("src.wandb.wandb.init") as mock_init,
        patch("src.wandb.wandb.log") as mock_log,
        patch("src.wandb.wandb.finish") as mock_finish,
    ):
        log_ood_calibration_results(
            report,
            model_path="fake/model",
            cache_path="fake/cache.parquet",
            target_fp_rate=0.01,
            thresholds=thresholds,
        )

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["job_type"] == "ood-calibration"
    # The resolved per-model thresholds (not Settings.OOD_*) must be what's logged as
    # "current" -- and knn is now included, which it wasn't before this task.
    config = mock_init.call_args.kwargs["config"]
    assert config["current_mahalanobis_threshold"] == 0.001  # noqa: PLR2004
    assert config["current_cosine_threshold"] == 13.7366  # noqa: PLR2004
    assert config["current_knn_threshold"] == 16.7908  # noqa: PLR2004
    assert config["current_tfidf_threshold"] == 2.5  # noqa: PLR2004
    mock_log.assert_called_once_with(
        {
            "ood/fp_rate_mahalanobis": 0.2951,
            "ood/fp_rate_cosine": 0.0104,
            "ood/suggested_mahalanobis_threshold": 0.0,
            "ood/suggested_cosine_threshold": 13.7186,
            "ood/fp_rate_knn": 0.0087,
            "ood/suggested_knn_threshold": 4.2,
            "ood/fp_rate_tfidf": 0.0093,
            "ood/suggested_tfidf_threshold": 2.71,
        }
    )
    mock_finish.assert_called_once()
