from unittest.mock import MagicMock, patch

from src.schema import CalibrationReport, PredictResult
from src.wandb import log_ood_calibration_results, log_predict_folder_results


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


def test_log_ood_calibration_results_logs_summary_metrics() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.2951,
        fp_rate_cosine=0.0104,
        fp_rate_knn=0.0087,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=13.7186,
        suggested_knn_threshold=4.2,
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
        )

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["job_type"] == "ood-calibration"
    mock_log.assert_called_once_with(
        {
            "ood/fp_rate_mahalanobis": 0.2951,
            "ood/fp_rate_cosine": 0.0104,
            "ood/suggested_mahalanobis_threshold": 0.0,
            "ood/suggested_cosine_threshold": 13.7186,
        }
    )
    mock_finish.assert_called_once()
