from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from click.testing import CliRunner, Result

from src.cli.ood_calibration import build_calibration_report, evaluate_ood_calibration_cmd
from src.embeddings import LoadedModel
from src.ood import OodThresholds, load_stats, save_stats
from src.ood import knn_mean_distance as real_knn_mean_distance
from src.schema import ClassEmbeddingStats
from src.settings import Settings


def _fake_extract_embeddings_and_predictions(
    _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
) -> tuple[npt.NDArray[np.float64], list[int]]:
    return np.zeros((len(texts), 8), dtype=np.float64), [0] * len(texts)


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
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0)

    report = build_calibration_report(
        p_values, z_scores, knn_distances, target_fp_rate=0.25, thresholds=thresholds
    )

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
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0)

    report = build_calibration_report(
        p_values, z_scores, knn_distances, target_fp_rate=0.25, thresholds=thresholds
    )

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

    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    report = build_calibration_report(
        p_values, z_scores, knn_distances, target_fp_rate=0.01, thresholds=thresholds
    )

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
    model_path.mkdir(exist_ok=True)
    save_stats(_make_stats(), model_path / "ood_stats.npz")

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=_make_stats()),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_fake_extract_embeddings_and_predictions,
        ),
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


def _run_calibration_with_stats_write_thresholds(
    tmp_path: Path, stats: ClassEmbeddingStats
) -> tuple[Result, MagicMock]:
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": [f"decreto {i}" for i in range(20)] + [f"ordenanza {i}" for i in range(20)],
            "label": ["decreto"] * 20 + ["ordenanza"] * 20,
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    save_stats(stats, model_path / "ood_stats.npz")

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=stats),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_fake_extract_embeddings_and_predictions,
        ),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results"),
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
                "--write-thresholds",
            ],
        )
    return result, MagicMock()


def test_evaluate_ood_calibration_cmd_write_thresholds_persists_to_stats_file(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    result, _ = _run_successful_calibration(tmp_path, extra_args=["--write-thresholds"])
    assert result.exit_code == 0
    written = load_stats(stats_path)
    assert written.cosine_threshold is not None
    assert written.knn_distance_threshold is not None


def test_evaluate_ood_calibration_cmd_without_flag_does_not_write(tmp_path: Path) -> None:
    # save_stats() needs the parent dir to already exist -- _run_successful_calibration's
    # own mkdir is exist_ok=True precisely so it can share this pre-created model_path.
    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    stats_path = model_path / "ood_stats.npz"
    save_stats(_make_stats(), stats_path)
    before = stats_path.read_bytes()
    result, _ = _run_successful_calibration(tmp_path, extra_args=[])
    assert result.exit_code == 0
    assert stats_path.read_bytes() == before


def test_evaluate_ood_calibration_cmd_write_thresholds_refuses_degenerate_maha_threshold(
    tmp_path: Path,
) -> None:
    # Only 4 reference points -- 1/(4+1) = 0.2, comfortably above any target-FP-rate
    # percentile this test's p_values could produce, so the suggested Mahalanobis threshold
    # is guaranteed to be at/below the floor. cosine/knn thresholds should still get written.
    # Centroids are shifted off the origin (unlike _make_stats()'s [0]*8/[5]*8) because the
    # fake extractor always returns the zero vector: with a centroid sitting exactly at
    # zero, the query would coincide with it (distance 0, the least-anomalous case possible)
    # and could never rank as extreme. Reference points are placed exactly on their own
    # class's centroid (distance 0) so every query -- always farther away -- ranks more
    # extreme than all of them, landing precisely on the floor.
    tiny_stats = _make_stats().model_copy(
        update={
            "centroids": np.array([[3.0] * 8, [8.0] * 8]),
            "knn_train_embeddings": np.array([[3.0] * 8] * 2 + [[8.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    assert "Refusing to write" in result.output
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_p_threshold is None  # unchanged -- tiny_stats had no prior value
    assert written.cosine_threshold is not None
    assert written.knn_distance_threshold is not None


def test_evaluate_ood_calibration_cmd_write_thresholds_persists_calibrated_status(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    result, _ = _run_successful_calibration(tmp_path, extra_args=["--write-thresholds"])
    assert result.exit_code == 0
    written = load_stats(stats_path)
    assert written.mahalanobis_threshold_status == "calibrated"


def test_evaluate_ood_calibration_cmd_write_thresholds_refused_status_when_degenerate(
    tmp_path: Path,
) -> None:
    tiny_stats = _make_stats().model_copy(
        update={
            "centroids": np.array([[3.0] * 8, [8.0] * 8]),
            "knn_train_embeddings": np.array([[3.0] * 8] * 2 + [[8.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_threshold_status == "refused_degenerate"


def test_evaluate_ood_calibration_cmd_write_thresholds_keeps_calibrated_status_on_refusal(
    tmp_path: Path,
) -> None:
    # Guard refuses the new suggestion but a real value from a PRIOR successful calibration
    # already exists -- status must stay "calibrated", not flip to "refused_degenerate",
    # since nothing about the currently-persisted value actually changed or became invalid.
    tiny_stats = _make_stats().model_copy(
        update={
            "centroids": np.array([[3.0] * 8, [8.0] * 8]),
            "knn_train_embeddings": np.array([[3.0] * 8] * 2 + [[8.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
            "mahalanobis_p_threshold": 0.0005,
            "mahalanobis_threshold_status": "calibrated",
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_p_threshold == pytest.approx(0.0005)
    assert written.mahalanobis_threshold_status == "calibrated"


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


def _fake_predict_matching_text_label(
    _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Predicts 0 ("decreto") or 1 ("ordenanza") based on which word appears in each text --
    unlike the constant-0 shared fake, this lets a single stats object exercise BOTH a
    present and a missing class within the same test run, which the NaN/skip test below
    needs (a fake that always predicts the same class can't produce a partial skip)."""
    return (
        np.zeros((len(texts), 8), dtype=np.float64),
        [0 if "decreto" in t else 1 for t in texts],
    )


def _run_calibration_with_stats(
    tmp_path: Path,
    stats: ClassEmbeddingStats,
    *,
    predict_fn: Callable[..., tuple[npt.NDArray[np.float64], list[int]]] = (
        _fake_extract_embeddings_and_predictions
    ),
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

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=stats),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=predict_fn,
        ),
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
    # Needs predicted labels that actually vary per doc (the shared constant-0 fake would
    # make every doc predict the same class, collapsing this into all-NaN or no-NaN).
    result, _ = _run_calibration_with_stats(
        tmp_path, _make_stats_missing_class(0), predict_fn=_fake_predict_matching_text_label
    )

    assert result.exit_code == 0
    assert "Skipping" in result.output or "skipping" in result.output.lower()


def test_evaluate_ood_calibration_cmd_fails_when_no_training_data(
    tmp_path: Path,
) -> None:
    # Both classes missing k-NN training data means compute_train_mahalanobis_distances()
    # (which pools every class's points into one combined reference set) is empty too --
    # the command must fail on that guard before ever reaching k-NN calibration.
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
    assert "no training data" in str(result.output).lower()
    assert "cannot calibrate mahalanobis" in str(result.output).lower()


def _make_stats_third_class_has_knn_points() -> ClassEmbeddingStats:
    """decreto (0) and ordenanza (1) -- the only classes that appear in the test corpus --
    both have zero stored k-NN training points, so every test doc gets a NaN knn_distance.
    A third class ('otro', label 2, never present in the cache) has points, purely so
    compute_train_mahalanobis_distances() -- which pools every class's points into one
    combined reference set -- is non-empty. This isolates the k-NN NaN-exhaustion path
    from the separate no-training-data guard tested above."""
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza", "otro"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8, [10.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.array([[10.0] * 8] * 5),
        knn_train_labels=[2] * 5,
    )


def test_evaluate_ood_calibration_cmd_fails_when_every_doc_has_nan_knn_distance(
    tmp_path: Path,
) -> None:
    result, _ = _run_calibration_with_stats(tmp_path, _make_stats_third_class_has_knn_points())

    assert result.exit_code != 0
    assert "cannot calibrate k-nn" in str(result.output).lower()


def test_evaluate_ood_calibration_cmd_uses_empirical_not_chi2_p_value(tmp_path: Path) -> None:
    with patch(
        "src.cli.ood_calibration.mahalanobis_empirical_p_value", return_value=0.5
    ) as mock_empirical:
        result, _ = _run_successful_calibration(tmp_path, extra_args=[])
    assert result.exit_code == 0
    mock_empirical.assert_called()


def test_evaluate_ood_calibration_cmd_uses_predicted_label_for_knn_not_true_label(
    tmp_path: Path,
) -> None:
    # 20 "decreto" (label 0) + 20 "ordenanza" (label 1) docs, but every document's forward
    # pass predicts label 1 regardless of its true label -- if the command still scores
    # k-NN using the true label, the "decreto" test docs would be scored against class 0's
    # neighbors (5 points); if it correctly uses the predicted label, every doc scores
    # against class 1's neighbors instead. knn_mean_distance is spied on to assert the
    # label id it actually received.
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

    def _always_predicts_ordenanza(
        _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        return np.zeros((len(texts), 8), dtype=np.float64), [1] * len(texts)

    seen_label_ids: list[int] = []

    def _spy_knn_mean_distance(
        embedding: npt.NDArray[np.float64],
        stats: ClassEmbeddingStats,
        predicted_label_id: int,
        *,
        k: int,
    ) -> float:
        seen_label_ids.append(predicted_label_id)
        return real_knn_mean_distance(embedding, stats, predicted_label_id, k=k)

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=_make_stats()),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_always_predicts_ordenanza,
        ),
        patch("src.cli.ood_calibration.knn_mean_distance", side_effect=_spy_knn_mean_distance),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results"),
    ):
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
            ["--model-path", str(model_path), "--model", "beto", "--cache-path", str(cache_path)],
        )

    assert result.exit_code == 0
    # Every seen label id must be 1 (the mocked prediction) -- never 0, even for the
    # "decreto" (true label 0) test documents.
    assert seen_label_ids
    assert set(seen_label_ids) == {1}
