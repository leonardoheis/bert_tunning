import numpy as np

from src.schema import (
    CalibrationReport,
    ClassEmbeddingStats,
    Hyperparams,
    OodMetrics,
    PredictResult,
    flatten_predict_result,
)


def test_predict_result_svm_scores_defaults_to_none() -> None:
    assert PredictResult(label="decreto", confidence=0.9, certain=True).svm_scores is None


def test_predict_result_svm_scores_populates_via_snake_case_kwarg() -> None:
    # Regression guard: PredictResult has alias_generator=to_camel -- without
    # populate_by_name=True, a snake_case svm_scores= kwarg would silently drop to the
    # default (None) instead of raising. See CLAUDE.md's populate_by_name gotcha.
    result = PredictResult(
        label="decreto", confidence=0.9, certain=True, svm_scores={"decreto": 1.2}
    )
    assert result.svm_scores == {"decreto": 1.2}


def test_class_embedding_stats_tfidf_fields_default_to_absent() -> None:
    stats = ClassEmbeddingStats(
        class_names=["a", "b"],
        pca_mean=np.zeros(4),
        pca_components=np.eye(4),
        centroids=np.zeros((2, 4)),
        covariance_inv=np.eye(4),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((2, 4)),
        knn_train_labels=[0, 1],
    )
    assert stats.tfidf_vocabulary_terms == []
    assert len(stats.tfidf_idf) == 0
    assert stats.tfidf_centroids.size == 0
    assert stats.tfidf_cosine_calibration_mean == 0.0
    assert stats.tfidf_cosine_calibration_std == 1.0
    assert stats.tfidf_threshold is None


def test_calibration_report_tfidf_fields_default_to_zero() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.0,
        fp_rate_cosine=0.0,
        fp_rate_knn=0.0,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=0.0,
        suggested_knn_threshold=0.0,
    )
    assert report.fp_rate_tfidf == 0.0
    assert report.suggested_tfidf_threshold == 0.0


def test_hyperparams_accepts_snake_case_construction() -> None:
    # Regression guard: Hyperparams(alias_generator=to_camel) without populate_by_name=True
    # rejects snake_case kwargs with a Pydantic ValidationError instead of accepting them --
    # exactly how src/training/pipeline.py constructs it (Hyperparams(model=..., batch_size=...)).
    # Broke every training run at the post-training reporting step until populate_by_name=True
    # was added; this test would have caught it.
    hyperparams = Hyperparams(
        model="beto",
        epochs=15,
        batch_size=8,
        grad_accum=8,
        effective_batch=64,
        learning_rate=2e-5,
        warmup_steps=100,
        precision="bf16",
        train_docs=1344,
        num_classes=9,
    )
    assert hyperparams.batch_size == 8  # noqa: PLR2004
    assert hyperparams.num_classes == 9  # noqa: PLR2004


def test_ood_metrics_tfidf_cosine_z_defaults_to_none() -> None:
    metrics = OodMetrics(
        mahalanobis_p_value=0.5,
        mahalanobis_p_value_theoretical=0.6,
        cosine_z=1.0,
        knn_distance=2.0,
        in_distribution=True,
    )
    assert metrics.tfidf_cosine_z is None


def test_flatten_predict_result_merges_ood_metrics_to_top_level() -> None:
    result = PredictResult(
        filename="a.pdf",
        label="decreto",
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=0.6,
            cosine_z=1.0,
            knn_distance=2.0,
            in_distribution=True,
        ),
    )
    row = flatten_predict_result(result)
    assert row["mahalanobis_p_value"] == 0.5  # noqa: PLR2004
    assert row["knn_distance"] == 2.0  # noqa: PLR2004
    assert row["in_distribution"] is True
    assert "ood_metrics" not in row


def test_flatten_predict_result_fills_none_when_ood_metrics_absent() -> None:
    result = PredictResult(filename="a.pdf", label="decreto")
    row = flatten_predict_result(result)
    assert row["mahalanobis_p_value"] is None
    assert row["knn_distance"] is None
    assert row["in_distribution"] is None
