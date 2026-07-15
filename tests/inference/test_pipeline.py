import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.exceptions import BertTunningError
from src.inference.classify import (
    BertTunningClassifier,
    ConfidenceTier,
    OodEvidence,
    OodScores,
    OodThresholds,
    decide_review_route,
    is_out_of_distribution,
)
from src.inference.pipeline import predict_pdf
from src.ood import compute_tfidf_stats, save_stats
from src.schema import ClassEmbeddingStats, ExtractionMetadata, PredictResult
from src.settings import Settings


def _make_stats() -> ClassEmbeddingStats:
    # 700 points/class, not 5 -- the empirical Mahalanobis p-value's minimum possible
    # value is 1/(N+1). OOD_MAHALANOBIS_P_THRESHOLD was recalibrated to 0.001
    # (2026-07-12, BETO v2), so N must exceed ~999 for a query point to ever be able
    # to drop below it regardless of how far away it actually is.
    n_per_class = 700
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        # All points sit exactly on their own class's centroid (distance 0) -- degenerate
        # but sufficient here, since these tests only need "near" vs. "far" distinguishable.
        knn_train_embeddings=np.array([[0.0] * 8] * n_per_class + [[5.0] * 8] * n_per_class),
        knn_train_labels=[0] * n_per_class + [1] * n_per_class,
    )


def _make_tight_cosine_stats() -> ClassEmbeddingStats:
    # A tighter cosine_calibration_std than _make_stats() — models in practice cluster
    # tightly around their centroid's direction, so a modest directional deviation is
    # already many standard deviations away. Used to isolate a cosine-only anomaly
    # without also tripping the Mahalanobis signal. std is small enough that the fixed
    # raw cosine distance produced by the test embedding still exceeds
    # Settings.OOD_COSINE_THRESHOLD after calibration.
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[5.0] * 8, [-5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=0.001,
        knn_train_embeddings=np.array([[5.0] * 8] * 5 + [[-5.0] * 8] * 5),
        knn_train_labels=[0] * 5 + [1] * 5,
    )


def _make_stats_with_isolated_knn_cluster() -> ClassEmbeddingStats:
    # Centroids sit exactly where the mocked [CLS] embedding for "decreto" lands (the origin),
    # so Mahalanobis and cosine both see a perfect in-distribution match. But the individual
    # k-NN training points stored for "decreto" are a tight cluster far away from that centroid
    # — a decoupling that's only possible because ClassEmbeddingStats stores centroids and
    # knn_train_embeddings as independent fields. This isolates the k-NN signal: it should fire
    # even though the other two signals pass.
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.array([[50.0] * 8] * 5 + [[5.0] * 8] * 5),
        knn_train_labels=[0] * 5 + [1] * 5,
    )


def _make_stats_with_no_knn_training_data_for_decreto() -> ClassEmbeddingStats:
    # "decreto" (label 0) has zero stored k-NN training points, so knn_mean_distance
    # returns NaN for any document predicted as "decreto" — the fail-safe case.
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.array([[5.0] * 8] * 5),
        knn_train_labels=[1] * 5,
    )


def _make_mock_classifier() -> BertTunningClassifier:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    tokenizer.return_value = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(1, 512, dtype=torch.long),
        "attention_mask": torch.ones(1, 512, dtype=torch.long),
    }

    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.return_value.logits = torch.tensor([[2.0, 0.5]])
    model.return_value.hidden_states = [torch.zeros(1, 512, 8)]

    with patch("torch.cuda.is_available", return_value=False):
        return BertTunningClassifier("fake/model/path", tokenizer=tokenizer, model=model)


def test_is_out_of_distribution_false_when_all_signals_pass() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0)
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is False


def test_is_out_of_distribution_true_when_mahalanobis_fires() -> None:
    scores = OodScores(mahalanobis_p=0.0001, cosine_z=0.0, knn_distance=1.0)
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_true_when_cosine_fires() -> None:
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=Settings.OOD_COSINE_THRESHOLD + 1, knn_distance=1.0
    )
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_true_when_knn_fires() -> None:
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=0.0, knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD + 1
    )
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_true_when_knn_distance_is_nan() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=float("nan"))
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_false_when_tfidf_signal_absent_and_others_pass() -> None:
    # tfidf_cosine_z=nan (signal not available for this model) must not make the document
    # anomalous by itself -- NaN here means "skip, fail open," the OPPOSITE of
    # knn_distance's NaN, which means "fail closed, treat as anomalous." The two NaNs
    # represent different situations (whole-model signal absence vs. one document's
    # predicted class having zero training points) and are handled with opposite polarity
    # on purpose -- see the Global Constraints note in this plan.
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=float("nan")
    )
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0)
    assert is_out_of_distribution(scores, thresholds) is False


def test_is_out_of_distribution_true_when_tfidf_z_exceeds_threshold() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=10.0)
    thresholds = OodThresholds(
        mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0, tfidf_cosine_z=2.5
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_false_when_tfidf_z_below_threshold() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=1.0)
    thresholds = OodThresholds(
        mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0, tfidf_cosine_z=2.5
    )
    assert is_out_of_distribution(scores, thresholds) is False


def test_confidence_tier_from_confidence_at_or_above_threshold_is_confident() -> None:
    assert ConfidenceTier.from_confidence(0.70, 0.70) is ConfidenceTier.CONFIDENT
    assert ConfidenceTier.from_confidence(0.90, 0.70) is ConfidenceTier.CONFIDENT


def test_confidence_tier_from_confidence_below_threshold_is_uncertain() -> None:
    assert ConfidenceTier.from_confidence(0.50, 0.70) is ConfidenceTier.UNCERTAIN


def test_ood_evidence_from_in_distribution_true_is_not_anomalous() -> None:
    assert OodEvidence.from_in_distribution(in_distribution=True) is OodEvidence.NOT_ANOMALOUS


def test_ood_evidence_from_in_distribution_none_is_not_anomalous() -> None:
    assert OodEvidence.from_in_distribution(in_distribution=None) is OodEvidence.NOT_ANOMALOUS


def test_ood_evidence_from_in_distribution_false_is_anomalous() -> None:
    assert OodEvidence.from_in_distribution(in_distribution=False) is OodEvidence.ANOMALOUS


def test_decide_review_route_accept_when_confident_and_not_anomalous() -> None:
    route = decide_review_route(
        confidence_tier=ConfidenceTier.CONFIDENT, ood_evidence=OodEvidence.NOT_ANOMALOUS
    )
    assert route == "accept"


def test_decide_review_route_llm_judge_when_uncertain_and_not_anomalous() -> None:
    route = decide_review_route(
        confidence_tier=ConfidenceTier.UNCERTAIN, ood_evidence=OodEvidence.NOT_ANOMALOUS
    )
    assert route == "llm_judge"


def test_decide_review_route_human_review_when_anomalous_regardless_of_confidence() -> None:
    assert (
        decide_review_route(
            confidence_tier=ConfidenceTier.CONFIDENT, ood_evidence=OodEvidence.ANOMALOUS
        )
        == "human_review"
    )
    assert (
        decide_review_route(
            confidence_tier=ConfidenceTier.UNCERTAIN, ood_evidence=OodEvidence.ANOMALOUS
        )
        == "human_review"
    )


def test_predict_text_returns_expected_keys() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("Decreto numero 123")
    assert result.label in {"decreto", "ordenanza"}
    assert isinstance(result.confidence, float)
    assert isinstance(result.certain, bool)
    assert isinstance(result.all_scores, dict)
    # Regression guard: PredictResult(all_scores=...) is constructed with a snake_case
    # kwarg while alias_generator=to_camel is set — without populate_by_name=True on the
    # model, Pydantic v2 silently drops unrecognized-alias kwargs to their default ({})
    # instead of raising, so this must assert non-empty, not just isinstance.
    assert result.all_scores == {"decreto": 0.8176, "ordenanza": 0.1824}


def test_predict_text_certain_above_threshold() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    # softmax([2.0, 0.5]) ≈ [0.818, 0.182], above threshold 0.70
    assert result.certain is True


def test_predict_text_without_stats_leaves_ood_fields_none() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None


def test_predict_text_degrades_gracefully_when_no_knn_training_data() -> None:
    # ood_stats.npz with populated centroids but empty knn_train_embeddings/labels (e.g. a
    # hand-edited or partially-corrupted stats file) -- OOD scoring must be skipped entirely,
    # the same as when _ood_stats is None, not raise ValueError from downstream ranking code.
    clf = _make_mock_classifier()
    clf._ood_stats = ClassEmbeddingStats(  # noqa: SLF001
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
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None


def test_predict_text_with_stats_populates_ood_fields() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert isinstance(result.ood_metrics.mahalanobis_p_value, float)
    assert isinstance(result.ood_metrics.mahalanobis_p_value_theoretical, float)
    assert isinstance(result.ood_metrics.cosine_z, float)
    assert isinstance(result.ood_metrics.in_distribution, bool)


def test_predict_text_attaches_tfidf_cosine_z_when_available() -> None:
    clf = _make_mock_classifier()
    texts = ["decreto rosario municipal"] * 5 + ["ordenanza cordoba concejo"] * 5
    labels = [0] * 5 + [1] * 5
    tfidf = compute_tfidf_stats(texts, labels, ["decreto", "ordenanza"], max_features=20)
    clf._ood_stats = _make_stats().model_copy(  # noqa: SLF001
        update={
            "tfidf_vocabulary_terms": tfidf.vocabulary_terms,
            "tfidf_idf": tfidf.idf,
            "tfidf_centroids": tfidf.centroids,
            "tfidf_cosine_calibration_mean": tfidf.cosine_calibration_mean,
            "tfidf_cosine_calibration_std": tfidf.cosine_calibration_std,
        }
    )
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("decreto rosario municipal")
    # OodMetrics.tfidf_cosine_z stays Optional -- predict_text is where the internal NaN
    # sentinel translates back to None, same as save_stats/load_stats already do at the
    # .npz storage boundary for the scalar threshold fields.
    assert result.ood_metrics is not None
    assert result.ood_metrics.tfidf_cosine_z is not None


def test_predict_text_leaves_tfidf_cosine_z_none_when_stats_predate_feature() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001 -- tfidf_vocabulary_terms defaults to []
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.tfidf_cosine_z is None


def test_predict_text_mahalanobis_p_value_is_empirical() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (mock hidden_states), exactly centroid A -- distance 0,
    # so the empirical p-value must be exactly 1.0 (all 1400 reference points have
    # distance >= 0).
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value == pytest.approx(1.0)


def test_predict_text_in_distribution_when_matching_a_centroid_exactly() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # first centroid in _make_stats() — i.e. a perfectly in-distribution point.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.in_distribution is True
    assert result.review_route == "accept"


def test_predict_text_review_route_llm_judge_when_uncertain_and_in_distribution() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # softmax([0.55, 0.45]) ≈ [0.525, 0.475], below the 0.70 threshold -- uncertain -- but
    # the [CLS] embedding (zeros, from the mock hidden_states) still matches the "decreto"
    # centroid exactly, so in_distribution stays True.
    clf.model.return_value.logits = torch.tensor([[0.55, 0.45]])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.certain is False
    assert result.ood_metrics is not None
    assert result.ood_metrics.in_distribution is True
    assert result.review_route == "llm_judge"


def test_predict_text_review_route_accept_without_ood_stats() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None
    assert result.review_route == "accept"


def test_predict_text_review_route_llm_judge_without_ood_stats_when_uncertain() -> None:
    clf = _make_mock_classifier()
    clf.model.return_value.logits = torch.tensor([[0.55, 0.45]])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.certain is False
    assert result.ood_metrics is None
    assert result.review_route == "llm_judge"


def test_predict_text_flags_out_of_distribution_via_mahalanobis_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # A point far from both centroids: with all 1400 reference (training) distances equal
    # to 0, the empirical p-value for any nonzero-distance query collapses to
    # 1 / (1400 + 1) ≈ 0.000714, safely below OOD_MAHALANOBIS_P_THRESHOLD (0.001) -- but
    # pointing in the exact same direction as centroid B, so cosine distance is ~0 and
    # only the Mahalanobis signal should fire.
    far_embedding = torch.full((1, 512, 8), 100.0)
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [far_embedding]
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value < Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"


def test_predict_text_flags_out_of_distribution_via_cosine_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_tight_cosine_stats()  # noqa: SLF001
    # A point close to centroid ["decreto"] ([5]*8) in Euclidean/Mahalanobis terms (squared
    # distance is 8, well under the chi-squared critical value for df=8) but rotated just
    # enough in direction to be several cosine-calibration standard deviations away — so
    # only the cosine signal should fire.
    embedding = torch.zeros(1, 512, 8)
    embedding[0, 0, :] = torch.tensor([6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [embedding]
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value >= Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z > Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"


def test_predict_text_flags_out_of_distribution_via_knn_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats_with_isolated_knn_cluster()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # "decreto" centroid — Mahalanobis and cosine both pass — but the "decreto" k-NN training
    # points are stored far away ([50]*8), so only the k-NN signal should fire.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value >= Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.knn_distance > Settings.OOD_KNN_DISTANCE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"


def test_predict_text_flags_out_of_distribution_when_knn_distance_is_nan() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats_with_no_knn_training_data_for_decreto()  # noqa: SLF001
    # [CLS] embedding is all zeros, exactly the "decreto" centroid — Mahalanobis and cosine
    # both pass — but "decreto" has zero k-NN training points, so knn_mean_distance returns
    # NaN. `nan > threshold` is False in Python, so without an explicit guard this would
    # silently pass as in-distribution; it must instead be treated as anomalous (fail safe).
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert np.isnan(result.ood_metrics.knn_distance)
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"


def test_predict_pdf_attaches_extraction_metadata() -> None:
    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    fake_result = PredictResult(label="decreto", confidence=0.9, certain=True)
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier") as mock_clf_cls,
    ):
        mock_clf = MagicMock()
        mock_clf.predict_text.return_value = fake_result
        mock_clf_cls.return_value = mock_clf
        result = predict_pdf("fake/model", "doc.pdf")

    mock_clf.predict_text.assert_called_once_with("hola mundo")
    assert result.extracted_text == "hola mundo"
    assert result.extractor_used == "OCRExtractor"


def test_predict_pdf_returns_extraction_failed_result_when_text_missing() -> None:
    fake_extraction = ExtractionMetadata(text=None, extractor_used=None, char_count=0)
    with (
        patch("src.inference.pipeline.extract_pdf_with_metadata", return_value=fake_extraction),
        patch("src.inference.pipeline.BertTunningClassifier"),
    ):
        result = predict_pdf("fake/model", "doc.pdf")

    assert result.label is None
    assert result.error == "empty/unreadable document"
    assert result.extracted_text == ""
    assert result.extractor_used == ""
    assert result.review_route == "human_review"


def test_load_ood_stats_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert BertTunningClassifier._load_ood_stats(str(tmp_path)) is None  # noqa: SLF001


def test_load_ood_stats_returns_stats_when_file_present(tmp_path: Path) -> None:
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")
    loaded = BertTunningClassifier._load_ood_stats(str(tmp_path))  # noqa: SLF001
    assert loaded is not None
    assert loaded.class_names == ["decreto", "ordenanza"]


def test_classifier_raises_when_ood_stats_class_names_mismatch_model_id2label(
    tmp_path: Path,
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats()  # class_names=["decreto", "ordenanza"] -- swap the order below
    mismatched_stats = stats.model_copy(update={"class_names": ["ordenanza", "decreto"]})
    save_stats(mismatched_stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(BertTunningError, match="do not match"),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)


def test_classifier_loads_fine_when_ood_stats_class_names_match(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    save_stats(_make_stats(), tmp_path / "ood_stats.npz")  # class_names already match order

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001


def test_classifier_raises_when_ood_stats_model_identity_mismatches(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    # Same class_names/order as the loaded model (passes the existing class-mapping check),
    # but computed from a different model_type -- this is the exact gap the class-name-only
    # check can't catch.
    stats = _make_stats().model_copy(update={"model_type": "bert", "model_hidden_size": 768})
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(BertTunningError, match="different model architecture"),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)


def test_classifier_loads_fine_when_ood_stats_model_identity_matches(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    stats = _make_stats().model_copy(update={"model_type": "xlm-roberta", "model_hidden_size": 768})
    save_stats(stats, tmp_path / "ood_stats.npz")

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001


def test_classifier_skips_model_identity_check_when_stats_predate_the_field(
    tmp_path: Path,
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    # _make_stats() doesn't set model_type/model_hidden_size -- both default to None,
    # simulating an ood_stats.npz written before this field existed. No raise expected.
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001


def test_classifier_skips_validation_when_no_ood_stats(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is None  # noqa: SLF001


def test_classifier_warns_when_thresholds_fall_back_to_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    # _make_stats() leaves all three thresholds at their None default -- fully uncalibrated.
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert any("falling back to Settings.OOD_*" in record.message for record in caplog.records)
    assert any("mahalanobis_p_threshold" in record.message for record in caplog.records)
    assert any("cosine_threshold" in record.message for record in caplog.records)
    assert any("knn_distance_threshold" in record.message for record in caplog.records)


def test_classifier_does_not_warn_when_thresholds_are_fully_calibrated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats().model_copy(
        update={
            "mahalanobis_p_threshold": 0.001,
            "mahalanobis_threshold_status": "calibrated",
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 16.7908,
        }
    )
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any("falling back to Settings.OOD_*" in record.message for record in caplog.records)


def test_classifier_logs_info_not_warning_when_mahalanobis_threshold_refused_degenerate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats().model_copy(
        update={
            "mahalanobis_threshold_status": "refused_degenerate",
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 16.7908,
        }
    )
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.INFO),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any(
        record.levelno == logging.WARNING and "mahalanobis" in record.message.lower()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.INFO and "refused" in record.message.lower()
        for record in caplog.records
    )


def test_classifier_does_not_warn_when_no_ood_stats(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any("falling back to Settings.OOD_*" in record.message for record in caplog.records)
