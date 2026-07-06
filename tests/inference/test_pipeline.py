from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.inference.classify import BertTunningClassifier
from src.inference.ood import save_stats
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
    )


def _make_tight_cosine_stats() -> ClassEmbeddingStats:
    # A tighter cosine_calibration_std than _make_stats() — models in practice cluster
    # tightly around their centroid's direction, so a modest directional deviation is
    # already many standard deviations away. Used to isolate a cosine-only anomaly
    # without also tripping the Mahalanobis signal.
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[5.0] * 8, [-5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=0.005,
    )


def _make_mock_classifier() -> BertTunningClassifier:
    asc_path = "src.inference.classify.AutoModelForSequenceClassification.from_pretrained"
    with (
        patch("src.inference.classify.AutoTokenizer.from_pretrained") as mock_tok,
        patch(asc_path) as mock_mdl,
        patch("torch.cuda.is_available", return_value=False),
    ):
        tokenizer = MagicMock()
        tokenizer.model_max_length = 512
        tokenizer.return_value = MagicMock()
        tokenizer.return_value.to.return_value = {
            "input_ids": torch.zeros(1, 512, dtype=torch.long),
            "attention_mask": torch.ones(1, 512, dtype=torch.long),
        }
        mock_tok.return_value = tokenizer

        model = MagicMock()
        model.config.id2label = {0: "decreto", 1: "ordenanza"}
        logits = torch.tensor([[2.0, 0.5]])
        model.return_value.logits = logits
        model.return_value.hidden_states = [torch.zeros(1, 512, 8)]
        mock_mdl.return_value = model

        model.config.max_position_embeddings = 512

        clf = BertTunningClassifier.__new__(BertTunningClassifier)
        clf.tokenizer = tokenizer
        clf.model = model
        clf.threshold = 0.70
        clf.device = "cpu"
        clf.max_length = 512
        clf._ood_stats = None  # noqa: SLF001
        return clf


def test_predict_text_returns_expected_keys() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("Decreto numero 123")
    assert result.label in {"decreto", "ordenanza"}
    assert isinstance(result.confidence, float)
    assert isinstance(result.certain, bool)
    assert isinstance(result.all_scores, dict)


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
    assert result.mahalanobis_p_value is None
    assert result.cosine_z is None
    assert result.in_distribution is None


def test_predict_text_with_stats_populates_ood_fields() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert isinstance(result.mahalanobis_p_value, float)
    assert isinstance(result.cosine_z, float)
    assert isinstance(result.in_distribution, bool)


def test_predict_text_in_distribution_when_matching_a_centroid_exactly() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # first centroid in _make_stats() — i.e. a perfectly in-distribution point.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.in_distribution is True


def test_predict_text_flags_out_of_distribution_via_mahalanobis_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # A point far from both centroids in Euclidean/Mahalanobis terms (squared distance to
    # the nearest centroid [5]*8 is 8*95**2, vastly over the chi-squared critical value for
    # df=8), but pointing in the exact same direction as that centroid — cosine distance to
    # it is ~0 — so only the Mahalanobis signal should fire.
    far_embedding = torch.full((1, 512, 8), 100.0)
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [far_embedding]
        result = clf.predict_text("anything")
    assert result.mahalanobis_p_value is not None
    assert result.cosine_z is not None
    assert result.mahalanobis_p_value < Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.in_distribution is False


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
    assert result.mahalanobis_p_value is not None
    assert result.cosine_z is not None
    assert result.mahalanobis_p_value >= Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.cosine_z > Settings.OOD_COSINE_THRESHOLD
    assert result.in_distribution is False


def test_load_ood_stats_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert BertTunningClassifier._load_ood_stats(str(tmp_path)) is None  # noqa: SLF001


def test_load_ood_stats_returns_stats_when_file_present(tmp_path: Path) -> None:
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")
    loaded = BertTunningClassifier._load_ood_stats(str(tmp_path))  # noqa: SLF001
    assert loaded is not None
    assert loaded.class_names == ["decreto", "ordenanza"]
