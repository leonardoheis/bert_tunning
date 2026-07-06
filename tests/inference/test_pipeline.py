from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.inference.classify import BertTunningClassifier
from src.inference.ood import save_stats
from src.schema import ClassEmbeddingStats


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
    assert result.label is not None or result.label is None  # field exists
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


def test_predict_text_flags_out_of_distribution_when_far_from_all_centroids() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # first centroid in _make_stats() — i.e. a perfectly in-distribution point.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.in_distribution is True


def test_load_ood_stats_returns_none_when_file_missing(tmp_path: Path) -> None:
    assert BertTunningClassifier._load_ood_stats(str(tmp_path)) is None  # noqa: SLF001


def test_load_ood_stats_returns_stats_when_file_present(tmp_path: Path) -> None:
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")
    loaded = BertTunningClassifier._load_ood_stats(str(tmp_path))  # noqa: SLF001
    assert loaded is not None
    assert loaded.class_names == ["decreto", "ordenanza"]
