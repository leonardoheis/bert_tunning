from unittest.mock import MagicMock, patch

import torch

from src.inference.classify import ClassiflowClassifier


def _make_mock_classifier() -> ClassiflowClassifier:
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
        mock_mdl.return_value = model

        clf = ClassiflowClassifier.__new__(ClassiflowClassifier)
        clf.tokenizer = tokenizer
        clf.model = model
        clf.threshold = 0.70
        clf.device = "cpu"
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
