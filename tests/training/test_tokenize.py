from unittest.mock import MagicMock

import torch

from src.training.tokenize import BertTunningDataset, prepare_text


def _mock_tokenizer(max_length: int = 512) -> MagicMock:
    tok = MagicMock()
    tok.model_max_length = max_length
    tok.encode.return_value = list(range(600))
    tok.decode.return_value = "decoded text"
    tok.return_value = {
        "input_ids": torch.zeros(2, max_length, dtype=torch.long),
        "attention_mask": torch.ones(2, max_length, dtype=torch.long),
    }
    return tok


def test_prepare_text_first_returns_unchanged() -> None:
    tok = _mock_tokenizer()
    assert prepare_text("hello world", tok, strategy="first") == "hello world"


def test_prepare_text_middle_decodes() -> None:
    tok = _mock_tokenizer(max_length=512)
    result = prepare_text("x" * 5000, tok, strategy="middle")
    assert isinstance(result, str)


DATASET_SIZE = 2


def test_dataset_len() -> None:
    tok = _mock_tokenizer()
    ds = BertTunningDataset(["doc1", "doc2"], [0, 1], tok, max_length=512)
    assert len(ds) == DATASET_SIZE


def test_dataset_getitem_keys() -> None:
    tok = _mock_tokenizer()
    ds = BertTunningDataset(["doc1", "doc2"], [0, 1], tok, max_length=512)
    item = ds[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
