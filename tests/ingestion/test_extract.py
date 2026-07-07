from unittest.mock import patch

import pytest

from src.exceptions import BertTunningError
from src.ingestion.extract import extract_pdf, extract_pdf_with_metadata


def _stub_extractor(name: str, *, text: str = "", raises: bool = False) -> object:
    """Build an extractor whose class __name__ is `name`, without mutating shared state
    (a plain reusable stub class would need to mutate its own __class__.__name__, which
    corrupts every other instance/reference to that same class)."""

    def extract(self: object, pdf_path: str) -> str:  # noqa: ARG001
        if raises:
            msg = f"{name} failed"
            raise BertTunningError(msg)
        return text

    return type(name, (), {"extract": extract})()


def test_extract_pdf_with_metadata_reports_markitdown_success() -> None:
    with patch(
        "src.ingestion.extract._CHAIN",
        [_stub_extractor("MarkItDownExtractor", text="decreto numero uno " * 10)],
    ):
        result = extract_pdf_with_metadata("fake.pdf")
    assert result.text is not None
    assert result.extractor_used == "MarkItDownExtractor"
    assert result.char_count == len(result.text)


def test_extract_pdf_with_metadata_reports_none_when_text_too_short() -> None:
    with patch("src.ingestion.extract._CHAIN", [_stub_extractor("MarkItDownExtractor", text="hi")]):
        result = extract_pdf_with_metadata("fake.pdf")
    assert result.text is None
    assert result.extractor_used is None


def test_extract_pdf_with_metadata_attributes_to_second_extractor_after_first_fails() -> None:
    chain = [
        _stub_extractor("MarkItDownExtractor", raises=True),
        _stub_extractor("OCRExtractor", text="decreto numero uno " * 10),
    ]
    with patch("src.ingestion.extract._CHAIN", chain):
        result = extract_pdf_with_metadata("fake.pdf")
    assert result.text is not None
    assert result.extractor_used == "OCRExtractor"


def test_extract_pdf_with_metadata_raises_when_every_extractor_fails() -> None:
    chain = [
        _stub_extractor("MarkItDownExtractor", raises=True),
        _stub_extractor("OCRExtractor", raises=True),
    ]
    with patch("src.ingestion.extract._CHAIN", chain), pytest.raises(BertTunningError):
        extract_pdf_with_metadata("fake.pdf")


def test_extract_pdf_still_returns_plain_string_or_none() -> None:
    with patch(
        "src.ingestion.extract._CHAIN",
        [_stub_extractor("MarkItDownExtractor", text="decreto numero uno " * 10)],
    ):
        text = extract_pdf("fake.pdf")
    assert isinstance(text, str)
