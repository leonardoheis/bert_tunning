from unittest.mock import patch

from src.ingestion.extract import extract_pdf, extract_pdf_with_metadata


class _StubExtractor:
    def __init__(self, name: str, text: str) -> None:
        self.__class__.__name__ = name
        self._text = text

    def extract(self, pdf_path: str) -> str:  # noqa: ARG002
        return self._text


def test_extract_pdf_with_metadata_reports_markitdown_success() -> None:
    with patch(
        "src.ingestion.extract._CHAIN",
        [_StubExtractor("MarkItDownExtractor", "decreto numero uno " * 10)],
    ):
        result = extract_pdf_with_metadata("fake.pdf")
    assert result.text is not None
    assert result.extractor_used == "MarkItDownExtractor"
    assert result.char_count == len(result.text)


def test_extract_pdf_with_metadata_reports_none_when_text_too_short() -> None:
    with patch("src.ingestion.extract._CHAIN", [_StubExtractor("MarkItDownExtractor", "hi")]):
        result = extract_pdf_with_metadata("fake.pdf")
    assert result.text is None
    assert result.extractor_used is None


def test_extract_pdf_still_returns_plain_string_or_none() -> None:
    with patch(
        "src.ingestion.extract._CHAIN",
        [_StubExtractor("MarkItDownExtractor", "decreto numero uno " * 10)],
    ):
        text = extract_pdf("fake.pdf")
    assert isinstance(text, str)
