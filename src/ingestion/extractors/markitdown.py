from markitdown import MarkItDown

from src.ingestion._text import clean_text
from src.ingestion.exceptions import PDFExtractionError
from src.ingestion.extractors._base import ExtractorBase


class MarkItDownExtractor(ExtractorBase):
    def __init__(self) -> None:
        self._md = MarkItDown()

    def extract(self, pdf_path: str) -> str:
        try:
            result = self._md.convert(pdf_path)
            return clean_text(result.text_content or "")
        except Exception as exc:
            raise PDFExtractionError(pdf_path, exc) from exc
