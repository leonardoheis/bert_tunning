import logging
from pathlib import Path

from src.exceptions import BertTunningError
from src.ingestion._text import clean_text
from src.ingestion.extractors import ExtractorBase, MarkItDownExtractor, OCRExtractor
from src.schema import ExtractionMetadata
from src.settings import Settings

__all__ = ["clean_text", "extract_pdf", "extract_pdf_with_metadata"]

log = logging.getLogger(__name__)

_CHAIN: list[ExtractorBase] = [MarkItDownExtractor(), OCRExtractor()]


def extract_pdf_with_metadata(
    pdf_path: str, *, use_ocr_fallback: bool = True
) -> ExtractionMetadata:
    chain = _CHAIN if use_ocr_fallback else _CHAIN[:1]
    name = Path(pdf_path).name
    text, failed = "", 0
    extractor_used: str | None = None

    for extractor in chain:
        if len(text) >= Settings.MIN_TEXT_FOR_OCR:
            break
        try:
            text = extractor.extract(pdf_path)
            extractor_used = type(extractor).__name__
        except BertTunningError as e:
            log.warning("%s failed on %s: %s", type(extractor).__name__, name, e)
            failed += 1

    if failed == len(chain):
        msg = f"All extractors failed on {name}"
        raise BertTunningError(msg)

    if len(text) < Settings.MIN_USABLE_TEXT:
        log.warning("Skipping %s — could not extract usable text", name)
        return ExtractionMetadata(text=None, extractor_used=None, char_count=len(text))

    return ExtractionMetadata(text=text, extractor_used=extractor_used, char_count=len(text))


def extract_pdf(pdf_path: str, *, use_ocr_fallback: bool = True) -> str | None:
    return extract_pdf_with_metadata(pdf_path, use_ocr_fallback=use_ocr_fallback).text
