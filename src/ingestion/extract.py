import logging
from pathlib import Path

from src.exceptions import BertTunningError
from src.ingestion._text import clean_text
from src.ingestion.extractors import ExtractorBase, MarkItDownExtractor, OCRExtractor
from src.settings import Settings

__all__ = ["clean_text", "extract_pdf"]

log = logging.getLogger(__name__)

_CHAIN: list[ExtractorBase] = [MarkItDownExtractor(), OCRExtractor()]


def extract_pdf(pdf_path: str, *, use_ocr_fallback: bool = True) -> str | None:
    chain = _CHAIN if use_ocr_fallback else _CHAIN[:1]
    name = Path(pdf_path).name
    text, failed = "", 0

    for extractor in chain:
        if len(text) >= Settings.MIN_TEXT_FOR_OCR:
            break
        try:
            text = extractor.extract(pdf_path)
        except BertTunningError as e:
            log.warning("%s failed on %s: %s", type(extractor).__name__, name, e)
            failed += 1

    if failed == len(chain):
        msg = f"All extractors failed on {name}"
        raise BertTunningError(msg)

    if len(text) < Settings.MIN_USABLE_TEXT:
        log.warning("Skipping %s — could not extract usable text", name)
        return None

    return text
