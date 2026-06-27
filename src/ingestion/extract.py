import logging
import re
from pathlib import Path

import easyocr
import fitz
import numpy as np
import torch
from markitdown import MarkItDown

log = logging.getLogger(__name__)

_md = MarkItDown()
_ocr_reader: easyocr.Reader | None = None

_MIN_TEXT_FOR_OCR = 50
_MIN_USABLE_TEXT = 20


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader  # noqa: PLW0603
    if _ocr_reader is None:
        log.info("Initializing EasyOCR reader (first use — may take ~10s)")
        _ocr_reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
        log.info("EasyOCR reader ready")
    return _ocr_reader


def _ocr_fallback(pdf_path: str) -> str:
    try:
        reader = _get_ocr_reader()
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            pix = page.get_pixmap(dpi=200)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            results = reader.readtext(img, detail=0, paragraph=True)
            text += " ".join(results) + "\n"
        doc.close()
        return text.strip()
    except Exception:
        log.exception("OCR error on %s", Path(pdf_path).name)
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\f", " ").replace("\xa0", " ")
    text = re.sub(r"\|[-: ]+\|[-: |]+", "", text)
    text = re.sub(r"^\|.*\|$", "", text, flags=re.MULTILINE)
    text = re.sub(r"#+ ", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(pdf_path: str, *, use_ocr_fallback: bool = True) -> str | None:
    try:
        result = _md.convert(pdf_path)
        text = clean_text(result.text_content or "")
    except Exception:  # noqa: BLE001
        log.warning("MarkItDown failed on %s — trying OCR", Path(pdf_path).name)
        text = ""

    if len(text) < _MIN_TEXT_FOR_OCR and use_ocr_fallback:
        log.info("Scanned PDF detected, running OCR: %s", Path(pdf_path).name)
        text = clean_text(_ocr_fallback(pdf_path))

    if len(text) < _MIN_USABLE_TEXT:
        log.warning("Skipping %s — could not extract usable text", Path(pdf_path).name)
        return None

    return text
