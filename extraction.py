import logging
import re
from pathlib import Path
from typing import Optional

import fitz
import easyocr
import numpy as np
import torch
from markitdown import MarkItDown

log = logging.getLogger(__name__)

_md = MarkItDown()

# Initialized once on first OCR call — loading the EasyOCR model takes ~10s
_ocr_reader: Optional[easyocr.Reader] = None


def _get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
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
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )
            results = reader.readtext(img, detail=0, paragraph=True)
            text += " ".join(results) + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        log.error("OCR error on %s: %s", Path(pdf_path).name, e)
        return ""


def clean_text(text: str) -> str:
    text = text.replace("\f", " ")
    text = text.replace("\xa0", " ")
    text = re.sub(r'\|[-: ]+\|[-: |]+', '', text)
    text = re.sub(r'^\|.*\|$', '', text, flags=re.M)
    text = re.sub(r'#+ ', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def extract_pdf(pdf_path: str, use_ocr_fallback: bool = True) -> Optional[str]:
    try:
        result = _md.convert(pdf_path)
        text = clean_text(result.text_content or "")
    except Exception as e:
        log.warning("MarkItDown error on %s: %s", Path(pdf_path).name, e)
        text = ""

    if len(text) < 50 and use_ocr_fallback:
        log.info("Scanned PDF detected, running OCR: %s", Path(pdf_path).name)
        text = clean_text(_ocr_fallback(pdf_path))

    if len(text) < 20:
        log.warning("Skipping %s — could not extract usable text", Path(pdf_path).name)
        return None

    return text
