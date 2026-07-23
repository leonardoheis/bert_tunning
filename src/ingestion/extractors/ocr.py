import logging
import threading

import easyocr
import fitz
import numpy as np
import torch

from src.ingestion._text import clean_text
from src.ingestion.exceptions import OCRError
from src.ingestion.extractors._base import ExtractorBase

log = logging.getLogger(__name__)


class OCRExtractor(ExtractorBase):
    def __init__(self) -> None:
        self._reader: easyocr.Reader | None = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Eagerly initializes the EasyOCR reader -- called once at server startup so the
        model download/integrity-check happens before any request can race it."""
        self._get_reader()

    def _get_reader(self) -> easyocr.Reader:
        # Real double-checked locking -- functools.lru_cache's lock only protects its
        # cache dict from corruption, it does NOT serialize the wrapped call itself, so
        # two threads can both miss the cache and both construct easyocr.Reader(...)
        # concurrently (reproduced in production as an EOFError, then a FileNotFoundError,
        # from two concurrent inits racing EasyOCR's own model download/integrity check).
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    log.info("Initializing EasyOCR reader (first use — may take ~10s)")
                    self._reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
                    log.info("EasyOCR reader ready")
        return self._reader

    def extract(self, pdf_path: str) -> str:
        try:
            reader = self._get_reader()
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
            return clean_text(text.strip())
        except Exception as exc:
            raise OCRError(pdf_path, exc) from exc
