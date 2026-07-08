import functools
import logging

import easyocr
import fitz
import numpy as np
import torch

from src.ingestion._text import clean_text
from src.ingestion.exceptions import OCRError
from src.ingestion.extractors._base import ExtractorBase

log = logging.getLogger(__name__)


class OCRExtractor(ExtractorBase):
    # ponytail: lru_cache is CPython's own lock-protected memoization, so a zero-arg
    # instance method decorated this way already gives lazy, thread-safe, compute-once
    # initialization — no need to hand-roll double-checked locking for the same guarantee.
    @functools.lru_cache(maxsize=1)  # noqa: B019
    def _get_reader(self) -> easyocr.Reader:
        log.info("Initializing EasyOCR reader (first use — may take ~10s)")
        reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
        log.info("EasyOCR reader ready")
        return reader

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
