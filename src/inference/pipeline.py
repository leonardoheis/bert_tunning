import logging
from pathlib import Path

import pandas as pd

from src.inference.classify import BertTunningClassifier
from src.ingestion.extract import extract_pdf
from src.schema import PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)


def predict_pdf(
    model_path: str,
    pdf_path: str,
    *,
    threshold: float = Settings.PREDICT_THRESHOLD,
    use_ocr: bool = True,
) -> PredictResult:
    clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
    log.info("Classifying: %s", Path(pdf_path).name)
    text = extract_pdf(pdf_path, use_ocr_fallback=use_ocr)

    if text is None:
        log.warning("Could not extract text from %s", Path(pdf_path).name)
        return PredictResult(
            label=None,
            confidence=Settings.PREDICT_CONFIDENCE,
            certain=False,
            error="empty/unreadable document",
            filename=Path(pdf_path).name,
        )

    result = clf.predict_text(text)
    result = result.model_copy(update={"filename": Path(pdf_path).name})
    log.info("%s → %s (%.2f%%)", Path(pdf_path).name, result.label, result.confidence * 100)
    return result


def predict_folder(
    model_path: str,
    folder_path: str,
    *,
    threshold: float = Settings.PREDICT_THRESHOLD,
    use_ocr: bool = True,
) -> pd.DataFrame:
    clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
    pdfs = sorted(Path(folder_path).glob("*.pdf"))
    log.info("Classifying %d PDFs in %s", len(pdfs), folder_path)

    results: list[PredictResult] = []
    for pdf in pdfs:
        text = extract_pdf(str(pdf), use_ocr_fallback=use_ocr)
        if text is None:
            results.append(
                PredictResult(
                    filename=pdf.name,
                    label=None,
                    confidence=Settings.PREDICT_CONFIDENCE,
                    certain=False,
                    error="empty/unreadable document",
                )
            )
            continue
        r = clf.predict_text(text)
        r = r.model_copy(update={"filename": pdf.name})
        results.append(r)

    log.info("Folder classification complete")
    return pd.DataFrame([r.model_dump() for r in results])
