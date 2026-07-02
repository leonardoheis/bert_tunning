import logging
from pathlib import Path

from src.inference.classify import BertTunningClassifier
from src.ingestion.extract import extract_pdf
from src.schema import PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)


def _extraction_failed(filename: str) -> PredictResult:
    return PredictResult(
        filename=filename,
        label=None,
        confidence=Settings.PREDICT_CONFIDENCE,
        certain=False,
        error="empty/unreadable document",
    )


def predict_pdf(
    model_path: str,
    pdf_path: str,
    *,
    threshold: float = Settings.PREDICT_THRESHOLD,
    use_ocr: bool = True,
) -> PredictResult:
    clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
    name = Path(pdf_path).name
    log.info("Classifying: %s", name)
    text = extract_pdf(pdf_path, use_ocr_fallback=use_ocr)

    if not text:
        log.warning("Could not extract text from %s", name)
        return _extraction_failed(name)

    result = clf.predict_text(text)
    result = result.model_copy(update={"filename": name})
    log.info("%s → %s (%.2f%%)", name, result.label, result.confidence * 100)
    return result


def predict_folder(
    model_path: str,
    folder_path: str,
    *,
    threshold: float = Settings.PREDICT_THRESHOLD,
    use_ocr: bool = True,
) -> list[PredictResult]:
    clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
    pdfs = sorted(Path(folder_path).glob("*.pdf"))
    log.info("Classifying %d PDFs in %s", len(pdfs), folder_path)

    results: list[PredictResult] = []
    for pdf in pdfs:
        text = extract_pdf(str(pdf), use_ocr_fallback=use_ocr)
        if not text:
            results.append(_extraction_failed(pdf.name))
            continue
        r = clf.predict_text(text)
        r = r.model_copy(update={"filename": pdf.name})
        results.append(r)

    log.info("Folder classification complete")
    return results
