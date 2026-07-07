import logging
from pathlib import Path

from src.inference.classify import BertTunningClassifier
from src.ingestion.extract import extract_pdf_with_metadata
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
    extraction = extract_pdf_with_metadata(pdf_path, use_ocr_fallback=use_ocr)

    if not extraction.text:
        log.warning("Could not extract text from %s", name)
        return _extraction_failed(name)

    result = clf.predict_text(extraction.text)
    result = result.model_copy(
        update={
            "filename": name,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
        }
    )
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
        extraction = extract_pdf_with_metadata(str(pdf), use_ocr_fallback=use_ocr)
        if not extraction.text:
            results.append(_extraction_failed(pdf.name))
            continue
        r = clf.predict_text(extraction.text)
        r = r.model_copy(
            update={
                "filename": pdf.name,
                "extracted_text": extraction.text,
                "extractor_used": extraction.extractor_used or "",
            }
        )
        results.append(r)

    log.info("Folder classification complete")
    return results
