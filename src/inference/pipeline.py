import logging
from pathlib import Path

from src.inference.classify import BertTunningClassifier
from src.ingestion._text import detect_foreign_municipality
from src.ingestion.extract import extract_pdf_with_metadata
from src.schemas import ExtractionMetadata, PredictResult
from src.settings import Settings

log = logging.getLogger(__name__)

# Point value each named smell contributes to PredictResult.risk_score, summed. OOD
# signals and an unreadable document are weighted equally (3) -- each is independently
# "this prediction can't be trusted," not a matter of degree. foreign_municipality (2) is
# strong but categorical evidence, not a statistical signal. low_confidence (1) is the
# weakest signal alone -- confidence being borderline doesn't by itself mean much.
_SMELL_WEIGHTS: dict[str, int] = {
    "low_mahalanobis_p": 3,
    "high_cosine_z": 3,
    "high_knn_distance": 3,
    "high_tfidf_z": 3,
    "unreadable_document": 3,
    "foreign_municipality": 2,
    "low_svm_margin": 2,
    "low_confidence": 1,
}


def _compute_risk_score(smells: list[str]) -> int:
    return sum(_SMELL_WEIGHTS.get(smell, 0) for smell in smells)


def extraction_failed(filename: str) -> PredictResult:
    """The empty/unreadable-document result -- one place defining this rule, reused by
    predict_pdf, predict_folder, and the /predict API route (src/api/routes/predict).
    Never goes through attach_metadata() (there's no extraction to attach metadata
    from), so its smells/risk_score are set directly here rather than derived."""
    smells = ["unreadable_document"]
    return PredictResult(
        filename=filename,
        label=None,
        confidence=Settings.PREDICT_CONFIDENCE,
        certain=False,
        error="empty/unreadable document",
        review_route="human_review",
        smells=smells,
        risk_score=_compute_risk_score(smells),
    )


def attach_metadata(
    result: PredictResult, filename: str, extraction: ExtractionMetadata
) -> PredictResult:
    foreign_match = detect_foreign_municipality(extraction.text or "")
    smells = list(result.smells)
    if foreign_match is not None:
        smells.append("foreign_municipality")
    return result.model_copy(
        update={
            "filename": filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
            "smells": smells,
            "risk_score": _compute_risk_score(smells),
        }
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
        return extraction_failed(name)

    result = clf.predict_text(extraction.text)
    result = attach_metadata(result, name, extraction)
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
            results.append(extraction_failed(pdf.name))
            continue
        r = clf.predict_text(extraction.text)
        r = attach_metadata(r, pdf.name, extraction)
        results.append(r)

    log.info("Folder classification complete")
    return results
