import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.inference.classify import BertTunningClassifier
from src.inference.pipeline import extraction_failed
from src.ingestion._text import detect_foreign_municipality
from src.ingestion.extract import extract_pdf_with_metadata
from src.schema import PredictResult
from src.settings import Settings

from .schemas import PredictResponse

router = APIRouter(tags=["Prediction"])


_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Reads file in bounded chunks instead of one unconditional await file.read() -- an
    unbounded read loads the entire upload into worker memory before any size check ever
    runs, so a sufficiently large upload can exhaust memory regardless of what happens
    afterward. FastAPI/uvicorn impose no default request-body limit on their own."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"File exceeds the {max_bytes} byte upload limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _get_clf(request: Request) -> BertTunningClassifier:
    return cast("BertTunningClassifier", request.app.state.clf)


def _to_predict_response(result: PredictResult) -> PredictResponse:
    data = result.model_dump()
    return PredictResponse(
        filename=data["filename"],
        label=data["label"],
        confidence=data["confidence"],
        certain=data["certain"],
        all_scores=data["all_scores"],
        error=data["error"] or None,
        ood_metrics=result.ood_metrics,
        extracted_text=data["extracted_text"],
        extractor_used=data["extractor_used"],
        review_route=data["review_route"],
        foreign_municipality=data["foreign_municipality"],
        foreign_municipality_context=data["foreign_municipality_context"],
    )


@router.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    clf: Annotated[BertTunningClassifier, Depends(_get_clf)],
) -> PredictResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await _read_upload_bounded(file, Settings.MAX_UPLOAD_SIZE_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        extraction = await asyncio.to_thread(
            extract_pdf_with_metadata, tmp_path, use_ocr_fallback=True
        )
    finally:
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)

    if not extraction.text:
        return _to_predict_response(extraction_failed(file.filename))

    result = await asyncio.to_thread(clf.predict_text, extraction.text)
    foreign_match = detect_foreign_municipality(extraction.text or "")
    result = result.model_copy(
        update={
            "filename": file.filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
        }
    )
    return _to_predict_response(result)
