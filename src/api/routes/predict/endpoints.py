import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from src.inference.classify import BertTunningClassifier
from src.ingestion.extract import extract_pdf

from .schemas import PredictResponse

router = APIRouter(tags=["Prediction"])


def _get_clf(request: Request) -> BertTunningClassifier:
    return cast("BertTunningClassifier", request.app.state.clf)


@router.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    clf: Annotated[BertTunningClassifier, Depends(_get_clf)],
) -> PredictResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        text = await asyncio.to_thread(extract_pdf, tmp_path, use_ocr_fallback=True)
    finally:
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)

    if text is None:
        return PredictResponse(
            filename=file.filename,
            label=None,
            confidence=0.0,
            certain=False,
            error="empty/unreadable document",
        )

    result = await asyncio.to_thread(clf.predict_text, text)
    result = result.model_copy(update={"filename": file.filename})
    data = result.model_dump()
    return PredictResponse(
        filename=data["filename"],
        label=data["label"],
        confidence=data["confidence"],
        certain=data["certain"],
        all_scores=data["all_scores"],
        error=data["error"] or None,
    )
