import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.inference.pipeline import predict_pdf

router = APIRouter()


class PredictResponse(BaseModel):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = {}
    error: str | None = None


_model_path: str = ""
_threshold: float = 0.70


def configure(model_path: str, threshold: float = 0.70) -> None:
    global _model_path, _threshold  # noqa: PLW0603
    _model_path = model_path
    _threshold = threshold


@router.post("/predict")
async def predict(file: Annotated[UploadFile, File()]) -> PredictResponse:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        result = predict_pdf(_model_path, tmp_path, threshold=_threshold, use_ocr=True)
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # noqa: ASYNC240

    return PredictResponse(**result)
