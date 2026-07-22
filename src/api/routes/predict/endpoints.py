import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

from src.inference.classify import BertTunningClassifier
from src.inference.pipeline import extraction_failed
from src.ingestion._text import detect_foreign_municipality
from src.ingestion.extract import extract_pdf_with_metadata
from src.schema import PredictResult
from src.settings import Settings

from .schemas import PredictJob, PredictJobCreated, PredictResponse

log = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# ponytail: in-memory dict, grows for the life of the process (no eviction). Fine for a
# dev tool serving one person's batch uploads; add TTL-based eviction if this ever becomes
# a long-lived shared server with many users.
_JOBS: dict[str, PredictJob] = {}

# Bounds how many jobs actually run (extraction + classification) at once -- see
# Settings.PREDICT_MAX_CONCURRENCY. Module-level is correct here: this Dockerfile runs a
# single process (CMD ["python", "-m", "src"]), one event loop, so this genuinely caps
# total concurrency for the whole pod.
_PREDICT_SEMAPHORE = asyncio.Semaphore(Settings.PREDICT_MAX_CONCURRENCY)


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
        svm_scores=data["svm_scores"],
        svm_predicted_label=data["svm_predicted_label"],
        svm_agrees_with_prediction=data["svm_agrees_with_prediction"],
    )


async def _run_prediction_job(
    job_id: str, tmp_path: str, filename: str, clf: BertTunningClassifier
) -> None:
    try:
        async with _PREDICT_SEMAPHORE:
            log.info("[%s] Classifying: %s", job_id, filename)
            _JOBS[job_id] = PredictJob(stage="extracting")
            extraction = await asyncio.to_thread(
                extract_pdf_with_metadata, tmp_path, use_ocr_fallback=True
            )
            if not extraction.text:
                log.warning("[%s] Could not extract text from %s", job_id, filename)
                _JOBS[job_id] = PredictJob(
                    stage="done", result=_to_predict_response(extraction_failed(filename))
                )
                return

            _JOBS[job_id] = PredictJob(stage="classifying")
            result = await asyncio.to_thread(clf.predict_text, extraction.text)
            foreign_match = detect_foreign_municipality(extraction.text or "")
            result = result.model_copy(
                update={
                    "filename": filename,
                    "extracted_text": extraction.text,
                    "extractor_used": extraction.extractor_used or "",
                    "foreign_municipality": foreign_match.name if foreign_match else None,
                    "foreign_municipality_context": (
                        foreign_match.context if foreign_match else None
                    ),
                }
            )
            log.info(
                "[%s] %s -> %s (%.2f%%)",
                job_id,
                filename,
                result.label,
                result.confidence * 100,
            )
            _JOBS[job_id] = PredictJob(stage="done", result=_to_predict_response(result))
    except Exception as exc:
        # job record; an uncaught exception here has no other way to reach the client than
        # through this record
        log.exception("[%s] Prediction failed for %s", job_id, filename)
        _JOBS[job_id] = PredictJob(stage="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)


@router.post("/predict")
async def predict(
    file: Annotated[UploadFile, File()],
    clf: Annotated[BertTunningClassifier, Depends(_get_clf)],
    background_tasks: BackgroundTasks,
) -> PredictJobCreated:
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await _read_upload_bounded(file, Settings.MAX_UPLOAD_SIZE_BYTES)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    job_id = uuid4().hex
    _JOBS[job_id] = PredictJob(stage="queued")
    background_tasks.add_task(_run_prediction_job, job_id, tmp_path, file.filename, clf)
    return PredictJobCreated(job_id=job_id)


@router.get("/predict/status/{job_id}")
async def predict_status(job_id: str) -> PredictJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job
