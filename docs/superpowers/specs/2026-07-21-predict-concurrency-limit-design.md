# Bound Concurrent `/predict` Jobs — Design Spec

## Motivation

Uploading 6+ files at once crashes the pod. The frontend's batch upload already fires every
file's `/predict` call at once (`Promise.allSettled`, all requests in flight together — see
`App.tsx`'s `handleSubmit`), and since the prediction-progress work
(`docs/superpowers/specs/2026-07-20-frontend-prediction-progress-design.md`), each request
starts a `BackgroundTasks` job (`_run_prediction_job`) that runs unthrottled: nothing in the
current code caps how many of those jobs actually execute at the same time. Each one does a
full BERT forward pass and, for scanned PDFs, an OCR fallback — the heaviest parts of this
codebase, memory-wise. Six-plus of those running simultaneously multiplies memory use until
the container's memory limit (or the host's, if none is set) is exceeded and the kernel
SIGKILLs the process. This is an OOM kill, not an application exception — `_run_prediction_job`
already wraps its work in `try/except Exception` (`endpoints.py`), which is exactly why that
existing handling doesn't save it: a SIGKILL from outside the process can't be caught by
anything running inside it.

The fix has to live server-side, not client-side, since it needs to protect the pod
regardless of who's calling `/predict` or how many files they submit at once (this frontend
today, potentially something else tomorrow).

## Scope

- Backend: cap how many `_run_prediction_job` runs execute concurrently, queuing the rest.
- A new `queued` stage so a job waiting for a free slot is represented honestly, instead of
  falsely showing `extracting` before any extraction has actually started.
- Frontend: recognize the new stage in the existing progress indicator. No client-side
  submission throttling — the server-side cap is the actual protection; limiting the client
  too would just be two places to keep in sync for no added safety.
- **Out of scope:** setting an actual pod/container memory limit (infrastructure, not this
  repo) — this spec makes the app behave safely *within whatever memory it's given*, which is
  the correct fix regardless of what limit ends up configured, or if none is.

## Design

### `Settings.PREDICT_MAX_CONCURRENCY` (`src/settings.py`)

```python
# How many /predict jobs (extraction + classification) may run at the same time. Each job
# does a full BERT forward pass and, for scanned PDFs, OCR -- running too many at once is
# what was crashing the pod (OOM kill) when 6+ files were uploaded in one batch. No pod
# memory limit is currently set, so this defaults conservatively; raise it once you've
# measured actual memory headroom.
PREDICT_MAX_CONCURRENCY: int = 2
```

Placed in the `# ── Server ──` section, next to `MAX_UPLOAD_SIZE_BYTES` — same kind of
resource-safety knob.

### Semaphore in `src/api/routes/predict/endpoints.py`

```python
_PREDICT_SEMAPHORE = asyncio.Semaphore(Settings.PREDICT_MAX_CONCURRENCY)
```

Module-level, matching `_JOBS`' existing scope — correct for this deployment shape (the
Dockerfile runs a single process, `CMD ["python", "-m", "src"]`, one event loop; a
module-level `asyncio.Semaphore` genuinely bounds total concurrency here, unlike in a
multi-worker setup where each worker would get its own).

`_run_prediction_job` acquires it around the actual heavy work — not around the upload's
temp-file write, which already happens synchronously in the request handler before the job
is even queued, and isn't the expensive part:

```python
async def _run_prediction_job(
    job_id: str, tmp_path: str, filename: str, clf: BertTunningClassifier
) -> None:
    try:
        async with _PREDICT_SEMAPHORE:
            _JOBS[job_id] = PredictJob(stage="extracting")
            extraction = await asyncio.to_thread(
                extract_pdf_with_metadata, tmp_path, use_ocr_fallback=True
            )
            if not extraction.text:
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
            _JOBS[job_id] = PredictJob(stage="done", result=_to_predict_response(result))
    except Exception as exc:  # noqa: BLE001
        _JOBS[job_id] = PredictJob(stage="error", error=str(exc))
    finally:
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)
```

The `POST /predict` handler sets the job's initial stage to `queued` (not `extracting`, which
now only gets set once the semaphore is actually acquired):

```python
    job_id = uuid4().hex
    _JOBS[job_id] = PredictJob(stage="queued")
    background_tasks.add_task(_run_prediction_job, job_id, tmp_path, file.filename, clf)
    return PredictJobCreated(job_id=job_id)
```

### `PredictJob.stage` (`src/api/routes/predict/schemas.py`)

```python
class PredictJob(BaseSchema):
    stage: Literal["queued", "extracting", "classifying", "done", "error"]
    result: PredictResponse | None = None
    error: str | None = None
```

### Frontend

`frontend/src/types/api.ts`: `PredictStage` gains `"queued"`.

`frontend/src/components/PredictionProgress.tsx`: `STEPS` gains `"queued"` as the first
step — the stepper dots already render generically off this array, so no other change is
needed there.

## Backward compatibility

Purely additive: one new `Literal` value on an existing field. Nothing currently branches on
the *absence* of `"queued"` in a way that would break — the frontend's stepper renders
whatever stage index it's given, and no test asserts the exhaustive set of possible stages.

## Touch list

| Path | Change |
|---|---|
| `src/settings.py` | add `PREDICT_MAX_CONCURRENCY` |
| `src/api/routes/predict/schemas.py` | `PredictJob.stage` gains `"queued"` |
| `src/api/routes/predict/endpoints.py` | module-level `_PREDICT_SEMAPHORE`; `predict()` sets initial stage `"queued"`; `_run_prediction_job` acquires the semaphore around the real work |
| `frontend/src/types/api.ts` | `PredictStage` gains `"queued"` |
| `frontend/src/components/PredictionProgress.tsx` | `STEPS` gains `"queued"` |

## Open questions

None outstanding. `PREDICT_MAX_CONCURRENCY` defaults to `2` since no pod memory limit is
currently known/set (confirmed) — raise it via `.env` once actual headroom is measured; no
code change needed to tune it.
