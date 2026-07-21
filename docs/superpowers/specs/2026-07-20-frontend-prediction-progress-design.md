# Frontend: Per-File Prediction Progress Indicator — Design Spec

## Motivation

Second of the four frontend increments (table → **progress indicator** → config page → visual
redesign) — this replaces the originally-drafted "timeline chart" spec. That draft plotted
*finished* results (confidence/OOD scores) after a batch completed; what was actually wanted
is visibility *while* each file is still being processed — which pipeline stage a file is
currently in (extracting text, then classifying), not a chart of settled metrics. The chart
idea is dropped, not deferred — this is a correction, not an additional feature.

Today `/predict` (`src/api/routes/predict/endpoints.py`) is one blocking call: it reads the
upload, extracts text, classifies, and returns the full result in a single response. Nothing
in that path is observable from outside until it's completely done, so there's no signal for
the frontend to render mid-flight.

## Scope

- Backend: `/predict` changes from "do the work, return the result" to "start the work, return
  a job id" plus a new `GET /predict/status/{job_id}` the frontend polls.
- Frontend: replaces the current single `fetch` + `await response.json()` per file with
  "submit, then poll until done," and renders each in-flight file's current stage.
- Two real stages exist in the pipeline as written today — `extracting` (`extract_pdf_with_metadata`,
  including the possible OCR fallback, the slower of the two) and `classifying`
  (`BertTunningClassifier.predict_text` — forward pass + OOD scoring + SVM scoring, not
  further split; splitting classify.py's internals into sub-stages isn't justified by
  anything in this request). Terminal states: `done` (with the full result) or `error`.

## Design

### Backend: job store + background execution

A module-level in-memory dict is enough — this is a single-process dev server with one human
uploading batches, not a multi-worker production API needing shared job state.

```python
# ponytail: in-memory dict, grows for the life of the process (no eviction). Fine for a
# dev tool serving one person's batch uploads; add TTL-based eviction if this ever becomes
# a long-lived shared server with many users.
_JOBS: dict[str, PredictJob] = {}
```

```python
# src/api/routes/predict/schemas.py
class PredictJob(BaseSchema):
    stage: Literal["extracting", "classifying", "done", "error"]
    result: PredictResponse | None = None
    error: str | None = None

class PredictJobCreated(BaseSchema):
    job_id: str
```

```python
# src/api/routes/predict/endpoints.py
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
    _JOBS[job_id] = PredictJob(stage="extracting")
    background_tasks.add_task(_run_prediction_job, job_id, tmp_path, file.filename, clf)
    return PredictJobCreated(job_id=job_id)


@router.get("/predict/status/{job_id}")
async def predict_status(job_id: str) -> PredictJob:
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job


async def _run_prediction_job(
    job_id: str, tmp_path: str, filename: str, clf: BertTunningClassifier
) -> None:
    try:
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
    except Exception as exc:  # noqa: BLE001 -- boundary between a background task and its job record; an uncaught exception here has no other way to reach the client than through this record
        _JOBS[job_id] = PredictJob(stage="error", error=str(exc))
    finally:
        await asyncio.to_thread(Path(tmp_path).unlink, missing_ok=True)
```

The existing 400 (bad extension) / 413 (oversized upload) checks stay synchronous, before a
job is even created — those are upload-validation failures, not pipeline-stage failures, and
returning them immediately (not via polling) is unchanged from today's behavior.

### Frontend: submit + poll

```typescript
// frontend/src/api.ts
export type PredictStage = "extracting" | "classifying" | "done" | "error";

export async function predict(
  file: File,
  onStage?: (stage: PredictStage) => void,
): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);
  const created = await fetch("/predict", { method: "POST", body: formData }).then((r) =>
    r.json(),
  );

  while (true) {
    const job = await fetch(`/predict/status/${created.jobId}`).then((r) => r.json());
    onStage?.(job.stage);
    if (job.stage === "done") return job.result as PredictResponse;
    if (job.stage === "error") throw new Error(job.error ?? "Prediction failed");
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}
```

`App.tsx` gains a `stages: Record<string, PredictStage>` state (keyed by filename) alongside
the existing `results` state, updated via each file's `onStage` callback, and renders a new
`PredictionProgress` component for files still in flight (dropped from `stages` once a
terminal state moves it into `results`).

```typescript
// frontend/src/components/PredictionProgress.tsx (new)
const STEPS: PredictStage[] = ["extracting", "classifying", "done"];

export function PredictionProgress({ stages }: { stages: Record<string, PredictStage> }) {
  const inFlight = Object.entries(stages).filter(([, stage]) => stage !== "done");
  if (inFlight.length === 0) return null;

  return (
    <ul className="mb-4 space-y-1 text-sm">
      {inFlight.map(([filename, stage]) => (
        <li key={filename} className="flex items-center gap-2">
          <span className="w-48 truncate">{filename}</span>
          {stage === "error" ? (
            <span className="text-red-600">error</span>
          ) : (
            <span className="flex gap-1">
              {STEPS.map((step) => (
                <span
                  key={step}
                  className={`h-2 w-2 rounded-full ${
                    STEPS.indexOf(step) <= STEPS.indexOf(stage) ? "bg-blue-600" : "bg-gray-300"
                  }`}
                />
              ))}
            </span>
          )}
          <span className="text-gray-500">{stage}</span>
        </li>
      ))}
    </ul>
  );
}
```

500ms poll interval, no backoff — matches this project's existing bias toward the simplest
thing that works for a low-volume internal tool (same bar as the no-lock model-swap
simplification in the config-page spec).

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/predict/schemas.py` | add `PredictJob`, `PredictJobCreated` |
| `src/api/routes/predict/endpoints.py` | `/predict` returns a job id + runs in background; new `GET /predict/status/{job_id}` |
| `frontend/src/api.ts` | `predict()` submits then polls, takes an `onStage` callback |
| `frontend/src/App.tsx` | `stages` state, passes `onStage` through `handleSubmit` |
| `frontend/src/components/PredictionProgress.tsx` (new) | per-file stage indicator |

## Backward compatibility

`/predict`'s response shape changes (`PredictResponse` directly → `PredictJobCreated`) — a
breaking change to the HTTP contract. The only consumer is this frontend (verified: no CLI
command or other code calls `/predict` over HTTP; `predict`/`predict-folder` run inference
in-process, not through the API), so this is safe within this repo, but worth flagging for
anyone hitting `/predict` directly (e.g. via `curl` or Postman during manual testing).

## Open question for spec review

None outstanding — this replaces the prior spec's unconfirmed "timeline chart" assumption
with the progress-indicator interpretation you confirmed directly.
