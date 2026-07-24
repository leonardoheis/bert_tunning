# Piece 3/4: API Wiring for the Registry — Design Spec

Part of a 4-piece decomposition (see the other 3 specs dated 2026-07-23):
1. Risk score/smells fields (`PredictResult`) — prerequisite.
2. Registry module + CLI wiring — prerequisite (this piece reuses `src/registry.py` as-is).
3. **This spec** — wiring `/predict` (the API/frontend path) into the same registry.
4. Registry query/update endpoints (`GET /predictions`, `PATCH /predictions/{id}`).

## Motivation

Piece 2 built `src/registry.py` and wired the single-threaded CLI path into it. The
API path is different in one way that actually matters: `/predict` jobs run as
`BackgroundTask`s, gated by `Settings.PREDICT_MAX_CONCURRENCY` (default `2`) — real
concurrent writers, not just "could theoretically run at the same time as another
process" the way two CLI invocations might. Piece 2's WAL-mode + short-lived-connection
design was already chosen with this in mind, so this piece is mostly "call the same
function from a second place" — but it's worth its own testable increment specifically to
verify that concurrency assumption holds under this project's actual concurrent-job path,
not just two independent terminals.

## Scope

- `_run_prediction_job` (`src/api/routes/predict/endpoints.py`) calls
  `record_prediction(result, source="api")`.
- **Out of scope**: the query/update HTTP endpoints (piece 4) — this piece only adds
  writes, nothing reads the registry back yet (beyond the direct `sqlite3` inspection
  used for testing, same as piece 2).

## Design

In `_run_prediction_job`, right after building the final `result` (the same point
`_JOBS[job_id] = PredictJob(stage="done", ...)` already gets set):

```python
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
record_prediction(result, source="api")
_JOBS[job_id] = PredictJob(stage="done", result=_to_predict_response(result))
```

Same addition on the `extraction_failed(...)` early-return branch. `record_prediction()`
is a synchronous, blocking SQLite call — already running inside `_run_prediction_job`,
which itself already runs inside the `_PREDICT_SEMAPHORE`-gated section of a
`BackgroundTask` (not the request-handling coroutine directly), so it doesn't need its
own `asyncio.to_thread` wrap the way `extract_pdf_with_metadata`/`clf.predict_text` do —
those are wrapped because they're slow (seconds); a single SQLite insert is not, and
wrapping it would add overhead for no benefit.

## How to test this piece

Start the API (`uv run python main.py serve --model-path ...` or the Docker container),
upload a batch of files through the frontend (or `curl -F file=@doc.pdf http://localhost:8000/predict`
followed by polling `/predict/status/{job_id}`), then inspect the registry:

```powershell
sqlite3 data/bert_tunning_registry.db "SELECT id, source, filename, risk_score, smells FROM predictions WHERE source='api' ORDER BY id DESC LIMIT 10;"
```

Confirm:
- Every prediction made through the frontend/API produces a row with `source='api'`.
- Uploading several files at once (exercising `PREDICT_MAX_CONCURRENCY`'s actual
  concurrent-job path, not just theoretical concurrency) produces exactly one row per
  file, no duplicates, no `database is locked` errors in the server log.
- CLI (`source='cli'`) and API (`source='api'`) rows coexist correctly in the same table
  — run a `predict` CLI command and an API upload back to back, confirm both appear.

## Backward compatibility

Purely additive — no existing behavior changes, only a new side effect on an
already-internal function (`_run_prediction_job`). No response shape change for
`/predict`/`/predict/status/{job_id}`.

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/predict/endpoints.py` | `_run_prediction_job` calls `record_prediction(..., source="api")` on both the success and extraction-failed paths |
