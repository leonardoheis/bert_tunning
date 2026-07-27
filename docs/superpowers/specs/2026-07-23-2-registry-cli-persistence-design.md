# Piece 2/4: SQLite Registry + CLI Persistence — Design Spec

Part of a 4-piece decomposition (see the other 3 specs dated 2026-07-23):
1. Risk score/smells fields (`PredictResult`) — prerequisite for this piece.
2. **This spec** — the SQLite registry module, wired into the CLI (`predict`/`predict-folder`).
3. API wiring (SQLite writes from `/predict`).
4. Registry query/update endpoints (`GET /predictions`, `PATCH /predictions/{id}`).

**Additional prerequisite, added later:** `2026-07-27-smell-review-suggested-design.md`
adds one more field to `PredictResult` (`smell_review_suggested`). Implement it before this
piece so the registry persists the final field set from the start, not a schema patch
after the fact.

## Motivation

Review status (piece 4) needs to survive across requests and container restarts — this
project has no database today, everything else is files (`.npz`, `.joblib`, parquet) or
in-memory state explicitly fine to lose (`_JOBS`). A manual review decision isn't fine to
lose. Confirmed: SQLite via stdlib `sqlite3`, no new dependency — and it becomes the
registry for **every** prediction, CLI and API both, not just a side table for review
status alone. There has to be something to look up before you can set a status on it. This
piece builds the registry module and wires the CLI side first — the smaller, single-
threaded half — before piece 3 adds the API side, which has real concurrency to think
about (`PREDICT_MAX_CONCURRENCY` allows more than one job writing at once).

## Scope

- New `src/registry.py`: SQLite schema, `record_prediction()`, `update_review_status()`,
  `list_predictions()` (the latter two used by piece 4, built now since they're trivial
  alongside the schema/connection code).
- `predict_pdf`/`predict_folder`/`extraction_failed` (`src/inference/pipeline.py`) call
  `record_prediction(result, source="cli")`.
- **Out of scope**: API wiring (piece 3), the query/update HTTP endpoints (piece 4,
  though the underlying functions are built here), any frontend change.

## Design

### `src/registry.py` (new)

Top-level, not under `inference/`, for the same used-by-both-CLI-and-API reason as
`ood.py`/`svm_reviewer.py`/`wandb.py`.

```python
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.schemas import PredictResult, flatten_predict_result
from src.settings import Settings

_DB_PATH = Path(Settings.CACHE_PATH).parent / "bert_tunning_registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('api', 'cli')),
    filename TEXT NOT NULL,
    label TEXT,
    confidence REAL,
    certain INTEGER,
    error TEXT,
    mahalanobis_p_value REAL,
    mahalanobis_p_value_theoretical REAL,
    cosine_z REAL,
    knn_distance REAL,
    tfidf_cosine_z REAL,
    in_distribution INTEGER,
    foreign_municipality TEXT,
    review_route TEXT,
    extractor_used TEXT,
    svm_predicted_label TEXT,
    svm_agrees_with_prediction INTEGER,
    risk_score INTEGER NOT NULL DEFAULT 0,
    smells TEXT NOT NULL,              -- JSON list
    all_scores TEXT,                   -- JSON dict
    svm_scores TEXT,                   -- JSON dict
    review_status TEXT NOT NULL DEFAULT 'pending'
        CHECK (review_status IN ('pending', 'confirmed', 'corrected')),
    corrected_label TEXT                -- set when review_status = 'corrected'
);
"""


# ponytail: a short-lived connection per call, not one shared long-lived connection --
# sqlite3 connections aren't safe to share across threads, and this project's write
# volume (one row per prediction, human-scale) doesn't justify a connection pool. WAL
# mode below is what actually matters for concurrent-write safety at this volume.
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    return conn


def record_prediction(result: PredictResult, *, source: Literal["api", "cli"]) -> int:
    """Inserts one row, returns its id."""
    row = flatten_predict_result(result)
    with _connect() as conn:
        cursor = conn.execute(
            """INSERT INTO predictions (
                created_at, source, filename, label, confidence, certain, error,
                mahalanobis_p_value, mahalanobis_p_value_theoretical, cosine_z,
                knn_distance, tfidf_cosine_z, in_distribution, foreign_municipality,
                review_route, extractor_used, svm_predicted_label,
                svm_agrees_with_prediction, risk_score, smells, all_scores, svm_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(tz=timezone.utc).isoformat(),
                source,
                row["filename"], row["label"], row["confidence"], row["certain"],
                row["error"], row["mahalanobis_p_value"], row["mahalanobis_p_value_theoretical"],
                row["cosine_z"], row["knn_distance"], row["tfidf_cosine_z"],
                row["in_distribution"], row["foreign_municipality"], row["review_route"],
                row["extractor_used"], row["svm_predicted_label"],
                row["svm_agrees_with_prediction"], row["risk_score"],
                json.dumps(row["smells"]), json.dumps(row["all_scores"]),
                json.dumps(row["svm_scores"]),
            ),
        )
        return cursor.lastrowid


def update_review_status(
    prediction_id: int,
    status: Literal["pending", "confirmed", "corrected"],
    *,
    corrected_label: str | None = None,
) -> bool:
    """Returns False if prediction_id doesn't exist -- callers turn that into a 404.
    corrected_label is only meaningful when status="corrected" -- piece 4's endpoint
    validates that pairing before calling this, this function just stores whatever it's
    given (e.g. explicitly clearing corrected_label back to None if a status is reverted
    from "corrected" to "pending")."""
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE predictions SET review_status = ?, corrected_label = ? WHERE id = ?",
            (status, corrected_label, prediction_id),
        )
        return cursor.rowcount > 0


def list_predictions(
    *, review_status: str | None = None, limit: int = 50, offset: int = 0
) -> list[dict]:
    query = "SELECT * FROM predictions"
    params: list[object] = []
    if review_status is not None:
        query += " WHERE review_status = ?"
        params.append(review_status)
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute(query, params).fetchall()]
```

`bert_tunning_registry.db` (+ `-wal`/`-shm` WAL sidecar files) lives in `data/`, next to
`bert_tunning_cache.parquet` — already a git-ignored directory per `CLAUDE.md`'s
"Git-Ignored Directories," no `.gitignore` change needed.

### CLI wiring (`src/inference/pipeline.py`)

```python
def predict_pdf(...) -> PredictResult:
    ...
    result = clf.predict_text(extraction.text)
    result = _attach_metadata(result, name, extraction)
    log.info("%s → %s (%.2f%%)", name, result.label, result.confidence * 100)
    record_prediction(result, source="cli")
    return result
```

Same one-line addition in `predict_folder`'s loop and in the `extraction_failed()` early-
return branches of both functions — a failed extraction is itself a fact worth keeping,
not just a transient error shown once and discarded.

**Failure handling, disclosed simplification**: `record_prediction()` failing (disk full,
permissions) is allowed to propagate for now, matching how every other genuinely
unexpected I/O failure in this codebase behaves (fail loud, not silently). Revisit if
registry writes ever become flaky enough in practice to justify catching and logging
instead — not pre-optimizing for a failure mode that hasn't been observed.

## How to test this piece

```powershell
uv run python main.py predict path/to/document.pdf
uv run python main.py predict-folder path/to/folder
```

Then inspect the database directly:

```powershell
sqlite3 data/bert_tunning_registry.db "SELECT id, source, filename, label, risk_score, smells, review_status FROM predictions ORDER BY id DESC LIMIT 5;"
```

Confirm:
- Every `predict`/`predict-folder` run produces one new row per document, `source='cli'`.
- `risk_score`/`smells` match what piece 1 printed to the console/CSV for the same document.
- `review_status` defaults to `'pending'` on every new row.
- Running `predict` concurrently from two terminals (simulate with two PowerShell windows)
  doesn't corrupt the file or raise `database is locked` — confirms WAL mode is doing its
  job before piece 3 adds real async concurrency on top.
- Deleting `data/bert_tunning_registry.db` and re-running `predict` recreates it cleanly
  (schema is `CREATE TABLE IF NOT EXISTS`, no separate migration step needed).

## Backward compatibility

- Purely additive — nothing currently reads from the registry, so this piece can't break
  any existing behavior, only add a side effect to `predict`/`predict-folder`.
- No `.gitignore` change needed (`data/` already excluded).

## Touch list

| Path | Change |
|---|---|
| `src/registry.py` (new) | `record_prediction()`, `update_review_status()`, `list_predictions()`, SQLite schema |
| `src/inference/pipeline.py` | `predict_pdf`/`predict_folder`/`extraction_failed` call `record_prediction(..., source="cli")` |
| `data/` | New `bert_tunning_registry.db` (+ WAL sidecar files) at runtime — already gitignored |
