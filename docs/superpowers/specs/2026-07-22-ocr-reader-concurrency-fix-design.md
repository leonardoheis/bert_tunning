# Fix EasyOCR Reader Race Condition — Design Spec

## Motivation

Reproduced in production logs on a fresh container: two concurrent `/predict` jobs both
needed OCR for the first time at nearly the same moment. EasyOCR detected an MD5 mismatch
on its model file and started deleting-and-re-downloading it; the second concurrent call
tried to read the same file mid-delete and crashed with `FileNotFoundError: [Errno 2] No
such file or directory: '/root/.EasyOCR//model/craft_mlt_25k.pth'`. An earlier occurrence
of the same underlying race surfaced as a different symptom (`EOFError` reading a
partially-written file after a container restart mid-download).

**Root cause**: `OCRExtractor._get_reader()` (`src/ingestion/extractors/ocr.py:20-25`)
relies on `functools.lru_cache(maxsize=1)` for "lazy, thread-safe, compute-once
initialization" — but that's incorrect. `lru_cache`'s internal lock only protects the
cache *dictionary* from corruption; it does not serialize the wrapped function call
itself. Two threads can both miss the cache and both call `easyocr.Reader(...)`
concurrently before either finishes populating the cache. This directly contradicts what
`CLAUDE.md` documents as the resolved design ("A `threading.Lock()` with double-checked
locking ensures EasyOCR is initialized exactly once across threads") — the code was at
some point refactored to `lru_cache` (see `task/39-ponytail-cleanup` in this repo's
history) on the mistaken assumption it gave an equivalent guarantee. `threading.Lock` is
the correct primitive here, not `asyncio.Lock`: `extract_pdf_with_metadata` runs inside
`asyncio.to_thread(...)` (`_run_prediction_job`), so concurrent `/predict` jobs call
`OCRExtractor.extract()` from separate OS threads in a pool, not separate asyncio tasks
on one event loop — `asyncio.Lock` would not touch this race at all.

## Scope

- Fix the race with real double-checked locking (`threading.Lock`), restoring the
  guarantee `CLAUDE.md` already documents as intended.
- Additionally pre-warm the reader once at server startup, so the model
  download/integrity-check happens before any request can race it at all, and so the
  first real user request isn't the one that eats a multi-minute download.
- **Deliberately out of scope**: baking the model into the Docker image at build time
  (eliminates runtime downloads entirely — the more robust long-term fix, but a bigger
  Dockerfile change deserving its own decision, not bundled into this bug fix). Retry-
  with-backoff around reader construction was considered and rejected — it would mask a
  genuine corruption behind a "worked on retry" success instead of fixing the race that
  causes it.

## Design

### `OCRExtractor` — real double-checked locking (`src/ingestion/extractors/ocr.py`)

Replaces `functools.lru_cache(maxsize=1)` with an explicit `threading.Lock()` held for
the *entire* `easyocr.Reader(...)` construction — a second concurrent caller blocks until
the first is completely done, instead of racing it. Adds a public `warm()` method so
startup pre-warming (below) doesn't need to reach into a private method from another
module.

```python
import logging
import threading

import easyocr
import fitz
import numpy as np
import torch

from src.ingestion._text import clean_text
from src.ingestion.exceptions import OCRError
from src.ingestion.extractors._base import ExtractorBase

log = logging.getLogger(__name__)


class OCRExtractor(ExtractorBase):
    def __init__(self) -> None:
        self._reader: easyocr.Reader | None = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Eagerly initializes the EasyOCR reader -- called once at server startup so the
        model download/integrity-check happens before any request can race it."""
        self._get_reader()

    def _get_reader(self) -> easyocr.Reader:
        if self._reader is None:
            with self._lock:
                if self._reader is None:
                    log.info("Initializing EasyOCR reader (first use — may take ~10s)")
                    self._reader = easyocr.Reader(["es"], gpu=torch.cuda.is_available())
                    log.info("EasyOCR reader ready")
        return self._reader

    def extract(self, pdf_path: str) -> str:
        try:
            reader = self._get_reader()
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )
                results = reader.readtext(img, detail=0, paragraph=True)
                text += " ".join(results) + "\n"
            doc.close()
            return clean_text(text.strip())
        except Exception as exc:
            raise OCRError(pdf_path, exc) from exc
```

### Warming the *shared* instance, not a throwaway one (`src/ingestion/extract.py`)

`_CHAIN` already holds the one `OCRExtractor` instance actually used by
`extract_pdf_with_metadata` (constructed once at module import). Startup pre-warming has
to reach that exact instance — constructing a fresh `OCRExtractor()` elsewhere and
warming *that* would warm a throwaway reader nobody uses, leaving the real one still lazy.
A small module-level helper keeps `_CHAIN` itself private to this module rather than
exposing it directly to `src/api/app.py`:

```python
def warm_ocr_reader() -> None:
    """Eagerly initializes the shared OCRExtractor's EasyOCR reader -- see
    OCRExtractor.warm() for why this needs to happen once, before any request."""
    for extractor in _CHAIN:
        if isinstance(extractor, OCRExtractor):
            extractor.warm()
            return
```

Added to `__all__` alongside the module's existing exports.

### Pre-warming at startup (`src/api/app.py`)

```python
from src.ingestion.extract import warm_ocr_reader

def create_app(model_path: str, threshold: float = 0.70) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.clf = BertTunningClassifier(model_path, confidence_threshold=threshold)
        await asyncio.to_thread(warm_ocr_reader)
        yield
    ...
```

`asyncio.to_thread` avoids blocking the event loop during EasyOCR's model
download/verification, which the container logs show can take multiple minutes on a slow
network — matching the same pattern already used for blocking calls elsewhere in this
codebase (`extract_pdf_with_metadata`, `clf.predict_text` in `_run_prediction_job`). This
means the server won't finish starting up (won't accept traffic) until the OCR reader is
ready — an intentional trade-off: a slightly slower `docker run`/`serve` startup in
exchange for zero chance of the race recurring and no request ever eating that latency
itself.

`main.py`'s `serve_cmd` (local CLI) goes through the same `create_app()`, so it gets this
fix automatically — no separate change needed there.

## Backward compatibility

- No public API/schema changes. `OCRExtractor.extract()`'s signature and behavior are
  unchanged; only how `_get_reader()` initializes is different.
- Server startup takes longer on a cold cache (first-ever run, or after the EasyOCR model
  cache is cleared) — the download that previously happened lazily on whichever request
  needed it first now happens unconditionally at startup instead. Acceptable for this
  internal tool; flagging directly rather than treating it as free.
- No test currently exercises `OCRExtractor` directly (confirmed: no `tests/ingestion/test_ocr.py` exists) — this fix should add one for the double-checked-locking behavior itself (e.g. concurrent calls to `_get_reader()` only construct the underlying reader once), not just rely on the existing `extract_pdf_with_metadata`-level tests.

## Touch list

| Path | Change |
|---|---|
| `src/ingestion/extractors/ocr.py` | Replace `functools.lru_cache` with `threading.Lock`-based double-checked locking; add `warm()` |
| `src/ingestion/extract.py` | Add `warm_ocr_reader()`, exported via `__all__` |
| `src/api/app.py` | `lifespan` calls `await asyncio.to_thread(warm_ocr_reader)` after constructing `app.state.clf` |
| `tests/ingestion/test_ocr.py` (new) | Test that concurrent `_get_reader()` calls only construct the reader once |
