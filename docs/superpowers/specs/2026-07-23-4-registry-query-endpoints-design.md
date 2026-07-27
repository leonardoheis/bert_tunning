# Piece 4/4: Registry Query + Review Endpoints — Design Spec

Part of a 4-piece decomposition (see the other 3 specs dated 2026-07-23):
1. Risk score/smells fields — prerequisite.
2. Registry module + CLI wiring — prerequisite (`list_predictions()`/`update_review_status()` already built here).
3. API wiring — prerequisite (rows worth querying only exist once this piece is in place).
4. **This spec** — the HTTP endpoints over the registry: listing/filtering and setting review status + a corrected label.

Grounded against the same example workbook as piece 1 — its `Review status` and
`Corrected label` columns confirmed the review workflow is a manual status
(`pending`/`confirmed`/`corrected`) plus, specifically when correcting, what the actual
label should have been.

## Motivation

Pieces 2/3 make every prediction persist with a `review_status` defaulting to `"pending"`.
Nothing yet lets a human see the backlog of pending predictions or act on one. This piece
is the two endpoints that make the registry actually useful: list/filter what's pending,
and record a decision.

## Scope

- `GET /predictions` — list, filterable by `review_status`, paginated.
- `PATCH /predictions/{id}` — set `review_status`, and `corrected_label` when the status
  is `"corrected"`.
- **Out of scope**: any frontend UI consuming these endpoints (its own follow-up spec,
  explicitly deferred back in piece 1's parent scope discussion) — this piece is
  curl/Swagger-testable only, matching the pattern piece 2/3 already used.

## Design

### Schemas (`src/api/routes/registry/schemas.py`, new)

```python
from typing import Literal

from src.api.schema import BaseSchema

ReviewStatus = Literal["pending", "confirmed", "corrected"]


class PredictionRecord(BaseSchema):
    id: int
    created_at: str
    source: Literal["api", "cli"]
    filename: str
    label: str | None
    confidence: float | None
    risk_score: int
    smells: list[str]
    review_status: ReviewStatus
    corrected_label: str | None
    review_route: str | None
    in_distribution: bool | None
    # ... remaining columns from the predictions table, same fields already in
    # flatten_predict_result()'s output


class UpdateReviewRequest(BaseSchema):
    review_status: ReviewStatus
    corrected_label: str | None = None
```

### Endpoints (`src/api/routes/registry/endpoints.py`, new)

```python
from fastapi import APIRouter, HTTPException

from src.registry import list_predictions, update_review_status

from .schemas import PredictionRecord, UpdateReviewRequest

router = APIRouter(tags=["Registry"])


@router.get("/predictions")
def get_predictions(
    review_status: str | None = None, limit: int = 50, offset: int = 0
) -> list[PredictionRecord]:
    rows = list_predictions(review_status=review_status, limit=limit, offset=offset)
    return [PredictionRecord(**_deserialize(row)) for row in rows]


@router.patch("/predictions/{prediction_id}")
def update_prediction_review(prediction_id: int, body: UpdateReviewRequest) -> PredictionRecord:
    if body.review_status == "corrected" and not body.corrected_label:
        raise HTTPException(
            status_code=400, detail="corrected_label is required when review_status is 'corrected'"
        )
    updated = update_review_status(
        prediction_id, body.review_status, corrected_label=body.corrected_label
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Unknown prediction id")
    row = list_predictions(limit=1, offset=0)  # simplest re-fetch; see open question below
    return PredictionRecord(**_deserialize(row[0]))
```

`_deserialize()` un-JSONs the `smells`/`all_scores`/`svm_scores` columns and converts
SQLite's `0`/`1` integers back to `bool` for `certain`/`in_distribution`/
`svm_agrees_with_prediction` — small helper, not worth its own module.

Registered in `src/api/routes/__init__.py`'s `ROUTERS` tuple, alongside `health_router`/
`prediction_router`.

**Open question for spec review**: `update_prediction_review`'s re-fetch after update
(`list_predictions(limit=1, offset=0)`) is wrong as sketched — it fetches the *most recent*
row, not necessarily the one just updated. `list_predictions()` (piece 2) doesn't currently
support filtering by `id`. Fix before implementing: either add a `get_prediction(id)`
function to `src/registry.py` (cleanest — a single-row lookup by primary key, small
addition to piece 2's module), or have `update_review_status()` return the updated row
directly instead of a bool. Leaning toward `get_prediction(id)` since "fetch one record" is
a reasonable registry operation to have anyway (e.g. useful outside this endpoint too), but
flagging rather than deciding silently since it changes piece 2's already-written module.

## How to test this piece

With some predictions already in the registry (from pieces 2/3's testing):

```bash
curl http://localhost:8000/predictions
curl "http://localhost:8000/predictions?reviewStatus=pending&limit=10"
curl -X PATCH http://localhost:8000/predictions/1 \
  -H "Content-Type: application/json" \
  -d '{"reviewStatus": "confirmed"}'
curl -X PATCH http://localhost:8000/predictions/2 \
  -H "Content-Type: application/json" \
  -d '{"reviewStatus": "corrected", "correctedLabel": "ordenanza"}'
```

Confirm:
- `GET /predictions` returns everything, newest first, matching what pieces 2/3 wrote
  directly to SQLite.
- `?reviewStatus=pending` actually filters (only rows still at the default show up).
- `PATCH .../1` with `reviewStatus: "confirmed"` updates that row's status; a follow-up
  `GET /predictions` reflects it.
- `PATCH .../2` with `reviewStatus: "corrected"` but no `correctedLabel` returns 400 (per
  the validation above).
- `PATCH` on an id that doesn't exist returns 404.
- Swagger UI (`/docs`) shows both new endpoints correctly, request/response schemas
  render sensibly.

## Backward compatibility

Purely additive — two new routes, no existing endpoint's behavior changes.

## Touch list

| Path | Change |
|---|---|
| `src/api/routes/registry/` (new package) | `endpoints.py`, `schemas.py` — `GET /predictions`, `PATCH /predictions/{id}` |
| `src/api/routes/__init__.py` | Register the new router in `ROUTERS` |
| `src/registry.py` | Add `get_prediction(id)` (see open question) or adjust `update_review_status()`'s return shape |
