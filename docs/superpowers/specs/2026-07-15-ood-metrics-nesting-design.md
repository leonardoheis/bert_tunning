# OodMetrics nesting — design spec

**Date:** 2026-07-15
**Status:** Approved, not yet implemented

## Problem

`PredictResult`'s `tfidf_cosine_z: float | None` currently returns `None` for two
different reasons that no caller can distinguish:

1. No `ood_stats.npz` loaded at all for this model — `mahalanobis_p_value`,
   `mahalanobis_p_value_theoretical`, `cosine_z`, `knn_distance`, and `in_distribution`
   are also all `None` in this case (`src/inference/classify.py`, early return at
   `if self._ood_stats is None:`).
2. `ood_stats.npz` **is** loaded and the other four fields **are** populated, but this
   specific stats file predates the TF-IDF signal (`self._tfidf_vectorizer is None` →
   `NaN` → converted to `None` at the `tfidf_cosine_z` field specifically, in the same
   function's `model_copy(update={...})` block).

Found via a `/stop-using-none` audit of the branch. This is the one field, among the
five OOD score fields plus `in_distribution`, with a genuinely overloaded `None` —
every other field on `PredictResult` has exactly one reason to be `None`.

## Design

### New type — `src/schema.py`

```python
class OodMetrics(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, frozen=True, populate_by_name=True)

    mahalanobis_p_value: float
    mahalanobis_p_value_theoretical: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float | None = None  # None means: this stats file predates TF-IDF
    in_distribution: bool
```

Same `alias_generator=to_camel` / `populate_by_name=True` convention as `BaseSchema`
(`src/api/schema.py`) and `PredictResult` — defined once, reused directly by both the
internal `PredictResult` and the API `PredictResponse`, avoiding a duplicate class.

### `PredictResult` (`src/schema.py`)

The five flat fields (`mahalanobis_p_value`, `mahalanobis_p_value_theoretical`,
`cosine_z`, `knn_distance`, `tfidf_cosine_z`) and `in_distribution` are removed and
replaced by:

```python
ood_metrics: OodMetrics | None = None
```

`None` now means only "no `ood_stats.npz` loaded for this model" — unambiguous.
`ood_metrics.tfidf_cosine_z is None` means only "this stats file predates TF-IDF" —
also unambiguous, now that the outer wrapper has absorbed the other reason.

`review_route` is untouched — it's always set regardless of OOD availability
(`decide_review_route` already handles the no-OOD-signal case via
`OodEvidence.NOT_ANOMALOUS`), so it was never ambiguous and stays a top-level field.

### `src/inference/classify.py` (`predict_text`)

The `model_copy(update={...})` block that currently sets five flat keys instead
constructs one `OodMetrics(...)` and sets `"ood_metrics": OodMetrics(...)`.

### API — `src/api/routes/predict/schemas.py`, `src/api/routes/predict/endpoints.py`

`PredictResponse` drops its five flat fields and gains `ood_metrics: OodMetrics | None
= None` (reusing the same class from `src/schema.py`). `_to_predict_response` changes
from five explicit `data["x"]` mappings to passing through `result.ood_metrics`.

**JSON shape change:** `{"mahalanobisPValue": ..., "cosineZ": ..., "knnDistance": ...,
"tfidfCosineZ": ..., "inDistribution": ...}` at the top level becomes:

```json
{"oodMetrics": {"mahalanobisPValue": ..., "mahalanobisPValueTheoretical": ...,
  "cosineZ": ..., "knnDistance": ..., "tfidfCosineZ": ..., "inDistribution": ...} | null}
```

This is a breaking change to the `/predict` response shape. Confirmed acceptable —
the API has no external/production consumers yet (pre-release).

### CLI — `src/cli/predict.py`

```python
if result.mahalanobis_p_value is not None:
    ...
```
becomes
```python
if result.ood_metrics is not None:
    ...  # reads result.ood_metrics.x for each field
```

### W&B — `src/wandb.py`

`_PREDICTION_COLUMNS` keeps its current flat, unprefixed names (`mahalanobis_p_value`,
`cosine_z`, `knn_distance`, `tfidf_cosine_z`, `in_distribution`, plus
`mahalanobis_p_value_theoretical` already present) — no column renaming.

`log_predict_folder_results`'s row-building currently does a flat `model_dump()` +
`row[col] for col in _PREDICTION_COLUMNS` lookup. Since `model_dump()` will now nest
these five fields under `ood_metrics`, that flat lookup breaks. The row-building logic
needs to special-case the OOD columns, reaching into
`r.ood_metrics.x if r.ood_metrics else None` for each one, while the remaining columns
(`filename`, `label`, `confidence`, `certain`, `review_route`, `extractor_used`,
`error`, `foreign_municipality`, `foreign_municipality_context`) keep using the flat
`model_dump()` lookup unchanged.

### Tests

Every test constructing `PredictResult(..., cosine_z=4.2)` (or any of the other four
OOD fields) becomes `PredictResult(..., ood_metrics=OodMetrics(cosine_z=4.2, ...))` —
all `OodMetrics` fields are required except `tfidf_cosine_z`, so every such test needs
to supply all four other values too. Affects `tests/inference/`,
`tests/api/test_predict.py`, `tests/test_wandb.py`, and any `tests/test_ood.py` cases
constructing `PredictResult` directly with OOD fields (most of that file constructs
`ClassEmbeddingStats`/uses `OodScores`/`OodThresholds` instead, which are unaffected —
this only touches `PredictResult` construction sites).

## Out of scope

- `foreign_municipality` / `foreign_municipality_context` (PR #47) are a separate,
  already-discussed case with no real ambiguity — not touched by this design.
- No changes to `OodScores`, `OodThresholds`, `ClassEmbeddingStats`, or the OOD
  math/calibration functions in `src/ood.py` — this is purely a `PredictResult`/API
  presentation-layer restructuring.

## Branch / PR

New branch off `task/50-tfidf-ood-signal` (same base as PR #47), separate PR — this is
an unrelated concern from foreign-municipality detection and shouldn't be bundled into
PR #47.
