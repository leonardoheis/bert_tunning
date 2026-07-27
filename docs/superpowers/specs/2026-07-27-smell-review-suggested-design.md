# Smell-Triggered Human Review Suggestion — Design Spec

**Recommended execution order:** implement this **before** registry pieces 2-4
(`2026-07-23-2/3/4-registry-*-design.md`), even though those specs were written earlier.
The registry persists `PredictResult` rows; this spec adds one more field to that shape
(`smell_review_suggested`). Landing it first means the registry captures the final field
set from day one instead of needing a schema patch once piece 2 has already shipped. See
the note added to piece 2's header cross-referencing this spec.

## Motivation

`docs/superpowers/specs/2026-07-26-smell-thresholds-design.md` deliberately kept
`risk_score`/`smells` presentational — never wired into `review_route`, so a permissive
smell profile could surface early-warning signals without silently becoming a second
decision-maker. Reviewing real batches against that design (see this session's document
walkthroughs) surfaced documents like `ordenanza_6729_1998.pdf` and `decreto_841_2026.pdf`
where `risk_score` hit `12`/`16` — every OOD smell firing at once — while
`review_route="accept"`, because the *decision* thresholds (calibrated for a low false-
positive rate) didn't trip. There was no way to surface "this scored dangerously high on
the smell profile" without opening the CSV and reading the `risk_score` column by hand.

This spec adds exactly one new signal: a boolean flag, computed from `risk_score` alone,
that a human reviewing results can filter/sort on directly — without changing what
`review_route` means or reopening the "smells must never become a decision" guarantee.

## Scope

- New `Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD: int = 5` — env-overridable like every
  other `Settings.*` threshold in this project.
- New `PredictResult.smell_review_suggested: bool = False` —
  `True` when `risk_score > Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD` **and**
  `ood_metrics` isn't explicitly `in_distribution=False`. Strict `>`, not `>=` — a
  document scoring exactly the threshold value stays `False`.
  - "Not explicitly `in_distribution=False`" means: `True` when `ood_metrics is None` (no
    `ood_stats.npz` loaded, same permissive-default-on-missing-artifact convention as
    `OodEvidence.from_in_distribution()`) **or** `ood_metrics.in_distribution is True`.
    Only an explicit `False` excludes it — the rationale being that `in_distribution=False`
    already gets `review_route="human_review"` today (see `decide_review_route()`), so this
    flag would add zero new information there; it exists specifically for the gap where the
    *decision* says fine but the smell profile disagrees.
- **Deliberately independent of `review_route`/`classifier_disagreement`** — confirmed with
  the user during brainstorming (see "Open question" below for the considered alternative).
  `smell_review_suggested=True` can coexist with `review_route="human_review"` from an
  unrelated cause (SVM disagreement); that's accepted redundancy for this iteration, not a
  bug.
- **Threshold lives in `Settings`, not `smell_thresholds.json`** — also confirmed with the
  user. Unlike the four per-signal smell thresholds (which are inherently per-model,
  calibrated against that model's own embedding space), a risk-score cutoff over the
  *aggregate* weighted sum is a review-policy choice, not a per-model statistical
  calibration — it doesn't need the same "lives next to `ood_stats.npz`" treatment.
- Computed in `attach_metadata()` (`src/inference/pipeline.py`) — the existing seam where
  `risk_score` itself is finalized, with `result.ood_metrics`/`result.risk_score` both
  already in scope.
- Full surface parity with `risk_score`/`smells` (the exact five places those two fields
  already reach, per this project's own established convention — and per the two
  known-gap bugs already hit twice for related fields, see `CLAUDE.md`'s
  `_PREDICTION_COLUMNS` history):
  1. `PredictResult` (backend)
  2. `PredictResponse` (`src/api/routes/predict/schemas.py`)
  3. `_PREDICTION_COLUMNS` (`src/wandb.py`)
  4. `predict-folder` CSV — automatic via `flatten_predict_result()`, no separate wiring
  5. Frontend: `types/api.ts`, `utils/flatten.ts`, `utils/csv.ts`'s `RESULT_COLUMNS`
     (drives both the on-screen table and the CSV export)
  6. `predict` CLI single-document output (`src/cli/predict.py`) — one more `click.echo`
     line, matching the `risk_score`/`smells` lines already there

**Out of scope:**
- Gating on `review_route`/`classifier_disagreement` (considered during brainstorming as
  "option B" — skip the flag whenever a human is already routed there for any reason, not
  just OOD). Deferred, not rejected — worth revisiting once this flag has real usage data.
- Moving the threshold into `smell_thresholds.json`. Deferred for the reason above; revisit
  if per-model tuning of this specific cutoff ever becomes a real need.
- Any change to `decide_review_route()`'s own return values or logic — untouched.
- Persisting this field anywhere beyond in-memory `PredictResult` — that's the registry
  work (pieces 2-4), which should pick up this field automatically once it exists (see
  "Recommended execution order" above).

## Design

### `Settings` (`src/settings.py`)

```python
SMELL_REVIEW_RISK_SCORE_THRESHOLD: int = 5
```

Placed alongside the other threshold settings, not under the `OOD_*` prefix — this isn't
an OOD signal, it's a policy over the aggregate `risk_score`.

### `_compute_smell_review_suggested()` (`src/inference/pipeline.py`)

```python
def _compute_smell_review_suggested(risk_score: int, ood_metrics: OodMetrics | None) -> bool:
    """True when risk_score alone crosses Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD, for
    documents the official decision didn't already send to a human. Deliberately independent
    of review_route/classifier_disagreement -- see the spec's Scope for why. `ood_metrics`
    being None (no ood_stats.npz loaded) behaves the same as in_distribution=True; only an
    explicit False excludes it, since that path already gets review_route="human_review"."""
    if ood_metrics is not None and ood_metrics.in_distribution is False:
        return False
    return risk_score > Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD
```

### `attach_metadata()` (`src/inference/pipeline.py`)

```python
def attach_metadata(
    result: PredictResult, filename: str, extraction: ExtractionMetadata
) -> PredictResult:
    foreign_match = detect_foreign_municipality(extraction.text or "")
    smells = list(result.smells)
    if foreign_match is not None:
        smells.append("foreign_municipality")
    risk_score = _compute_risk_score(smells)
    return result.model_copy(
        update={
            "filename": filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
            "smells": smells,
            "risk_score": risk_score,
            "smell_review_suggested": _compute_smell_review_suggested(
                risk_score, result.ood_metrics
            ),
        }
    )
```

`extraction_failed()` needs no special-case: its hardcoded `risk_score=3` is below the
default threshold of `5`, so `PredictResult.smell_review_suggested`'s own `False` default
is already correct there without an explicit set.

### `PredictResult` (`src/schemas.py`)

```python
smell_review_suggested: bool = False
```

Placed next to `risk_score`, same "0/False is the not-yet-computed default" convention
`risk_score`'s own docstring already documents.

## How to test this piece

- A document with `risk_score > 5` and `in_distribution=True` (or no `ood_stats.npz`
  loaded) shows `smell_review_suggested=True`.
- A document with `risk_score > 5` but `in_distribution=False` shows
  `smell_review_suggested=False` (already `review_route="human_review"` via the existing
  path — no new information from this flag).
- A document with `risk_score` exactly `5` shows `smell_review_suggested=False` (strict
  `>`, boundary case).
- `extraction_failed()` (unreadable document, `risk_score=3`) shows
  `smell_review_suggested=False`.
- CSV/API/frontend/W&B: confirm the field round-trips through `flatten_predict_result()`,
  the `/predict` response, `PredictionsTable.tsx`'s columns, and the W&B predictions table
  — same regression shape as the `svm_scores`/`svm_predicted_label` W&B-table gap this
  project already hit once.

## Backward compatibility

- `PredictResult.smell_review_suggested` is additive with a `False` default — every
  existing `PredictResult(...)` construction (17+ call sites across tests) keeps working
  unchanged.
- `Settings.SMELL_REVIEW_RISK_SCORE_THRESHOLD` is a new setting with a default — nothing
  needs a `.env` change to keep existing behavior undefined; the feature is simply off
  (`False` for everyone) until a document's `risk_score` actually exceeds `5`.
- No existing field's meaning changes. `review_route`, `risk_score`, and `smells` are
  read, never written, by this feature.

## Touch list

| Path | Change |
|---|---|
| `src/settings.py` | New `SMELL_REVIEW_RISK_SCORE_THRESHOLD: int = 5` |
| `src/schemas.py` | `PredictResult.smell_review_suggested: bool = False` |
| `src/inference/pipeline.py` | New `_compute_smell_review_suggested()`; `attach_metadata()` sets the new field |
| `src/api/routes/predict/schemas.py` | `PredictResponse.smell_review_suggested: bool = False` |
| `src/wandb.py` | `_PREDICTION_COLUMNS` gains `"smell_review_suggested"` |
| `src/cli/predict.py` | One more `click.echo` line in `predict_cmd` |
| `frontend/src/types/api.ts` | `PredictResponse.smellReviewSuggested: boolean` |
| `frontend/src/utils/flatten.ts` | `FlatResultRow` + `NULL_ROW` + `flattenResult()` |
| `frontend/src/utils/csv.ts` | `RESULT_COLUMNS` gains the new column (drives table + CSV) |
| Tests | `tests/inference/test_pipeline.py` (the four boundary cases above), `tests/cli/test_commands.py`, `tests/test_wandb.py`, `tests/api/test_predict.py` |
