# Piece 1/4: Risk Score and Smells on `PredictResult` — Design Spec

Part of a 4-piece decomposition (see the other 3 specs dated 2026-07-23) so each piece can
be implemented, tested, and reviewed independently:
1. **This spec** — `risk_score`/`smells` fields, no persistence yet.
2. Registry module + CLI wiring (SQLite writes from `predict`/`predict-folder`).
3. API wiring (SQLite writes from `/predict`).
4. Registry query/update endpoints (`GET /predictions`, `PATCH /predictions/{id}`).

Grounded against a real example workbook the user shared
(`review_workbook.xlsx`) — its columns (`Tier`, `Risk score`, `Smells`, `Review status`,
`Corrected label`, followed by every existing CSV-export field) confirmed `risk_score` is
a **numeric**, weighted-sum value, not a `low`/`medium`/`high` label, and that
`foreign_municipality` counts as a smell too. The workbook's own numbers were hand-typed
example data (no formulas — confirmed by unzipping the `.xlsx` and checking for `<f>`
tags), not a precise algorithm to reverse-engineer; the weighting below was proposed and
confirmed separately.

## Motivation

- **Risk score** — a derived, presentational summary, not a new model decision. This
  project deliberately keeps the four OOD signals (Mahalanobis, cosine, k-NN, TF-IDF)
  unblended, OR'd into `in_distribution` rather than combined into one score. A weighted
  point total across *named smells* doesn't touch that decision — `in_distribution` is
  still computed exactly as before — it just gives a human scanning results one number
  instead of reading a list.
- **Smells** — per-prediction, which specific condition(s) fired: `low_mahalanobis_p`,
  `high_cosine_z`, `high_knn_distance`, `high_tfidf_z`, `foreign_municipality`, and
  `low_confidence` (not an OOD signal — the base softmax confidence tier). Today only the
  aggregate `in_distribution` boolean is exposed; nothing says *which* condition flagged a
  given document.

## Scope

- `PredictResult` gains `risk_score: int` and `smells: list[str]`, computed once across
  `predict_text()` (the OOD-signal + confidence smells) and `_attach_metadata()` (adding
  `foreign_municipality` once it's known, and finalizing the risk score) — flowing
  automatically through every existing consumer (API response, CSV export,
  `predict-folder`, the frontend table) with zero extra plumbing, the same pattern
  `svm_scores`/`svm_predicted_label` already followed.
- Small bundled refactor: `_run_prediction_job` (`src/api/routes/predict/endpoints.py`)
  currently duplicates `_attach_metadata()`'s logic inline instead of calling it. Fixed as
  part of this piece, since `_attach_metadata()` is the natural, and now only, place the
  `foreign_municipality` smell and the final `risk_score` get computed — leaving the
  duplicate in place would mean fixing this twice, and the two copies silently drifting
  apart is exactly the class of bug this project has hit before (`_PREDICTION_COLUMNS`
  missing a field once already).
- **Out of scope**: persistence (piece 2), any new endpoint, any frontend UI change,
  renaming `llm_judge` (`Tier` in the workbook is a display-only label — confirmed with
  the user; `review_route`'s actual values are untouched).

## Design

### `is_out_of_distribution()` — extract the per-signal breakdown it already computes

`is_out_of_distribution()` (`src/inference/ood_scorer.py`) already computes
`maha_anomalous`/`cosine_anomalous`/`knn_anomalous`/`tfidf_anomalous` as locals — including
calibration-gating and the NaN fail-open/fail-closed asymmetry between k-NN and TF-IDF —
before OR'ing them into one bool. Duplicating that logic elsewhere for `smells` risks the
two implementations silently disagreeing (`in_distribution=True` but a smell claims a
signal fired, or vice versa). Extract the shared computation instead, so there is exactly
one place that decides "is this specific signal anomalous":

```python
class OodSignalBreakdown(NamedTuple):
    mahalanobis: bool
    cosine: bool
    knn_distance: bool
    tfidf: bool


def ood_signal_breakdown(
    scores: OodScores,
    thresholds: OodThresholds,
    calibration_status: OodCalibrationStatus = _ALL_CALIBRATED,
    *,
    allow_uncalibrated_fallback: bool = True,
) -> OodSignalBreakdown:
    # exact body of what is_out_of_distribution() computes today, unchanged
    ...
    return OodSignalBreakdown(maha_anomalous, cosine_anomalous, knn_anomalous, tfidf_anomalous)


def is_out_of_distribution(
    scores: OodScores,
    thresholds: OodThresholds,
    calibration_status: OodCalibrationStatus = _ALL_CALIBRATED,
    *,
    allow_uncalibrated_fallback: bool = True,
) -> bool:
    breakdown = ood_signal_breakdown(
        scores, thresholds, calibration_status,
        allow_uncalibrated_fallback=allow_uncalibrated_fallback,
    )
    return any(breakdown)
```

Purely a refactor — `is_out_of_distribution()`'s signature, behavior, and every existing
caller are unchanged.

### `OodMetrics.smells` — the OOD-specific smells, computed where thresholds are already resolved

`OodScorer.score()` already calls `resolve_ood_thresholds()` and has every raw score in
scope. It calls `ood_signal_breakdown()` and translates the four booleans into names:

```python
_SMELL_NAMES = {
    "mahalanobis": "low_mahalanobis_p",
    "cosine": "high_cosine_z",
    "knn_distance": "high_knn_distance",
    "tfidf": "high_tfidf_z",
}

def _smells_from_breakdown(breakdown: OodSignalBreakdown) -> list[str]:
    return [name for field, name in _SMELL_NAMES.items() if getattr(breakdown, field)]
```

`OodMetrics` (`src/schema.py`) gains `smells: list[str] = []` — populated in `score()`
alongside `in_distribution`.

### `PredictResult.smells` (partial) — `predict_text()`

```python
smells = list(ood_metrics.smells) if ood_metrics is not None else []
if not certain:
    smells.append("low_confidence")
```

`PredictResult` (`src/schema.py`) gains `smells: list[str] = []` and
`risk_score: int = 0`. `predict_text()` sets `smells` to the OOD + confidence smells above
— **not yet including `foreign_municipality`**, since that's only known after extraction
metadata is attached, a step that happens after `predict_text()` returns in every real
call site. `risk_score` is deliberately left at its `0` default here, not computed twice —
see below for where it's finalized.

### Weighted risk score + the `foreign_municipality` smell — `_attach_metadata()`

`_attach_metadata()` (`src/inference/pipeline.py`) is already the shared point where
`detect_foreign_municipality()` runs, for both `predict_pdf` and `predict_folder`. It now
also appends the `foreign_municipality` smell (when present) and computes the final,
complete `risk_score` from the now-final `smells` list:

```python
_SMELL_WEIGHTS: dict[str, int] = {
    "low_mahalanobis_p": 3,
    "high_cosine_z": 3,
    "high_knn_distance": 3,
    "high_tfidf_z": 3,
    "foreign_municipality": 2,
    "low_confidence": 1,
}

def _compute_risk_score(smells: list[str]) -> int:
    return sum(_SMELL_WEIGHTS.get(smell, 0) for smell in smells)


def _attach_metadata(
    result: PredictResult, filename: str, extraction: ExtractionMetadata
) -> PredictResult:
    foreign_match = detect_foreign_municipality(extraction.text or "")
    smells = list(result.smells)
    if foreign_match is not None:
        smells.append("foreign_municipality")
    return result.model_copy(
        update={
            "filename": filename,
            "extracted_text": extraction.text,
            "extractor_used": extraction.extractor_used or "",
            "foreign_municipality": foreign_match.name if foreign_match else None,
            "foreign_municipality_context": foreign_match.context if foreign_match else None,
            "smells": smells,
            "risk_score": _compute_risk_score(smells),
        }
    )
```

`extraction_failed()` (`src/inference/pipeline.py`) gets an explicit `risk_score` too —
it never goes through `_attach_metadata()` (there's no extraction to attach metadata
from), so it sets a fixed high score directly (matching its existing
`review_route="human_review"`): `risk_score=3` felt arbitrary to hardcode independently,
so it reuses `_compute_risk_score(["low_confidence"])`-equivalent reasoning — **open
question below**, since "empty/unreadable document" doesn't cleanly map to any existing
smell name.

**`_run_prediction_job` (`src/api/routes/predict/endpoints.py`) now calls
`_attach_metadata()` directly** instead of its previous inline duplicate of the same
update — the bundled refactor from Scope above. Its `model_copy(update={...})` block
(`filename`, `extracted_text`, `extractor_used`, `foreign_municipality`,
`foreign_municipality_context`) is replaced by `result = _attach_metadata(result, filename, extraction)`.

## How to test this piece

No persistence yet, so this is testable entirely through existing entry points:

```powershell
uv run python main.py predict path/to/document.pdf
```

Check the printed output (or `predict-folder`'s CSV) for the new `riskScore`/`smells`
fields/columns. Confirm:
- A document flagged by, say, the cosine signal shows `high_cosine_z` in `smells` *and*
  `inDistribution=false` — never one without the other (the exact bug the shared
  `ood_signal_breakdown()` extraction is meant to prevent).
- `riskScore` matches manual arithmetic over `smells` using the weight table (3 per OOD
  smell, 2 for `foreign_municipality`, 1 for `low_confidence`).
- A document naming a non-`OOD_TRAINED_MUNICIPALITY` city shows `foreign_municipality` in
  `smells` and `+2` reflected in `riskScore`, even on a document that otherwise has zero
  OOD smells (mirrors the real motivating case from `CLAUDE.md`'s TF-IDF known-limitation
  note — a foreign jurisdiction a purely-statistical signal might miss).
- A model with no `ood_stats.npz` still returns only `low_confidence`/`foreign_municipality`
  smells, never the four OOD ones — same as `in_distribution` staying absent today.
- Test via the API too (`serve` + upload), to confirm `_run_prediction_job`'s
  `_attach_metadata()` call produces identical `smells`/`riskScore` to the CLI path for
  the same document — this is what the bundled refactor is actually verifying.

## Backward compatibility

- Both new `PredictResult` fields are additive with defaults (`[]`/`0`) — every existing
  `PredictResult(...)` construction (17+ call sites across tests) keeps working unchanged.
- `OodMetrics.smells` is additive with a `[]` default — same precedent as the existing
  `*_calibration_status` fields ("defaulted so none of the 9 existing direct
  `OodMetrics(...)` test constructors need updating").
- `is_out_of_distribution()`'s public signature and behavior are unchanged.
- `_run_prediction_job`'s switch to calling `_attach_metadata()` must produce byte-for-byte
  identical `foreign_municipality`/`foreign_municipality_context` values to its old inline
  logic — it's the same `detect_foreign_municipality()` call either way, but worth an
  explicit regression test given it's touching a currently-working API path.

## Open question for spec review

`extraction_failed()`'s `risk_score` — no smell name cleanly describes "the document
couldn't be read at all." Options: reuse `low_confidence`'s weight (`1`) since
`certain=False` there too; invent a new `unreadable_document` smell/weight; or hardcode a
fixed high score independent of the weight table, since an unreadable document is already
maximally urgent regardless of what a weighted sum would say. Leaning toward a new
`unreadable_document` smell (weight `3`, same tier as an OOD signal) so `smells` stays the
single source of truth `risk_score` is always derived from, rather than a special case —
confirm before implementing.

## Touch list

| Path | Change |
|---|---|
| `src/inference/ood_scorer.py` | Extract `ood_signal_breakdown()` out of `is_out_of_distribution()`; `OodScorer.score()` computes `OodMetrics.smells` from it |
| `src/schema.py` | `OodMetrics.smells: list[str] = []`; `PredictResult.smells: list[str] = []`, `PredictResult.risk_score: int = 0` |
| `src/inference/classify.py` | `predict_text()` computes the OOD + `low_confidence` smells |
| `src/inference/pipeline.py` | `_attach_metadata()` adds the `foreign_municipality` smell and computes the final `risk_score`; `extraction_failed()` sets its own `risk_score`/`smells` (see open question) |
| `src/api/routes/predict/endpoints.py` | `_run_prediction_job` calls `_attach_metadata()` instead of duplicating its logic inline |
