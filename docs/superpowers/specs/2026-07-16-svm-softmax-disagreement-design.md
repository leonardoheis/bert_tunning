# SVM/Softmax Classifier Disagreement — Design Spec

## Motivation

Two distinct classifier failure modes were identified while designing the SVM independent reviewer:

1. **OOD (Option A):** the document isn't really any of the trained classes at all. Already covered by five existing signals (Mahalanobis, cosine, k-NN, TF-IDF, `foreign_municipality`).
2. **Misclassification among known classes (Option B):** the document genuinely is one of the trained classes, but softmax picked the wrong one. Nothing currently checks this.

The SVM independent reviewer (shipped, `docs/superpowers/specs/2026-07-15-svm-independent-reviewer-design.md`) already computes a second, independently-trained opinion on every prediction — `svm_scores`, a per-class decision-function margin. Comparing its top pick against softmax's top pick is the cheapest possible check for Option B: no new training, no new artifact, reuses data already computed inside `predict_text()`.

## Why this is not part of the OOD ensemble

`in_distribution` answers one specific question: "does this document resemble anything the model was ever trained on at all." A classifier disagreement answers a *different* question: "this document genuinely is a known class, but the two independently-trained classifiers can't agree on which one." A document can be perfectly in-distribution and still trigger a disagreement. Folding the two together would mean `in_distribution=False` stops meaning one specific thing — the exact ambiguity `OodMetrics` was nested (rather than left flat) to avoid in the first place (see that field's docstring in `src/schema.py`).

## Why it should still drive `review_route`

`review_route` exists precisely to turn multiple *independent* signals into one actionable lane without a human eyeballing every field. It already has one trigger for `human_review` (OOD firing, **regardless of confidence** — the whole point being that a confident-but-wrong prediction is the dangerous case). A classifier disagreement is a second, independent reason to distrust a prediction, and belongs in the same lane for the identical reason: a human should look at it regardless of how confident softmax was.

## Design

### New helper: `svm_top_label()` (`src/svm_reviewer.py`)

```python
def svm_top_label(scores: dict[str, float]) -> str:
    """The class whose one-vs-rest SVM scored this embedding highest -- the SVM
    reviewer's own "prediction," for comparison against softmax's argmax."""
    return max(scores, key=scores.get)  # type: ignore[arg-type]
```

Pure function, no I/O, same placement rationale as `svm_scores()` itself.

### New `PredictResult` fields (`src/schema.py`)

```python
svm_predicted_label: str | None = None
svm_agrees_with_prediction: bool | None = None
```

Both `None` when `svm_scores` itself is `None` (no `svm_classifiers.joblib` loaded) — same graceful-degradation pattern as every other optional signal in this schema. Two fields, not one, because the user explicitly wants a ready-to-filter comparison column (`svm_agrees_with_prediction`) *and* wants to see what the SVM thought instead when it disagrees (`svm_predicted_label`) — mirrors the existing `foreign_municipality` + `foreign_municipality_context` pairing (a flag/value plus its supporting detail).

### `predict_text()` wiring (`src/inference/classify.py`)

Computed once, right after `svm_scores_result` (already computed today, before either `decide_review_route()` call):

```python
svm_predicted_label = svm_top_label(svm_scores_result) if svm_scores_result is not None else None
svm_agrees = None if svm_predicted_label is None else svm_predicted_label == label
classifier_disagreement = svm_agrees is False  # None (no SVM loaded) and True (agrees) both mean "don't trigger"
```

**Both** `decide_review_route()` call sites inside `predict_text()` need `classifier_disagreement` passed through — not just the second one. The first call (before OOD stats are even checked) is what actually gets returned when `self._ood_stats is None`; missing it there would silently drop the disagreement signal for any model without `ood_stats.npz`.

### `decide_review_route()` signature change (`src/inference/classify.py`)

```python
def decide_review_route(
    *,
    confidence_tier: ConfidenceTier,
    ood_evidence: OodEvidence,
    classifier_disagreement: bool = False,
) -> str:
    if ood_evidence is OodEvidence.ANOMALOUS or classifier_disagreement:
        return "human_review"
    if confidence_tier is ConfidenceTier.CONFIDENT:
        return "accept"
    return "llm_judge"
```

Defaults to `False` so existing callers/tests that don't pass it see no behavior change.

**Updated routing table:**

| `in_distribution` | classifiers disagree | `certain` | `review_route` |
|---|---|---|---|
| `False` | — | — | `human_review` (OOD always wins, unchanged) |
| `True`/`None` | **True** | — | `human_review` (**new** — disagreement alone is enough, regardless of confidence) |
| `True`/`None` | False/`None` | `True` | `accept` (unchanged) |
| `True`/`None` | False/`None` | `False` | `llm_judge` (unchanged) |

### API, CLI, W&B plumbing

- `PredictResponse` (`src/api/routes/predict/schemas.py`): add `svm_predicted_label`/`svm_agrees_with_prediction`; wire through `_to_predict_response()` in `endpoints.py`.
- `predict-folder` CSV: both fields are scalars (`str | None`, `bool | None`), so — unlike `svm_scores` (a dict) — they flow into the CSV automatically via `flatten_predict_result()`'s full `model_dump()`. **No special case needed there.**
- `predict-folder --log-wandb` table: **does need explicit plumbing** — add both to `_PREDICTION_COLUMNS` in `src/wandb.py`. This is the exact gap already found and fixed once for `svm_scores` in this session; the CSV and the W&B table do not share one automatic path, only the CSV does.
- `predict` CLI single-document output (`src/cli/predict.py`): print alongside the existing "SVM reviewer" section, e.g. a line noting disagreement when `svm_agrees_with_prediction is False`.

## Out of scope

- Not folding disagreement into `in_distribution` — see rationale above.
- No new `review_route` value (e.g. a distinct `"classifier_disagreement"` lane) — reuses the existing `human_review` lane. Classiflow can already distinguish *why* a document landed there by reading `svm_agrees_with_prediction`/`in_distribution` directly; a new lane would be a decision this repo doesn't need to make.
- No new CLI command, no new training artifact, no calibration/threshold (same rationale as the SVM reviewer itself — this is a direct label comparison, not a continuous score needing a cutoff).

## Testing

- `svm_top_label()`: returns the correct max-margin class.
- `decide_review_route()`: `classifier_disagreement=True` forces `human_review` regardless of `confidence_tier`/`ood_evidence` (including when both would otherwise say `accept`); `classifier_disagreement=False` preserves all existing routing behavior (regression guard for the default-value backward-compatibility claim above).
- `predict_text()`: `svm_predicted_label`/`svm_agrees_with_prediction` populated correctly when SVM classifiers are loaded, both `None` when absent; disagreement forces `human_review` even when `certain=True` and `in_distribution=True`; agreement doesn't suppress an OOD-driven `human_review` (OOD and disagreement are independent triggers for the same lane, not mutually exclusive).
- API test: response includes both new camelCase fields.
- `src/wandb.py`: both new fields present in the `predict-folder --log-wandb` table (the same regression pattern used to catch `svm_scores`' earlier omission).

## Touch list

| File | Change |
|---|---|
| `src/svm_reviewer.py` | Add `svm_top_label()`. |
| `src/schema.py` | Add `PredictResult.svm_predicted_label`, `.svm_agrees_with_prediction`. |
| `src/inference/classify.py` | Compute both in `predict_text()`; extend `decide_review_route()` signature; pass `classifier_disagreement` at both call sites. |
| `src/api/routes/predict/schemas.py` | Add both fields to `PredictResponse`. |
| `src/api/routes/predict/endpoints.py` | Wire both through `_to_predict_response()`. |
| `src/cli/predict.py` | Print disagreement in single-`predict` output. |
| `src/wandb.py` | Add both to `_PREDICTION_COLUMNS`. |
| `README.md` / `CLAUDE.md` | Document the new fields and updated routing table. |
