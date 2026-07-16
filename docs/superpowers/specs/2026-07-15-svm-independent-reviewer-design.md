# SVM Independent Reviewer — Design Spec

## Motivation

`bert_tunning`'s softmax classifier can fail in two distinct ways:

1. **OOD (Option A, in scope here):** the document isn't really any of the trained classes at all (a payment receipt, a decree from an untrained municipality) but softmax always emits a confident label anyway. The existing signals (Mahalanobis, cosine, k-NN, TF-IDF, `foreign_municipality`) already target this.
2. **Misclassification among known classes (Option B, explicitly out of scope for this spec):** the document genuinely is one of the trained classes, but softmax picked the wrong one. None of the current signals check this. Deferred — may become its own spec later.

This feature adds a fifth, independent check aimed at Option A, inspired by Peña et al. 2023 ("Leveraging LLMs for Topic Classification in the Domain of Public Affairs"), which found that one-vs-rest SVMs trained on frozen transformer `[CLS]` embeddings handled their imbalanced-class problem better than a softmax/NN head or Random Forest — particularly on classes with very few samples, which is directly relevant here (`otro`=46, `declaracion_concejo_municipal`=37, `resolucion`=170 in BETO v2's training set).

## Why this isn't part of the OOD ensemble

The project's downstream consumer, **Classiflow** (an agentic workflow that will consume this trained model plus a conjunction of OOD and independent classifiers), needs raw evidence to weigh itself — not another vote pre-baked into a single `in_distribution` boolean. So this signal is designed the same way `foreign_municipality` is: computed and exposed on every prediction, **never merged into `in_distribution`**, and — unlike `foreign_municipality` — never reduced to a single value either. It's exposed as a full per-class score dict, mirroring `all_scores` (softmax's own per-class breakdown), so Classiflow can compare "what softmax thinks" against "what each class's own SVM boundary thinks."

**No calibration, no threshold.** Because this signal doesn't gate any decision in this repo, there's nothing to calibrate — no `Settings.OOD_*`-style config, no `evaluate-ood-calibration` involvement. This is a deliberate scope reduction relative to the other four signals, each of which needed a threshold tuned against a held-out split.

## Design

### Training time (`src/training/pipeline.py`)

After step 7 (where `ood_stats.npz` is built), fit one one-vs-rest `sklearn.svm.SVC` per class on the training split's **raw, pre-PCA** `[CLS]` embeddings — the same array `extract_embeddings()` already produces, before it's fed into `compute_class_stats`'s PCA step. No new embedding extraction pass.

- Kernel: RBF (matches the paper's validated config).
- `class_weight="balanced"` — consistent with this project's existing imbalance handling (`compute_class_weight("balanced", ...)` already used for the softmax head's `CrossEntropyLoss`), and directly relevant given `otro`/`declaracion_concejo_municipal`'s small sample counts.
- No PCA reduction — matches the paper's proven config, and avoids coupling this signal's feature space to a dimensionality (`OOD_PCA_COMPONENTS`) chosen for Mahalanobis's covariance-estimation needs, which this signal doesn't share.

### New module: `src/svm_reviewer.py`

Parallel to `src/ingestion/_text.py` (self-contained, not bolted onto `ood.py`, since this is a genuinely independent concern):

- `fit_svm_classifiers(embeddings, labels, class_names) -> dict[str, SVC]`
- `save_svm_classifiers(classifiers, path)` / `load_svm_classifiers(path) -> dict[str, SVC] | None`
- `svm_scores(embedding, classifiers) -> dict[str, float]` — calls `.decision_function()` per class.

### New artifact: `svm_classifiers.joblib`

Saved next to `ood_stats.npz` in `<output_dir>/final/`. `joblib` is already an installed transitive dependency (scikit-learn requires it) — no new package. Not a `.npz` array file since `SVC` objects need pickling.

### Schema (`src/schema.py`)

Add to `PredictResult`, alongside the existing `all_scores: dict[str, float]` field:

```python
svm_scores: dict[str, float] | None = None
```

`None` when `svm_classifiers.joblib` isn't present next to the loaded model (mirrors how `ood_metrics` itself is `None` when `ood_stats.npz` is missing — see `OodMetrics`'s docstring on why nesting-vs-flat matters for `None` meaning exactly one thing). `svm_scores` is **not** nested inside `OodMetrics` — it isn't an OOD signal, it's evidence about per-class membership, a different axis entirely, and doesn't share `OodMetrics`'s reason for existing (disambiguating "no stats file" from "stats file predates this field").

`svm_scores` deliberately follows `all_scores`' existing flat-dict treatment, not `OodMetrics`'s flatten-to-columns treatment (`flatten_predict_result()` in `src/schema.py`) — `all_scores` already goes into the CSV/W&B table as a single stringified-dict column via plain `model_dump()`, and `svm_scores` should behave the same way (same shape, same consumer expectations), not get special per-class-column flattening the way `ood_metrics`' five *scalar* fields did.

### Inference (`src/inference/classify.py`)

`BertTunningClassifier.__init__` optionally loads `svm_classifiers.joblib` if present (mirrors the existing optional `ood_stats.npz` load). `predict_text()` computes `svm_scores` from the **same** `[CLS]` embedding already computed for the softmax forward pass — no extra model inference.

### Validation

A twin of `_validate_ood_stats_class_mapping()` for the SVM classifiers dict, checked at classifier construction — so a stale or mismatched `svm_classifiers.joblib` fails loudly (`BertTunningError`) instead of silently scoring against the wrong classes. This bug class already bit `ood_stats.npz` once; the fix pattern is proven and directly reusable.

### CLI

**New standalone command: `compute-svm-classifiers`** (mirrors `compute-ood-stats`'s shape — same `--model-path`/`--model`/`--cache-path`/`--seed` args, same reconstructed-split requirement) — independent of `compute-ood-stats`, since regenerating the SVM reviewer shouldn't require recomputing OOD stats, and vice versa.

**`predict`/`predict-folder` CLI output** (`src/cli/predict.py`): print/export `svm_scores` the same way `all_scores` is already handled.

**CSV and W&B — no extra plumbing needed.** `flatten_predict_result()` (`src/schema.py`) is the single function that builds both the `predict-folder` CSV row and the `predict-folder --log-wandb` table row — it does `result.model_dump(exclude={"ood_metrics"})`, so any new top-level `PredictResult` field, including `svm_scores`, flows into both automatically once added to the schema. No change needed to `flatten_predict_result()` itself, `src/wandb.py`, or the CSV export path.

**API** (`src/api/routes/predict/schemas.py`): add `svmScores` (camelCase alias) to the response schema. Remember the `populate_by_name=True` gotcha already documented in `CLAUDE.md` for any Pydantic model with `alias_generator=to_camel` constructed via snake_case kwargs.

## Out of scope

- Option B (misclassification-among-known-classes) — may become its own spec later, but a per-class SVM built for Option A doesn't obviously answer Option B's question ("did softmax pick the *right* known class") without a different aggregation step; not addressed here.
- Any threshold, calibration CLI flag, or `Settings.OOD_SVM_*`-style config — deliberately not needed, see above.
- Folding `svm_scores` into `in_distribution` or `review_route` — deliberately excluded; this is raw evidence for Classiflow, not a decision made in this repo.

## Touch list

| File | Change |
|---|---|
| `src/svm_reviewer.py` | **New.** fit/save/load/score functions. |
| `src/training/pipeline.py` | Fit + save SVM classifiers after `ood_stats.npz` step. |
| `src/schema.py` | Add `PredictResult.svm_scores`. |
| `src/inference/classify.py` | Optional load at `__init__`; compute `svm_scores` in `predict_text()`; class-mapping validation. |
| `src/cli/svm_classifiers.py` | **New.** `compute-svm-classifiers` command. |
| `src/cli/predict.py` | Print/export `svm_scores`. |
| `src/api/routes/predict/schemas.py` | Add `svmScores`. |
| `main.py` | Register `compute-svm-classifiers` command. |
| `README.md` / `CLAUDE.md` | Document the new signal, consistent with how the other four are documented. |

## Testing

- `src/svm_reviewer.py`: fit/save/load round-trip, `svm_scores()` output shape.
- `predict_text()`: `svm_scores` populated when artifact present, `None` when absent.
- Class-mapping validation: mismatch raises `BertTunningError`.
- API response includes `svmScores`.
- `predict-folder` CSV includes `svm_scores` as a single dict-serialized column, matching `all_scores`' existing treatment.
- `compute-svm-classifiers` CLI: backfills the artifact for an already-trained model without retraining.
