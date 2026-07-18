# Consolidate SVM Training Workflows — Design Spec

## Motivation

A code review found genuine duplicated operational logic between
`_fit_svm_reviewer()` (`src/training/pipeline.py:39`, called from `run()` during training)
and `_run_compute_svm_classifiers()` (`src/cli/svm_classifiers.py:35`, the standalone
`compute-svm-classifiers` backfill command). Both do the identical sequence: fit one-vs-rest
SVM classifiers on training embeddings → extract validation embeddings at the `"first"`
chunk strategy → evaluate held-out balanced accuracy → log it → build a per-class training
document count dict (the exact same dict comprehension, verified byte-for-byte identical in
both files). Only what happens with the result differs: `training/pipeline.py` logs into an
already-open W&B training run; `cli/svm_classifiers.py` saves the classifiers to disk and
optionally logs to a standalone W&B run.

**Recommendation being implemented:** extract one reusable SVM training/evaluation
operation; let `training/pipeline.py` and `cli/svm_classifiers.py` each decide when to
invoke it and what to do with the result.

## Touch list

| File | What changes |
|---|---|
| `src/svm_reviewer.py` | New `SvmTrainingResult` + `fit_and_evaluate_svm_reviewer()` — the shared fit+evaluate+count operation. Adds `import logging` + module `log`, since the "held-out balanced accuracy" log line moves here. |
| `src/training/pipeline.py` | `_fit_svm_reviewer()` shrinks to: extract val embeddings (its own call shape, reusing already-extracted `train_embeddings`) → call the new shared function → log to the already-open W&B run |
| `src/cli/svm_classifiers.py` | `_run_compute_svm_classifiers()`'s SVM portion shrinks to: extract val embeddings (its own call shape, via `embed_texts()`) → call the new shared function → save to disk → optionally log to a standalone W&B run |
| `tests/test_svm_reviewer.py` | New tests for `fit_and_evaluate_svm_reviewer()` |

**Not touched:** `fit_svm_classifiers()`, `evaluate_svm_classifiers()`, `save_svm_classifiers()`
(already pure/separate, unchanged) — the new function is a thin composition of these two
existing pure functions, following the same "pure math, no I/O beyond logging" shape.
`WandbLogger.log_svm_results()`/`log_svm_classifiers_results()` stay separate, matching
CLAUDE.md's own documented reasoning for keeping those two apart ("a long-lived class vs.
fire-and-forget... forcing them into a common interface would either bolt unrelated methods
onto one class or throw away type safety") — this spec doesn't revisit that decision, it
only removes the *duplicated fit+evaluate* logic sitting upstream of both.

## Design

### `SvmTrainingResult` + `fit_and_evaluate_svm_reviewer()` (new, `src/svm_reviewer.py`)

```python
class SvmTrainingResult(NamedTuple):
    classifiers: dict[str, SVC]
    val_accuracy: dict[str, float]
    train_class_counts: dict[str, int]


def fit_and_evaluate_svm_reviewer(
    train_embeddings: npt.NDArray[np.float64],
    train_labels: list[int],
    val_embeddings: npt.NDArray[np.float64],
    val_labels: list[int],
    class_names: list[str],
) -> SvmTrainingResult:
    """Fits one-vs-rest SVM classifiers and immediately evaluates held-out balanced
    accuracy on val_embeddings (never the data they were fit on) -- the fit+evaluate+count
    sequence training/pipeline.py and cli/svm_classifiers.py both need, previously
    duplicated in both files. Callers stay responsible for extracting the embeddings
    themselves (the two callers use different embedding-extraction call shapes -- one
    already has train_embeddings computed for OOD stats, the other doesn't) and for what
    happens with the result (persisting to disk, logging to an already-open W&B run vs. a
    standalone one) -- this function only owns the fit+evaluate+count math, the same
    pure-computation shape as fit_svm_classifiers/evaluate_svm_classifiers above it."""
    classifiers = fit_svm_classifiers(train_embeddings, train_labels, class_names)
    val_accuracy = evaluate_svm_classifiers(classifiers, val_embeddings, val_labels, class_names)
    log.info(
        "SVM reviewer held-out balanced accuracy (val split): %s",
        {k: round(v, 4) for k, v in val_accuracy.items()},
    )
    train_labels_arr = np.asarray(train_labels)
    train_class_counts = {
        name: int((train_labels_arr == idx).sum()) for idx, name in enumerate(class_names)
    }
    return SvmTrainingResult(
        classifiers=classifiers, val_accuracy=val_accuracy, train_class_counts=train_class_counts
    )
```

`train_labels`/`val_labels` stay `list[int]`, matching `fit_svm_classifiers`'s/
`evaluate_svm_classifiers`'s own existing parameter types — converted to a numpy array
internally only for the boolean comparison the count computation needs.

### `training/pipeline.py`'s `_fit_svm_reviewer()` (shrinks)

```python
def _fit_svm_reviewer(
    loaded: LoadedModel,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    class_names: list[str],
    train_embeddings: npt.NDArray[np.float64],
    wb: WandbLogger,
) -> dict[str, SVC]:
    val_embeddings = extract_embeddings(
        loaded,
        [prepare_text(t, loaded.tokenizer, "first") for t in val_df["text"]],
        max_length=model_cfg.max_tokens,
    )
    result = fit_and_evaluate_svm_reviewer(
        train_embeddings,
        train_df["label_id"].tolist(),
        val_embeddings,
        val_df["label_id"].tolist(),
        class_names,
    )
    wb.log_svm_results(result.val_accuracy, result.train_class_counts)
    return result.classifiers
```

### `cli/svm_classifiers.py`'s `_run_compute_svm_classifiers()` (SVM portion shrinks)

```python
val_embeddings = embed_texts(
    split.loaded, split.val_df, chunk_strategy="first", max_tokens=model_cfg.max_tokens
)
result = fit_and_evaluate_svm_reviewer(
    embeddings,
    split.train_df["label_id"].tolist(),
    val_embeddings,
    split.val_df["label_id"].tolist(),
    split.classes,
)
log.info("Fit %d one-vs-rest SVM reviewers", len(result.classifiers))

out_path = Path(opts.model_path) / "svm_classifiers.joblib"
save_svm_classifiers(result.classifiers, out_path)
log.info("Saved SVM reviewer classifiers -> %s", out_path)

if opts.log_wandb:
    log_svm_classifiers_results(
        model_path=opts.model_path,
        cache_path=opts.cache_path,
        model_key=opts.model_key,
        svm_val_accuracy=result.val_accuracy,
        train_class_counts=result.train_class_counts,
    )
```

`log.info("Fit %d one-vs-rest SVM reviewers", ...)` moves to right after the call (reading
`len(result.classifiers)`) instead of immediately after the old inline `fit_svm_classifiers`
call — same informational content, now describing the result of one call instead of two.

## Backward compatibility

- Both callers' observable behavior (W&B payloads, `svm_classifiers.joblib` contents, log
  line content) is unchanged — this is a pure internal reshuffle of who calls what, not a
  change to what gets computed, persisted, or reported.
- `training/pipeline.py`'s characterization test
  (`tests/training/test_pipeline.py::test_run_orchestrates_training_pipeline_end_to_end`,
  added in the training-pipeline-decomposition PR) should pass unmodified — it doesn't mock
  `fit_svm_classifiers`/`evaluate_svm_classifiers`, only the embedding extraction feeding
  into them, so it exercises this exact fit+evaluate+count path for real already and is a
  strong safety net for this change too.
