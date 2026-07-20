# Consolidate Embedding Batch Execution — Design Spec

## Motivation

A code review found `extract_embeddings()` and `extract_embeddings_and_predictions()`
(`src/embeddings.py`) duplicating eval-mode setup, batching, tokenization, device transfer,
and `no_grad()` handling — the two functions are near-identical except for how the forward
pass itself is run and whether predictions are collected.

**Recommendation being implemented:** share one private batch/tokenization helper while
keeping the two public functions exactly as they are today (same signatures, same callers,
same behavior).

## What must NOT change

The two functions' forward-pass calls are **deliberately different**, not accidental
duplication, and this spec does not touch that difference:

- `extract_embeddings()` calls `loaded.model.base_model(**inputs).last_hidden_state` —
  skipping the classification head entirely. Its own existing docstring cross-reference
  says why: "its other callers (compute-ood-stats, training) only ever need embeddings,
  never predictions" — this is a real performance choice (fewer FLOPs per batch across
  potentially thousands of documents), not something to unify away.
- `extract_embeddings_and_predictions()` calls the full
  `loaded.model(**inputs, output_hidden_states=True)` because it needs `outputs.logits` for
  predictions too — it cannot skip the classification head.

Unifying these onto one shared forward-pass call (e.g. always calling the full model) would
be a real, silent performance regression for `extract_embeddings`'s callers. The
consolidation therefore targets only the genuinely byte-for-byte-identical part: batching
texts into chunks, tokenizing, and moving to device — plus the equally-identical
[CLS]-embedding extraction line that follows each forward pass.

## Touch list

| File | What changes |
|---|---|
| `src/embeddings.py` | Two new private helpers (`_batched_inputs`, `_cls_embedding`); `extract_embeddings`/`extract_embeddings_and_predictions` call them instead of duplicating the loop body |

**Not touched:** both public functions' signatures, return types, and forward-pass strategy.
Every caller is unaffected — same public API in, same output out. Verified via grep,
`extract_embeddings` has exactly two call sites (`src/training/pipeline.py`,
`src/cli/_ood_common.py`'s `embed_texts()` wrapper — itself called from `compute-ood-stats`
and `compute-svm-classifiers`); `extract_embeddings_and_predictions` has one
(`src/cli/_ood_common.py`'s `embed_texts_and_predict()`, used by `evaluate-ood-calibration`).
Neither is called from `src/inference/classify.py` — `predict_text()`'s own embedding
extraction is a separate, inline forward pass, out of scope for this spec.

## Design

### `_batched_inputs` (new, private)

```python
def _batched_inputs(
    loaded: LoadedModel, texts: list[str], *, max_length: int, batch_size: int
) -> Iterator[BatchEncoding]:
    """Yields tokenized, device-moved inputs one batch at a time -- the identical
    batching/tokenization/device-transfer machinery both extract_embeddings and
    extract_embeddings_and_predictions need. Does NOT run the forward pass -- the two
    callers deliberately use different forward-pass strategies (see module-level note on
    why extract_embeddings skips the classification head) and must stay free to do so."""
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        yield loaded.tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(loaded.device)
```

### `_cls_embedding` (new, private)

```python
def _cls_embedding(hidden_state: torch.Tensor) -> npt.NDArray[np.float64]:
    return hidden_state[:, 0, :].cpu().numpy().astype(np.float64)
```

The one-line `[CLS]`-token extraction, identical in both functions today, given a name
instead of being repeated.

### `extract_embeddings` (unchanged signature/behavior, new body)

```python
def extract_embeddings(
    loaded: LoadedModel, texts: list[str], *, max_length: int, batch_size: int = 16
) -> npt.NDArray[np.float64]:
    loaded.model.eval()
    batches: list[npt.NDArray[np.float64]] = []
    with torch.no_grad():
        for inputs in _batched_inputs(loaded, texts, max_length=max_length, batch_size=batch_size):
            hidden = loaded.model.base_model(**inputs).last_hidden_state  # type: ignore[operator]
            batches.append(_cls_embedding(hidden))
    return np.vstack(batches)
```

### `extract_embeddings_and_predictions` (unchanged signature/behavior, new body)

```python
def extract_embeddings_and_predictions(
    loaded: LoadedModel, texts: list[str], *, max_length: int, batch_size: int = 16
) -> tuple[npt.NDArray[np.float64], list[int]]:
    loaded.model.eval()
    embedding_batches: list[npt.NDArray[np.float64]] = []
    predicted_ids: list[int] = []
    with torch.no_grad():
        for inputs in _batched_inputs(loaded, texts, max_length=max_length, batch_size=batch_size):
            outputs = loaded.model(**inputs, output_hidden_states=True)
            embedding_batches.append(_cls_embedding(outputs.hidden_states[-1]))
            predicted_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return np.vstack(embedding_batches), predicted_ids
```

## Backward compatibility

- Both public functions keep identical signatures, return types, and (given the same
  input) identical output — this is a pure internal reshuffle.
- Existing tests (`tests/test_ood.py::test_extract_embeddings_returns_correct_shape`,
  `tests/test_ood.py::test_extract_embeddings_and_predictions_returns_matching_lengths`)
  should pass unmodified — they assert on the public functions' output shape, not their
  internals.
- Wider blast radius worth calling out explicitly: `extract_embeddings` is used by
  training, `compute-ood-stats`, and `compute-svm-classifiers` (via `embed_texts()`) — this
  consolidation touches shared code with three real call paths even though the diff itself
  is small. Run the full test suite, not just `tests/test_ood.py`, before considering this
  done.
