# Training Pipeline: Characterization Test + Decomposition — Design Spec

## Motivation

A code review found `training.pipeline.run()` (`src/training/pipeline.py:75`) to be a
~160-line orchestration function performing label encoding, splitting, class weighting,
tokenization, model construction, `TrainingArguments`/`Trainer` setup, training, embedding
extraction, OOD stats computation, SVM fitting, evaluation, W&B logging, and final artifact
persistence — eleven distinct concerns in one function body. There are strong unit tests
around its individual dependencies (`make_split`, `BertTunningDataset`, `compute_class_stats`,
`fit_svm_classifiers`, etc.) but **no test exercises `run()` itself end-to-end** — nothing
locks down its current orchestration behavior (call order, what gets passed to what, what
gets persisted where). Refactoring it later without that safety net is unnecessarily risky:
a change to the extraction could silently reorder steps or drop an argument with nothing to
catch it.

**Recommendation being implemented:** write a characterization test against `run()` as it
exists today, then extract it into eight cohesive functions — both in the same piece of
work, since the test's entire purpose is making the extraction safe; shipping them
separately would leave a window where either the safety net exists with nothing to protect,
or the extraction happens without it.

## Complete touch list

| File | What changes |
|---|---|
| `src/training/pipeline.py` | `run()` decomposed into 8 new private functions + itself becomes a thin orchestrator; `_fit_svm_reviewer` (already extracted) is reused unchanged |
| `tests/training/test_pipeline.py` | **New file** — the characterization test(s) |

**Not touched:** every dependency `run()` calls (`make_split`, `BertTunningDataset`,
`compute_class_stats`, `fit_svm_classifiers`, `evaluate_svm_classifiers`, `run_evaluation`,
`WandbLogger`, `save_stats`, `save_svm_classifiers`) — all already have their own test
coverage and keep their existing signatures. `src/training/__init__.py`'s `run` export is
unaffected — the public entry point's signature doesn't change.

## Design

### Extraction shape — 8 functions, `run()` as thin orchestrator

| Function | Replaces | Returns |
|---|---|---|
| `_encode_labels(df)` | Label encoding — mutates `df["label_id"]` in place, identical to today | `_LabelEncoding(le, label2id, id2label, num_labels)` |
| `_split_and_weight(df, num_labels, seed)` | `make_split` + `compute_class_weight` | `_SplitBundle(train_df, val_df, test_df, class_weights)` |
| `_build_datasets(train_df, val_df, test_df, model_cfg, chunk_strategy)` | Tokenizer load + the `_texts` closure + 3× `BertTunningDataset` | `_DatasetBundle(tokenizer, train_ds, val_ds, test_ds)` |
| `_build_model(model_cfg, num_labels, id2label, label2id)` | `AutoModelForSequenceClassification.from_pretrained` | `model` |
| `_compute_hyperparams(model_cfg, request, train_ds, train_df, num_labels)` | Precision detection + steps/warmup calc + `Hyperparams` construction | `_HyperparamsBundle(hyperparams, use_bf16, use_fp16, warmup_steps)` |
| `_build_trainer(model, tokenizer, train_ds, val_ds, class_weights, model_cfg, request, hp_bundle, report_to)` | `TrainingArguments` + `WeightedTrainer` construction | `Trainer` |
| `_generate_auxiliary_artifacts(model, tokenizer, model_cfg, train_df, val_df, class_names, wb)` | Embedding extraction + `compute_class_stats` + `_fit_svm_reviewer` (reused unchanged) | `_AuxiliaryArtifacts(ood_stats, svm_classifiers)` |
| `_persist_artifacts(trainer, tokenizer, ood_stats, svm_classifiers, save_path)` | Final `save_model`/`save_pretrained`/`save_stats`/`save_svm_classifiers` calls | `None` |

`run()` itself becomes a ~25-line sequential story: encode → split+weight → datasets →
model → hyperparams → `wb.init` → trainer → `trainer.train()` → auxiliary artifacts →
`run_evaluation`/`wb.log_results`/`wb.finish` → persist → return `(trainer, le)`.

**`_build_trainer` takes `report_to: str`, not the whole `wb` object** — it only needs the
one string `TrainingArguments` requires, not a dependency on `WandbLogger` existing at all.
`wb.init(hyperparams)` stays an explicit call in `run()` itself (it's a cross-cutting
concern used again later for `log_results`/`finish`, not part of "building a trainer").

**Bundle types are plain `NamedTuple`s**, not Pydantic models — matching the existing
convention already used in this codebase for internal-only return types (`_PcaReduction`,
`_TfidfStats` in `src/ood.py`), since these never cross a validation boundary.

```python
class _LabelEncoding(NamedTuple):
    le: LabelEncoder
    label2id: dict[str, int]
    id2label: dict[int, str]
    num_labels: int


class _SplitBundle(NamedTuple):
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    class_weights: torch.Tensor


class _DatasetBundle(NamedTuple):
    tokenizer: PreTrainedTokenizerBase
    train_ds: BertTunningDataset
    val_ds: BertTunningDataset
    test_ds: BertTunningDataset


class _HyperparamsBundle(NamedTuple):
    hyperparams: Hyperparams
    use_bf16: bool
    use_fp16: bool
    warmup_steps: int


class _AuxiliaryArtifacts(NamedTuple):
    ood_stats: OodArtifact
    svm_classifiers: "dict[str, SVC]"
```

### Characterization test strategy (`tests/training/test_pipeline.py`)

The test must be strong enough to actually catch a mistake during extraction, not just
exist as a checkbox — so the mocking boundary is deliberately narrow: **only mock what's
genuinely expensive, network-bound, or already has its own dedicated test file elsewhere**;
let everything else run for real against a small synthetic fixture (mirroring
`tests/training/test_split.py`'s `balanced_df` fixture — 3 classes × 20 docs).

**Mocked** (via `monkeypatch.setattr("src.training.pipeline.<name>", ...)`):
- `AutoTokenizer.from_pretrained` / `AutoModelForSequenceClassification.from_pretrained` —
  real calls hit HuggingFace's network/cache; always mocked in this test suite for exactly
  this reason. Returns a `MagicMock` tokenizer and a `MagicMock` model with `.config.model_type`,
  `.config.hidden_size`, and `.device` set to real values (not further `MagicMock`s) so
  downstream Pydantic validation (`ArtifactMetadata`) doesn't choke on them.
- `WeightedTrainer` (the class) — constructing a *real* `Trainer` and calling `.train()` on
  it would run an actual (if tiny) training loop against a mock model, which either fails
  or is needlessly slow/flaky. Patched to a `MagicMock` class so `WeightedTrainer(...)`
  returns a `MagicMock` trainer instance; `_build_trainer`'s own logic (constructing
  `TrainingArguments`) still runs for real.
- `extract_embeddings` — real embedding extraction needs a genuine forward pass; mocked to
  return `rng.normal(size=(len(texts), 8))` via `side_effect` so both call sites (training
  embeddings in `run()`, validation embeddings inside the existing `_fit_svm_reviewer`) get
  shape-correct output regardless of split size.
- `run_evaluation` — needs `trainer.predict()`, which isn't real once `WeightedTrainer` is
  mocked. No dedicated test file exists for `run_evaluation` today (confirmed via `ls
  tests/training/`) — mocked here as a pre-existing gap, not something this task expands
  scope to fix.

**Real** (exercised for real against the synthetic fixture, not mocked):
- `_encode_labels`, `_split_and_weight` (`make_split` + `compute_class_weight`),
  `_build_datasets` (`BertTunningDataset` construction against the mocked-but-configured
  tokenizer) — all pure/cheap, and exactly the wiring this test exists to lock down.
- `_generate_auxiliary_artifacts`'s internals (`compute_class_stats`,
  `fit_svm_classifiers`/`evaluate_svm_classifiers` via `_fit_svm_reviewer`) — run for real
  against the mocked `extract_embeddings`' fixed-shape output; both already have their own
  extensive test coverage (`tests/test_ood.py`, `tests/test_svm_reviewer.py`), but running
  them for real here (cheaply, on a tiny fixture) verifies `run()` actually wires them
  together correctly, which is the whole point of an orchestration test.
- `WandbLogger` — exercised via `request.use_wandb=False`, the real no-op code path (not
  mocked), so no network risk while still verifying `run()` calls `wb.init`/`wb.log_results`/
  `wb.finish`/`log_svm_results` without raising.
- `save_stats`/`save_svm_classifiers` — run for real against `tmp_path`, actually writing
  `ood_stats.npz`/`svm_classifiers.joblib` and letting the test assert the files exist —
  strong end-to-end coverage of the full auxiliary-artifact-through-persistence path.
- `trainer.save_model()`/`tokenizer.save_pretrained()` — captured automatically since
  `trainer`/`tokenizer` are already `MagicMock`s; asserted via `assert_called_with`.

**Assertions the test makes:**
1. `run()` returns `(trainer, le)` — the same trainer instance `WeightedTrainer(...)`
   produced, and a `LabelEncoder` whose `.classes_` matches the fixture's 3 labels.
2. `AutoModelForSequenceClassification.from_pretrained` was called with `num_labels`,
   `id2label`, `label2id` matching the fixture's label encoding.
3. `trainer.train()` was called.
4. `trainer.save_model()`/`tokenizer.save_pretrained()` were called with the expected
   `save_path` (`Path(request.output_dir) / "final"`).
5. `ood_stats.npz`/`svm_classifiers.joblib` exist at that same path afterward, and
   `load_stats()`/`load_svm_classifiers()` can read them back with the fixture's 3 class
   names.

## Backward compatibility

- `run()`'s public signature (`df, model_cfg, request) -> tuple[Trainer, LabelEncoder]`) is
  unchanged — `src/training/__init__.py`'s export and every caller are unaffected.
- The characterization test is written **against today's `run()` first**, run to confirm it
  passes, then the extraction happens, then the same test must pass unchanged (or with only
  mechanical import updates if the new private functions need direct unit tests added later)
  — that sequencing is what makes it a genuine characterization test rather than a test
  written to match whatever the refactor produces.

## Known follow-up, out of scope for this spec

`run_evaluation` (`src/training/evaluate.py`) has no dedicated test file
(`tests/training/test_evaluate.py` doesn't exist) — this task mocks it rather than
exercising it for real (see "Mocked" above), which is a pre-existing coverage gap, not one
this task's scope extends to fixing.
