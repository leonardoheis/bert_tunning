# OOD Remediation Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four issues found in a critical review of PR #43 (`task/49-ood-review-remediation`): (P1) uncalibrated models silently inherit BETO v2's thresholds with no visibility, (P1) the class-mapping validator can't tell two different model architectures apart when they share a label set, (P2) `save_stats()` writes `ood_stats.npz` non-atomically, (P2) W&B still logs global `Settings.OOD_*` thresholds instead of the resolved per-model values, and never logs a k-NN threshold at all.

**Architecture:** Land these directly on the existing branch `task/49-ood-review-remediation` (PR #43 already open) — this is a continuation of that PR's work, not a new branch. Fix `save_stats()`'s atomicity first (Task 1), since Task 2's new identity fields build on top of the same function. Add a coarse `(model_type, hidden_size)` fingerprint to `ClassEmbeddingStats`, populated at both `compute_class_stats()` call sites and validated at classifier construction (Task 2) — this catches the realistic mistake in this codebase (all three registered models can share an identical `class_names` list since they're commonly trained on the same corpus). Add a one-time startup warning (not a hard failure) when a loaded model's thresholds are falling back to `Settings.OOD_*` (Task 3) — deliberately not a fail-fast, since failing startup would also break BETO v2's own currently-correct Mahalanobis fallback (its threshold is `None` because the degenerate-threshold guard correctly refused to write a floor-adjacent value, not because it was never calibrated). Thread resolved `OodThresholds` into W&B logging, including the previously-missing k-NN threshold (Task 4).

**Tech Stack:** Same as the rest of the project — Pydantic v2, Click, FastAPI, PyTorch/Transformers, pytest, numpy.

## Global Constraints

- This plan continues PR #43 / branch `task/49-ood-review-remediation` in the existing worktree at `C:/Users/leona/source/repos/bert_tunning-task49` — do not create a new branch or worktree.
- `poe check` (lint + typecheck + test) must pass after every task.
- Every new/changed function gets a WHY-comment docstring in this codebase's existing style (explains the non-obvious reasoning, not what the code does).
- Backward compatibility: existing `ood_stats.npz` files (including the currently-committed BETO v2 one) must keep loading without error. New optional fields default to `None`/absent and are validated only when present.
- No retraining required for any task in this plan.
- `mahalanobis_p_value_theoretical` still never participates in the OOD decision — unrelated to this plan, don't touch it.
- The three-signal OR decision logic (`is_out_of_distribution`) is unchanged by this plan.

---

### Task 1: Atomic `save_stats()` write with load-back verification

**Files:**
- Modify: `src/ood.py` (`save_stats`)
- Test: `tests/test_ood.py`

**Interfaces:**
- Produces: `save_stats(stats, path)` — same public signature, now writes via a temp file + verified load-back + atomic rename instead of writing directly to `path`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py` (near the existing `test_save_and_load_stats_roundtrip*` tests):

```python
def test_save_stats_leaves_original_file_untouched_if_write_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_atomic_original.npz")
    try:
        # A real, previously-good file already exists at `path`.
        save_stats(stats, path)
        original_bytes = path.read_bytes()

        # Simulate a crash mid-write: np.savez succeeds but the written file is corrupt/
        # incomplete, so load_stats(tmp_path) inside save_stats must raise before the
        # original file is ever touched.
        def _broken_savez(*_args: object, **_kwargs: object) -> None:
            msg = "simulated write failure"
            raise OSError(msg)

        monkeypatch.setattr(np, "savez", _broken_savez)
        with pytest.raises(OSError, match="simulated write failure"):
            save_stats(stats, path)

        assert path.read_bytes() == original_bytes  # untouched
        assert not path.with_name(path.name + ".tmp").exists()  # tmp file cleaned up
    finally:
        path.unlink(missing_ok=True)
        path.with_name(path.name + ".tmp").unlink(missing_ok=True)


def test_save_stats_does_not_leave_tmp_file_on_success() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_atomic_success.npz")
    try:
        save_stats(stats, path)
        assert path.exists()
        assert not path.with_name(path.name + ".tmp").exists()
    finally:
        path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_ood.py -v -k "atomic"`
Expected: FAIL — `test_save_stats_leaves_original_file_untouched_if_write_fails` fails because today's `save_stats` writes directly to `path` via `np.savez(str(path), ...)`, so a failure mid-write can corrupt `path` itself (there's nothing here to assert cleanly yet — the point is this test currently has no `.tmp` file to check for and the "untouched" guarantee doesn't hold). `test_save_stats_does_not_leave_tmp_file_on_success` currently passes trivially (no tmp file exists because none is ever created) — that's expected; it becomes a real regression guard after Step 3.

- [ ] **Step 3: Rewrite `save_stats` in `src/ood.py`**

Add `import os` to the top of `src/ood.py` (alongside the existing `from pathlib import Path`).

Replace the existing `save_stats` function with:

```python
def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
    # npz has no native "missing key" for a single scalar the way a dict does, and no None --
    # NaN is the serialization sentinel for "not yet calibrated," round-tripped back to None
    # by load_stats's _optional_threshold.
    maha_threshold = (
        np.nan if stats.mahalanobis_p_threshold is None else stats.mahalanobis_p_threshold
    )
    cosine_thresh = np.nan if stats.cosine_threshold is None else stats.cosine_threshold
    knn_thresh = np.nan if stats.knn_distance_threshold is None else stats.knn_distance_threshold

    # Write to a temp file first, verify it actually loads back, then atomically replace the
    # real path -- np.savez writing directly to `path` left a window where a crash, disk-full,
    # or kill mid-write corrupts the ONLY copy of ood_stats.npz, which predict/serve also read
    # from. os.replace() is atomic on both POSIX and Windows when src/dst are on the same
    # filesystem (always true here -- same directory).
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            np.savez(
                f,
                class_names=np.array(stats.class_names),
                pca_mean=stats.pca_mean,
                pca_components=stats.pca_components,
                centroids=stats.centroids,
                covariance_inv=stats.covariance_inv,
                cosine_calibration_mean=stats.cosine_calibration_mean,
                cosine_calibration_std=stats.cosine_calibration_std,
                knn_train_embeddings=stats.knn_train_embeddings,
                knn_train_labels=np.array(stats.knn_train_labels),
                mahalanobis_p_threshold=maha_threshold,
                cosine_threshold=cosine_thresh,
                knn_distance_threshold=knn_thresh,
            )
        load_stats(tmp_path)  # fail fast on a corrupt/incomplete write, before touching `path`
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
```

Note: passing an open file object (`f`) rather than a string path to `np.savez` is required here, not just style — `np.savez` auto-appends a `.npz` extension to string/`Path` filenames that don't already end in `.npz` (so a string `tmp_path` ending in `.npz.tmp` would otherwise get saved as `....npz.tmp.npz`, not the path we opened). Passing a file object bypasses that filename logic entirely; the bytes go exactly where `open()` pointed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ood.py -v`
Expected: PASS, all tests including the 2 new ones and every pre-existing `save_stats`/`load_stats` round-trip test (their behavior is unchanged from the caller's perspective).

- [ ] **Step 5: Run full check and commit**

```bash
cd C:/Users/leona/source/repos/bert_tunning-task49
uv run poe check
git add src/ood.py tests/test_ood.py
git commit -m "fix: write ood_stats.npz atomically with load-back verification"
git push
```

---

### Task 2: Per-model identity fingerprint (`model_type` + `hidden_size`)

**Files:**
- Modify: `src/schema.py` (`ClassEmbeddingStats`)
- Modify: `src/ood.py` (`compute_class_stats`, `save_stats`, `load_stats`)
- Modify: `src/training/pipeline.py` (`run`)
- Modify: `src/cli/ood_stats.py` (`_run_compute_ood_stats`)
- Modify: `src/inference/classify.py` (`BertTunningClassifier`)
- Test: `tests/test_ood.py`, `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1's atomic `save_stats`.
- Produces: `ClassEmbeddingStats.model_type: str | None`, `.model_hidden_size: int | None`; `compute_class_stats(..., model_type: str | None = None, model_hidden_size: int | None = None)`; `BertTunningClassifier._validate_ood_stats_model_identity()`, called once at construction alongside the existing `_validate_ood_stats_class_mapping()`.

**Why `model_type` + `hidden_size`, not something stronger:** this project's three registered models (`src/training/models/{xlm_roberta,beto,minilm}.py`) are commonly trained on the identical corpus and label set, so `class_names` (a property of the *data*, not the *model*) is often identical across all three — the existing Task 3 validator (from PR #43) can't distinguish them. `model_type` (`"xlm-roberta"` vs `"bert"`) and `hidden_size` (768 for XLM-RoBERTa-base and BETO, 384 for the registered MiniLM) together disambiguate all three currently-registered models without requiring a new CLI flag or introspecting fragile HF checkpoint metadata (e.g. `config._name_or_path`, which gets overwritten after `trainer.save_model()` and can't be trusted). This is a coarse fingerprint, not a cryptographic guarantee — two different checkpoints of the exact same architecture and hidden size would still collide — but it catches the realistic mistake in this codebase.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py`:

```python
def test_save_and_load_stats_roundtrip_includes_model_identity() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(
        embeddings, labels, class_names, n_components=8,
        model_type="bert", model_hidden_size=768,
    )
    path = Path("test_stats_identity.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.model_type == "bert"
        assert loaded.model_hidden_size == 768
    finally:
        path.unlink(missing_ok=True)


def test_save_and_load_stats_roundtrip_model_identity_defaults_to_none() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_identity_none.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.model_type is None
        assert loaded.model_hidden_size is None
    finally:
        path.unlink(missing_ok=True)


def test_load_stats_handles_legacy_file_without_model_identity_fields() -> None:
    # A pre-this-task ood_stats.npz (including one written by Task 1's atomic save_stats,
    # which predates the model_type/model_hidden_size keys) has neither key at all.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_identity_legacy.npz")
    try:
        with open(path, "wb") as f:
            np.savez(
                f,
                class_names=np.array(stats.class_names),
                pca_mean=stats.pca_mean,
                pca_components=stats.pca_components,
                centroids=stats.centroids,
                covariance_inv=stats.covariance_inv,
                cosine_calibration_mean=stats.cosine_calibration_mean,
                cosine_calibration_std=stats.cosine_calibration_std,
                knn_train_embeddings=stats.knn_train_embeddings,
                knn_train_labels=np.array(stats.knn_train_labels),
                mahalanobis_p_threshold=np.nan,
                cosine_threshold=np.nan,
                knn_distance_threshold=np.nan,
            )
        loaded = load_stats(path)
        assert loaded.model_type is None
        assert loaded.model_hidden_size is None
    finally:
        path.unlink(missing_ok=True)
```

Add to `tests/inference/test_pipeline.py` (near the existing `test_classifier_raises_when_ood_stats_class_names_mismatch_model_id2label` tests):

```python
def test_classifier_raises_when_ood_stats_model_identity_mismatches(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    # Same class_names/order as the loaded model (passes the existing class-mapping check),
    # but computed from a different model_type -- this is the exact gap the class-name-only
    # check can't catch.
    stats = _make_stats().model_copy(update={"model_type": "bert", "model_hidden_size": 768})
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(BertTunningError, match="different model architecture"),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)


def test_classifier_loads_fine_when_ood_stats_model_identity_matches(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    stats = _make_stats().model_copy(
        update={"model_type": "xlm-roberta", "model_hidden_size": 768}
    )
    save_stats(stats, tmp_path / "ood_stats.npz")

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001


def test_classifier_skips_model_identity_check_when_stats_predate_the_field(
    tmp_path: Path,
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512
    model.config.model_type = "xlm-roberta"
    model.config.hidden_size = 768

    # _make_stats() doesn't set model_type/model_hidden_size -- both default to None,
    # simulating an ood_stats.npz written before this field existed. No raise expected.
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_ood.py tests/inference/test_pipeline.py -v -k "model_identity or identity"`
Expected: FAIL — `ClassEmbeddingStats` has no `model_type`/`model_hidden_size` fields yet, `compute_class_stats` doesn't accept those kwargs, and `BertTunningClassifier` has no identity validation.

- [ ] **Step 3: Add the fields to `ClassEmbeddingStats`**

In `src/schema.py`, inside `ClassEmbeddingStats` (after `knn_distance_threshold`):

```python
    knn_distance_threshold: float | None = None
    # Coarse per-model identity fingerprint -- written by compute_class_stats() from the
    # model that produced the embeddings, validated at classifier construction in
    # BertTunningClassifier._validate_ood_stats_model_identity(). class_names alone can't
    # distinguish two different model architectures trained on the same corpus/label set
    # (true for every model in this project's registry). None means "predates this field" --
    # the identity check is skipped entirely, not enforced as absent.
    model_type: str | None = None
    model_hidden_size: int | None = None
```

- [ ] **Step 4: Thread `model_type`/`model_hidden_size` through `compute_class_stats`**

In `src/ood.py`, change the `compute_class_stats` signature and return:

```python
def compute_class_stats(
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
    model_type: str | None = None,
    model_hidden_size: int | None = None,
) -> ClassEmbeddingStats:
    pca_result = _reduce_dimensionality(embeddings, n_components)
    reduced = pca_result.reduced
    labels_arr = np.asarray(labels)

    centroids = np.stack([reduced[labels_arr == k].mean(axis=0) for k in range(len(class_names))])
    centered = reduced - centroids[labels_arr]
    covariance = (centered.T @ centered) / reduced.shape[0]
    covariance_reg = covariance + covariance_epsilon * np.eye(covariance.shape[0])
    covariance_inv = np.linalg.inv(covariance_reg)

    cosine_scores = np.array(
        [_cosine_min_distance_raw(reduced[i], centroids) for i in range(reduced.shape[0])]
    )

    return ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=pca_result.mean,
        pca_components=pca_result.components,
        centroids=centroids,
        covariance_inv=covariance_inv,
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
        knn_train_embeddings=reduced,
        knn_train_labels=labels_arr.tolist(),
        model_type=model_type,
        model_hidden_size=model_hidden_size,
    )
```

- [ ] **Step 5: Update `save_stats`/`load_stats` in `src/ood.py`**

In `save_stats` (the version Task 1 just rewrote), add the two new sentinel locals right after the threshold sentinels, and add the two new keys to the `np.savez(...)` call:

```python
    maha_threshold = (
        np.nan if stats.mahalanobis_p_threshold is None else stats.mahalanobis_p_threshold
    )
    cosine_thresh = np.nan if stats.cosine_threshold is None else stats.cosine_threshold
    knn_thresh = np.nan if stats.knn_distance_threshold is None else stats.knn_distance_threshold
    # "" / -1 are the None-sentinels for a string/int field, the same role NaN plays for the
    # threshold floats above -- npz has no native optional-scalar support.
    model_type = "" if stats.model_type is None else stats.model_type
    model_hidden_size = -1 if stats.model_hidden_size is None else stats.model_hidden_size
```

and inside the `np.savez(f, ...)` call, add after `knn_distance_threshold=knn_thresh,`:

```python
                model_type=model_type,
                model_hidden_size=model_hidden_size,
```

Add two new helper functions right after `_optional_threshold`, and extend `load_stats`:

```python
def _optional_str(data: npt.NDArray[np.str_]) -> str | None:
    value = str(data)
    return None if value == "" else value


def _optional_int(data: npt.NDArray[np.int_]) -> int | None:
    value = int(data)
    return None if value == -1 else value
```

In `load_stats`, add after the `knn_distance_threshold=...` field:

```python
        model_type=_optional_str(data["model_type"]) if "model_type" in data.files else None,
        model_hidden_size=_optional_int(data["model_hidden_size"])
        if "model_hidden_size" in data.files
        else None,
```

- [ ] **Step 6: Pass `model_type`/`model_hidden_size` at both `compute_class_stats` call sites**

In `src/training/pipeline.py`'s `run()`, change:

```python
    ood_stats = compute_class_stats(
        train_embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        n_components=Settings.OOD_PCA_COMPONENTS,
    )
```

to:

```python
    ood_stats = compute_class_stats(
        train_embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=model.config.model_type,
        model_hidden_size=model.config.hidden_size,
    )
```

(`model` is already in scope in `run()` — it's the `AutoModelForSequenceClassification` instance created earlier in the same function.)

In `src/cli/ood_stats.py`'s `_run_compute_ood_stats()`, change:

```python
    stats = compute_class_stats(
        embeddings,
        split.train_df["label_id"].tolist(),
        split.classes,
        n_components=Settings.OOD_PCA_COMPONENTS,
    )
```

to:

```python
    stats = compute_class_stats(
        embeddings,
        split.train_df["label_id"].tolist(),
        split.classes,
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=split.loaded.model.config.model_type,
        model_hidden_size=split.loaded.model.config.hidden_size,
    )
```

- [ ] **Step 7: Add the identity validation to `BertTunningClassifier`**

In `src/inference/classify.py`, in `__init__`, change:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

to:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        self._validate_ood_stats_model_identity()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

Add, right after `_validate_ood_stats_class_mapping`:

```python
    def _validate_ood_stats_model_identity(self) -> None:
        """class_names alone can't distinguish two different model architectures when both
        were trained on the identical corpus/label set -- true for every model in this
        project's registry (xlm-roberta/beto/minilm all commonly train on the same
        FOLDER_TO_LABEL classes). model_type + hidden_size is a coarse fingerprint that
        catches the realistic mistake (copying one model's ood_stats.npz next to a
        different architecture) without a new CLI flag or fragile checkpoint-metadata
        introspection. Skipped entirely when the stats predate this field (both None) --
        this is an additional check layered on top of the class-mapping one, not a
        replacement for it, and not a hard requirement for older artifacts."""
        if self._ood_stats is None:
            return
        if self._ood_stats.model_type is None or self._ood_stats.model_hidden_size is None:
            return
        actual_type = self.model.config.model_type
        actual_hidden_size = self.model.config.hidden_size
        if (
            self._ood_stats.model_type != actual_type
            or self._ood_stats.model_hidden_size != actual_hidden_size
        ):
            msg = (
                f"ood_stats.npz was computed from model_type={self._ood_stats.model_type!r}, "
                f"hidden_size={self._ood_stats.model_hidden_size}, but the loaded model is "
                f"model_type={actual_type!r}, hidden_size={actual_hidden_size} -- this "
                "ood_stats.npz belongs to a different model architecture. Regenerate it for "
                "this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ood.py tests/inference/test_pipeline.py -v`
Expected: PASS, all tests.

- [ ] **Step 9: Run full check and commit**

```bash
uv run poe check
git add src/schema.py src/ood.py src/training/pipeline.py src/cli/ood_stats.py src/inference/classify.py tests/test_ood.py tests/inference/test_pipeline.py
git commit -m "fix: add per-model identity fingerprint to ood_stats.npz, validated at construction"
git push
```

---

### Task 3: Warn (not fail) when a model's OOD thresholds fall back to Settings

**Files:**
- Modify: `src/inference/classify.py` (`BertTunningClassifier`)
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `ClassEmbeddingStats.mahalanobis_p_threshold`/`.cosine_threshold`/`.knn_distance_threshold` (already exist from PR #43).
- Produces: `BertTunningClassifier._warn_on_uncalibrated_thresholds()`, called once at construction.

**Why a warning, not a hard failure:** the reviewed alternative ("disable the signal or fail startup") is too strong as default behavior — it would also break BETO v2's own currently-correct Mahalanobis fallback. BETO v2's `ood_stats.npz` has `mahalanobis_p_threshold=None` not because it was never calibrated, but because `evaluate-ood-calibration --write-thresholds`'s degenerate-threshold guard correctly refused to persist a floor-adjacent value (see `_write_calibrated_thresholds` in `src/cli/ood_calibration.py`) — `Settings.OOD_MAHALANOBIS_P_THRESHOLD` *is* BETO v2's own calibrated value in this case, just represented via the fallback rather than a written field. Failing startup (or disabling the signal) for that case would be actively wrong. A warning makes the fallback visible and auditable without breaking a model that's already correctly configured.

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_pipeline.py`:

```python
def test_classifier_warns_when_thresholds_fall_back_to_settings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    # _make_stats() leaves all three thresholds at their None default -- fully uncalibrated.
    save_stats(_make_stats(), tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert any("falling back to Settings.OOD_*" in record.message for record in caplog.records)
    assert any("mahalanobis_p_threshold" in record.message for record in caplog.records)
    assert any("cosine_threshold" in record.message for record in caplog.records)
    assert any("knn_distance_threshold" in record.message for record in caplog.records)


def test_classifier_does_not_warn_when_thresholds_are_fully_calibrated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats().model_copy(
        update={
            "mahalanobis_p_threshold": 0.001,
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 16.7908,
        }
    )
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any("falling back to Settings.OOD_*" in record.message for record in caplog.records)


def test_classifier_does_not_warn_when_no_ood_stats(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.WARNING),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any("falling back to Settings.OOD_*" in record.message for record in caplog.records)
```

Add `import logging` to `tests/inference/test_pipeline.py` if not already present (check the top of the file first — it very likely already imports `logging` given the codebase's logging conventions; only add if missing).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/inference/test_pipeline.py -v -k "warns_when_thresholds or does_not_warn"`
Expected: FAIL — no such warning exists yet.

- [ ] **Step 3: Add the warning to `BertTunningClassifier`**

In `src/inference/classify.py`, in `__init__`, change:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        self._validate_ood_stats_model_identity()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

to:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        self._validate_ood_stats_model_identity()
        self._warn_on_uncalibrated_thresholds()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

Add, right after `_validate_ood_stats_model_identity`:

```python
    def _warn_on_uncalibrated_thresholds(self) -> None:
        """resolve_ood_thresholds()'s silent Settings.OOD_* fallback is intentional backward
        compatibility, not something to hide from whoever operates this service -- a model
        that's never been through `evaluate-ood-calibration --write-thresholds` (or, like
        BETO v2's Mahalanobis threshold, had a write correctly refused by the
        degenerate-threshold guard) silently inherits whichever model Settings.OOD_* happens
        to be calibrated for. This does not fail startup -- an uncalibrated model is still
        usable, just with potentially miscalibrated OOD decisions -- but it must not be
        silent either. Runs once at construction, not per-request."""
        if self._ood_stats is None:
            return
        uncalibrated = [
            name
            for name, value in (
                ("mahalanobis_p_threshold", self._ood_stats.mahalanobis_p_threshold),
                ("cosine_threshold", self._ood_stats.cosine_threshold),
                ("knn_distance_threshold", self._ood_stats.knn_distance_threshold),
            )
            if value is None
        ]
        if uncalibrated:
            log.warning(
                "ood_stats.npz has no per-model value for %s -- falling back to Settings.OOD_* "
                "(calibrated for a specific model, not necessarily this one). Run "
                "evaluate-ood-calibration --write-thresholds for this model to silence this.",
                ", ".join(uncalibrated),
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/inference/test_pipeline.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run full check and commit**

```bash
uv run poe check
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "feat: warn at classifier construction when OOD thresholds fall back to Settings"
git push
```

---

### Task 4: W&B logs resolved per-model thresholds, including k-NN

**Files:**
- Modify: `src/wandb.py` (`log_ood_calibration_results`)
- Modify: `src/cli/ood_calibration.py` (`_run_ood_calibration`)
- Test: `tests/test_wandb.py`

**Interfaces:**
- Consumes: `OodThresholds` (existing, `src/ood.py`).
- Produces: `log_ood_calibration_results(report, *, model_path, cache_path, target_fp_rate, thresholds: OodThresholds)` — new required `thresholds` keyword parameter.

- [ ] **Step 1: Write the failing test**

Replace the existing test in `tests/test_wandb.py`:

```python
def test_log_ood_calibration_results_logs_summary_metrics() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.2951,
        fp_rate_cosine=0.0104,
        fp_rate_knn=0.0087,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=13.7186,
        suggested_knn_threshold=4.2,
    )
    thresholds = OodThresholds(mahalanobis_p=0.001, cosine_z=13.7366, knn_distance=16.7908)
    with (
        patch("src.wandb.wandb.init") as mock_init,
        patch("src.wandb.wandb.log") as mock_log,
        patch("src.wandb.wandb.finish") as mock_finish,
    ):
        log_ood_calibration_results(
            report,
            model_path="fake/model",
            cache_path="fake/cache.parquet",
            target_fp_rate=0.01,
            thresholds=thresholds,
        )

    mock_init.assert_called_once()
    assert mock_init.call_args.kwargs["job_type"] == "ood-calibration"
    # The resolved per-model thresholds (not Settings.OOD_*) must be what's logged as
    # "current" -- and knn is now included, which it wasn't before this task.
    config = mock_init.call_args.kwargs["config"]
    assert config["current_mahalanobis_threshold"] == 0.001
    assert config["current_cosine_threshold"] == 13.7366
    assert config["current_knn_threshold"] == 16.7908
    mock_log.assert_called_once_with(
        {
            "ood/fp_rate_mahalanobis": 0.2951,
            "ood/fp_rate_cosine": 0.0104,
            "ood/suggested_mahalanobis_threshold": 0.0,
            "ood/suggested_cosine_threshold": 13.7186,
            "ood/fp_rate_knn": 0.0087,
            "ood/suggested_knn_threshold": 4.2,
        }
    )
```

Add `OodThresholds` to `tests/test_wandb.py`'s imports (`from src.ood import OodThresholds`).

Remove the unused `mock_finish` warning if ruff flags it (it was already unused-by-assertion in the original test — check whether the original file uses `# noqa` or similar; if the existing test already had this pattern, keep it identical).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_wandb.py -v -k "log_ood_calibration_results"`
Expected: FAIL — `log_ood_calibration_results` doesn't accept a `thresholds` keyword yet.

- [ ] **Step 3: Update `log_ood_calibration_results` in `src/wandb.py`**

Add `OodThresholds` to the imports (`from src.ood import OodThresholds`).

Replace the function:

```python
def log_ood_calibration_results(
    report: CalibrationReport,
    *,
    model_path: str,
    cache_path: str,
    target_fp_rate: float,
    thresholds: OodThresholds,
) -> None:
    """Log an evaluate-ood-calibration run's summary metrics to W&B.

    `thresholds` must be the resolved per-model OodThresholds (resolve_ood_thresholds(stats)),
    not Settings.OOD_* directly -- otherwise a W&B dashboard comparing calibration runs across
    models with different per-model thresholds shows the identical "current threshold" for
    every model, silently wrong once any model has calibrated values written via
    --write-thresholds. Also logs the k-NN threshold, which this function previously omitted
    entirely.
    """
    wandb.init(
        entity=Settings.WANDB_ENTITY,
        project=Settings.WANDB_PROJECT,
        job_type="ood-calibration",
        config={
            "model_path": model_path,
            "cache_path": cache_path,
            "target_fp_rate": target_fp_rate,
            "current_mahalanobis_threshold": thresholds.mahalanobis_p,
            "current_cosine_threshold": thresholds.cosine_z,
            "current_knn_threshold": thresholds.knn_distance,
        },
    )
    wandb.log(
        {
            "ood/fp_rate_mahalanobis": report.fp_rate_maha,
            "ood/fp_rate_cosine": report.fp_rate_cosine,
            "ood/suggested_mahalanobis_threshold": report.suggested_maha_threshold,
            "ood/suggested_cosine_threshold": report.suggested_cosine_threshold,
            "ood/fp_rate_knn": report.fp_rate_knn,
            "ood/suggested_knn_threshold": report.suggested_knn_threshold,
        }
    )
    wandb.finish()
    log.info(
        "Logged OOD calibration results to W&B (%s/%s)",
        Settings.WANDB_ENTITY,
        Settings.WANDB_PROJECT,
    )
```

- [ ] **Step 4: Update the call site in `src/cli/ood_calibration.py`**

`_run_ood_calibration` already computes `current_thresholds = resolve_ood_thresholds(stats)` (from PR #43's own review-fix commit). Change the `log_ood_calibration_results` call:

```python
    if opts.log_wandb:
        log_ood_calibration_results(
            report,
            model_path=opts.model_path,
            cache_path=opts.cache_path,
            target_fp_rate=opts.target_fp_rate,
            thresholds=current_thresholds,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_wandb.py tests/cli/test_ood_calibration.py -v`
Expected: PASS, all tests.

- [ ] **Step 6: Run full check and commit**

```bash
uv run poe check
git add src/wandb.py src/cli/ood_calibration.py tests/test_wandb.py
git commit -m "fix: log resolved per-model OOD thresholds to W&B, including k-NN"
git push
```

---

### Task 5: Documentation

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `CLAUDE.md`**

Add a new paragraph to "Key Technical Decisions", after the three paragraphs added by PR #43 (per-model OOD thresholds / predicted-label k-NN calibration / class-mapping validation):

```markdown
**Follow-up hardening: model identity fingerprint, atomic writes, uncalibrated-threshold visibility, W&B parity (2026-07-13)**
A critical review of the per-model OOD threshold work above found four gaps. (1) `_validate_ood_stats_class_mapping()` only compared `class_names`, which is a property of the training corpus, not the model — since this project's three registered models (`xlm-roberta`, `beto`, `minilm`) are commonly trained on the identical label set, a `beto` model's `ood_stats.npz` copied next to an `xlm-roberta` checkpoint would pass that check trivially. `ClassEmbeddingStats` now also carries an optional `model_type`/`model_hidden_size` fingerprint (from `model.config.model_type`/`.hidden_size` at `compute_class_stats()` time), validated by a new `_validate_ood_stats_model_identity()` — skipped entirely for stats predating the field, enforced when present. (2) `save_stats()` wrote directly to `ood_stats.npz` with no atomicity; an interrupted write could corrupt the only copy a running server reads from. It now writes to a temp file, verifies the temp file loads back with `load_stats()`, then `os.replace()`s it onto the real path. (3) `resolve_ood_thresholds()`'s Settings fallback was completely silent — a freshly trained, never-calibrated model would inherit BETO v2's thresholds with no visibility. `BertTunningClassifier` now logs one `WARNING` at construction naming which specific thresholds are falling back. This is deliberately a warning, not a startup failure or a disabled signal: BETO v2's own `mahalanobis_p_threshold` is `None` because `--write-thresholds`'s degenerate-guard correctly refused to persist a floor-adjacent value, not because it was never calibrated — `Settings.OOD_MAHALANOBIS_P_THRESHOLD` genuinely is BETO v2's calibrated value in that case, so failing startup would break a correctly-configured model. (4) `log_ood_calibration_results()` logged `Settings.OOD_*` directly instead of the resolved per-model thresholds (the same gap already fixed in `evaluate-ood-calibration`'s own console/log output during PR #43's review), and never logged a k-NN threshold at all — both fixed, threading the same `OodThresholds` the CLI's log lines use.
```

- [ ] **Step 2: Commit**

```bash
uv run poe check
git add CLAUDE.md
git commit -m "docs: document identity fingerprint, atomic writes, threshold-fallback warning, W&B parity"
git push
```

---

### Task 6: Update the PR

Not a code task.

- [ ] **Step 1: Update PR #43's description**

Add a section to the existing PR description (`gh pr edit 43 --body ...`, preserving the existing content and appending) summarizing this follow-up work: the four issues found in review, and that they're now fixed on the same branch. Do not open a new PR — this plan's commits land on the existing `task/49-ood-review-remediation` branch.

- [ ] **Step 2: Note remaining known limitations**

The `(model_type, hidden_size)` fingerprint (Task 2) is coarse, not cryptographic — document this expectation is already captured in the code's docstrings and Task 5's `CLAUDE.md` paragraph; no further action needed, just confirm it's not overstated as a complete fix anywhere in the PR description.
