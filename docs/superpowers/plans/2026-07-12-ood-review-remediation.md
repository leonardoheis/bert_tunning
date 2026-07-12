# OOD Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four verified defects in the OOD detection feature: (P1) global thresholds applied regardless of which model/corpus calibrated them, (P2a) k-NN calibration scoring against the true label instead of the model's own prediction, (P2b) no validation that `ood_stats.npz` belongs to the loaded model, (P3) unbounded `/predict` upload reads.

**Architecture:** Store calibrated thresholds *inside* each model's own `ood_stats.npz` (new optional fields on `ClassEmbeddingStats`), resolved at decision time with a fallback to `Settings.OOD_*` for artifacts that predate this change. Fix k-NN calibration to run the model's real forward pass and use its predicted label, matching production exactly. Validate `ood_stats.npz`'s class ordering against the loaded model's `id2label` once, at classifier construction (fail fast, not per-request). Bound the `/predict` upload read with a configurable byte cap.

**Tech Stack:** Same as the rest of the project — Pydantic v2, Click, FastAPI, PyTorch/Transformers, pytest.

## Global Constraints

- `Settings.OOD_MAHALANOBIS_P_THRESHOLD` / `OOD_COSINE_THRESHOLD` / `OOD_KNN_DISTANCE_THRESHOLD` remain the fallback defaults for any `ood_stats.npz` that doesn't carry per-model thresholds yet — this is a backward-compatible addition, not a hard break. Existing committed artifacts keep working via fallback until explicitly regenerated (Task 8).
- `mahalanobis_p_value_theoretical` (chi²) never participates in the OOD decision — unchanged from prior work, do not wire it into `is_out_of_distribution()` or any threshold resolution.
- The three-signal OR decision logic itself is unchanged — this plan changes *which thresholds* are compared against, not *how* the comparison combines.
- `poe check` (lint + typecheck + test) must pass after every task.
- Every new/changed Pydantic model or function gets a WHY-comment docstring in this codebase's existing style (explains the non-obvious reasoning, not what the code does) — see any function in `src/ood.py` for the established tone.
- `ClassEmbeddingStats` has no `alias_generator`/`populate_by_name` — new fields on it are plain snake_case, no camelCase alias gotcha applies.
- No retraining required for any task in this plan.

---

### Task 1: Optional per-model threshold fields on `ClassEmbeddingStats`

**Files:**
- Modify: `src/schema.py` (`ClassEmbeddingStats`)
- Modify: `src/ood.py` (`save_stats`, `load_stats`)
- Test: `tests/test_ood.py`

**Interfaces:**
- Produces: `ClassEmbeddingStats.mahalanobis_p_threshold: float | None`, `.cosine_threshold: float | None`, `.knn_distance_threshold: float | None` — all default `None`. Later tasks read these via `resolve_ood_thresholds()` (Task 2) and write them via `--write-thresholds` (Task 5).

- [ ] **Step 1: Write the failing round-trip tests**

Add to `tests/test_ood.py` (near the existing `test_save_and_load_stats_roundtrip*` tests):

```python
def test_save_and_load_stats_roundtrip_includes_thresholds() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={
            "mahalanobis_p_threshold": 0.001,
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 26.125,
        }
    )
    path = Path("test_stats_thresholds.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold == pytest.approx(0.001)
        assert loaded.cosine_threshold == pytest.approx(13.7366)
        assert loaded.knn_distance_threshold == pytest.approx(26.125)
    finally:
        path.unlink(missing_ok=True)


def test_save_and_load_stats_roundtrip_thresholds_default_to_none() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_no_thresholds.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold is None
        assert loaded.cosine_threshold is None
        assert loaded.knn_distance_threshold is None
    finally:
        path.unlink(missing_ok=True)


def test_load_stats_handles_legacy_file_without_threshold_fields() -> None:
    # A pre-this-change ood_stats.npz has no threshold keys at all (not even as NaN) --
    # load_stats must not KeyError, and must resolve all three to None.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_legacy.npz")
    try:
        np.savez(
            str(path),
            class_names=np.array(stats.class_names),
            pca_mean=stats.pca_mean,
            pca_components=stats.pca_components,
            centroids=stats.centroids,
            covariance_inv=stats.covariance_inv,
            cosine_calibration_mean=stats.cosine_calibration_mean,
            cosine_calibration_std=stats.cosine_calibration_std,
            knn_train_embeddings=stats.knn_train_embeddings,
            knn_train_labels=np.array(stats.knn_train_labels),
        )
        loaded = load_stats(path)
        assert loaded.mahalanobis_p_threshold is None
        assert loaded.cosine_threshold is None
        assert loaded.knn_distance_threshold is None
    finally:
        path.unlink(missing_ok=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_ood.py -v -k "roundtrip_includes_thresholds or roundtrip_thresholds_default or legacy_file_without_threshold"`
Expected: FAIL — `ClassEmbeddingStats` has no `mahalanobis_p_threshold` field yet (`model_copy` with an unknown key raises a Pydantic validation error), and `save_stats`/`load_stats` don't write/read the new keys.

- [ ] **Step 3: Add the fields to `ClassEmbeddingStats`**

In `src/schema.py`, inside `ClassEmbeddingStats` (after `knn_train_labels`):

```python
    knn_train_labels: list[int]  # length n_train_docs, parallel to knn_train_embeddings
    # Per-model calibrated thresholds -- written by `evaluate-ood-calibration --write-thresholds`
    # (src/cli/ood_calibration.py), read via resolve_ood_thresholds() (src/ood.py). None means
    # "not yet calibrated for this specific model" -- resolve_ood_thresholds() falls back to
    # Settings.OOD_* in that case. Fixes thresholds calibrated for one model (e.g. BETO v2)
    # being silently applied to a different model's differently-scaled embedding space.
    mahalanobis_p_threshold: float | None = None
    cosine_threshold: float | None = None
    knn_distance_threshold: float | None = None
```

- [ ] **Step 4: Update `save_stats`/`load_stats` in `src/ood.py`**

Replace the existing `save_stats` and `load_stats` functions with:

```python
def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
    # npz has no native "missing key" for a single scalar the way a dict does, and no None --
    # NaN is the serialization sentinel for "not yet calibrated," round-tripped back to None
    # by load_stats's _optional_threshold.
    np.savez(
        str(path),
        class_names=np.array(stats.class_names),
        pca_mean=stats.pca_mean,
        pca_components=stats.pca_components,
        centroids=stats.centroids,
        covariance_inv=stats.covariance_inv,
        cosine_calibration_mean=stats.cosine_calibration_mean,
        cosine_calibration_std=stats.cosine_calibration_std,
        knn_train_embeddings=stats.knn_train_embeddings,
        knn_train_labels=np.array(stats.knn_train_labels),
        mahalanobis_p_threshold=np.nan if stats.mahalanobis_p_threshold is None else stats.mahalanobis_p_threshold,
        cosine_threshold=np.nan if stats.cosine_threshold is None else stats.cosine_threshold,
        knn_distance_threshold=np.nan if stats.knn_distance_threshold is None else stats.knn_distance_threshold,
    )


def _optional_threshold(data: npt.NDArray[np.float64]) -> float | None:
    value = float(data)
    return None if np.isnan(value) else value


def load_stats(path: Path) -> ClassEmbeddingStats:
    data = np.load(str(path), allow_pickle=False)
    return ClassEmbeddingStats(
        class_names=data["class_names"].tolist(),
        pca_mean=data["pca_mean"],
        pca_components=data["pca_components"],
        centroids=data["centroids"],
        covariance_inv=data["covariance_inv"],
        cosine_calibration_mean=float(data["cosine_calibration_mean"]),
        cosine_calibration_std=float(data["cosine_calibration_std"]),
        knn_train_embeddings=data["knn_train_embeddings"],
        knn_train_labels=data["knn_train_labels"].tolist(),
        # "in data.files" -- not data.get() (npz's NpzFile has no .get) -- lets a
        # pre-this-change ood_stats.npz (missing these keys entirely, not just NaN) still
        # load instead of KeyError-ing every predict/serve call until it's regenerated.
        mahalanobis_p_threshold=_optional_threshold(data["mahalanobis_p_threshold"])
        if "mahalanobis_p_threshold" in data.files
        else None,
        cosine_threshold=_optional_threshold(data["cosine_threshold"])
        if "cosine_threshold" in data.files
        else None,
        knn_distance_threshold=_optional_threshold(data["knn_distance_threshold"])
        if "knn_distance_threshold" in data.files
        else None,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ood.py -v`
Expected: PASS, all tests including the 3 new ones.

- [ ] **Step 6: Run full check and commit**

```bash
uv run poe check
git add src/schema.py src/ood.py tests/test_ood.py
git commit -m "feat: add optional per-model OOD threshold fields to ClassEmbeddingStats"
```

---

### Task 2: `resolve_ood_thresholds()` + `is_out_of_distribution()` takes explicit thresholds

**Files:**
- Modify: `src/ood.py` (new `OodThresholds`, `resolve_ood_thresholds`)
- Modify: `src/inference/classify.py` (`is_out_of_distribution` signature, `predict_text`)
- Test: `tests/test_ood.py`, `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `ClassEmbeddingStats` fields from Task 1.
- Produces: `OodThresholds` (NamedTuple: `mahalanobis_p: float`, `cosine_z: float`, `knn_distance: float`), `resolve_ood_thresholds(stats: ClassEmbeddingStats) -> OodThresholds` in `src/ood.py`. `is_out_of_distribution(scores: OodScores, thresholds: OodThresholds) -> bool` (was `is_out_of_distribution(scores: OodScores) -> bool`) — every caller/test must pass `thresholds` explicitly now.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py`:

```python
def test_resolve_ood_thresholds_falls_back_to_settings_when_stats_thresholds_none() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    thresholds = resolve_ood_thresholds(stats)
    assert thresholds.mahalanobis_p == Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert thresholds.cosine_z == Settings.OOD_COSINE_THRESHOLD
    assert thresholds.knn_distance == Settings.OOD_KNN_DISTANCE_THRESHOLD


def test_resolve_ood_thresholds_uses_stats_values_when_present() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={
            "mahalanobis_p_threshold": 0.002,
            "cosine_threshold": 5.0,
            "knn_distance_threshold": 10.0,
        }
    )
    thresholds = resolve_ood_thresholds(stats)
    assert thresholds.mahalanobis_p == pytest.approx(0.002)
    assert thresholds.cosine_z == pytest.approx(5.0)
    assert thresholds.knn_distance == pytest.approx(10.0)
```

Add `resolve_ood_thresholds` and `Settings` to the `from src.ood import (...)` block and add `from src.settings import Settings` if not already imported in `tests/test_ood.py`.

Update the 4 existing tests in `tests/inference/test_pipeline.py` that call `is_out_of_distribution` directly:

```python
def test_is_out_of_distribution_false_when_all_signals_pass() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0)
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is False


def test_is_out_of_distribution_true_when_mahalanobis_fires() -> None:
    scores = OodScores(mahalanobis_p=0.0001, cosine_z=0.0, knn_distance=1.0)
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_true_when_cosine_fires() -> None:
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=Settings.OOD_COSINE_THRESHOLD + 1, knn_distance=1.0
    )
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_true_when_knn_fires() -> None:
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=0.0, knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD + 1
    )
    thresholds = OodThresholds(
        mahalanobis_p=Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=Settings.OOD_COSINE_THRESHOLD,
        knn_distance=Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
    assert is_out_of_distribution(scores, thresholds) is True
```

Add `OodThresholds` to the `from src.inference.classify import (...)` block in `tests/inference/test_pipeline.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_ood.py tests/inference/test_pipeline.py -v -k "resolve_ood_thresholds or is_out_of_distribution"`
Expected: FAIL — `resolve_ood_thresholds`/`OodThresholds` don't exist yet; `is_out_of_distribution` doesn't accept a second argument yet.

- [ ] **Step 3: Add `OodThresholds` + `resolve_ood_thresholds` to `src/ood.py`**

Add `from src.settings import Settings` to `src/ood.py`'s imports (top of file, alongside `from src.schema import ClassEmbeddingStats`). Then add, near `knn_mean_distance`:

```python
class OodThresholds(NamedTuple):
    """Resolved OOD decision thresholds for one specific model -- either the values
    evaluate-ood-calibration wrote into that model's own ood_stats.npz (via
    --write-thresholds), or Settings.OOD_* as a fallback for stats files that predate
    per-model calibration. is_out_of_distribution() must never read Settings.OOD_* directly
    again -- always go through resolve_ood_thresholds(), or a freshly trained model silently
    inherits whichever model's thresholds happen to be in Settings."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float


def resolve_ood_thresholds(stats: ClassEmbeddingStats) -> OodThresholds:
    """Falls back to Settings.OOD_* per-field, only for whichever threshold
    evaluate-ood-calibration hasn't written yet (None) -- a stats file with all three set
    never touches Settings at all."""
    return OodThresholds(
        mahalanobis_p=stats.mahalanobis_p_threshold
        if stats.mahalanobis_p_threshold is not None
        else Settings.OOD_MAHALANOBIS_P_THRESHOLD,
        cosine_z=stats.cosine_threshold
        if stats.cosine_threshold is not None
        else Settings.OOD_COSINE_THRESHOLD,
        knn_distance=stats.knn_distance_threshold
        if stats.knn_distance_threshold is not None
        else Settings.OOD_KNN_DISTANCE_THRESHOLD,
    )
```

- [ ] **Step 4: Update `is_out_of_distribution()` and `predict_text()` in `src/inference/classify.py`**

Add `resolve_ood_thresholds` and `OodThresholds` to the `from src.ood import (...)` block.

Replace `is_out_of_distribution`:

```python
def is_out_of_distribution(scores: OodScores, thresholds: OodThresholds) -> bool:
    """Any one of the three OOD signals firing is enough -- a deliberate OR, not a
    weighted blend (see README's "OOD scoring internals" for why). NaN in knn_distance
    means the predicted class had zero training points to compare against; treated as
    anomalous, fail-safe, since `nan > threshold` would otherwise silently pass.
    `thresholds` comes from resolve_ood_thresholds(stats) -- per-model calibrated values
    when available, Settings.OOD_* fallback otherwise. Never reads Settings directly here,
    or a model's decisions silently use whichever thresholds happen to be configured for a
    completely different model.
    """
    maha_anomalous = scores.mahalanobis_p < thresholds.mahalanobis_p
    cosine_anomalous = scores.cosine_z > thresholds.cosine_z
    knn_anomalous = (
        bool(np.isnan(scores.knn_distance)) or scores.knn_distance > thresholds.knn_distance
    )
    log.debug(
        "OOD signals: mahalanobis_p=%.6f (threshold=%.6f, anomalous=%s), "
        "cosine_z=%.4f (threshold=%.4f, anomalous=%s), "
        "knn_distance=%.4f (threshold=%.4f, anomalous=%s)",
        scores.mahalanobis_p,
        thresholds.mahalanobis_p,
        maha_anomalous,
        scores.cosine_z,
        thresholds.cosine_z,
        cosine_anomalous,
        scores.knn_distance,
        thresholds.knn_distance,
        knn_anomalous,
    )
    return maha_anomalous or cosine_anomalous or knn_anomalous
```

In `predict_text`, replace:

```python
        maha_p_theoretical = mahalanobis_chi2_p_value_from_distance(
            squared_distance, self._ood_stats
        )
        in_distribution = not is_out_of_distribution(scores)
```

with:

```python
        maha_p_theoretical = mahalanobis_chi2_p_value_from_distance(
            squared_distance, self._ood_stats
        )
        thresholds = resolve_ood_thresholds(self._ood_stats)
        in_distribution = not is_out_of_distribution(scores, thresholds)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ood.py tests/inference/test_pipeline.py -v`
Expected: PASS, all tests.

- [ ] **Step 6: Run full check and commit**

```bash
uv run poe check
git add src/ood.py src/inference/classify.py tests/test_ood.py tests/inference/test_pipeline.py
git commit -m "feat: resolve OOD thresholds per-model instead of reading Settings globally"
```

---

### Task 3: Validate `ood_stats.npz` class mapping at classifier construction

**Files:**
- Modify: `src/inference/classify.py` (`BertTunningClassifier.__init__`)
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `self.model.config.id2label` (already available after `__init__`'s model load), `self._ood_stats.class_names` (Task 1's existing field, unchanged shape).
- Produces: `BertTunningClassifier._validate_ood_stats_class_mapping()` — raises `BertTunningError` (from `src.exceptions`) on mismatch, called once during `__init__`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_pipeline.py`:

```python
def test_classifier_raises_when_ood_stats_class_names_mismatch_model_id2label(
    tmp_path: Path,
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats()  # class_names=["decreto", "ordenanza"] -- swap the order below
    mismatched_stats = stats.model_copy(update={"class_names": ["ordenanza", "decreto"]})
    save_stats(mismatched_stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(BertTunningError, match="do not match"),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)


def test_classifier_loads_fine_when_ood_stats_class_names_match(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    save_stats(_make_stats(), tmp_path / "ood_stats.npz")  # class_names already match order

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is not None  # noqa: SLF001


def test_classifier_skips_validation_when_no_ood_stats(tmp_path: Path) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    with patch("torch.cuda.is_available", return_value=False):
        clf = BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)
    assert clf._ood_stats is None  # noqa: SLF001
```

Add `BertTunningError` (from `src.exceptions`) and `save_stats` (from `src.ood`) to `tests/inference/test_pipeline.py`'s imports if not already present.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/inference/test_pipeline.py -v -k "class_names_mismatch or class_names_match or skips_validation"`
Expected: FAIL (or the mismatch test passes for the wrong reason) — no validation exists yet, so the mismatched-class construction currently succeeds without raising.

- [ ] **Step 3: Add the validation to `BertTunningClassifier`**

Add `from src.exceptions import BertTunningError` to `src/inference/classify.py`'s imports.

In `__init__`, replace:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

with:

```python
        self._ood_stats = self._load_ood_stats(model_path)
        self._validate_ood_stats_class_mapping()
        log.info("Classifier ready on %s (max_length=%d)", self.device, self.max_length)
```

Add, right after `_load_ood_stats`:

```python
    def _validate_ood_stats_class_mapping(self) -> None:
        """ood_stats.npz's class_names must match this model's id2label -- by count AND by
        ordered index, since knn_mean_distance() indexes stats.knn_train_labels directly by
        the model's own predicted label id (see predict_text). A silently mismatched or
        stale ood_stats.npz would score every prediction's k-NN signal against the wrong
        class's neighbors with no error. Fails fast here, once, at classifier construction
        (server startup or CLI invocation) -- not per-request, so a bad artifact can't reach
        production traffic at all rather than corrupting scores silently."""
        if self._ood_stats is None:
            return
        id2label: dict[int, str] = self.model.config.id2label
        expected = [id2label[i] for i in range(len(id2label))]
        if self._ood_stats.class_names != expected:
            msg = (
                f"ood_stats.npz class_names {self._ood_stats.class_names} do not match "
                f"this model's id2label {expected} (order matters, not just the set) -- "
                "OOD scoring would silently score against the wrong classes. Regenerate "
                "ood_stats.npz for this exact model with compute-ood-stats."
            )
            raise BertTunningError(msg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/inference/test_pipeline.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run full check and commit**

```bash
uv run poe check
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "fix: validate ood_stats.npz class mapping against the loaded model at construction"
```

---

### Task 4: k-NN calibration uses the model's own prediction, not the true label

**Files:**
- Modify: `src/ood.py` (new `extract_embeddings_and_predictions`)
- Modify: `src/cli/_ood_common.py` (new `embed_texts_and_predict`)
- Modify: `src/cli/ood_calibration.py` (`_run_ood_calibration`)
- Test: `tests/test_ood.py`, `tests/cli/test_ood_calibration.py`

**Interfaces:**
- Consumes: `LoadedModel` (existing, `src/ood.py`).
- Produces: `extract_embeddings_and_predictions(loaded, texts, *, max_length, batch_size=16) -> tuple[npt.NDArray[np.float64], list[int]]` (`src/ood.py`); `embed_texts_and_predict(loaded, df, *, chunk_strategy, max_tokens) -> tuple[npt.NDArray[np.float64], list[int]]` (`src/cli/_ood_common.py`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py`:

```python
def test_extract_embeddings_and_predictions_returns_matching_lengths() -> None:
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {
        "input_ids": torch.zeros(2, 8, dtype=torch.long),
        "attention_mask": torch.ones(2, 8, dtype=torch.long),
    }
    model = MagicMock()
    model.return_value.hidden_states = [torch.zeros(2, 8, 4)]
    model.return_value.logits = torch.tensor([[2.0, 0.5], [0.1, 3.0]])

    loaded = LoadedModel(model=model, tokenizer=tokenizer, device="cpu")
    embeddings, predicted_ids = extract_embeddings_and_predictions(
        loaded, ["doc one", "doc two"], max_length=8
    )
    assert embeddings.shape == (2, 4)
    assert predicted_ids == [0, 1]  # argmax of each row above
```

Add `extract_embeddings_and_predictions` to `tests/test_ood.py`'s `from src.ood import (...)` block, and `from unittest.mock import MagicMock` / `import torch` if not already present (both are already used elsewhere in this file per existing tests).

Add to `tests/cli/test_ood_calibration.py`, a new test proving predicted (not true) labels drive the k-NN calibration call, and update the shared fake used by `_run_successful_calibration`/`_run_calibration_with_stats`:

```python
def _fake_extract_embeddings_and_predictions(
    _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
) -> tuple[npt.NDArray[np.float64], list[int]]:
    return np.zeros((len(texts), 8), dtype=np.float64), [0] * len(texts)
```

Replace every `patch("src.cli._ood_common.extract_embeddings", side_effect=_fake_extract_embeddings)` in this file (in `_run_successful_calibration` and `_run_calibration_with_stats`) with:

```python
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_fake_extract_embeddings_and_predictions,
        ),
```

Then add:

```python
def test_evaluate_ood_calibration_cmd_uses_predicted_label_for_knn_not_true_label(
    tmp_path: Path,
) -> None:
    # 20 "decreto" (label 0) + 20 "ordenanza" (label 1) docs, but every document's forward
    # pass predicts label 1 regardless of its true label -- if the command still scores
    # k-NN using the true label, the "decreto" test docs would be scored against class 0's
    # neighbors (5 points); if it correctly uses the predicted label, every doc scores
    # against class 1's neighbors instead. knn_mean_distance is spied on to assert the
    # label id it actually received.
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": [f"decreto {i}" for i in range(20)] + [f"ordenanza {i}" for i in range(20)],
            "label": ["decreto"] * 20 + ["ordenanza"] * 20,
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    (model_path / "ood_stats.npz").touch()

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    def _always_predicts_ordenanza(
        _loaded: LoadedModel, texts: list[str], **_kwargs: int | str
    ) -> tuple[npt.NDArray[np.float64], list[int]]:
        return np.zeros((len(texts), 8), dtype=np.float64), [1] * len(texts)

    from src.ood import knn_mean_distance as real_knn_mean_distance

    seen_label_ids: list[int] = []

    def _spy_knn_mean_distance(
        embedding: npt.NDArray[np.float64],
        stats: ClassEmbeddingStats,
        predicted_label_id: int,
        *,
        k: int,
    ) -> float:
        seen_label_ids.append(predicted_label_id)
        return real_knn_mean_distance(embedding, stats, predicted_label_id, k=k)

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=_make_stats()),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_always_predicts_ordenanza,
        ),
        patch("src.cli.ood_calibration.knn_mean_distance", side_effect=_spy_knn_mean_distance),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results"),
    ):
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
            ["--model-path", str(model_path), "--model", "beto", "--cache-path", str(cache_path)],
        )

    assert result.exit_code == 0
    # Every seen label id must be 1 (the mocked prediction) -- never 0, even for the
    # "decreto" (true label 0) test documents.
    assert seen_label_ids
    assert set(seen_label_ids) == {1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_ood.py tests/cli/test_ood_calibration.py -v -k "extract_embeddings_and_predictions or predicted_label_for_knn"`
Expected: FAIL — `extract_embeddings_and_predictions`/`embed_texts_and_predict` don't exist; `_run_ood_calibration` still uses true `label_id`.

- [ ] **Step 3: Add `extract_embeddings_and_predictions` to `src/ood.py`**

Add, right after `extract_embeddings`:

```python
def extract_embeddings_and_predictions(
    loaded: LoadedModel,
    texts: list[str],
    *,
    max_length: int,
    batch_size: int = 16,
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Like extract_embeddings, but also returns each document's predicted label id from
    the same forward pass -- for evaluate-ood-calibration, which must score the k-NN signal
    against the model's actual prediction (mirroring predict_text exactly), not the
    document's true label. extract_embeddings alone can't do this: it calls
    loaded.model.base_model(...) to skip the classification head entirely, since its other
    callers (compute-ood-stats, training) only ever need embeddings, never predictions."""
    loaded.model.eval()
    embedding_batches: list[npt.NDArray[np.float64]] = []
    predicted_ids: list[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = loaded.tokenizer(
                batch,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            ).to(loaded.device)
            outputs = loaded.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            embedding_batches.append(hidden[:, 0, :].cpu().numpy().astype(np.float64))
            predicted_ids.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
    return np.vstack(embedding_batches), predicted_ids
```

- [ ] **Step 4: Add `embed_texts_and_predict` to `src/cli/_ood_common.py`**

Add `extract_embeddings_and_predictions` to the `from src.ood import (...)` block. Add, right after `embed_texts`:

```python
def embed_texts_and_predict(
    loaded: LoadedModel, df: pd.DataFrame, *, chunk_strategy: str, max_tokens: int
) -> tuple[npt.NDArray[np.float64], list[int]]:
    """Sibling to embed_texts() for callers that also need each document's predicted label
    -- currently only evaluate-ood-calibration, for reproducing predict_text()'s exact k-NN
    scoring input (the model's own prediction, not the document's true label)."""
    texts = [prepare_text(t, loaded.tokenizer, chunk_strategy) for t in df["text"]]
    return extract_embeddings_and_predictions(loaded, texts, max_length=max_tokens)
```

- [ ] **Step 5: Update `_run_ood_calibration` in `src/cli/ood_calibration.py`**

Change the import line:

```python
from src.cli._ood_common import embed_texts, reconstruct_split_and_load_model
```

to:

```python
from src.cli._ood_common import embed_texts_and_predict, reconstruct_split_and_load_model
```

Replace:

```python
    embeddings = embed_texts(
        split.loaded,
        split.test_df,
        chunk_strategy=opts.chunk_strategy,
        max_tokens=model_cfg.max_tokens,
    )

    p_values = np.array(
        [mahalanobis_empirical_p_value(e, stats, train_distances) for e in embeddings]
    )
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    label_ids = split.test_df["label_id"].to_numpy()
    knn_distances = np.array(
        [
            knn_mean_distance(e, stats, int(lbl), k=Settings.OOD_KNN_NEIGHBORS)
            for e, lbl in zip(embeddings, label_ids, strict=True)
        ]
    )
```

with:

```python
    embeddings, predicted_ids = embed_texts_and_predict(
        split.loaded,
        split.test_df,
        chunk_strategy=opts.chunk_strategy,
        max_tokens=model_cfg.max_tokens,
    )

    p_values = np.array(
        [mahalanobis_empirical_p_value(e, stats, train_distances) for e in embeddings]
    )
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    # Predicted label, not the document's true label -- predict_text() in production always
    # scores knn_mean_distance against the model's own prediction, so calibration must
    # reproduce exactly that, including the k-NN penalty a misclassified in-distribution
    # document actually gets in production when it's scored against the wrong class's
    # neighbors. Using the true label here would understate the real false-positive rate.
    knn_distances = np.array(
        [
            knn_mean_distance(e, stats, pred_id, k=Settings.OOD_KNN_NEIGHBORS)
            for e, pred_id in zip(embeddings, predicted_ids, strict=True)
        ]
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_ood.py tests/cli/test_ood_calibration.py -v`
Expected: PASS, all tests.

- [ ] **Step 7: Run full check and commit**

```bash
uv run poe check
git add src/ood.py src/cli/_ood_common.py src/cli/ood_calibration.py tests/test_ood.py tests/cli/test_ood_calibration.py
git commit -m "fix: calibrate k-NN OOD signal against the model's predicted label, not the true label"
```

---

### Task 5: `--write-thresholds` persists calibrated thresholds into `ood_stats.npz`

**Files:**
- Modify: `src/cli/ood_calibration.py` (`OodCalibrationOptions`, CLI option, `_run_ood_calibration`, new `_write_calibrated_thresholds`)
- Test: `tests/cli/test_ood_calibration.py`

**Interfaces:**
- Consumes: `CalibrationReport` (existing, `src/schema.py`), `save_stats`/`ClassEmbeddingStats` (Task 1).
- Produces: `evaluate-ood-calibration --write-thresholds` flag.

- [ ] **Step 1: Write the failing tests**

Add `save_stats` and `load_stats` to `tests/cli/test_ood_calibration.py`'s `from src.ood import ...` block (currently only `LoadedModel` is imported from there — this file will need `from src.ood import LoadedModel, load_stats, save_stats` after this task and Task 4 combined).

Add to `tests/cli/test_ood_calibration.py`:

```python
def test_evaluate_ood_calibration_cmd_write_thresholds_persists_to_stats_file(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    result, _ = _run_successful_calibration(tmp_path, extra_args=["--write-thresholds"])
    assert result.exit_code == 0
    written = load_stats(stats_path)
    assert written.cosine_threshold is not None
    assert written.knn_distance_threshold is not None


def test_evaluate_ood_calibration_cmd_without_flag_does_not_write(tmp_path: Path) -> None:
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    save_stats(_make_stats(), stats_path)
    before = stats_path.read_bytes()
    result, _ = _run_successful_calibration(tmp_path, extra_args=[])
    assert result.exit_code == 0
    assert stats_path.read_bytes() == before


def test_evaluate_ood_calibration_cmd_write_thresholds_refuses_degenerate_maha_threshold(
    tmp_path: Path,
) -> None:
    # Only 4 reference points -- 1/(4+1) = 0.2, comfortably above any target-FP-rate
    # percentile this test's p_values could produce, so the suggested Mahalanobis threshold
    # is guaranteed to be at/below the floor. cosine/knn thresholds should still get written.
    tiny_stats = _make_stats().model_copy(
        update={
            "knn_train_embeddings": np.array([[0.0] * 8] * 2 + [[5.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    assert "Refusing to write" in result.output
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_p_threshold is None  # unchanged -- tiny_stats had no prior value
    assert written.cosine_threshold is not None
    assert written.knn_distance_threshold is not None
```

Add a `--write-thresholds`-enabled variant of `_run_calibration_with_stats` right after the existing one:

```python
def _run_calibration_with_stats_write_thresholds(
    tmp_path: Path, stats: ClassEmbeddingStats
) -> tuple[Result, MagicMock]:
    cache_path = tmp_path / "cache.parquet"
    pd.DataFrame(
        {
            "text": [f"decreto {i}" for i in range(20)] + [f"ordenanza {i}" for i in range(20)],
            "label": ["decreto"] * 20 + ["ordenanza"] * 20,
        }
    ).to_parquet(cache_path)

    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    save_stats(stats, model_path / "ood_stats.npz")

    mock_model = MagicMock()
    mock_model.config.id2label = {0: "decreto", 1: "ordenanza"}

    with (
        patch("src.cli._ood_common.AutoTokenizer.from_pretrained"),
        patch(
            "src.cli._ood_common.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.cli.ood_calibration.load_stats", return_value=stats),
        patch(
            "src.cli._ood_common.extract_embeddings_and_predictions",
            side_effect=_fake_extract_embeddings_and_predictions,
        ),
        patch("torch.cuda.is_available", return_value=False),
        patch("src.cli.ood_calibration.log_ood_calibration_results"),
    ):
        result = CliRunner().invoke(
            evaluate_ood_calibration_cmd,
            [
                "--model-path",
                str(model_path),
                "--model",
                "beto",
                "--cache-path",
                str(cache_path),
                "--write-thresholds",
            ],
        )
    return result, MagicMock()
```

The two non-degenerate tests above use `_run_successful_calibration`, which currently creates its `ood_stats.npz` with a bare `.touch()` (an empty file — fine when `load_stats` is mocked and nothing ever reads the file for real, but `--write-thresholds` calls the real `save_stats()`, which needs somewhere to write). In `_run_successful_calibration`, replace:

```python
    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    (model_path / "ood_stats.npz").touch()
```

with:

```python
    model_path = tmp_path / "fake-model"
    model_path.mkdir()
    save_stats(_make_stats(), model_path / "ood_stats.npz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/cli/test_ood_calibration.py -v -k "write_thresholds"`
Expected: FAIL — no `--write-thresholds` option exists yet.

- [ ] **Step 3: Add the flag and persistence logic to `src/cli/ood_calibration.py`**

Add `save_stats` to the `from src.ood import (...)` block.

Add to `OodCalibrationOptions`:

```python
    write_thresholds: bool = False
```

Add the CLI option (after `--target-fp-rate`, before `--log-wandb`):

```python
@click.option(
    "--write-thresholds",
    is_flag=True,
    default=False,
    help=(
        "Persist the suggested thresholds into this model's ood_stats.npz, so "
        "predict/predict-folder/serve use them instead of falling back to Settings.OOD_*"
    ),
)
```

Add, after `build_calibration_report`:

```python
def _write_calibrated_thresholds(
    stats: ClassEmbeddingStats,
    stats_path: Path,
    report: CalibrationReport,
    n_train: int,
) -> None:
    """Writes evaluate-ood-calibration's suggested thresholds back into this model's own
    ood_stats.npz, so resolve_ood_thresholds() uses per-model calibrated values instead of
    falling back to Settings.OOD_* -- the fix for thresholds calibrated against one model
    being silently applied to every other model. Refuses to write a Mahalanobis threshold at
    or below the empirical p-value's own resolution floor (1/(n_train+1)) -- that threshold
    would be mathematically unreachable (the signal could never fire), the exact bug this
    project hit once already with an unchecked suggested value."""
    floor = 1 / (n_train + 1)
    maha_threshold = report.suggested_maha_threshold
    if maha_threshold <= floor:
        log.warning(
            "Refusing to write suggested Mahalanobis threshold %.6f: at or below this "
            "model's empirical resolution floor %.6f (n_train=%d). The signal would never "
            "fire. Keeping the existing value (%s).",
            maha_threshold,
            floor,
            n_train,
            stats.mahalanobis_p_threshold,
        )
        maha_threshold = stats.mahalanobis_p_threshold

    updated = stats.model_copy(
        update={
            "mahalanobis_p_threshold": maha_threshold,
            "cosine_threshold": report.suggested_cosine_threshold,
            "knn_distance_threshold": report.suggested_knn_threshold,
        }
    )
    save_stats(updated, stats_path)
    log.info(
        "Wrote calibrated thresholds to %s: mahalanobis_p=%s, cosine=%.4f, knn_distance=%.4f",
        stats_path,
        maha_threshold,
        report.suggested_cosine_threshold,
        report.suggested_knn_threshold,
    )
```

In `_run_ood_calibration`, after the `report = build_calibration_report(...)` line, add:

```python
    if opts.write_thresholds:
        _write_calibrated_thresholds(stats, stats_path, report, len(train_distances))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/cli/test_ood_calibration.py -v`
Expected: PASS, all tests.

- [ ] **Step 5: Run full check and commit**

```bash
uv run poe check
git add src/cli/ood_calibration.py tests/cli/test_ood_calibration.py
git commit -m "feat: add --write-thresholds to persist calibrated thresholds per-model"
```

---

### Task 6: Bounded `/predict` upload read

**Files:**
- Modify: `src/settings.py` (`MAX_UPLOAD_SIZE_BYTES`)
- Modify: `src/api/routes/predict/endpoints.py`
- Test: `tests/api/test_predict.py`

**Interfaces:**
- Produces: `Settings.MAX_UPLOAD_SIZE_BYTES: int` (default `25 * 1024 * 1024`); `_read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes` in `endpoints.py`, raising `HTTPException(413)` when exceeded.

- [ ] **Step 1: Write the failing test**

This file has no shared fixtures — every test builds its own `app`/`TestClient`/inline PDF bytes inline (see e.g. `test_predict_rejects_non_pdf`). Add, matching that exact pattern:

```python
def test_predict_endpoint_rejects_upload_exceeding_max_size() -> None:
    app = create_app(model_path="fake/path")
    app.state.clf = MagicMock()  # satisfy dependency; classifier is irrelevant to this assertion
    with patch("src.api.routes.predict.endpoints.Settings.MAX_UPLOAD_SIZE_BYTES", 10):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )
    assert response.status_code == HTTPStatus.REQUEST_ENTITY_TOO_LARGE
```

`b"%PDF-1.4 fake content"` is 22 bytes, comfortably over the monkeypatched 10-byte cap, so this triggers the check on the first chunk read regardless of `_UPLOAD_CHUNK_SIZE`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/api/test_predict.py -v -k "exceeding_max_size"`
Expected: FAIL — no size check exists yet, so the request currently succeeds (200) instead of 413.

- [ ] **Step 3: Add `MAX_UPLOAD_SIZE_BYTES` to `src/settings.py`**

Add, near `API_PORT`/`HOST` in the "Server" section:

```python
    # 25 MB -- municipal decree/ordinance PDFs in this corpus are small (a few hundred KB
    # typical); this is generous headroom, not a tight budget. Enforced by
    # src/api/routes/predict/endpoints.py's chunked read, not by FastAPI/uvicorn, which
    # impose no default body-size limit on their own.
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024
```

- [ ] **Step 4: Add the bounded read to `src/api/routes/predict/endpoints.py`**

Add `from src.settings import Settings` to the imports.

Add, above `predict()`:

```python
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MB


async def _read_upload_bounded(file: UploadFile, max_bytes: int) -> bytes:
    """Reads file in bounded chunks instead of one unconditional await file.read() -- an
    unbounded read loads the entire upload into worker memory before any size check ever
    runs, so a sufficiently large upload can exhaust memory regardless of what happens
    afterward. FastAPI/uvicorn impose no default request-body limit on their own."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK_SIZE)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413, detail=f"File exceeds the {max_bytes} byte upload limit"
            )
        chunks.append(chunk)
    return b"".join(chunks)
```

Replace:

```python
    contents = await file.read()
```

with:

```python
    contents = await _read_upload_bounded(file, Settings.MAX_UPLOAD_SIZE_BYTES)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run python -m pytest tests/api/test_predict.py -v`
Expected: PASS, all tests (including pre-existing ones — small uploads still round-trip fine through the chunked reader).

- [ ] **Step 6: Run full check and commit**

```bash
uv run poe check
git add src/settings.py src/api/routes/predict/endpoints.py tests/api/test_predict.py
git commit -m "fix: bound /predict upload reads instead of loading unbounded content into memory"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `README.md`**

In the "OOD scoring internals" section (the "Why the thresholds are calibrated, not theory-derived" subsection added previously), add a new paragraph after the existing "Current calibrated values" table:

```markdown
**Thresholds are per-model, not global.** Each model's `ood_stats.npz` can carry its own calibrated `mahalanobis_p_threshold`/`cosine_threshold`/`knn_distance_threshold` (written by `evaluate-ood-calibration --write-thresholds`). `resolve_ood_thresholds()` (`src/ood.py`) reads these from whichever `ood_stats.npz` the loaded classifier has, falling back to `Settings.OOD_*` only for artifacts that haven't been calibrated yet. This means BETO v2's calibrated values never leak into a different model's decisions — training XLM-RoBERTa or MiniLM and running `evaluate-ood-calibration --write-thresholds` against that specific checkpoint gives it its own thresholds, scoped to its own embedding space and corpus size.
```

Under "Out-of-distribution detection", add:

```markdown
`BertTunningClassifier` validates that a loaded `ood_stats.npz`'s `class_names` match the model's own `id2label` (by order, not just the set) at construction time — a mismatch raises `BertTunningError` immediately (server startup or CLI invocation), rather than silently scoring every prediction's k-NN signal against the wrong class.

`/predict` bounds the upload read to `MAX_UPLOAD_SIZE_BYTES` (default 25 MB), read in 1 MB chunks — an unbounded `await file.read()` would otherwise load an arbitrarily large upload into worker memory before any check ran.
```

- [ ] **Step 2: Update `CLAUDE.md`**

Add to the Settings table:

```markdown
| `MAX_UPLOAD_SIZE_BYTES` | `26214400` (25 MB) | `/predict` rejects uploads larger than this with a 413, read in bounded chunks |
```

Add a new paragraph to "Key Technical Decisions", after the existing Mahalanobis-related paragraphs:

```markdown
**Per-model OOD thresholds, not a single global config (2026-07-12)**
`Settings.OOD_MAHALANOBIS_P_THRESHOLD`/`OOD_COSINE_THRESHOLD`/`OOD_KNN_DISTANCE_THRESHOLD` were being applied identically regardless of which model was loaded — but they were calibrated specifically against BETO v2's embedding space, corpus size, and empirical rank floor (see the Mahalanobis resolution-floor paragraph above). A different model (a freshly trained XLM-RoBERTa or MiniLM, or a BETO v2 retrained on a different corpus) would receive apparently-valid but statistically unrelated OOD decisions. `ClassEmbeddingStats` (`src/schema.py`) now carries optional `mahalanobis_p_threshold`/`cosine_threshold`/`knn_distance_threshold` fields, `None` by default. `resolve_ood_thresholds()` (`src/ood.py`) reads them from whichever `ood_stats.npz` is loaded, falling back to `Settings.OOD_*` only when they're unset — backward compatible with artifacts that predate this change. `is_out_of_distribution()` now takes an explicit `OodThresholds` instead of reading `Settings` directly. `evaluate-ood-calibration --write-thresholds` persists its suggested values back into that exact model's `ood_stats.npz`, refusing to write a Mahalanobis threshold at or below that model's own empirical resolution floor (the same degenerate-suggestion class of bug this project hit once already) — cosine/k-NN still get written even when Mahalanobis is refused.

**k-NN calibration scores against the model's predicted label, not the true label (2026-07-12)**
`evaluate-ood-calibration` was passing each held-out test document's *true* label into `knn_mean_distance()`, while `predict_text()` in production always passes the model's *predicted* label (`pred_idx = argmax(probs)`). For a misclassified in-distribution document, these differ — and misclassified documents are exactly the ones likely to have large k-NN distances against the wrong class's neighbors, so the true-label shortcut made the reported empirical false-positive rate optimistic relative to what production actually experiences. `extract_embeddings_and_predictions()` (`src/ood.py`) now runs the model's real forward pass (mirroring `predict_text`'s own `output_hidden_states=True` call, not the `base_model`-only path `extract_embeddings` uses for training/`compute-ood-stats`, which never need predictions) and returns each document's `argmax` prediction alongside its embedding. `embed_texts_and_predict()` (`src/cli/_ood_common.py`) wraps it for `evaluate-ood-calibration`'s use; Mahalanobis and cosine calibration are unaffected, since neither takes a predicted label as input.

**`ood_stats.npz` class mapping validated at classifier construction (2026-07-12)**
`BertTunningClassifier` previously loaded whatever `ood_stats.npz` sat next to a model's checkpoint with zero validation that it actually belonged to that model — a copied, stale, or reordered file would silently score embeddings against the wrong centroids, and `knn_mean_distance()` would interpret a predicted label id against the wrong class entirely (it indexes `stats.knn_train_labels` directly by that id). `_validate_ood_stats_class_mapping()` now compares `stats.class_names` (ordered) against `model.config.id2label` (ordered by index) once, at classifier construction — raising `BertTunningError` immediately rather than corrupting every subsequent prediction's OOD scores silently. The existing class-mismatch check in `src/cli/_ood_common.py` (used by `compute-ood-stats`/`evaluate-ood-calibration`) only ever covered the CLI backfill/calibration path, never the ordinary `predict`/`predict-folder`/`serve` path that most traffic actually goes through.
```

- [ ] **Step 3: Commit**

```bash
uv run poe check
git add README.md CLAUDE.md
git commit -m "docs: document per-model OOD thresholds, predicted-label k-NN calibration, class-mapping validation, upload limit"
```

---

### Task 8: Regenerate committed `ood_stats.npz` artifacts with per-model thresholds (operational)

Not a code task — run against whichever models have `ood_stats.npz` committed to the repo (BETO v2 at minimum; BETO v1 if still in active use):

```powershell
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model_beto_v2/final --model beto --cache-path ./data/bert_tunning_cache_con_otro_300.parquet --write-thresholds
```

- [ ] **Step 1: Run `evaluate-ood-calibration --write-thresholds` for BETO v2**

Confirm the log output shows written threshold values (not a "Refusing to write" warning for all three — cosine/knn should always write; check whether Mahalanobis's suggested value clears BETO v2's own resolution floor before trusting it, same math as the prior recalibration in this session).

- [ ] **Step 2: Spot-check `predict`/`predict-folder` still behaves sanely**

```powershell
uv run python main.py predict-folder .\samples --model-path ./models/bert_tunning_model_beto_v2/final
```

Confirm `mahalanobis_p_value`/`cosine_z`/`knn_distance`/`in_distribution` are populated and the flagged/unflagged mix looks consistent with before this change (per-model thresholds should resolve to the same values `Settings.OOD_*` already held for BETO v2, since BETO v2 is what those defaults were calibrated against — this step is a regression check, not expected to change behavior for BETO v2 specifically).

- [ ] **Step 3: Commit the regenerated artifact**

```bash
git add models/bert_tunning_model_beto_v2/final/ood_stats.npz
git commit -m "chore: write per-model calibrated OOD thresholds into BETO v2's ood_stats.npz"
```

Report is complete — this task is done manually; there is no automated test for "the committed artifact now carries thresholds" beyond the spot-check above (Task 5's unit tests already cover the write mechanism itself).
