# TF-IDF Cosine-Centroid OOD Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth, independent OOD signal — cosine distance to per-class TF-IDF centroids — that catches lexical/vocabulary divergence (e.g. a document naming a different municipality) that the three existing embedding-based signals (Mahalanobis, cosine-on-BERT-embedding, k-NN) cannot separate, since they all operate on the `[CLS]` embedding's semantic "shape" rather than surface vocabulary.

**Architecture:** Mirrors the existing Mahalanobis/cosine/k-NN pattern exactly, in TF-IDF space instead of PCA-reduced BERT-embedding space: fit a `TfidfVectorizer` + per-class centroids at training time (`compute_class_stats`), persist the fitted vectorizer's vocabulary/idf/centroids as plain NumPy arrays inside the existing `ood_stats.npz` (no new artifact file, no new dependency — `TfidfVectorizer`'s `vocabulary_`/`idf_` round-trip losslessly through two arrays, verified working with `allow_pickle=False`), score new documents at inference time with the same z-score-against-training-distribution technique `cosine_z_score` already uses, and calibrate the threshold with the exact same percentile technique `build_calibration_report` already uses. The signal becomes a fourth OR-branch in `is_out_of_distribution` — never blended with the other three, matching this project's existing "OR, not a weighted blend" design decision.

**Tech Stack:** scikit-learn's `TfidfVectorizer` (already a transitive dependency — `sklearn.decomposition.PCA` and `sklearn.metrics.pairwise.cosine_distances` are already imported in `src/ood.py`), no new packages.

## Global Constraints

- **Backward compatibility is mandatory.** BETO v1's and BETO v2's committed `ood_stats.npz` files predate this feature and must keep working without regeneration. Every new field on `ClassEmbeddingStats` defaults to its own "absent" sentinel (empty collection for `tfidf_vocabulary_terms`/`tfidf_idf`/`tfidf_centroids`, `None` for the three scalar calibration/threshold fields — see the `stop-using-none` bullet below for why these differ), the same backward-compatible-default spirit `model_type`/`model_hidden_size` already established for the identity-fingerprint feature. When the TF-IDF signal is absent, it is **disabled** for that model (skipped in `is_out_of_distribution`, not treated as "always anomalous" — that fail-safe-anomalous behavior is reserved for `knn_distance`'s NaN case, which represents a specific document's predicted class having zero training points, a different situation from a whole model never having computed this signal at all).
- **No new artifact file.** `TfidfVectorizer.vocabulary_` (a `dict[str, int]`) and `.idf_` (a float array) must round-trip through `save_stats`/`load_stats`'s existing `allow_pickle=False` NumPy-array-only pattern — store the vocabulary as an ordered array of terms (`vectorizer.get_feature_names_out()`, index = feature id) plus the parallel `idf_` array, and reconstruct a fixed-vocabulary `TfidfVectorizer` from those two arrays at load time. This was verified working end-to-end before this plan was written (fit → extract terms + idf → reconstruct → transform produces bit-identical output to the original fitted vectorizer).
- **Never blend into one score.** `is_out_of_distribution` becomes a 4-way OR. Do not average, weight, or otherwise combine `tfidf_cosine_z` with the other three signals.
- **Reuse the existing percentile calibration technique exactly.** HIGH `tfidf_cosine_z` = anomalous (same direction as `cosine_z`/`knn_distance`), so the calibrated threshold is the `(1 - target_fp_rate)`-th percentile of in-distribution scores — do not invent a new calibration method.
- **`OodThresholds` and `OodScores` are both `NamedTuple`s with existing call sites across the test suite** (11 and 5 respectively, none of which pass a TF-IDF value). Give the new field a default (`Settings.OOD_TFIDF_COSINE_THRESHOLD` for `OodThresholds.tfidf_cosine_z`; `float("nan")` for `OodScores.tfidf_cosine_z` — see the `stop-using-none` bullet below) so none of those existing call sites need to change.
- **`CalibrationReport` gets two new required-shape fields** (`fp_rate_tfidf`, `suggested_tfidf_threshold`) with a `0.0` default for the same non-breaking reason — one existing test (`tests/test_wandb.py`) constructs it directly.
- Use `clean_text` (`from src.ingestion.extract import clean_text`, already imported this way in `src/inference/classify.py`) on text before fitting/transforming with the TF-IDF vectorizer, for consistency with what `predict_text` already does before BERT tokenization.
- Cache the reconstructed `TfidfVectorizer` for the classifier's process lifetime (mirror `BertTunningClassifier._train_mahalanobis_distances`'s `@cached_property` pattern) — do not rebuild it on every `predict_text` call.
- **Avoid `None` where an existing, more specific pattern already fits** (per `stop-using-none`): `tfidf_vocabulary_terms`/`tfidf_idf`/`tfidf_centroids` are collections — a real fitted vectorizer always has ≥1 term, so an **empty collection** (`[]` / a zero-length array) is an unambiguous, self-describing "not fitted yet" sentinel, exactly the "Nothing here" row in the skill's Quick Reference table. This also removes a whole NaN-sentinel translation layer from `save_stats`/`load_stats` for these three fields specifically (an empty array written to `.npz` and an empty array read back are already the same value — no None↔NaN round-trip needed).
  `tfidf_cosine_calibration_mean`/`.std` are **plain floats with placeholder defaults** (`0.0`/`1.0`), not `Optional` — despite superficially looking like the same "scalar, no safe zero" case `mahalanobis_p_threshold` et al. are. The test that actually distinguishes them: `mahalanobis_p_threshold`/`cosine_threshold`/`knn_distance_threshold`/`tfidf_threshold` are each set **independently**, later, by a separate calibration step (`evaluate-ood-calibration --write-thresholds`) that may or may not have run for any one of them — a caller (`resolve_ood_thresholds`, `_warn_on_uncalibrated_thresholds`) genuinely branches per-field on that. `tfidf_cosine_calibration_mean`/`.std`, by contrast, are **always** produced together with `tfidf_vocabulary_terms`/`tfidf_idf`/`tfidf_centroids` in the same `compute_tfidf_stats` call — never independently set, never independently checked. No caller reads `tfidf_cosine_calibration_mean` without having already confirmed `tfidf_vocabulary_terms` is non-empty (`build_tfidf_vectorizer`'s emptiness check gates every path that would use it). A `None` here would just be a second, redundant way to express a condition `tfidf_vocabulary_terms` already expresses — the "reason nobody branches on" case the skill says isn't worth modeling. `std` defaults to `1.0`, not `0.0`, purely as a defensive non-zero divisor if ever misused ungated; it is never actually read that way.
  `tfidf_threshold` stays `float | None`, matching `mahalanobis_p_threshold`/`cosine_threshold`/`knn_distance_threshold`'s existing, already-shipped precedent exactly, for the identical independently-calibrated reason.
- **`OodScores.tfidf_cosine_z` uses the NaN sentinel already established by its sibling `knn_distance` field in the same `NamedTuple`**, not `float | None` — keeps `OodScores` a uniform 4-tuple of plain floats with zero `Optional` fields, matching its own existing style. This is a *deliberately opposite* NaN convention from `knn_distance`'s: `knn_distance`'s NaN means "this specific document's predicted class had zero training points" and fails **closed** (treated as anomalous, since the model can't judge it at all). `tfidf_cosine_z`'s NaN means "this whole model's `ood_stats.npz` predates the TF-IDF signal" and fails **open** (skipped, not anomalous) — the same "whole model doesn't have this capability" situation `_ood_stats is None` already handles by skipping OOD scoring entirely for the other three signals. Two NaNs, two different reasons, two different handling rules — call out both explicitly in code comments so a future reader doesn't assume they share one convention. `PredictResult.tfidf_cosine_z` (the external/API-facing field) stays `float | None = None`, matching its three already-shipped siblings (`mahalanobis_p_value`, `cosine_z`, `knn_distance`) exactly — that's a genuine system/API boundary, the one place the skill says `None` still fits, and `predict_text` is where the internal NaN sentinel gets translated back to external `None`, mirroring how `save_stats`/`load_stats` already translate NaN↔`None` at the `.npz` storage boundary.

---

### Task 1: Settings + schema fields (foundation, no behavior yet)

**Files:**
- Modify: `src/settings.py`
- Modify: `src/schema.py`
- Test: `tests/test_settings_ood.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: `Settings.OOD_TFIDF_COSINE_THRESHOLD: float`, `Settings.OOD_TFIDF_MAX_FEATURES: int`; `ClassEmbeddingStats.tfidf_vocabulary_terms: list[str]` (empty = not fitted), `.tfidf_idf: Float64Array` (empty = not fitted), `.tfidf_centroids: Float64Array` (empty = not fitted), `.tfidf_cosine_calibration_mean: float` (default `0.0`, empty when not fitted), `.tfidf_cosine_calibration_std: float` (default `1.0`), `.tfidf_threshold: float | None`; `CalibrationReport.fp_rate_tfidf: float`, `.suggested_tfidf_threshold: float`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_settings_ood.py`:

```python
def test_ood_tfidf_settings_have_defaults() -> None:
    assert Settings.OOD_TFIDF_COSINE_THRESHOLD > 0
    assert Settings.OOD_TFIDF_MAX_FEATURES > 0
```

Add to `tests/test_schema.py`:

```python
def test_class_embedding_stats_tfidf_fields_default_to_absent() -> None:
    stats = ClassEmbeddingStats(
        class_names=["a", "b"],
        pca_mean=np.zeros(4),
        pca_components=np.eye(4),
        centroids=np.zeros((2, 4)),
        covariance_inv=np.eye(4),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((2, 4)),
        knn_train_labels=[0, 1],
    )
    assert stats.tfidf_vocabulary_terms == []
    assert len(stats.tfidf_idf) == 0
    assert stats.tfidf_centroids.size == 0
    assert stats.tfidf_cosine_calibration_mean == 0.0
    assert stats.tfidf_cosine_calibration_std == 1.0
    assert stats.tfidf_threshold is None


def test_calibration_report_tfidf_fields_default_to_zero() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.0,
        fp_rate_cosine=0.0,
        fp_rate_knn=0.0,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=0.0,
        suggested_knn_threshold=0.0,
    )
    assert report.fp_rate_tfidf == 0.0
    assert report.suggested_tfidf_threshold == 0.0
```

(Add `import numpy as np` and `from src.schema import CalibrationReport, ClassEmbeddingStats` to `tests/test_schema.py` if not already present — check the file's existing imports first, since `ClassEmbeddingStats` may already be imported there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_settings_ood.py tests/test_schema.py -v -k "tfidf"`
Expected: FAIL — `AttributeError` (Settings has no `OOD_TFIDF_COSINE_THRESHOLD`) and Pydantic validation errors (unexpected field access on `ClassEmbeddingStats`/`CalibrationReport`).

- [ ] **Step 3: Add the Settings fields**

In `src/settings.py`, immediately after the `OOD_KNN_DISTANCE_THRESHOLD`/`TARGET_FP_RATE` block:

```python
    # Uncalibrated placeholder, matching how OOD_COSINE_THRESHOLD/OOD_KNN_DISTANCE_THRESHOLD
    # started before their first evaluate-ood-calibration --write-thresholds run. Run that
    # command for any model using this signal before trusting it in production.
    OOD_TFIDF_COSINE_THRESHOLD: float = 2.5
    OOD_TFIDF_MAX_FEATURES: int = 5000
```

- [ ] **Step 4: Add the ClassEmbeddingStats fields**

In `src/schema.py`, add the `Field` import to the existing `from pydantic import ...` line:

```python
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
```

Then, inside `ClassEmbeddingStats`, immediately after the `mahalanobis_threshold_status` field:

```python
    # TF-IDF cosine-centroid signal (added 2026-07-14) -- a fourth OOD signal, independent
    # of the three above, operating on raw lexical vocabulary instead of BERT-embedding
    # space. Catches divergence the embedding-based signals structurally cannot (e.g. a
    # document naming a different municipality but sharing the same document-type shape).
    # An empty vocabulary/idf/centroids together means "this ood_stats.npz predates this
    # feature" -- a real fitted vectorizer always has >=1 term, so empty is an unambiguous
    # "not fitted" sentinel (no None needed: see stop-using-none's "Nothing here" case).
    # The signal is skipped entirely in is_out_of_distribution, not treated as anomalous.
    tfidf_vocabulary_terms: list[str] = []  # ordered; index = TF-IDF feature id
    tfidf_idf: Float64Array = Field(default_factory=lambda: np.zeros(0))
    tfidf_centroids: Float64Array = Field(default_factory=lambda: np.zeros((0, 0)))
    # These two stay Optional -- they're scalars with no safe zero-sentinel (a genuine
    # calibration mean of 0.0 must stay distinguishable from "never computed"), matching
    # the existing precedent mahalanobis_p_threshold/cosine_threshold/knn_distance_threshold
    # already use for the identical reason.
    tfidf_cosine_calibration_mean: float = 0.0
    tfidf_cosine_calibration_std: float = 1.0  # never 0.0 -- avoids a divide-by-zero if
    # ever read ungated, though every real caller already gates on tfidf_vocabulary_terms
    # Per-model calibrated threshold, same role as cosine_threshold/knn_distance_threshold --
    # no degenerate-guard status field needed, since that guard only ever applies to Mahalanobis.
    tfidf_threshold: float | None = None
```

- [ ] **Step 5: Add the CalibrationReport fields**

In `src/schema.py`, inside `CalibrationReport`:

```python
    fp_rate_tfidf: float = 0.0
    suggested_tfidf_threshold: float = 0.0
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_settings_ood.py tests/test_schema.py -v -k "tfidf"`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/settings.py src/schema.py tests/test_settings_ood.py tests/test_schema.py
git commit -m "feat: add TF-IDF OOD signal fields to Settings/ClassEmbeddingStats/CalibrationReport"
```

---

### Task 2: Pure TF-IDF stats/scoring functions in `src/ood.py`

**Files:**
- Modify: `src/ood.py`
- Test: `tests/test_ood.py`

**Interfaces:**
- Consumes: `ClassEmbeddingStats` (Task 1), `_cosine_min_distance_raw` (existing, `src/ood.py:41`).
- Produces: `compute_tfidf_stats(texts: list[str], labels: list[int], class_names: list[str], *, max_features: int) -> _TfidfStats` (internal `NamedTuple` with fields `vocabulary_terms: list[str]`, `idf: Float64Array`, `centroids: Float64Array`, `cosine_calibration_mean: float`, `cosine_calibration_std: float`); `build_tfidf_vectorizer(stats: ClassEmbeddingStats) -> TfidfVectorizer | None`; `tfidf_cosine_z_score(text: str, stats: ClassEmbeddingStats, vectorizer: TfidfVectorizer) -> float`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py`:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

from src.ood import build_tfidf_vectorizer, compute_tfidf_stats, tfidf_cosine_z_score


def _synthetic_texts() -> tuple[list[str], list[int], list[str]]:
    # Two lexically distinct clusters -- "decreto rosario" vocabulary vs. "ordenanza cordoba"
    # vocabulary -- so a same-class query with different vocabulary lands far from its
    # class's TF-IDF centroid, the exact failure mode this signal targets.
    decreto_docs = ["decreto rosario municipal intendente"] * 10
    ordenanza_docs = ["ordenanza cordoba concejo deliberante"] * 10
    texts = decreto_docs + ordenanza_docs
    labels = [0] * 10 + [1] * 10
    return texts, labels, ["decreto", "ordenanza"]


def test_compute_tfidf_stats_shapes() -> None:
    texts, labels, class_names = _synthetic_texts()
    stats = compute_tfidf_stats(texts, labels, class_names, max_features=50)
    n_terms = len(stats.vocabulary_terms)
    assert stats.idf.shape == (n_terms,)
    assert stats.centroids.shape == (2, n_terms)
    assert stats.cosine_calibration_std > 0


def test_build_tfidf_vectorizer_reconstructs_fitted_transform() -> None:
    texts, labels, class_names = _synthetic_texts()
    stats_partial = compute_tfidf_stats(texts, labels, class_names, max_features=50)
    stats = ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=np.zeros(1),
        pca_components=np.eye(1),
        centroids=np.zeros((2, 1)),
        covariance_inv=np.eye(1),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((2, 1)),
        knn_train_labels=[0, 1],
        tfidf_vocabulary_terms=stats_partial.vocabulary_terms,
        tfidf_idf=stats_partial.idf,
        tfidf_centroids=stats_partial.centroids,
        tfidf_cosine_calibration_mean=stats_partial.cosine_calibration_mean,
        tfidf_cosine_calibration_std=stats_partial.cosine_calibration_std,
    )
    vectorizer = build_tfidf_vectorizer(stats)
    assert vectorizer is not None

    reference = TfidfVectorizer(max_features=50)
    reference.fit(texts)
    query = "decreto rosario municipal intendente"
    np.testing.assert_allclose(
        vectorizer.transform([query]).toarray(),
        reference.transform([query]).toarray(),
    )


def test_build_tfidf_vectorizer_returns_none_when_stats_predate_feature() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    assert build_tfidf_vectorizer(stats) is None


def test_tfidf_cosine_z_score_higher_for_lexically_divergent_same_class_query() -> None:
    texts, labels, class_names = _synthetic_texts()
    stats_partial = compute_tfidf_stats(texts, labels, class_names, max_features=50)
    stats = ClassEmbeddingStats(
        class_names=class_names,
        pca_mean=np.zeros(1),
        pca_components=np.eye(1),
        centroids=np.zeros((2, 1)),
        covariance_inv=np.eye(1),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((2, 1)),
        knn_train_labels=[0, 1],
        tfidf_vocabulary_terms=stats_partial.vocabulary_terms,
        tfidf_idf=stats_partial.idf,
        tfidf_centroids=stats_partial.centroids,
        tfidf_cosine_calibration_mean=stats_partial.cosine_calibration_mean,
        tfidf_cosine_calibration_std=stats_partial.cosine_calibration_std,
    )
    vectorizer = build_tfidf_vectorizer(stats)
    assert vectorizer is not None

    matching_z = tfidf_cosine_z_score("decreto rosario municipal intendente", stats, vectorizer)
    # Same words as the OTHER class's training vocabulary -- lexically divergent from
    # whichever centroid it's nearest to, so its z-score should be higher (more anomalous)
    # than a query using words seen during training.
    divergent_z = tfidf_cosine_z_score("otro texto completamente distinto aqui", stats, vectorizer)
    assert divergent_z > matching_z
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ood.py -v -k "tfidf"`
Expected: FAIL — `ImportError` (`compute_tfidf_stats`/`build_tfidf_vectorizer`/`tfidf_cosine_z_score` don't exist yet).

- [ ] **Step 3: Implement**

In `src/ood.py`, add the import and the three functions. Add to the existing `sklearn` import block:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

Add near the top, after `_PcaReduction`:

```python
class _TfidfStats(NamedTuple):
    """Internal return type for compute_tfidf_stats -- merged into ClassEmbeddingStats by
    the caller (compute_class_stats), same convention as _PcaReduction."""

    vocabulary_terms: list[str]
    idf: npt.NDArray[np.float64]
    centroids: npt.NDArray[np.float64]
    cosine_calibration_mean: float
    cosine_calibration_std: float
```

Add after `compute_class_stats` (after line 86 in the current file):

```python
def compute_tfidf_stats(
    texts: list[str], labels: list[int], class_names: list[str], *, max_features: int = 5000
) -> _TfidfStats:
    """Fits a TF-IDF vectorizer + per-class centroids on raw training text -- a signal
    independent of compute_class_stats' BERT-embedding space, operating on surface
    vocabulary instead. Catches lexical divergence (e.g. a different municipality's name)
    that a shared document-type "shape" in embedding space cannot separate."""
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(cleaned).toarray()
    labels_arr = np.asarray(labels)

    centroids = np.stack([X[labels_arr == k].mean(axis=0) for k in range(len(class_names))])
    cosine_scores = np.array(
        [_cosine_min_distance_raw(X[i], centroids) for i in range(X.shape[0])]
    )

    return _TfidfStats(
        vocabulary_terms=vectorizer.get_feature_names_out().tolist(),
        idf=vectorizer.idf_,
        centroids=centroids,
        cosine_calibration_mean=float(cosine_scores.mean()),
        cosine_calibration_std=float(cosine_scores.std() + 1e-9),
    )


def build_tfidf_vectorizer(stats: ClassEmbeddingStats) -> TfidfVectorizer | None:
    """Reconstructs a fixed-vocabulary TfidfVectorizer from the two arrays load_stats/
    save_stats round-trip through ood_stats.npz -- verified to produce bit-identical
    .transform() output to the originally-fitted vectorizer. Returns None when this
    model's ood_stats.npz predates the TF-IDF signal (tfidf_vocabulary_terms is empty),
    so callers can treat the signal as disabled rather than crash on missing data."""
    if not stats.tfidf_vocabulary_terms:  # empty list = not fitted, see Task 1's field comment
        return None
    vocabulary = {term: i for i, term in enumerate(stats.tfidf_vocabulary_terms)}
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    vectorizer.idf_ = stats.tfidf_idf
    return vectorizer


def tfidf_cosine_z_score(
    text: str, stats: ClassEmbeddingStats, vectorizer: TfidfVectorizer
) -> float:
    """Cosine distance to the nearest TF-IDF centroid, z-scored against the training set --
    same technique as cosine_z_score, different vector space. Caller must have already
    confirmed build_tfidf_vectorizer(stats) is not None."""
    point = vectorizer.transform([clean_text(text)]).toarray()[0]
    cosine_raw = _cosine_min_distance_raw(point, stats.tfidf_centroids)
    return (cosine_raw - stats.tfidf_cosine_calibration_mean) / stats.tfidf_cosine_calibration_std
```

Add the `clean_text` import to `src/ood.py`'s import block:

```python
from src.ingestion.extract import clean_text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ood.py -v -k "tfidf"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ood.py tests/test_ood.py
git commit -m "feat: add TF-IDF cosine-centroid stats/scoring functions"
```

---

### Task 3: Persist TF-IDF fields through `save_stats`/`load_stats`

**Files:**
- Modify: `src/ood.py`
- Test: `tests/test_ood.py`

**Interfaces:**
- Consumes: `ClassEmbeddingStats.tfidf_*` fields (Task 1).
- Produces: `save_stats`/`load_stats` round-trip all six new fields; loading a legacy `.npz` (missing the new keys) still works, with the three collection fields defaulting to empty and the three scalar fields defaulting to `None`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py`:

```python
def test_save_and_load_stats_roundtrip_includes_tfidf_fields(tmp_path: Path) -> None:
    texts, labels, class_names = _synthetic_texts()
    embeddings = np.random.default_rng(0).normal(size=(20, 16))
    tfidf = compute_tfidf_stats(texts, labels, class_names, max_features=50)
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={
            "tfidf_vocabulary_terms": tfidf.vocabulary_terms,
            "tfidf_idf": tfidf.idf,
            "tfidf_centroids": tfidf.centroids,
            "tfidf_cosine_calibration_mean": tfidf.cosine_calibration_mean,
            "tfidf_cosine_calibration_std": tfidf.cosine_calibration_std,
            "tfidf_threshold": 2.5,
        }
    )
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)

    assert loaded.tfidf_vocabulary_terms == stats.tfidf_vocabulary_terms
    np.testing.assert_allclose(loaded.tfidf_idf, stats.tfidf_idf)
    np.testing.assert_allclose(loaded.tfidf_centroids, stats.tfidf_centroids)
    assert loaded.tfidf_cosine_calibration_mean == pytest.approx(
        stats.tfidf_cosine_calibration_mean
    )
    assert loaded.tfidf_cosine_calibration_std == pytest.approx(stats.tfidf_cosine_calibration_std)
    assert loaded.tfidf_threshold == pytest.approx(2.5)


def test_load_stats_handles_legacy_file_without_tfidf_fields(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)  # tfidf_* fields are all at their empty/None defaults --
    # this IS the legacy-shape file (compute_class_stats here predates Task 4's texts= param)

    loaded = load_stats(path)

    assert loaded.tfidf_vocabulary_terms == []
    assert len(loaded.tfidf_idf) == 0
    assert loaded.tfidf_centroids.size == 0
    assert loaded.tfidf_cosine_calibration_mean == 0.0
    assert loaded.tfidf_cosine_calibration_std == 1.0
    assert loaded.tfidf_threshold is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ood.py -v -k "tfidf_fields"`
Expected: `test_save_and_load_stats_roundtrip_includes_tfidf_fields` FAILs (round-tripped values are the empty/default values, not the fitted ones) — `test_load_stats_handles_legacy_file_without_tfidf_fields` currently passes vacuously (fields are already always at their defaults) but must stay passing after Step 3.

- [ ] **Step 3: Implement persistence**

In `src/ood.py`'s `save_stats`, five of the six new fields need NO None-translation at all -- `stats.tfidf_vocabulary_terms`/`.tfidf_idf`/`.tfidf_centroids`/`.tfidf_cosine_calibration_mean`/`.tfidf_cosine_calibration_std` are all plain (possibly-empty/placeholder-default) values per Task 1's field defaults, so they pass straight into `np.savez` unchanged. Only `tfidf_threshold` still needs the existing NaN-sentinel translation, since `.npz` has no native `None`:

```python
    tfidf_threshold = np.nan if stats.tfidf_threshold is None else stats.tfidf_threshold
```

Add all six to the `np.savez(...)` call (alongside the existing `mahalanobis_threshold_status=...` line) -- note five pass the `ClassEmbeddingStats` fields through directly, no local variable needed:

```python
                tfidf_vocabulary_terms=np.array(stats.tfidf_vocabulary_terms),
                tfidf_idf=stats.tfidf_idf,
                tfidf_centroids=stats.tfidf_centroids,
                tfidf_cosine_calibration_mean=stats.tfidf_cosine_calibration_mean,
                tfidf_cosine_calibration_std=stats.tfidf_cosine_calibration_std,
                tfidf_threshold=tfidf_threshold,
```

In `load_stats`, add the six fields to the `ClassEmbeddingStats(...)` construction. Five of them only need the legacy-file guard (`"key" in data.files` -- a pre-this-feature `.npz` doesn't have the key at all), with no None-vs-value branching, since a value written by `save_stats` reads back as the same value directly:

```python
        tfidf_vocabulary_terms=(
            data["tfidf_vocabulary_terms"].tolist() if "tfidf_vocabulary_terms" in data.files else []
        ),
        tfidf_idf=(data["tfidf_idf"] if "tfidf_idf" in data.files else np.zeros(0)),
        tfidf_centroids=(
            data["tfidf_centroids"] if "tfidf_centroids" in data.files else np.zeros((0, 0))
        ),
        tfidf_cosine_calibration_mean=(
            float(data["tfidf_cosine_calibration_mean"])
            if "tfidf_cosine_calibration_mean" in data.files
            else 0.0
        ),
        tfidf_cosine_calibration_std=(
            float(data["tfidf_cosine_calibration_std"])
            if "tfidf_cosine_calibration_std" in data.files
            else 1.0
        ),
        tfidf_threshold=(
            _optional_threshold(data["tfidf_threshold"])
            if "tfidf_threshold" in data.files
            else None
        ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ood.py -v -k "tfidf"`
Expected: PASS (all Task 2 + Task 3 tfidf tests)

- [ ] **Step 5: Verify BETO v1/v2's committed `ood_stats.npz` still load**

Run:
```bash
uv run python -c "
from src.ood import load_stats
from pathlib import Path
for p in ['models/bert_tunning_model_beto/final/ood_stats.npz', 'models/bert_tunning_model_beto_v2/final/ood_stats.npz']:
    s = load_stats(Path(p))
    assert s.tfidf_vocabulary_terms == []
    print(p, 'OK')
"
```
Expected: both print `OK` with no exception.

- [ ] **Step 6: Run the full test suite**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ood.py tests/test_ood.py
git commit -m "feat: persist TF-IDF stats through save_stats/load_stats, backward compatible"
```

---

### Task 4: Wire TF-IDF fitting into `compute_class_stats` and both training call sites

**Files:**
- Modify: `src/ood.py`
- Modify: `src/training/pipeline.py`
- Modify: `src/cli/ood_stats.py`
- Test: `tests/test_ood.py`

**Interfaces:**
- Consumes: `compute_tfidf_stats` (Task 2).
- Produces: `compute_class_stats(embeddings, labels, class_names, *, texts: list[str], n_components=64, covariance_epsilon=1e-6, model_type=None, model_hidden_size=None, max_tfidf_features=5000) -> ClassEmbeddingStats` — now populates the six `tfidf_*` fields.

**Global constraint reminder:** `texts` must be the SAME documents, in the SAME order, as `embeddings`/`labels` — both training call sites already have `train_df` in hand with a `"text"` column aligned to `train_df["label_id"]`, so this is a matter of passing `train_df["text"].tolist()` alongside the existing arguments, not re-deriving alignment.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ood.py`:

```python
def test_compute_class_stats_populates_tfidf_fields() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    texts = ["decreto rosario"] * 20 + ["ordenanza cordoba"] * 20
    stats = compute_class_stats(
        embeddings, labels, class_names, n_components=8, texts=texts
    )
    assert stats.tfidf_vocabulary_terms != []
    assert len(stats.tfidf_centroids) == len(class_names)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ood.py -v -k "populates_tfidf_fields"`
Expected: FAIL — `TypeError: compute_class_stats() got an unexpected keyword argument 'texts'`

- [ ] **Step 3: Implement**

In `src/ood.py`, change `compute_class_stats`'s signature to add `texts` and `max_tfidf_features`, and merge in the TF-IDF fields:

```python
def compute_class_stats(  # noqa: PLR0913 -- model_type/model_hidden_size/texts/
    # max_tfidf_features are optional trailing kwargs threaded through from the two call
    # sites (training/pipeline.py, cli/ood_stats.py); bundling them into a NamedTuple for
    # rarely-varying trailing kwargs would be more ceremony than the limit is worth here.
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    texts: list[str],
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
    model_type: str | None = None,
    model_hidden_size: int | None = None,
    max_tfidf_features: int = 5000,
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
    tfidf = compute_tfidf_stats(texts, labels, class_names, max_features=max_tfidf_features)

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
        tfidf_vocabulary_terms=tfidf.vocabulary_terms,
        tfidf_idf=tfidf.idf,
        tfidf_centroids=tfidf.centroids,
        tfidf_cosine_calibration_mean=tfidf.cosine_calibration_mean,
        tfidf_cosine_calibration_std=tfidf.cosine_calibration_std,
    )
```

**Update every other existing call site in `tests/test_ood.py`** that calls `compute_class_stats(embeddings, labels, class_names, ...)` without `texts=` — add `texts=["placeholder text"] * len(labels)` (or reuse `_synthetic_texts()`'s texts where a test already has them) to each. Search first: `grep -n "compute_class_stats(" tests/test_ood.py` to find every call site before editing, since `texts` is now a required keyword-only argument and every existing call breaks otherwise.

In `src/training/pipeline.py`, update the `compute_class_stats(...)` call (around line 161):

```python
    ood_stats = compute_class_stats(
        train_embeddings,
        train_df["label_id"].tolist(),
        list(le.classes_),
        texts=train_df["text"].tolist(),
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=model.config.model_type,
        model_hidden_size=model.config.hidden_size,
        max_tfidf_features=Settings.OOD_TFIDF_MAX_FEATURES,
    )
```

In `src/cli/ood_stats.py`, update the `compute_class_stats(...)` call (around line 51):

```python
    stats = compute_class_stats(
        embeddings,
        split.train_df["label_id"].tolist(),
        split.classes,
        texts=split.train_df["text"].tolist(),
        n_components=Settings.OOD_PCA_COMPONENTS,
        model_type=split.loaded.model.config.model_type,  # type: ignore[union-attr,arg-type]
        model_hidden_size=split.loaded.model.config.hidden_size,  # type: ignore[union-attr,arg-type]
        max_tfidf_features=Settings.OOD_TFIDF_MAX_FEATURES,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ood.py -v`
Expected: PASS (every test in the file, including the ones just updated with `texts=`)

- [ ] **Step 5: Run the full test suite**

Run: `uv run poe check`
Expected: PASS — this will surface any other `compute_class_stats` call site in `tests/training/` or elsewhere that also needs `texts=` added; fix each the same way.

- [ ] **Step 6: Commit**

```bash
git add src/ood.py src/training/pipeline.py src/cli/ood_stats.py tests/test_ood.py
git commit -m "feat: fit TF-IDF stats inside compute_class_stats, wire into both training call sites"
```

---

### Task 5: Wire the fourth signal into `BertTunningClassifier`

**Files:**
- Modify: `src/inference/classify.py`
- Modify: `src/ood.py` (only `resolve_ood_thresholds`/`OodThresholds`)
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `build_tfidf_vectorizer`, `tfidf_cosine_z_score` (Task 2), `ClassEmbeddingStats.tfidf_*` (Task 1/3).
- Produces: `OodThresholds.tfidf_cosine_z: float`; `OodScores.tfidf_cosine_z: float` (NaN sentinel — no `Optional`, see the Global Constraints `stop-using-none` bullet); `is_out_of_distribution` becomes a 4-way OR; `PredictResult.tfidf_cosine_z: float | None` (the one place this stays `Optional` — external API boundary, matching its three siblings); `BertTunningClassifier._tfidf_vectorizer` cached property (`TfidfVectorizer | None` — single-reason absence at a data-loading boundary, the legitimate `None` case per the same skill).

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_pipeline.py` (mirroring the existing `test_is_out_of_distribution_*` tests found via `grep -n "def test_is_out_of_distribution" tests/inference/test_pipeline.py` — read the surrounding ~10 lines of one such test first to match its exact `_make_stats_with_no_knn_training_data_for_decreto`-style fixture conventions before writing these):

```python
def test_is_out_of_distribution_false_when_tfidf_signal_absent_and_others_pass() -> None:
    # tfidf_cosine_z=nan (signal not available for this model) must not make the document
    # anomalous by itself -- NaN here means "skip, fail open," the OPPOSITE of
    # knn_distance's NaN, which means "fail closed, treat as anomalous." The two NaNs
    # represent different situations (whole-model signal absence vs. one document's
    # predicted class having zero training points) and are handled with opposite polarity
    # on purpose -- see the Global Constraints note in this plan.
    scores = OodScores(
        mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=float("nan")
    )
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0)
    assert is_out_of_distribution(scores, thresholds) is False


def test_is_out_of_distribution_true_when_tfidf_z_exceeds_threshold() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=10.0)
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0, tfidf_cosine_z=2.5)
    assert is_out_of_distribution(scores, thresholds) is True


def test_is_out_of_distribution_false_when_tfidf_z_below_threshold() -> None:
    scores = OodScores(mahalanobis_p=0.5, cosine_z=0.0, knn_distance=1.0, tfidf_cosine_z=1.0)
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0, tfidf_cosine_z=2.5)
    assert is_out_of_distribution(scores, thresholds) is False
```

Also add, placed after `test_predict_text_with_stats_populates_ood_fields` (same file, same `clf._ood_stats = ...` fixture-assignment convention that test and its neighbors already use):

```python
def test_predict_text_attaches_tfidf_cosine_z_when_available() -> None:
    from src.ood import compute_tfidf_stats

    clf = _make_mock_classifier()
    texts = ["decreto rosario municipal"] * 5 + ["ordenanza cordoba concejo"] * 5
    labels = [0] * 5 + [1] * 5
    tfidf = compute_tfidf_stats(texts, labels, ["decreto", "ordenanza"], max_features=20)
    clf._ood_stats = _make_stats().model_copy(  # noqa: SLF001
        update={
            "tfidf_vocabulary_terms": tfidf.vocabulary_terms,
            "tfidf_idf": tfidf.idf,
            "tfidf_centroids": tfidf.centroids,
            "tfidf_cosine_calibration_mean": tfidf.cosine_calibration_mean,
            "tfidf_cosine_calibration_std": tfidf.cosine_calibration_std,
        }
    )
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("decreto rosario municipal")
    # PredictResult.tfidf_cosine_z stays Optional at this external boundary (matching
    # mahalanobis_p_value/cosine_z/knn_distance) -- predict_text is where the internal
    # NaN sentinel translates back to None, same as save_stats/load_stats already do at
    # the .npz storage boundary for the scalar threshold fields.
    assert result.tfidf_cosine_z is not None


def test_predict_text_leaves_tfidf_cosine_z_none_when_stats_predate_feature() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001 -- tfidf_vocabulary_terms defaults to []
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.tfidf_cosine_z is None
```

- [ ] **Step 2: Run tests to verify the OodScores/OodThresholds ones fail**

Run: `uv run pytest tests/inference/test_pipeline.py -v -k "tfidf"`
Expected: FAIL — `TypeError: OodScores() got an unexpected keyword argument 'tfidf_cosine_z'` (and same for `OodThresholds`).

- [ ] **Step 3: Extend `OodThresholds` in `src/ood.py`**

```python
class OodThresholds(NamedTuple):
    mahalanobis_p: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float = Settings.OOD_TFIDF_COSINE_THRESHOLD
```

Extend `resolve_ood_thresholds`:

```python
def resolve_ood_thresholds(stats: ClassEmbeddingStats) -> OodThresholds:
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
        tfidf_cosine_z=stats.tfidf_threshold
        if stats.tfidf_threshold is not None
        else Settings.OOD_TFIDF_COSINE_THRESHOLD,
    )
```

- [ ] **Step 4: Extend `OodScores`/`is_out_of_distribution`/`PredictResult` in `src/inference/classify.py`**

```python
class OodScores(NamedTuple):
    """The four OOD signals -- always computed together, passed together, never used
    independently. tfidf_cosine_z uses the same NaN-sentinel convention as knn_distance,
    not Optional -- keeps this NamedTuple a uniform tuple of plain floats. The two NaNs
    mean different things and are handled with OPPOSITE polarity in is_out_of_distribution:
    knn_distance's NaN means "this document's predicted class has zero training points"
    and fails CLOSED (anomalous); tfidf_cosine_z's NaN means "this whole model's
    ood_stats.npz predates the TF-IDF signal" and fails OPEN (not anomalous, same as
    _ood_stats being None disables OOD scoring entirely for the other three signals)."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float = float("nan")
```

```python
def is_out_of_distribution(scores: OodScores, thresholds: OodThresholds) -> bool:
    maha_anomalous = scores.mahalanobis_p < thresholds.mahalanobis_p
    cosine_anomalous = scores.cosine_z > thresholds.cosine_z
    knn_anomalous = (
        bool(np.isnan(scores.knn_distance)) or scores.knn_distance > thresholds.knn_distance
    )
    # Opposite of knn_anomalous's NaN handling on purpose -- see OodScores' docstring.
    tfidf_anomalous = (
        not np.isnan(scores.tfidf_cosine_z) and scores.tfidf_cosine_z > thresholds.tfidf_cosine_z
    )
    log.debug(
        "OOD signals: mahalanobis_p=%.6f (threshold=%.6f, anomalous=%s), "
        "cosine_z=%.4f (threshold=%.4f, anomalous=%s), "
        "knn_distance=%.4f (threshold=%.4f, anomalous=%s), "
        "tfidf_cosine_z=%s (threshold=%.4f, anomalous=%s)",
        scores.mahalanobis_p,
        thresholds.mahalanobis_p,
        maha_anomalous,
        scores.cosine_z,
        thresholds.cosine_z,
        cosine_anomalous,
        scores.knn_distance,
        thresholds.knn_distance,
        knn_anomalous,
        scores.tfidf_cosine_z,
        thresholds.tfidf_cosine_z,
        tfidf_anomalous,
    )
    return maha_anomalous or cosine_anomalous or knn_anomalous or tfidf_anomalous
```

Add to `src/schema.py`'s `PredictResult` (alongside `knn_distance`):

```python
    tfidf_cosine_z: float | None = None
```

- [ ] **Step 5: Wire the cached vectorizer + scoring into `BertTunningClassifier`**

In `src/inference/classify.py`, add the import:

```python
from src.ood import (
    build_tfidf_vectorizer,
    tfidf_cosine_z_score,
)
```

Add a cached property, mirroring `_train_mahalanobis_distances`:

```python
    @cached_property
    def _tfidf_vectorizer(self) -> "TfidfVectorizer | None":
        """Reconstructed once per process lifetime, not per predict_text() call -- mirrors
        _train_mahalanobis_distances' caching rationale. None when ood_stats.npz predates
        the TF-IDF signal or there's no ood_stats.npz at all."""
        if self._ood_stats is None:
            return None
        return build_tfidf_vectorizer(self._ood_stats)
```

(Add `from sklearn.feature_extraction.text import TfidfVectorizer` under `TYPE_CHECKING` if mypy strict requires it for the string-quoted return annotation — check how `LoadedModel`/other sklearn-typed members in this file already handle this, and match that convention exactly rather than guessing.)

In `predict_text`, after the existing `scores = OodScores(...)` construction, compute the fourth value and include it. `tfidf_z` uses the NaN sentinel (matching `OodScores.tfidf_cosine_z`'s type) when the vectorizer is absent, not `None`:

```python
        tfidf_z = (
            tfidf_cosine_z_score(text, self._ood_stats, self._tfidf_vectorizer)
            if self._tfidf_vectorizer is not None
            else float("nan")
        )
        squared_distance = mahalanobis_min_distance(cls_embedding, self._ood_stats)
        scores = OodScores(
            mahalanobis_p=empirical_survival_p_value(squared_distance, train_distances),
            cosine_z=cosine_z_score(cls_embedding, self._ood_stats),
            knn_distance=knn_mean_distance(
                cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
            ),
            tfidf_cosine_z=tfidf_z,
        )
```

Add this to the `result.model_copy(update={...})` call's dict — this is where the internal NaN sentinel translates to the external `PredictResult.tfidf_cosine_z: float | None` (the same NaN↔`None` translation `save_stats`/`load_stats` already do at the `.npz` boundary for the scalar threshold fields):

```python
                "tfidf_cosine_z": (
                    None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)
                ),
```

Extend `_warn_on_uncalibrated_thresholds` to also check the TF-IDF threshold, but only when the TF-IDF signal actually exists for this model (don't warn about a signal that isn't in use at all):

```python
        if self._ood_stats.tfidf_centroids.size > 0 and self._ood_stats.tfidf_threshold is None:
            uncalibrated.append("tfidf_threshold")
```

(Add this line into the existing `uncalibrated` list-building block, alongside the `cosine_threshold`/`knn_distance_threshold` checks.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v -k "tfidf"`
Expected: PASS

- [ ] **Step 7: Run the full test suite**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/ood.py src/inference/classify.py src/schema.py tests/inference/test_pipeline.py
git commit -m "feat: wire TF-IDF cosine-centroid as a fourth OOD signal into BertTunningClassifier"
```

---

### Task 6: Extend `evaluate-ood-calibration` to calibrate/report/write the TF-IDF threshold

**Files:**
- Modify: `src/cli/ood_calibration.py`
- Modify: `src/wandb.py`
- Test: `tests/cli/test_ood_calibration.py`
- Test: `tests/test_wandb.py`

**Interfaces:**
- Consumes: `build_tfidf_vectorizer`, `tfidf_cosine_z_score` (Task 2), `OodThresholds.tfidf_cosine_z` (Task 5), `CalibrationReport.fp_rate_tfidf`/`.suggested_tfidf_threshold` (Task 1).
- Produces: `build_calibration_report` gains a required `tfidf_z_scores: npt.NDArray[np.float64]` parameter — no `None`/default: an empty array (`np.array([])`) is the "no TF-IDF data for this run" case, passed explicitly by the one caller rather than defaulted, so there's no mutable-default-argument lint concern (ruff `B006`/`B008`) and no `Optional` on this parameter at all; `_write_calibrated_thresholds` also persists `tfidf_threshold`; `_run_ood_calibration` computes TF-IDF scores for the test split when available (empty array otherwise); `log_ood_calibration_results` logs the TF-IDF FP rate/suggested threshold/current threshold.

- [ ] **Step 1: Write the failing tests**

Add to `tests/cli/test_ood_calibration.py` (place near the existing `test_build_calibration_report_*` tests, matching their exact style):

```python
def test_build_calibration_report_tfidf_percentile_direction() -> None:
    # Same HIGH-value-is-anomalous direction as cosine/knn.
    p_values = np.array([0.1, 0.2, 0.3, 0.4])
    z_scores = np.array([1.0, 2.0, 3.0, 4.0])
    knn_distances = np.array([1.0, 2.0, 3.0, 4.0])
    tfidf_z_scores = np.array([1.0, 2.0, 3.0, 4.0])
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0, tfidf_cosine_z=2.5)

    report = build_calibration_report(
        p_values, z_scores, knn_distances, target_fp_rate=0.25, thresholds=thresholds,
        tfidf_z_scores=tfidf_z_scores,
    )

    assert report.suggested_tfidf_threshold == pytest.approx(np.percentile(tfidf_z_scores, 75))
    assert report.suggested_tfidf_threshold > np.median(tfidf_z_scores)


def test_build_calibration_report_tfidf_fp_rate_zero_when_scores_empty() -> None:
    # When the model has no TF-IDF stats, the caller passes an empty array (not None --
    # see this task's Interfaces note on why tfidf_z_scores has no Optional/default at
    # all) and the TF-IDF FP rate/suggested threshold must be reported as 0.0, not crash
    # and not silently fabricate a number.
    p_values = np.array([0.1, 0.2])
    z_scores = np.array([1.0, 2.0])
    knn_distances = np.array([1.0, 2.0])
    thresholds = OodThresholds(mahalanobis_p=0.01, cosine_z=2.5, knn_distance=5.0)

    report = build_calibration_report(
        p_values, z_scores, knn_distances, target_fp_rate=0.25, thresholds=thresholds,
        tfidf_z_scores=np.array([]),
    )

    assert report.fp_rate_tfidf == 0.0
    assert report.suggested_tfidf_threshold == 0.0
```

Modify the existing `test_log_ood_calibration_results_logs_summary_metrics` in `tests/test_wandb.py` (lines 108-148) in place — add `fp_rate_tfidf`/`suggested_tfidf_threshold` to the `CalibrationReport(...)` construction, add `tfidf_cosine_z` to the `OodThresholds(...)` construction, and add the two new keys to both assertion blocks:

```python
def test_log_ood_calibration_results_logs_summary_metrics() -> None:
    report = CalibrationReport(
        fp_rate_maha=0.2951,
        fp_rate_cosine=0.0104,
        fp_rate_knn=0.0087,
        suggested_maha_threshold=0.0,
        suggested_cosine_threshold=13.7186,
        suggested_knn_threshold=4.2,
        fp_rate_tfidf=0.0093,
        suggested_tfidf_threshold=2.71,
    )
    thresholds = OodThresholds(
        mahalanobis_p=0.001, cosine_z=13.7366, knn_distance=16.7908, tfidf_cosine_z=2.5
    )
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
    config = mock_init.call_args.kwargs["config"]
    assert config["current_mahalanobis_threshold"] == 0.001  # noqa: PLR2004
    assert config["current_cosine_threshold"] == 13.7366  # noqa: PLR2004
    assert config["current_knn_threshold"] == 16.7908  # noqa: PLR2004
    assert config["current_tfidf_threshold"] == 2.5  # noqa: PLR2004
    mock_log.assert_called_once_with(
        {
            "ood/fp_rate_mahalanobis": 0.2951,
            "ood/fp_rate_cosine": 0.0104,
            "ood/suggested_mahalanobis_threshold": 0.0,
            "ood/suggested_cosine_threshold": 13.7186,
            "ood/fp_rate_knn": 0.0087,
            "ood/suggested_knn_threshold": 4.2,
            "ood/fp_rate_tfidf": 0.0093,
            "ood/suggested_tfidf_threshold": 2.71,
        }
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_ood_calibration.py tests/test_wandb.py -v -k "tfidf"`
Expected: FAIL — `TypeError: build_calibration_report() got an unexpected keyword argument 'tfidf_z_scores'`

- [ ] **Step 3: Extend `build_calibration_report`**

```python
def build_calibration_report(
    p_values: npt.NDArray[np.float64],
    z_scores: npt.NDArray[np.float64],
    knn_distances: npt.NDArray[np.float64],
    target_fp_rate: float,
    thresholds: OodThresholds,
    tfidf_z_scores: npt.NDArray[np.float64],
) -> CalibrationReport:
    fp_rate_tfidf = 0.0
    suggested_tfidf_threshold = 0.0
    if len(tfidf_z_scores) > 0:
        fp_rate_tfidf = float(np.mean(tfidf_z_scores > thresholds.tfidf_cosine_z))
        suggested_tfidf_threshold = float(
            np.percentile(tfidf_z_scores, (1 - target_fp_rate) * 100)
        )
    return CalibrationReport(
        fp_rate_maha=float(np.mean(p_values < thresholds.mahalanobis_p)),
        fp_rate_cosine=float(np.mean(z_scores > thresholds.cosine_z)),
        fp_rate_knn=float(np.mean(knn_distances > thresholds.knn_distance)),
        suggested_maha_threshold=float(np.percentile(p_values, target_fp_rate * 100)),
        suggested_cosine_threshold=float(np.percentile(z_scores, (1 - target_fp_rate) * 100)),
        suggested_knn_threshold=float(np.percentile(knn_distances, (1 - target_fp_rate) * 100)),
        fp_rate_tfidf=fp_rate_tfidf,
        suggested_tfidf_threshold=suggested_tfidf_threshold,
    )
```

Update its docstring to mention the TF-IDF branch follows the same HIGH-value-is-anomalous convention as cosine/k-NN.

- [ ] **Step 4: Extend `_write_calibrated_thresholds`**

Add `"tfidf_threshold": report.suggested_tfidf_threshold if report.suggested_tfidf_threshold > 0 else stats.tfidf_threshold,` to the `stats.model_copy(update={...})` call's dict — mirrors cosine/knn (no degenerate-guard needed here, matching the existing comment that the guard is Mahalanobis-only). Guard against writing `0.0` when there was no TF-IDF data for this run (keeps whatever was already persisted, or `None`, rather than overwriting a real prior calibration with a meaningless zero).

- [ ] **Step 5: Extend `_run_ood_calibration`**

After the existing `knn_distances`/`knn_valid` block, add:

```python
    tfidf_vectorizer = build_tfidf_vectorizer(stats)
    tfidf_z_scores = np.array([])  # "no TF-IDF data for this run" -- passed explicitly,
    # not defaulted, matching build_calibration_report's required (non-Optional) parameter
    if tfidf_vectorizer is not None:
        tfidf_z_scores = np.array(
            [
                tfidf_cosine_z_score(text, stats, tfidf_vectorizer)
                for text in split.test_df["text"]
            ]
        )
```

Pass `tfidf_z_scores=tfidf_z_scores` into the `build_calibration_report(...)` call. Add a log line after the existing k-NN log lines:

```python
    log.info(
        "TF-IDF cosine — current threshold=%.4f, empirical false-positive rate=%.2f%%",
        current_thresholds.tfidf_cosine_z,
        report.fp_rate_tfidf * 100,
    )
    log.info(
        "TF-IDF cosine — suggested threshold for %.1f%% target FP rate: %.4f",
        opts.target_fp_rate * 100,
        report.suggested_tfidf_threshold,
    )
```

Add the necessary imports (`build_tfidf_vectorizer`, `tfidf_cosine_z_score`) to `src/cli/ood_calibration.py`'s existing `from src.ood import (...)` block.

- [ ] **Step 6: Extend `log_ood_calibration_results`**

In `src/wandb.py`, add to the `config` dict: `"current_tfidf_threshold": thresholds.tfidf_cosine_z,`. Add to the `wandb.log({...})` dict: `"ood/fp_rate_tfidf": report.fp_rate_tfidf, "ood/suggested_tfidf_threshold": report.suggested_tfidf_threshold,`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_ood_calibration.py tests/test_wandb.py -v`
Expected: PASS

- [ ] **Step 8: Run the full test suite**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/cli/ood_calibration.py src/wandb.py tests/cli/test_ood_calibration.py tests/test_wandb.py
git commit -m "feat: calibrate, report, and persist the TF-IDF OOD threshold"
```

---

### Task 7: Regenerate BETO v1/v2 stats, verify doc3, document in CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`
- Regenerate: `models/bert_tunning_model_beto/final/ood_stats.npz`, `models/bert_tunning_model_beto_v2/final/ood_stats.npz`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: both committed model artifacts gain the TF-IDF fields; `CLAUDE.md` documents the fourth signal following the existing "Key Technical Decisions" prose style.

- [ ] **Step 1: Regenerate stats for both models**

```bash
uv run python main.py compute-ood-stats --model-path "models/bert_tunning_model_beto/final" --model beto --cache-path "./data/bert_tunning_cache_no_otro.parquet"
uv run python main.py compute-ood-stats --model-path "models/bert_tunning_model_beto_v2/final" --model beto --cache-path "./data/bert_tunning_cache_con_otro_300.parquet"
```

(Confirm the exact `--cache-path` for BETO v1 first — this plan's earlier investigation only confirmed BETO v2's cache path; check `README.md`'s dataset summary table or ask before running, per this project's "never guess a training artifact path" convention.)

- [ ] **Step 2: Calibrate the TF-IDF threshold for both models**

```bash
uv run python main.py evaluate-ood-calibration --model-path "models/bert_tunning_model_beto_v2/final" --model beto --cache-path "./data/bert_tunning_cache_con_otro_300.parquet" --write-thresholds
uv run python main.py evaluate-ood-calibration --model-path "models/bert_tunning_model_beto/final" --model beto --cache-path "./data/bert_tunning_cache_no_otro.parquet" --write-thresholds
```

- [ ] **Step 3: Verify doc3 (`document_predict_3.pdf`) is now caught by BETO v2**

Re-run the `predict-folder` command from earlier in this session against `samples/` with BETO v2, and confirm `document_predict_3.pdf`'s `in_distribution` is now `False` with a non-null `tfidf_cosine_z`. If it is NOT caught, do not silently accept — report back with the actual `tfidf_cosine_z` value and the resolved threshold, since that would mean the TF-IDF vocabulary itself doesn't contain enough of the distinguishing city-name tokens (a real, reportable finding, not a bug to paper over).

- [ ] **Step 4: Document in CLAUDE.md**

Add a new paragraph under "Key Technical Decisions", after the "k-NN class-conditional distance" entry, following that entry's exact prose style (motivating failure mode → mechanism → what changed → caveats):

```markdown
**TF-IDF cosine-centroid — a fourth OOD signal, catching lexical divergence the embedding-based signals cannot**
Mahalanobis, cosine, and k-NN all operate on the `[CLS]` embedding's semantic "shape" -- which means a document sharing the same document-type genre (e.g. a decree) but naming a different municipality than any in the training corpus is nearly indistinguishable from a genuine in-distribution document in that space, since BERT's embedding compresses away the specific place name in favor of "this is decree-shaped text." `compute_tfidf_stats()` (`src/ood.py`) fits a `TfidfVectorizer` + per-class centroids directly on training text's raw vocabulary instead, and `tfidf_cosine_z_score()` scores new documents the same way `cosine_z_score` already does (cosine distance to nearest centroid, z-scored against the training set) -- just in TF-IDF space, where a different city's name is a literal distinguishing feature rather than noise the embedding smooths over. Persisted through the existing `ood_stats.npz` with no new artifact file: `TfidfVectorizer.vocabulary_`/`.idf_` round-trip through two plain arrays (`tfidf_vocabulary_terms`, `tfidf_idf`), verified to reconstruct a vectorizer whose `.transform()` output is bit-identical to the originally-fitted one. All six `tfidf_*` fields default to an "absent" sentinel for backward compatibility -- an empty vocabulary/idf/centroids for the three collection fields (a real fitted vectorizer always has >=1 term, so empty is unambiguous), `None` for the three scalar calibration/threshold fields (no safe zero-sentinel for those) -- an `ood_stats.npz` predating this feature has the signal skipped entirely (not treated as anomalous by default), and must be regenerated via `compute-ood-stats` to gain it. Folded into the same OR as the other three signals (`in_distribution=False` if any of the four fire) -- never blended into one score, same rationale as the original Mahalanobis/cosine decision. `OOD_TFIDF_COSINE_THRESHOLD` (default `2.5`) is an uncalibrated placeholder like the original `OOD_COSINE_THRESHOLD`/`OOD_KNN_DISTANCE_THRESHOLD` were -- run `evaluate-ood-calibration --write-thresholds` before trusting it in production.
```

- [ ] **Step 5: Run the full test suite one final time**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md models/bert_tunning_model_beto/final/ood_stats.npz models/bert_tunning_model_beto_v2/final/ood_stats.npz
git commit -m "docs: document TF-IDF OOD signal; regenerate BETO v1/v2 ood_stats.npz with it"
```
