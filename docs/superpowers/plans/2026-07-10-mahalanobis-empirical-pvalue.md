# Mahalanobis Empirical P-Value Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the chi²-based `mahalanobis_p_value` (which assumes multivariate-Gaussian, shared-covariance class embeddings — an assumption a QQ-plot check showed is badly violated, explaining Mahalanobis's measured 20-30% false-positive rate vs. a 1% target) with an empirical (rank-based) p-value computed against the training set's own real Mahalanobis distances, while keeping the chi²-based value available as a separate, purely informational field.

**Architecture:** `src/ood.py` gains a shared rank-math helper (`empirical_survival_p_value`), a reference-distribution builder (`compute_train_mahalanobis_distances` — each training document's distance to its **own true class centroid**, matching how `compute_class_stats()` itself estimates the covariance matrix, NOT the nearest centroid), and `mahalanobis_empirical_p_value` (ranks a query's nearest-centroid distance against that reference). The existing chi²-based function is renamed to `mahalanobis_chi2_p_value` and kept for a new informational field. `BertTunningClassifier` computes the reference array once per process (lazily cached), not per-request. `is_out_of_distribution()`'s comparison direction and `OOD_MAHALANOBIS_P_THRESHOLD`'s role are unchanged — only how the p-value is computed changes, so the OR-of-three-signals design is untouched.

**Tech Stack:** Python 3.10, NumPy, Pydantic v2, pytest. No new dependencies.

## Global Constraints

- No retraining, no `ood_stats.npz` format change, no backfill of existing model files — `ClassEmbeddingStats` already stores everything needed (`centroids`, `covariance_inv`, `knn_train_embeddings`, `knn_train_labels`).
- `mahalanobis_p_value` keeps its name and keeps driving the `in_distribution` decision — only its computation changes from parametric (chi²) to empirical (rank-based). `Settings.OOD_MAHALANOBIS_P_THRESHOLD` and `is_out_of_distribution()`'s `scores.mahalanobis_p < threshold` comparison stay exactly as-is.
- The chi²-based value becomes a new field, `mahalanobis_p_value_theoretical`, and must never participate in `is_out_of_distribution()` or any threshold comparison — informational only, per the project's documented rule against diluting one signal with another.
- **Asymmetric by design, not an inconsistency**: the empirical *reference* distribution (training set) is built from each point's distance to its **true** label's centroid, matching `compute_class_stats()`'s own covariance estimation. The *query* distance (a new document being scored) still uses `mahalanobis_min_distance()` — the nearest centroid, exactly as it already did before this plan. This is preserved existing behavior, not derived from "the predicted label" (there is no per-class distance computed at inference time to begin with) — worth stating precisely so a future reader doesn't "fix" it into a predicted-class distance calculation that doesn't currently exist. Every function that touches this must document the asymmetry, not silently rely on it.
- `uv run poe check` (ruff + mypy strict + pytest) must pass **after every task, no exceptions** — achieved by doing the `mahalanobis_p_value` → `mahalanobis_chi2_p_value` rename as one mechanical, behavior-preserving pass across every call site inside Task 1 itself (not deferred, not bridged with a compatibility alias).
- Every Pydantic model touched that has `alias_generator=to_camel` must also have `populate_by_name=True` if constructed with snake_case kwargs anywhere (already true for `PredictResult`/`BaseSchema`).
- **Format before checking**: several code blocks in this plan (long function signatures, multi-name imports) may exceed the repo's 100-char ruff line-length target as typed. Run `uv run poe fmt` before `uv run poe check` in every task's implementation step — don't hand-wrap lines to guess at ruff's formatting.

---

### Task 1: Empirical Mahalanobis p-value math in `src/ood.py` (+ mechanical rename propagation)

**Files:**
- Modify: `src/ood.py`
- Modify: `src/inference/classify.py` (mechanical rename only — see Step 3b)
- Modify: `src/cli/ood_calibration.py` (mechanical rename only — see Step 3c)
- Test: `tests/test_ood.py`

**Interfaces:**
- Produces: `empirical_survival_p_value(distance: float, reference: npt.NDArray[np.float64]) -> float`, `compute_train_mahalanobis_distances(stats: ClassEmbeddingStats) -> npt.NDArray[np.float64]`, `mahalanobis_empirical_p_value(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats, train_distances: npt.NDArray[np.float64]) -> float`, `mahalanobis_chi2_p_value(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float` (renamed from `mahalanobis_p_value` — same body, same behavior).

- [ ] **Step 1: Write the failing tests**

Update the import block at the top of `tests/test_ood.py`:

```python
from src.ood import (
    LoadedModel,
    compute_class_stats,
    compute_train_mahalanobis_distances,
    cosine_min_distance,
    cosine_z_score,
    empirical_survival_p_value,
    extract_embeddings,
    knn_mean_distance,
    load_stats,
    mahalanobis_chi2_p_value,
    mahalanobis_empirical_p_value,
    mahalanobis_min_distance,
    save_stats,
)
from src.schema import ClassEmbeddingStats
```

(`ClassEmbeddingStats` may already be imported for other fixtures in this file — check before duplicating the import line.)

Replace the two existing `mahalanobis_p_value` tests with the chi2-renamed versions, and add the new tests:

```python
def test_mahalanobis_chi2_p_value_is_lower_for_far_point() -> None:
    # A low p-value means "unlikely to be in-distribution" — the far point should
    # score LOWER (more anomalous), not higher, unlike a distance/z-score metric.
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    maha_p_far = mahalanobis_chi2_p_value(far_point, stats)
    maha_p_known = mahalanobis_chi2_p_value(known_point, stats)
    assert maha_p_far < maha_p_known


def test_mahalanobis_chi2_p_value_is_bounded_between_zero_and_one() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    for point in (known_point, far_point):
        p_value = mahalanobis_chi2_p_value(point, stats)
        assert 0.0 <= p_value <= 1.0


def test_empirical_survival_p_value_matches_hand_computed_rank() -> None:
    reference = np.array([1.0, 3.0, 5.0, 9.0])
    # 2 of 4 reference values are >= 5.0, so p = (2 + 1) / (4 + 1) = 0.6.
    assert empirical_survival_p_value(5.0, reference) == pytest.approx(0.6)


def test_empirical_survival_p_value_raises_on_empty_reference() -> None:
    # Silently returning 1.0 ("maximally normal") for no reference data would be a
    # fail-open bug — exactly backwards for an anomaly-detection signal.
    with pytest.raises(ValueError, match="empty"):
        empirical_survival_p_value(5.0, np.array([]))


def test_compute_train_mahalanobis_distances_returns_one_value_per_training_doc() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    distances = compute_train_mahalanobis_distances(stats)
    assert distances.shape == (len(embeddings),)
    assert np.all(distances >= 0.0)


def test_compute_train_mahalanobis_distances_uses_true_label_not_nearest_centroid() -> None:
    # A training point labeled class_a but physically much closer to class_b's centroid --
    # its distance must be measured against its TRUE centroid (class_a, far -> large
    # distance), not whichever centroid is nearest (class_b, close -> small distance).
    # This mirrors compute_class_stats()'s own covariance estimation
    # (centered = reduced - centroids[labels_arr]), which uses true labels too.
    stats = ClassEmbeddingStats(
        class_names=["class_a", "class_b"],
        pca_mean=np.zeros(2),
        pca_components=np.eye(2),
        centroids=np.array([[0.0, 0.0], [10.0, 0.0]]),
        covariance_inv=np.eye(2),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        # labeled class_a (centroid [0,0]) but sits right next to class_b's centroid [10,0].
        knn_train_embeddings=np.array([[9.0, 0.0]]),
        knn_train_labels=[0],
    )
    distances = compute_train_mahalanobis_distances(stats)
    # True-label (class_a) squared distance: 9^2 = 81. Nearest-centroid (class_b) would
    # have been 1^2 = 1 -- if this assertion sees 1.0 instead of 81.0, the implementation
    # is using nearest-centroid instead of true-label distance.
    assert distances[0] == pytest.approx(81.0)


def test_mahalanobis_empirical_p_value_is_lower_for_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    train_distances = compute_train_mahalanobis_distances(stats)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    p_far = mahalanobis_empirical_p_value(far_point, stats, train_distances)
    p_known = mahalanobis_empirical_p_value(known_point, stats, train_distances)
    assert p_far < p_known


def test_mahalanobis_empirical_p_value_is_bounded_between_zero_and_one() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    train_distances = compute_train_mahalanobis_distances(stats)
    known_point = embeddings[0]
    far_point = np.full(16, 100.0)
    for point in (known_point, far_point):
        p_value = mahalanobis_empirical_p_value(point, stats, train_distances)
        assert 0.0 < p_value <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ood.py -v`
Expected: FAIL — `ImportError: cannot import name 'empirical_survival_p_value' from 'src.ood'` (and similarly for the other new names).

- [ ] **Step 3a: Implement the new math in `src/ood.py`**

Rename the existing `mahalanobis_p_value` function (currently right after `cosine_min_distance`) to `mahalanobis_chi2_p_value`, keeping its body unchanged except the docstring:

```python
def mahalanobis_chi2_p_value(embedding: npt.NDArray[np.float64], stats: ClassEmbeddingStats) -> float:
    """Theoretical p-value under the assumption that class-conditional embeddings are
    multivariate Gaussian with one shared covariance matrix: the squared Mahalanobis
    distance of such a point follows a chi-squared distribution with `df` equal to the
    embedding dimensionality. Kept as a transparent, purely informational value --
    NOT used to decide in_distribution. A QQ-plot check (see the playground notebook)
    showed this assumption is badly violated for this corpus (observed distances run
    ~5x larger than chi2 predicts), which is why mahalanobis_empirical_p_value exists
    and is the one that actually drives the anomaly decision. A LOW value here still
    means "far from centroid," it just isn't a trustworthy probability."""
    squared_distance = mahalanobis_min_distance(embedding, stats)
    degrees_of_freedom = stats.centroids.shape[1]
    return float(chi2.sf(squared_distance, df=degrees_of_freedom))


def empirical_survival_p_value(distance: float, reference: npt.NDArray[np.float64]) -> float:
    """The standard permutation-test empirical p-value: the fraction of `reference` values
    at least as extreme as `distance`, with the usual +1/+1 correction so the result is
    never exactly 0. Raises if `reference` is empty -- silently returning 1.0 ("maximally
    normal") for no reference data would be a fail-open bug, backwards for an
    anomaly-detection signal."""
    if len(reference) == 0:
        msg = "empirical_survival_p_value: reference array is empty, cannot rank against it"
        raise ValueError(msg)
    exceed_count = int(np.sum(reference >= distance))
    return (exceed_count + 1) / (len(reference) + 1)


def compute_train_mahalanobis_distances(stats: ClassEmbeddingStats) -> npt.NDArray[np.float64]:
    """Squared Mahalanobis distance from every training document (stats.knn_train_embeddings,
    already PCA-reduced) to its OWN TRUE class centroid (via stats.knn_train_labels) -- not
    the nearest centroid. This intentionally mirrors compute_class_stats()'s own covariance
    estimation (`centered = reduced - centroids[labels_arr]`), built from each point's
    deviation from its labeled class, not whichever centroid happens to be closest. Using
    nearest-centroid distance here would let ambiguous/boundary training points look
    artificially unremarkable (nearest distance <= true-label distance, always), corrupting
    the reference distribution's tail -- exactly where "how extreme is extreme" matters most.
    mahalanobis_empirical_p_value() below still scores a QUERY point's distance via
    mahalanobis_min_distance() (nearest centroid) -- inference has no true label to measure
    against, and nearest-centroid is that function's existing, unchanged, preserved
    behavior (NOT a distance to some "predicted-class centroid" -- no such per-class
    distance is computed at inference time). This asymmetry (reference: true-label
    distance; query: nearest-centroid distance) is intentional -- see "Global Constraints"
    in this plan."""
    labels_arr = np.asarray(stats.knn_train_labels)
    distances = np.empty(len(stats.knn_train_embeddings), dtype=np.float64)
    for i, point in enumerate(stats.knn_train_embeddings):
        centroid = stats.centroids[labels_arr[i]]
        diff = centroid - point
        distances[i] = float(diff @ stats.covariance_inv @ diff)
    return distances


def mahalanobis_empirical_p_value(
    embedding: npt.NDArray[np.float64],
    stats: ClassEmbeddingStats,
    train_distances: npt.NDArray[np.float64],
) -> float:
    """Empirical (rank-based) p-value for a query embedding: ranks its Mahalanobis distance
    to the nearest class centroid (mahalanobis_min_distance) against train_distances (each
    training document's distance to its own TRUE class centroid -- see
    compute_train_mahalanobis_distances). Makes no distributional assumption, unlike
    mahalanobis_chi2_p_value -- this is the value that drives is_out_of_distribution(). A LOW
    p-value means the document is anomalous, same comparison direction as the chi2 version
    it replaces."""
    distance = mahalanobis_min_distance(embedding, stats)
    return empirical_survival_p_value(distance, train_distances)
```

- [ ] **Step 3b: Mechanically propagate the rename in `src/inference/classify.py`**

This is a **rename only** — zero behavior change, `mahalanobis_p` still ends up chi²-based after this step. The switch to empirical happens in Task 3.

Change the import line:

```python
from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_chi2_p_value
```

Change the one call site inside `predict_text`'s `OodScores(...)` construction:

```python
        scores = OodScores(
            mahalanobis_p=mahalanobis_chi2_p_value(cls_embedding, self._ood_stats),
            cosine_z=cosine_z_score(cls_embedding, self._ood_stats),
```

- [ ] **Step 3c: Mechanically propagate the rename in `src/cli/ood_calibration.py`**

Same rule — rename only, zero behavior change.

Change the import line:

```python
from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_chi2_p_value
```

Change the one call site inside `_run_ood_calibration`:

```python
    p_values = np.array([mahalanobis_chi2_p_value(e, stats) for e in embeddings])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ood.py tests/inference/test_pipeline.py tests/cli/test_ood_calibration.py -v`
Expected: PASS — every existing test in `test_pipeline.py`/`test_ood_calibration.py` still passes unchanged, since Steps 3b/3c are pure renames.

- [ ] **Step 5: Run full check**

Run: `uv run poe fmt && uv run poe check`
Expected: PASS, cleanly, no exceptions.

- [ ] **Step 6: Commit**

```bash
git add src/ood.py src/inference/classify.py src/cli/ood_calibration.py tests/test_ood.py
git commit -m "feat: add empirical Mahalanobis p-value math; rename chi2 version everywhere"
```

---

### Task 2: `mahalanobis_p_value_theoretical` field — schema, API, CLI, W&B

**Files:**
- Modify: `src/schema.py`
- Modify: `src/api/routes/predict/schemas.py`
- Modify: `src/api/routes/predict/endpoints.py`
- Modify: `src/cli/predict.py`
- Modify: `src/wandb.py`
- Test: `tests/api/test_predict.py`
- Test: `tests/test_wandb.py`

**Interfaces:**
- Produces: `PredictResult.mahalanobis_p_value_theoretical: float | None`, matched by `PredictResponse`, `_PREDICTION_COLUMNS`, and the CLI echo. Not yet populated by `BertTunningClassifier` (that's Task 3) — this task only adds the field and its plumbing, using directly-constructed `PredictResult`s in tests. Deliberately independent of `src/inference/classify.py`.

- [ ] **Step 1: Write the failing test**

Add to `tests/api/test_predict.py`:

```python
def test_predict_endpoint_returns_theoretical_mahalanobis_p_value() -> None:
    expected_theoretical_p = 0.1708
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        mahalanobis_p_value_theoretical=expected_theoretical_p,
    )
    app.state.clf = mock_clf

    fake_extraction = ExtractionMetadata(
        text="hola mundo", extractor_used="OCRExtractor", char_count=10
    )
    with patch(
        "src.api.routes.predict.endpoints.extract_pdf_with_metadata", return_value=fake_extraction
    ):
        client = TestClient(app)
        response = client.post(
            "/predict",
            files={"file": ("doc.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        )

    assert response.status_code == HTTPStatus.OK
    assert response.json()["mahalanobisPValueTheoretical"] == expected_theoretical_p
```

Add to `tests/test_wandb.py`:

```python
def test_log_predict_folder_results_table_includes_theoretical_mahalanobis_column() -> None:
    expected_theoretical_p = 0.1708
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            mahalanobis_p_value_theoretical=expected_theoretical_p,
        ),
    ]
    mock_table = MagicMock()
    with (
        patch("src.wandb.wandb.init"),
        patch("src.wandb.wandb.Table", return_value=mock_table) as mock_table_cls,
        patch("src.wandb.wandb.log"),
        patch("src.wandb.wandb.finish"),
    ):
        log_predict_folder_results(results, model_path="fake/model", folder_path="fake/folder")

    row = _logged_row(mock_table_cls, mock_table)
    assert row["mahalanobis_p_value_theoretical"] == expected_theoretical_p
```

Add to `tests/cli/test_commands.py` — this covers the exact transitional state this task deliberately creates (`mahalanobis_p_value` populated, `mahalanobis_p_value_theoretical` still `None` until Task 3), following the same `patch("src.cli.predict.predict_pdf", ...)` mocking pattern the file already uses for `predict_folder`:

```python
def test_predict_cmd_echoes_n_a_when_theoretical_p_value_missing(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        mahalanobis_p_value=0.005,
        cosine_z=0.1,
        knn_distance=1.0,
        in_distribution=False,
        mahalanobis_p_value_theoretical=None,
    )

    with patch("src.cli.predict.predict_pdf", return_value=fake_result):
        result = CliRunner().invoke(predict_cmd, [str(pdf_path)])

    assert result.exit_code == 0
    assert "Mahalanobis p (chi2, theoretical): n/a" in result.output
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_predict.py tests/test_wandb.py -v`
Expected: FAIL — but **not** with a constructor error. `PredictResult`'s default Pydantic v2 `extra` behavior is `"ignore"`, so `PredictResult(..., mahalanobis_p_value_theoretical=0.17)` **silently drops** the unrecognized kwarg and constructs successfully — verified directly: `PredictResult(label="x", confidence=0.9, certain=True, mahalanobis_p_value_theoretical=0.17).model_dump().keys()` has no such key. The actual failures:
- `tests/api/test_predict.py`: `KeyError: 'mahalanobisPValueTheoretical'` on `response.json()["mahalanobisPValueTheoretical"]` — the field was silently dropped from the constructed `PredictResult`, so `PredictResponse` (which doesn't have the field yet either) never puts it in the JSON body.
- `tests/test_wandb.py`: `KeyError: 'mahalanobis_p_value_theoretical'` inside `_logged_row(...)[...]` — `_PREDICTION_COLUMNS` doesn't include that name yet, so it was never zipped into the row dict in the first place.

- [ ] **Step 3: Implement**

`src/schema.py` — add the field to `PredictResult`, right after `mahalanobis_p_value`:

```python
    mahalanobis_p_value: float | None = None
    mahalanobis_p_value_theoretical: float | None = None
    cosine_z: float | None = None
```

`src/api/routes/predict/schemas.py` — matching field on `PredictResponse`, right after `mahalanobis_p_value`:

```python
    mahalanobis_p_value: float | None = None
    mahalanobis_p_value_theoretical: float | None = None
    cosine_z: float | None = None
```

`src/api/routes/predict/endpoints.py` — add to `_to_predict_response`'s explicit field list, right after `mahalanobis_p_value`:

```python
        mahalanobis_p_value=data["mahalanobis_p_value"],
        mahalanobis_p_value_theoretical=data["mahalanobis_p_value_theoretical"],
        cosine_z=data["cosine_z"],
```

`src/wandb.py` — add to `_PREDICTION_COLUMNS`, right after `"mahalanobis_p_value"`:

```python
_PREDICTION_COLUMNS = [
    "filename",
    "label",
    "confidence",
    "certain",
    "mahalanobis_p_value",
    "mahalanobis_p_value_theoretical",
    "cosine_z",
    "knn_distance",
    "in_distribution",
    "review_route",
    "extractor_used",
    "error",
]
```

`src/cli/predict.py` — add a **null-safe** echo line in `predict_cmd`, right after the existing Mahalanobis line. `mahalanobis_p_value_theoretical` is independently nullable from `mahalanobis_p_value` — until Task 3 wires the classifier to populate it, real predictions would have `mahalanobis_p_value` set but `mahalanobis_p_value_theoretical` still `None`, which would crash an unguarded `:.6f` format:

```python
    if result.mahalanobis_p_value is not None:
        click.echo(f"  Mahalanobis p: {result.mahalanobis_p_value:.6f}")
        theoretical = result.mahalanobis_p_value_theoretical
        theoretical_str = f"{theoretical:.6f}" if theoretical is not None else "n/a"
        click.echo(f"  Mahalanobis p (chi2, theoretical): {theoretical_str}")
        click.echo(f"  Cosine Z     : {result.cosine_z:.4f}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_predict.py tests/test_wandb.py tests/cli/test_commands.py -v`
Expected: PASS

- [ ] **Step 5: Run full check**

Run: `uv run poe fmt && uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/schema.py src/api/routes/predict/schemas.py src/api/routes/predict/endpoints.py src/cli/predict.py src/wandb.py tests/api/test_predict.py tests/test_wandb.py tests/cli/test_commands.py
git commit -m "feat: add mahalanobis_p_value_theoretical field across API, CLI, and W&B"
```

---

### Task 3: Switch `BertTunningClassifier` to the empirical p-value

**Files:**
- Modify: `src/inference/classify.py`
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `compute_train_mahalanobis_distances(stats)`, `mahalanobis_empirical_p_value(embedding, stats, train_distances)`, `mahalanobis_chi2_p_value(embedding, stats)` (Task 1). `PredictResult.mahalanobis_p_value_theoretical` (Task 2, already exists).
- Produces: `predict_text()` now populates `mahalanobis_p_value` from the empirical function (the actual behavior change) and `mahalanobis_p_value_theoretical` from the already-wired chi2 function.

- [ ] **Step 1: Write the failing test**

First, widen `_make_stats()` in `tests/inference/test_pipeline.py` — the shared fixture used by several tests — from 5 to 100 points per class. The empirical p-value's minimum possible value is `1/(N+1)`; with only 5-10 reference points it can never drop below `OOD_MAHALANOBIS_P_THRESHOLD` (0.01) regardless of how far a query point is, which would silently break the existing "flags out of distribution via Mahalanobis only" test:

```python
def _make_stats() -> ClassEmbeddingStats:
    # 100 points/class, not 5 -- the empirical Mahalanobis p-value's minimum possible
    # value is 1/(N+1). With only 5-10 reference points it could never drop below
    # OOD_MAHALANOBIS_P_THRESHOLD (0.01) regardless of how far a query point actually is.
    n_per_class = 100
    return ClassEmbeddingStats(
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        # All points sit exactly on their own class's centroid (distance 0) -- degenerate
        # but sufficient here, since these tests only need "near" vs. "far" distinguishable.
        knn_train_embeddings=np.array([[0.0] * 8] * n_per_class + [[5.0] * 8] * n_per_class),
        knn_train_labels=[0] * n_per_class + [1] * n_per_class,
    )
```

Then update `test_predict_text_with_stats_populates_ood_fields` and add two new tests:

```python
def test_predict_text_with_stats_populates_ood_fields() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert isinstance(result.mahalanobis_p_value, float)
    assert isinstance(result.mahalanobis_p_value_theoretical, float)
    assert isinstance(result.cosine_z, float)
    assert isinstance(result.in_distribution, bool)


def test_predict_text_mahalanobis_p_value_is_empirical() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (mock hidden_states), exactly centroid A -- distance 0,
    # so the empirical p-value must be exactly 1.0 (all 200 reference points have
    # distance >= 0).
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.mahalanobis_p_value == pytest.approx(1.0)


def test_predict_text_flags_out_of_distribution_via_mahalanobis_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # A point far from both centroids: with all 200 reference (training) distances equal
    # to 0, the empirical p-value for any nonzero-distance query collapses to
    # 1 / (200 + 1) ≈ 0.005, safely below OOD_MAHALANOBIS_P_THRESHOLD (0.01) -- but
    # pointing in the exact same direction as centroid B, so cosine distance is ~0 and
    # only the Mahalanobis signal should fire.
    far_embedding = torch.full((1, 512, 8), 100.0)
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [far_embedding]
        result = clf.predict_text("anything")
    assert result.mahalanobis_p_value is not None
    assert result.cosine_z is not None
    assert result.mahalanobis_p_value < Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.in_distribution is False
    assert result.review_route == "human_review"
```

(The third test replaces the existing test of the same name — same assertions, updated comment; `torch.full((1, 512, 8), 100.0)` and the cosine assertions are unchanged.)

Need `import pytest` at the top of `tests/inference/test_pipeline.py` if not already present (check — it may not be, since prior tests didn't need it directly).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/inference/test_pipeline.py -k "populates_ood_fields or is_empirical or mahalanobis_only" -v`
Expected: FAIL — `mahalanobis_p_value_theoretical` is `None` (nothing populates it yet), and/or the far-point test's `mahalanobis_p_value` is still chi2-based (not yet below threshold via the empirical formula's specific behavior — though note the chi2 value likely already fires for this specific far point too, so the clearest failure signal is the `is_empirical` test, which checks the exact `1.0` value only the empirical formula produces for a distance-0 query against an all-zero reference).

- [ ] **Step 3: Implement in `src/inference/classify.py`**

Update the imports — add `functools.cached_property` and `numpy.typing`:

```python
from functools import cached_property
```

(alongside the existing `from enum import Enum` / `from pathlib import Path` / `from typing import Any, NamedTuple` block, keeping alphabetical stdlib order: `enum`, `functools`, `pathlib`, `typing`)

```python
import numpy.typing as npt
```

(alongside the existing `import numpy as np`)

```python
from src.ood import (
    compute_train_mahalanobis_distances,
    cosine_z_score,
    knn_mean_distance,
    load_stats,
    mahalanobis_chi2_p_value,
    mahalanobis_empirical_p_value,
)
```

(replaces Task 1's `from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_chi2_p_value`)

Update `OodScores`'s docstring to state the empirical/chi2 split explicitly:

```python
class OodScores(NamedTuple):
    """The three OOD signals -- always computed together, passed together, never used
    independently. Matches LoadedModel/_PcaReduction's convention in src/ood.py.
    mahalanobis_p is the empirical (rank-based) p-value, not the chi2 one -- see
    BertTunningClassifier.predict_text for where the chi2 value is separately attached
    to PredictResult.mahalanobis_p_value_theoretical, informational only."""

    mahalanobis_p: float
    cosine_z: float
    knn_distance: float
```

Add the cached reference-distance property to `BertTunningClassifier`, right after `_load_ood_stats`:

```python
    @cached_property
    def _train_mahalanobis_distances(self) -> npt.NDArray[np.float64] | None:
        """Computed lazily on first access (not in __init__), cached for the process
        lifetime -- avoids recomputing 1300+ training-point distances on every single
        predict_text() call. None when there's no ood_stats.npz to compute it from."""
        if self._ood_stats is None:
            return None
        return compute_train_mahalanobis_distances(self._ood_stats)
```

Update `predict_text`'s OOD-scoring block:

```python
        train_distances = self._train_mahalanobis_distances
        assert train_distances is not None
        scores = OodScores(
            mahalanobis_p=mahalanobis_empirical_p_value(cls_embedding, self._ood_stats, train_distances),
            cosine_z=cosine_z_score(cls_embedding, self._ood_stats),
            knn_distance=knn_mean_distance(
                cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
            ),
        )
        maha_p_theoretical = mahalanobis_chi2_p_value(cls_embedding, self._ood_stats)
        in_distribution = not is_out_of_distribution(scores)
        return result.model_copy(
            update={
                "mahalanobis_p_value": round(scores.mahalanobis_p, 6),
                "mahalanobis_p_value_theoretical": round(maha_p_theoretical, 6),
                "cosine_z": round(scores.cosine_z, 4),
                "knn_distance": round(scores.knn_distance, 4),
                "in_distribution": in_distribution,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(in_distribution=in_distribution),
                ),
            }
        )
```

The local `train_distances = self._train_mahalanobis_distances` assignment (not repeated inline `self._train_mahalanobis_distances` accesses) matters here: `cached_property` is a descriptor, and mypy strict does not reliably persist a `self.attr is not None` narrowing across multiple separate attribute reads the way it does for a plain local variable. Assign once, assert once, then only ever pass the local `train_distances` onward. It's guaranteed non-`None` at this call site (reached only after the `if self._ood_stats is None: return result` early return above it) — the `assert` documents that guarantee for mypy strict and for the next reader.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: PASS (all tests in the file — the widened `_make_stats()` fixture affects every test that uses it)

- [ ] **Step 5: Run full check**

Run: `uv run poe fmt && uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "feat: switch BertTunningClassifier's mahalanobis_p_value to the empirical formula"
```

---

### Task 4: `evaluate-ood-calibration` calibrates against the empirical p-value

**Files:**
- Modify: `src/cli/ood_calibration.py`
- Test: `tests/cli/test_ood_calibration.py`

**Interfaces:**
- Consumes: `compute_train_mahalanobis_distances(stats)`, `mahalanobis_empirical_p_value(embedding, stats, train_distances)` (Task 1).
- Produces: `build_calibration_report()`'s signature and `CalibrationReport`'s fields are unchanged — that function is already agnostic to what kind of p-values it receives (`fp_rate_maha`/`suggested_maha_threshold` are computed the same percentile way regardless).

- [ ] **Step 1: Write the failing test**

`build_calibration_report()`'s own tests need no changes — they pass raw `p_values`/`z_scores`/`knn_distances` arrays directly. Add this test verifying the calibration command actually calls the empirical (not chi2) function:

```python
def test_evaluate_ood_calibration_cmd_uses_empirical_not_chi2_p_value(tmp_path: Path) -> None:
    with patch(
        "src.cli.ood_calibration.mahalanobis_empirical_p_value", return_value=0.5
    ) as mock_empirical:
        result, _ = _run_successful_calibration(tmp_path, extra_args=[])
    assert result.exit_code == 0
    mock_empirical.assert_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_ood_calibration.py -k empirical_not_chi2 -v`
Expected: FAIL — `AttributeError: <module 'src.cli.ood_calibration'> does not have the attribute 'mahalanobis_empirical_p_value'` (not imported into that module yet).

- [ ] **Step 3: Implement in `src/cli/ood_calibration.py`**

Update the import:

```python
from src.ood import (
    compute_train_mahalanobis_distances,
    cosine_z_score,
    knn_mean_distance,
    load_stats,
    mahalanobis_empirical_p_value,
)
```

(replaces Task 1's `from src.ood import cosine_z_score, knn_mean_distance, load_stats, mahalanobis_chi2_p_value`)

Update `_run_ood_calibration`, right after `stats = load_stats(stats_path)`:

```python
    stats = load_stats(stats_path)
    train_distances = compute_train_mahalanobis_distances(stats)
```

And replace the `p_values` line:

```python
    p_values = np.array(
        [mahalanobis_empirical_p_value(e, stats, train_distances) for e in embeddings]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_ood_calibration.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run full check**

Run: `uv run poe fmt && uv run poe check`
Expected: PASS — every reference to the old `mahalanobis_p_value` name across `src/` is now gone (only `mahalanobis_chi2_p_value`, `mahalanobis_empirical_p_value`, and the unrelated `PredictResult.mahalanobis_p_value` field remain).

- [ ] **Step 6: Commit**

```bash
git add src/cli/ood_calibration.py tests/cli/test_ood_calibration.py
git commit -m "feat: calibrate evaluate-ood-calibration against empirical Mahalanobis p-value"
```

---

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update `README.md`'s "OOD scoring internals" table**

Find the table row for `mahalanobis_p_value` (added earlier this session) and update its description, plus add a row for the new theoretical field:

```markdown
| `mahalanobis_p_value` | Empirical (rank-based) p-value: the fraction of the training set's own documents whose Mahalanobis distance to their **own true class centroid** was at least as large as this query's distance to its **nearest** centroid. Makes no distributional assumption — replaced a chi²-based p-value after a QQ-plot check (see the playground notebook, `src/playground/pca_train_vs_predict.ipynb`) showed the underlying Gaussian/shared-covariance assumption is badly violated for this corpus (observed distances run ~5x larger than chi² predicts). | **LOW** p-value |
| `mahalanobis_p_value_theoretical` | The original chi²-based p-value, kept as a transparent secondary field — never used to decide `in_distribution`. A large gap between this and `mahalanobis_p_value` for a given document is itself diagnostic: it means the Gaussian assumption is failing badly for that document's region of the embedding space. | Informational only, not a decision input |
```

- [ ] **Step 2: Update `CLAUDE.md`'s "Key Technical Decisions" section**

Find the paragraph titled `**Mahalanobis chi-squared p-value + cosine z-score OOD detection...**` and add a follow-up paragraph immediately after it:

```markdown
**Mahalanobis empirical p-value replaces the chi² one for the actual decision (2026-07-10)**
A QQ-plot check of the training set's own Mahalanobis distances against the theoretical chi²(df) distribution showed the multivariate-Gaussian/shared-covariance assumption is badly violated for this corpus — observed distances run roughly 5x larger than chi² predicts, which is the direct explanation for Mahalanobis's measured 20-30% empirical false-positive rate against a 1% target (see `evaluate-ood-calibration` history above). `mahalanobis_p_value` is now computed empirically — `mahalanobis_empirical_p_value()` in `src/ood.py` ranks a query's raw Mahalanobis distance (to its **nearest** centroid, via `mahalanobis_min_distance()`) against `compute_train_mahalanobis_distances()`'s array of the training set's own distances (each training document to its **own true label's** centroid — deliberately not nearest-centroid, since that's what `compute_class_stats()`'s covariance estimation itself is built from; using nearest-centroid for the reference would let ambiguous/boundary training points look artificially unremarkable and corrupt the tail). This asymmetry (reference: true-label; query: nearest-centroid) is intentional, not a bug. The rank formula is the standard permutation-test empirical p-value: `(exceed_count + 1) / (N + 1)`, making no distributional assumption at all. `OOD_MAHALANOBIS_P_THRESHOLD` and `is_out_of_distribution()`'s comparison direction (`p < threshold` = anomalous) are unchanged — only how the p-value is computed changed. The original chi²-based value is kept as `mahalanobis_p_value_theoretical`, purely informational, explicitly never compared against a threshold or used in `is_out_of_distribution()` — combining it with the empirical value in the OR logic would be redundant, since both are monotonic transforms of the identical underlying distance. No `ood_stats.npz` format change or backfill was needed — `compute_train_mahalanobis_distances()` is a pure function of fields (`centroids`, `covariance_inv`, `knn_train_embeddings`, `knn_train_labels`) the file already stored for the k-NN signal. `BertTunningClassifier` computes the reference distance array once per process (`functools.cached_property`), not per request.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document Mahalanobis empirical p-value change"
```

---

### Task 6: Re-calibrate against production models (operational, not code)

**Files:** None — this is a command-line operational step, not a code change.

- [ ] **Step 1: Run calibration against BETO v2 (the default model)**

```powershell
uv run python main.py evaluate-ood-calibration `
  --model-path ./models/bert_tunning_model_beto_v2/final `
  --model beto `
  --cache-path ./data/bert_tunning_cache_con_otro_300.parquet `
  --target-fp-rate 0.01
```

Read the log output's `Mahalanobis — current threshold=..., empirical false-positive rate=...%` line. Since the empirical p-value is rank-based against the training set itself, expect this rate to already land much closer to 1% than the old chi²-based 20-30% — but confirm with the real number rather than assuming.

- [ ] **Step 2: Update `Settings.OOD_MAHALANOBIS_P_THRESHOLD` if needed**

If the reported empirical false-positive rate is not already close to the 1% target, update `src/settings.py`'s `OOD_MAHALANOBIS_P_THRESHOLD` to the `suggested_maha_threshold` value from Step 1's output, following the exact pattern already used for `OOD_COSINE_THRESHOLD`/`OOD_KNN_DISTANCE_THRESHOLD` (a comment citing the date, the model, the run's numbers).

- [ ] **Step 3: Repeat for any other production model** (e.g. BETO v1) if one is still in active use.

- [ ] **Step 4: Commit** (only if Step 2 changed `settings.py`)

```bash
git add src/settings.py
git commit -m "fix: recalibrate OOD_MAHALANOBIS_P_THRESHOLD for empirical p-value"
```
