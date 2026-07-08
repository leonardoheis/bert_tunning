# k-NN Class-Conditional OOD Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third, independent OOD/confidence signal — mean distance (in PCA space) to the `k=10` nearest training documents that share the model's *predicted* class — alongside the existing Mahalanobis chi-squared p-value and cosine z-score signals, combined with the same OR logic.

**Architecture:** Extends `ClassEmbeddingStats` to additionally persist every training document's PCA-reduced embedding and label (not just per-class centroids), enabling a class-conditional k-nearest-neighbor lookup at inference time. Reuses the existing PCA space, `ood_stats.npz` file, `compute-ood-stats`/`evaluate-ood-calibration` CLI commands, and `src/wandb.py` logging — none of those need new files, only new fields threaded through.

**Tech Stack:** Same as the existing OOD feature — `numpy` for brute-force k-NN (corpus size, ~150-300 docs per class, makes an indexed structure like FAISS/BallTree unnecessary), `scipy`/`sklearn` unchanged (no new dependency).

## Global Constraints

- Python ≥ 3.10, `X | Y` union types, not `Optional[X]`.
- Pydantic v2 for all schemas — `frozen=True`, `alias_generator=to_camel` + `populate_by_name=True` on every model touched by keyword construction (per the project's documented gotcha in `CLAUDE.md`).
- `ruff` (line-length=100) + `mypy --strict` must pass (`uv run poe check`) before every commit.
- **No retraining of existing models** — this reuses the same `[CLS]` embeddings already extracted for Mahalanobis/cosine; `compute-ood-stats` backfills existing model directories without retraining, same as before.
- **Commit/push ownership:** every task's implementer applies file changes and runs `uv run poe check`, then stops. Committing, pushing, and opening/editing a PR are the human's actions by default, performed only when explicitly delegated for that specific task.
- **Review every generated PR:** dispatch a code-review agent per this repo's policy; post PR comments only if something is worth flagging.
- **Design decision (locked in by the user, 2026-07-07):** the k-NN search is **class-conditional** — neighbors are drawn only from training documents of the class BETO/the model just predicted for this document, not the whole training set. This directly targets the failure mode already diagnosed for BETO models on this corpus: a single shared centroid+covariance badly represents heterogeneous classes (e.g. `otro`), and class-conditional k-NN makes no such shape assumption.
- **Design decision:** `evaluate-ood-calibration` uses each held-out test document's *true* label (from the cache) as the "predicted class" for k-NN purposes, not a live forward-pass prediction. This is a documented approximation — the test split is "known in-distribution by construction," so true label is a reasonable proxy for what a well-performing model would predict, and avoids adding a full classification forward pass to a command that currently only extracts embeddings.

---

## File Structure

```
src/schema.py                        MODIFY — ClassEmbeddingStats gains knn_train_embeddings/knn_train_labels;
                                                PredictResult gains knn_distance; CalibrationReport gains
                                                fp_rate_knn/suggested_knn_threshold
src/settings.py                      MODIFY — OOD_KNN_NEIGHBORS, OOD_KNN_DISTANCE_THRESHOLD
src/inference/ood.py                 MODIFY — compute_class_stats persists raw per-class embeddings;
                                                new knn_mean_distance(); save_stats/load_stats extended
src/inference/classify.py            MODIFY — predict_text computes the third signal, extends the OR
src/cli/ood_calibration.py           MODIFY — build_calibration_report/_run_ood_calibration compute the
                                                third signal's empirical FP rate + suggested threshold
src/cli/predict.py                   MODIFY — print knn_distance
src/api/routes/predict/schemas.py    MODIFY — PredictResponse gains knn_distance
src/api/routes/predict/endpoints.py  MODIFY — thread knn_distance through
src/wandb.py                         MODIFY — both logging functions include the new field/metric
tests/inference/test_ood.py          MODIFY — knn_mean_distance tests, save/load roundtrip
tests/inference/test_pipeline.py     MODIFY — predict_text triple-signal tests
tests/cli/test_ood_calibration.py    MODIFY — third-signal calibration tests
tests/api/test_predict.py            MODIFY — knn_distance passthrough test
tests/test_wandb.py                  MODIFY — new field in logged table/metrics
CLAUDE.md, README.md                 MODIFY — final documentation task
```

No new files — every change is additive to files the original OOD plan already touched, which keeps this smaller than that plan.

---

## Task 1: Storage layer — persist per-class training embeddings + core k-NN math

**Files:**
- Modify: `src/schema.py`
- Modify: `src/settings.py`
- Modify: `src/inference/ood.py`
- Test: `tests/inference/test_ood.py`

**Interfaces:**
- Consumes: nothing new — reuses `_project()`, `_reduce_dimensionality()` already in `src/inference/ood.py`
- Produces: `ClassEmbeddingStats.knn_train_embeddings: Float64Array` (shape `(n_train_docs, n_components)`), `ClassEmbeddingStats.knn_train_labels: list[int]` (length `n_train_docs`, parallel array); `knn_mean_distance(embedding, stats, predicted_label_id, *, k) -> float` — consumed by Task 2 and Task 4

- [ ] **Step 1: Write the failing tests**

Add to `tests/inference/test_ood.py`:

```python
def test_knn_mean_distance_is_zero_for_a_training_point_itself() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    # embeddings[0] is a class_a point; its own class-conditional 10-NN distance
    # should be small (it's one of its own neighbors, distance 0 to itself).
    dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=10)
    assert dist >= 0.0
    assert dist < 1.0  # class_a cluster has scale=0.1, so neighbor distances are small


def test_knn_mean_distance_is_larger_for_a_far_point() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    far_point = np.full(16, 100.0)
    near_dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=10)
    far_dist = knn_mean_distance(far_point, stats, predicted_label_id=0, k=10)
    assert far_dist > near_dist


def test_knn_mean_distance_handles_k_larger_than_class_size() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    # class_a has 20 members in the fixture; request more neighbors than exist.
    dist = knn_mean_distance(embeddings[0], stats, predicted_label_id=0, k=1000)
    assert dist >= 0.0  # falls back to using all available class members, not an error


def test_save_and_load_stats_roundtrip_includes_knn_fields(tmp_path: Path) -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = tmp_path / "ood_stats.npz"
    save_stats(stats, path)
    loaded = load_stats(path)
    np.testing.assert_allclose(loaded.knn_train_embeddings, stats.knn_train_embeddings)
    assert loaded.knn_train_labels == stats.knn_train_labels
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_ood.py -v`
Expected: FAIL with `ImportError: cannot import name 'knn_mean_distance'` / `TypeError: ClassEmbeddingStats() got unexpected keyword argument`

- [ ] **Step 3: Extend `src/schema.py`**

Modify `ClassEmbeddingStats`:

```python
class ClassEmbeddingStats(BaseModel):
    """Per-class embedding centroids + shared covariance for Mahalanobis/cosine OOD scoring,
    plus the raw per-class training embeddings needed for k-NN local-density scoring."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    class_names: list[str]
    pca_mean: Float64Array
    pca_components: Float64Array
    centroids: Float64Array
    covariance_inv: Float64Array
    cosine_calibration_mean: float
    cosine_calibration_std: float
    knn_train_embeddings: Float64Array  # (n_train_docs, n_components), PCA-reduced
    knn_train_labels: list[int]  # length n_train_docs, parallel to knn_train_embeddings
```

Modify `PredictResult` (add after `cosine_z`/`in_distribution`):

```python
    knn_distance: float | None = None
```

Modify `CalibrationReport`:

```python
class CalibrationReport(BaseModel):
    """Return value from build_calibration_report — empirical OOD threshold calibration."""

    model_config = ConfigDict(frozen=True)

    fp_rate_maha: float
    fp_rate_cosine: float
    fp_rate_knn: float
    suggested_maha_threshold: float
    suggested_cosine_threshold: float
    suggested_knn_threshold: float
```

- [ ] **Step 4: Add settings to `src/settings.py`**

Add alongside the existing `OOD_*` fields:

```python
    OOD_KNN_NEIGHBORS: int = 10
    OOD_KNN_DISTANCE_THRESHOLD: float = 5.0  # uncalibrated placeholder — run
    # evaluate-ood-calibration per-model before trusting this, same caveat as OOD_COSINE_THRESHOLD
```

- [ ] **Step 5: Modify `src/inference/ood.py`**

Modify `compute_class_stats` to persist the reduced embeddings/labels it already computes:

```python
def compute_class_stats(
    embeddings: npt.NDArray[np.float64],
    labels: list[int],
    class_names: list[str],
    *,
    n_components: int = 64,
    covariance_epsilon: float = 1e-6,
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
    )
```

Add the new scoring function (place after `cosine_z_score`):

```python
def knn_mean_distance(
    embedding: npt.NDArray[np.float64],
    stats: ClassEmbeddingStats,
    predicted_label_id: int,
    *,
    k: int = 10,
) -> float:
    """Mean Euclidean distance, in PCA space, to the k nearest training documents that share
    the predicted class. Unlike Mahalanobis (global shared covariance, assumes one Gaussian
    shape) and cosine (distance to a single centroid), this makes no assumption about the
    class's shape — it directly measures local density around the predicted class's own
    training examples, which matters for heterogeneous classes (e.g. a broad `otro`
    catch-all) that a single centroid represents poorly. A HIGH distance means anomalous —
    same comparison direction as cosine_z_score."""
    point = _project(embedding, stats)
    labels_arr = np.array(stats.knn_train_labels)
    class_points = stats.knn_train_embeddings[labels_arr == predicted_label_id]
    if class_points.shape[0] == 0:
        return float("nan")
    k_eff = min(k, class_points.shape[0])
    distances = np.linalg.norm(class_points - point, axis=1)
    nearest = np.partition(distances, k_eff - 1)[:k_eff]
    return float(nearest.mean())
```

Modify `save_stats`/`load_stats` to round-trip the two new fields:

```python
def save_stats(stats: ClassEmbeddingStats, path: Path) -> None:
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
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_ood.py -v`
Expected: PASS

- [ ] **Step 7: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/schema.py src/settings.py src/inference/ood.py tests/inference/test_ood.py
git commit -m "feat: persist per-class training embeddings and add knn_mean_distance for class-conditional OOD scoring"
```

**Manual verification note:** `compute-ood-stats`/training's `run()` call `compute_class_stats()` unchanged — no code change needed there, but every *newly generated* `ood_stats.npz` (via training or a re-run of `compute-ood-stats`) will now be larger (~1MB vs ~10KB) since it stores every training document's reduced embedding. Existing `ood_stats.npz` files from before this task **do not** have `knn_train_embeddings`/`knn_train_labels` and must be regenerated via `compute-ood-stats` before the k-NN signal will work for them — `load_stats` will raise a `KeyError` on an old file until regenerated.

---

## Task 2: Wire the third signal into `classify.py`

**Files:**
- Modify: `src/inference/classify.py`
- Test: `tests/inference/test_pipeline.py`

**Interfaces:**
- Consumes: `knn_mean_distance` (Task 1), `Settings.OOD_KNN_NEIGHBORS`/`OOD_KNN_DISTANCE_THRESHOLD` (Task 1)
- Produces: `PredictResult.knn_distance` populated; `in_distribution` now reflects a three-way OR

- [ ] **Step 1: Write the failing test**

Add to `tests/inference/test_pipeline.py` (reuse the existing `_make_stats()` fixture, which will need `knn_train_embeddings`/`knn_train_labels` added — see note below):

```python
def test_predict_text_flags_out_of_distribution_via_knn_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # A point close enough to the "decreto" centroid to pass Mahalanobis and cosine, but far
    # from every individual decreto training point stored for k-NN (a small, tight training
    # cluster far from this point) — only the k-NN signal should fire.
    ...
    result = clf.predict_text("anything")
    assert result.knn_distance is not None
    assert result.knn_distance > Settings.OOD_KNN_DISTANCE_THRESHOLD
    assert result.in_distribution is False
```

**Note:** `_make_stats()`/`_make_tight_cosine_stats()` in `tests/inference/test_pipeline.py` construct `ClassEmbeddingStats` directly and must be updated to pass `knn_train_embeddings`/`knn_train_labels` (e.g. a handful of points tightly clustered around each centroid) now that those are required fields — do this as part of this task, not a separate one, since every existing test in this file constructs stats via these two fixtures.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: FAIL — fixture missing required fields, then (after fixture fix) `AttributeError: 'PredictResult' object has no attribute 'knn_distance'`

- [ ] **Step 3: Modify `src/inference/classify.py`**

Extend the OOD block in `predict_text` (after the existing `maha_anomalous`/`cosine_anomalous` lines):

```python
        maha_p = mahalanobis_p_value(cls_embedding, self._ood_stats)
        cosine_z = cosine_z_score(cls_embedding, self._ood_stats)
        knn_dist = knn_mean_distance(
            cls_embedding, self._ood_stats, pred_idx, k=Settings.OOD_KNN_NEIGHBORS
        )
        maha_anomalous = maha_p < Settings.OOD_MAHALANOBIS_P_THRESHOLD
        cosine_anomalous = cosine_z > Settings.OOD_COSINE_THRESHOLD
        knn_anomalous = knn_dist > Settings.OOD_KNN_DISTANCE_THRESHOLD
        out_of_distribution = maha_anomalous or cosine_anomalous or knn_anomalous
        return result.model_copy(
            update={
                "mahalanobis_p_value": round(maha_p, 6),
                "cosine_z": round(cosine_z, 4),
                "knn_distance": round(knn_dist, 4),
                "in_distribution": not out_of_distribution,
            }
        )
```

Add `knn_mean_distance` to the existing `from src.inference.ood import ...` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "feat: fold k-NN class-conditional distance into predict_text's OOD OR logic"
```

---

## Task 3: Extend `evaluate-ood-calibration` for the third signal

**Files:**
- Modify: `src/cli/ood_calibration.py`
- Test: `tests/cli/test_ood_calibration.py`

**Interfaces:**
- Consumes: `knn_mean_distance` (Task 1), `CalibrationReport.fp_rate_knn`/`suggested_knn_threshold` (Task 1)
- Produces: `evaluate-ood-calibration` reports empirical FP rate + suggested threshold for k-NN alongside the existing two signals

- [ ] **Step 1: Write the failing test**

Add to `tests/cli/test_ood_calibration.py`:

```python
def test_build_calibration_report_knn_direction() -> None:
    # HIGH knn distance = anomalous, same direction as cosine — suggested threshold for a 25%
    # target rate is the 75th percentile of in-distribution distances.
    p_values = np.array([0.1, 0.2, 0.3, 0.4])
    z_scores = np.array([1.0, 2.0, 3.0, 4.0])
    knn_distances = np.array([1.0, 2.0, 3.0, 4.0])

    report = build_calibration_report(p_values, z_scores, knn_distances, target_fp_rate=0.25)

    assert report.suggested_knn_threshold == pytest.approx(np.percentile(knn_distances, 75))
    assert report.suggested_knn_threshold > np.median(knn_distances)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_ood_calibration.py -v`
Expected: FAIL — `build_calibration_report()` doesn't accept a `knn_distances` argument yet

- [ ] **Step 3: Modify `src/cli/ood_calibration.py`**

Extend `build_calibration_report`:

```python
def build_calibration_report(
    p_values: npt.NDArray[np.float64],
    z_scores: npt.NDArray[np.float64],
    knn_distances: npt.NDArray[np.float64],
    target_fp_rate: float,
) -> CalibrationReport:
    """Pure calibration math, isolated from model/IO for direct unit testing.

    Mahalanobis: LOW p-value = anomalous. Cosine and k-NN: HIGH value = anomalous — both use
    the `(1 - target_fp_rate)`-th percentile as the suggested threshold.
    """
    return CalibrationReport(
        fp_rate_maha=float(np.mean(p_values < Settings.OOD_MAHALANOBIS_P_THRESHOLD)),
        fp_rate_cosine=float(np.mean(z_scores > Settings.OOD_COSINE_THRESHOLD)),
        fp_rate_knn=float(np.mean(knn_distances > Settings.OOD_KNN_DISTANCE_THRESHOLD)),
        suggested_maha_threshold=float(np.percentile(p_values, target_fp_rate * 100)),
        suggested_cosine_threshold=float(np.percentile(z_scores, (1 - target_fp_rate) * 100)),
        suggested_knn_threshold=float(np.percentile(knn_distances, (1 - target_fp_rate) * 100)),
    )
```

In `_run_ood_calibration`, after the existing `p_values`/`z_scores` computation (using the **true label**, per the Global Constraints design decision):

```python
    label_ids = test_df["label_id"].to_numpy()
    p_values = np.array([mahalanobis_p_value(e, stats) for e in embeddings])
    z_scores = np.array([cosine_z_score(e, stats) for e in embeddings])
    knn_distances = np.array(
        [
            knn_mean_distance(e, stats, int(lbl), k=Settings.OOD_KNN_NEIGHBORS)
            for e, lbl in zip(embeddings, label_ids, strict=True)
        ]
    )
    report = build_calibration_report(p_values, z_scores, knn_distances, opts.target_fp_rate)
```

Add the corresponding `log.info(...)` lines for k-NN (mirroring the existing Mahalanobis/cosine ones), and add `knn_mean_distance` to the `from src.inference.ood import ...` line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_ood_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/cli/ood_calibration.py tests/cli/test_ood_calibration.py
git commit -m "feat: calibrate the k-NN distance threshold in evaluate-ood-calibration"
```

---

## Task 4: Surface `knn_distance` in CLI, API, and W&B logging

**Files:**
- Modify: `src/cli/predict.py`
- Modify: `src/api/routes/predict/schemas.py`
- Modify: `src/api/routes/predict/endpoints.py`
- Modify: `src/wandb.py`
- Test: `tests/api/test_predict.py`, `tests/test_wandb.py`

**Interfaces:**
- Consumes: `PredictResult.knn_distance` (Task 2), `CalibrationReport.fp_rate_knn`/`suggested_knn_threshold` (Task 3)
- Produces: `knn_distance` visible in single-document CLI output, `predict-folder` CSV (automatic via `model_dump()`, no code change needed there), `/predict` API response, and both W&B logging functions

- [ ] **Step 1: Write the failing tests**

Add to `tests/api/test_predict.py`:

```python
def test_predict_response_has_knn_field() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.knn_distance is None
```

Add to `tests/test_wandb.py` (extend the existing `test_log_ood_calibration_results_logs_summary_metrics`'s `CalibrationReport(...)` construction with `fp_rate_knn=...`/`suggested_knn_threshold=...`, and assert the new `mock_log` keys are present).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_predict.py tests/test_wandb.py -v`
Expected: FAIL — `knn_distance` not yet a `PredictResponse` field; `CalibrationReport` requires the two new fields

- [ ] **Step 3: Modify `src/cli/predict.py`**

In `predict_cmd`, extend the existing OOD-fields block:

```python
    if result.mahalanobis_p_value is not None:
        click.echo(f"  Mahalanobis p: {result.mahalanobis_p_value:.6f}")
        click.echo(f"  Cosine Z     : {result.cosine_z:.4f}")
        click.echo(f"  k-NN dist    : {result.knn_distance:.4f}")
        click.echo(f"  In-Dist.     : {result.in_distribution}")
```

- [ ] **Step 4: Modify `src/api/routes/predict/schemas.py`**

```python
    knn_distance: float | None = None
```

- [ ] **Step 5: Modify `src/api/routes/predict/endpoints.py`**

Add `knn_distance=data["knn_distance"]` to the `PredictResponse(...)` construction, and `"extracted_text": extraction.text, "extractor_used": ...` block is unaffected (already present from Task 7 of the prior plan).

- [ ] **Step 6: Modify `src/wandb.py`**

Add `"knn_distance"` to `_PREDICTION_COLUMNS` and `r.knn_distance` to the `table.add_data(...)` call in `log_predict_folder_results`. In `log_ood_calibration_results`, add to the logged dict:

```python
            "ood/fp_rate_knn": report.fp_rate_knn,
            "ood/suggested_knn_threshold": report.suggested_knn_threshold,
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_predict.py tests/test_wandb.py -v`
Expected: PASS

- [ ] **Step 8: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/cli/predict.py src/api/routes/predict/schemas.py src/api/routes/predict/endpoints.py src/wandb.py tests/api/test_predict.py tests/test_wandb.py
git commit -m "feat: surface knn_distance in CLI, API, and W&B logging"
```

---

## Task 5: Backfill verification for existing models

**Files:**
- No code changes expected — `compute-ood-stats` already calls `compute_class_stats()`, which Task 1 updated
- Test: manual verification only

**Interfaces:**
- Consumes: everything from Tasks 1-4
- Produces: confirmation that `compute-ood-stats` regenerates a working `ood_stats.npz` with the k-NN fields for every existing model, and that `evaluate-ood-calibration --log-wandb` and `predict-folder --log-wandb` work end-to-end against a regenerated file

This task exists because Task 1's note flags that **old `ood_stats.npz` files will break** (`KeyError` on `load_stats`) until regenerated — this task is the explicit checkpoint to re-run the backfill for every model directory currently in use, verify nothing else breaks, and catch anything Tasks 1-4 missed before writing documentation.

- [ ] **Step 1: Regenerate `ood_stats.npz` for each existing model**

```powershell
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model_beto/final --model beto --cache-path ./data/bert_tunning_cache_300.parquet
uv run python main.py compute-ood-stats --model-path ./models/bert_tunning_model_beto_v2/final --model beto --cache-path ./data/bert_tunning_cache_con_otro_300.parquet
# ... repeat for xlm-roberta v1/v2, minilm v1
```

Expected: succeeds, new `ood_stats.npz` is ~1MB instead of ~10KB.

- [ ] **Step 2: Run calibration against the regenerated file**

```powershell
uv run python main.py evaluate-ood-calibration --model-path ./models/bert_tunning_model_beto_v2/final --model beto --cache-path ./data/bert_tunning_cache_con_otro_300.parquet --log-wandb
```

Expected: reports empirical FP rate + suggested threshold for all three signals (Mahalanobis, cosine, k-NN); W&B run logs `ood/fp_rate_knn`/`ood/suggested_knn_threshold` alongside the existing metrics.

- [ ] **Step 3: Run a real prediction and confirm `knn_distance` appears**

```powershell
uv run python main.py predict-folder samples --model-path ./models/bert_tunning_model_beto_v2/final --output samples/results_beto_v2_knn.csv --log-wandb
```

Expected: CSV has a `knn_distance` column; W&B `predictions` table includes it.

- [ ] **Step 4: Update `Settings.OOD_KNN_DISTANCE_THRESHOLD`'s placeholder**

Based on Step 2's suggested threshold for whichever model is the project's primary one, update the default in `src/settings.py` from the `5.0` placeholder to the empirically suggested value (documented the same way `OOD_COSINE_THRESHOLD` was updated after calibration in this project's history) — or leave the placeholder and note in the commit message that per-model calibration is required before trusting the default, consistent with how `OOD_COSINE_THRESHOLD`/`OOD_MAHALANOBIS_P_THRESHOLD` are already treated.

- [ ] **Step 5: Commit** (only if Step 4 changed `settings.py`; otherwise this task produces no diff and is just a verification checkpoint)

```bash
git add src/settings.py
git commit -m "fix: calibrate OOD_KNN_DISTANCE_THRESHOLD default from empirical evaluate-ood-calibration run"
```

---

## Task 6: Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing (documentation only)
- Produces: nothing new — describes what Tasks 1-5 built

- [ ] **Step 1: Update `CLAUDE.md`**

Extend the existing "Mahalanobis chi-squared p-value + cosine z-score OOD detection" Key Technical Decisions entry (or add a new entry immediately after it) to describe the third signal:

```markdown
**k-NN class-conditional distance — a third OOD signal, complementary to Mahalanobis/cosine**
Both Mahalanobis (shared covariance) and cosine (single centroid) assume every class has one
coherent "shape." Heterogeneous classes — most notably `otro`, a broad catch-all — violate that
assumption, which is why Mahalanobis's empirically-measured false-positive rate for BETO models
came back around 20-30% instead of the intended 1% (see `evaluate-ood-calibration` history).
`knn_mean_distance()` sidesteps the shape assumption entirely: it stores every training
document's PCA-reduced embedding (not just per-class centroids) and, at inference, measures the
mean distance to the `k=10` (`OOD_KNN_NEIGHBORS`) nearest training documents *of the class the
model just predicted*. A HIGH distance is anomalous — same direction as cosine. Folded into the
existing OR logic: `in_distribution=False` if any of the three signals fire. `ood_stats.npz`
grew from ~10KB to ~1MB as a result (storing raw embeddings, not aggregates) — negligible at this
corpus size (~1,300-1,900 training docs). `OOD_KNN_DISTANCE_THRESHOLD` is calibrated the same way
as `OOD_COSINE_THRESHOLD` — run `evaluate-ood-calibration` per model before trusting the default.
```

Add to the Settings table:

```markdown
| `OOD_KNN_NEIGHBORS` | `10` | Number of same-predicted-class training documents used for the k-NN distance signal |
| `OOD_KNN_DISTANCE_THRESHOLD` | `5.0` (uncalibrated placeholder) | Mean k-NN distance above which a document is flagged `in_distribution=False` |
```

- [ ] **Step 2: Update `README.md`**

Extend the existing "Out-of-distribution detection" section's example JSON to include `knnDistance`, and add a sentence noting the three-signal OR (update the "opposite directions" note to "Mahalanobis points one direction; cosine and k-NN both point the other").

- [ ] **Step 3: Run full check**

Run: `uv run poe check`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document the k-NN class-conditional OOD signal"
```

---

## Self-Review

**Spec coverage:**
- "KNN to determine if the neighbors (at least 10) are near the embedding" → Task 1's `knn_mean_distance(..., k=10)`, `OOD_KNN_NEIGHBORS: int = 10`
- "As a 3rd option to evaluate, like OOD and cosine similarity" → folded into the same triple-signal OR in Task 2, same calibration command in Task 3, same CLI/API/W&B surfaces in Task 4
- Class-conditional scope (user's explicit design choice) → `knn_mean_distance` filters `knn_train_embeddings` by `predicted_label_id` before searching, not a global search

**Placeholder scan:** no TBD/TODO. `OOD_KNN_DISTANCE_THRESHOLD: float = 5.0` is an explicit uncalibrated placeholder, called out the same way `OOD_COSINE_THRESHOLD`'s original `2.5` default was — Task 5 exists specifically to replace it with a real calibrated value, not left unaddressed.

**Type consistency:** `knn_mean_distance(embedding, stats, predicted_label_id, *, k)` defined once in Task 1, consumed identically in Task 2 (`classify.py`, using `pred_idx`) and Task 3 (`ood_calibration.py`, using the test document's true `label_id`) — same function, same signature, two different sources for the "which class" argument, each documented. `CalibrationReport`'s three-signal shape defined in Task 1, consumed identically in Task 3's `build_calibration_report` and Task 4's `log_ood_calibration_results`.

**Task ordering / merge-conflict note:** Task 1 must land first — everything else depends on the extended `ClassEmbeddingStats`/`PredictResult`/`CalibrationReport`/`Settings`. Tasks 2 and 3 both depend only on Task 1 and touch disjoint files (`classify.py` vs. `ood_calibration.py`), so they can run in parallel once Task 1 merges. Task 4 depends on Task 2 (needs `PredictResult.knn_distance` populated) and Task 3 (needs `CalibrationReport.fp_rate_knn`). Task 5 depends on Task 4 (needs the full pipeline working to backfill/verify). Task 6 depends on everything.

**Ownership of commit/push:** every task's implementer applies file changes and runs `uv run poe check`, then stops — commit/push/PR are the human's action by default, exactly as established for the original OOD plan.
