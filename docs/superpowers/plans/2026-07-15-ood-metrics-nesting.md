# OodMetrics Nesting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `PredictResult`'s five flat, independently-`None`-able OOD score fields (`mahalanobis_p_value`, `mahalanobis_p_value_theoretical`, `cosine_z`, `knn_distance`, `tfidf_cosine_z`) plus `in_distribution` with one `ood_metrics: OodMetrics | None` field, so `None` at the outer level means exactly one thing ("no `ood_stats.npz` loaded for this model") and `ood_metrics.tfidf_cosine_z is None` means exactly one different thing ("this stats file predates the TF-IDF signal") — instead of both reasons currently colliding on the same flat `tfidf_cosine_z = None`.

**Architecture:** A new `OodMetrics` Pydantic model in `src/schema.py`, constructed once in `BertTunningClassifier.predict_text` (`src/inference/classify.py`) and threaded through as one field everywhere `PredictResult`'s OOD data currently flows: the API response schema, the CLI's printed output, and the W&B predictions table.

**Tech Stack:** Python 3.10, Pydantic v2, pytest, FastAPI, Click, wandb.

## Global Constraints

- Use `X | Y` union types, not `Optional[X]` (per CLAUDE.md).
- `alias_generator=to_camel` + `populate_by_name=True` required on every Pydantic model constructed with snake_case keyword arguments (per CLAUDE.md's documented gotcha) — `OodMetrics` needs both.
- Run `uv run poe check` (lint + typecheck + test) before every commit.
- Commit after each task; use `git add <specific files>`, never `git add -A`/`.`.
- This is a breaking change to the `/predict` API's JSON response shape — confirmed acceptable, no external consumers exist yet (pre-release).
- Full design rationale: `docs/superpowers/specs/2026-07-15-ood-metrics-nesting-design.md`.

---

### Task 1: Add `OodMetrics` and update `PredictResult`

**Files:**
- Modify: `src/schema.py`
- Test: `tests/test_schema.py`, `tests/test_settings_ood.py:17-21`

**Interfaces:**
- Produces: `OodMetrics` (Pydantic model, `src/schema.py`) with fields `mahalanobis_p_value: float`, `mahalanobis_p_value_theoretical: float`, `cosine_z: float`, `knn_distance: float`, `tfidf_cosine_z: float | None = None`, `in_distribution: bool`. `PredictResult.ood_metrics: OodMetrics | None = None` replaces the five removed flat fields plus `in_distribution`. Also produces `flatten_predict_result(result: PredictResult) -> dict[str, object]` — reconstructs a flat dict with `ood_metrics`'s fields merged back to the top level (`None`-filled when `ood_metrics is None`), for consumers that need one flat row per prediction (the W&B table in Task 5, the predict-folder CSV in Task 6) rather than a nested object. Written once here instead of duplicated in both consumers.

- [ ] **Step 1: Write the failing test for `OodMetrics` construction**

Add to `tests/test_schema.py` (new import line, new test at the end of the file):

```python
from src.schema import CalibrationReport, ClassEmbeddingStats, Hyperparams, OodMetrics
```

```python
def test_ood_metrics_tfidf_cosine_z_defaults_to_none() -> None:
    metrics = OodMetrics(
        mahalanobis_p_value=0.5,
        mahalanobis_p_value_theoretical=0.6,
        cosine_z=1.0,
        knn_distance=2.0,
        in_distribution=True,
    )
    assert metrics.tfidf_cosine_z is None


def test_flatten_predict_result_merges_ood_metrics_to_top_level() -> None:
    result = PredictResult(
        filename="a.pdf",
        label="decreto",
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=0.6,
            cosine_z=1.0,
            knn_distance=2.0,
            in_distribution=True,
        ),
    )
    row = flatten_predict_result(result)
    assert row["mahalanobis_p_value"] == 0.5  # noqa: PLR2004
    assert row["knn_distance"] == 2.0  # noqa: PLR2004
    assert row["in_distribution"] is True
    assert "ood_metrics" not in row


def test_flatten_predict_result_fills_none_when_ood_metrics_absent() -> None:
    result = PredictResult(filename="a.pdf", label="decreto")
    row = flatten_predict_result(result)
    assert row["mahalanobis_p_value"] is None
    assert row["knn_distance"] is None
    assert row["in_distribution"] is None
```

Also update this test file's import line to:

```python
from src.schema import (
    CalibrationReport,
    ClassEmbeddingStats,
    Hyperparams,
    OodMetrics,
    PredictResult,
    flatten_predict_result,
)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL with `ImportError: cannot import name 'OodMetrics'`

- [ ] **Step 3: Add `OodMetrics`, `flatten_predict_result`, and update `PredictResult` in `src/schema.py`**

In `src/schema.py`, replace the `class PredictResult(BaseModel):` block (currently lines 20-44) with:

```python
class OodMetrics(BaseModel):
    """Out-of-distribution scoring results — only present when a model has ood_stats.npz
    loaded. Nested on PredictResult (rather than five flat Optional fields) so that
    PredictResult.ood_metrics is None means exactly one thing — no ood_stats.npz loaded
    for this model — and tfidf_cosine_z's own None means exactly one different thing —
    this specific stats file predates the TF-IDF signal — instead of both reasons
    colliding on the same flat None. See
    docs/superpowers/specs/2026-07-15-ood-metrics-nesting-design.md for the full
    rationale (found via a /stop-using-none audit)."""

    model_config = ConfigDict(alias_generator=to_camel, frozen=True, populate_by_name=True)

    mahalanobis_p_value: float
    mahalanobis_p_value_theoretical: float
    cosine_z: float
    knn_distance: float
    tfidf_cosine_z: float | None = None
    in_distribution: bool


class PredictResult(BaseModel):
    """Return value from BertTunningClassifier.predict_text and predict_pdf."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        arbitrary_types_allowed=True,
        frozen=True,
        populate_by_name=True,
    )

    label: str | None = None
    confidence: float = 0.0
    certain: bool = False
    all_scores: dict[str, float] = {}
    filename: str = ""
    error: str = ""
    ood_metrics: OodMetrics | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: str = ""


_OOD_METRIC_FIELDS = (
    "mahalanobis_p_value",
    "mahalanobis_p_value_theoretical",
    "cosine_z",
    "knn_distance",
    "tfidf_cosine_z",
    "in_distribution",
)


def flatten_predict_result(result: PredictResult) -> dict[str, object]:
    """Flattens PredictResult.ood_metrics back into individual top-level keys, for
    consumers that need one flat row per prediction (the predict-folder CSV, the W&B
    predictions table) rather than a nested object -- pandas/wandb.Table don't
    recursively flatten a nested dict column/cell, so without this every OOD score would
    collapse into one unreadable stringified-dict value. None-fills every OOD field when
    ood_metrics itself is None (no ood_stats.npz loaded), matching the same shape a
    caller would have seen from the flat fields this replaced."""
    row = result.model_dump(exclude={"ood_metrics"})
    metrics = result.ood_metrics.model_dump() if result.ood_metrics is not None else {}
    for field in _OOD_METRIC_FIELDS:
        row[field] = metrics.get(field)
    return row
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (all three new tests in this file)

- [ ] **Step 5: Update the now-broken default-fields test**

Replace `tests/test_settings_ood.py:17-21`:

```python
def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.mahalanobis_p_value is None
    assert result.cosine_z is None
    assert result.in_distribution is None
```

with:

```python
def test_predict_result_ood_fields_default_to_none() -> None:
    result = PredictResult(label="decreto", confidence=0.9, certain=True)
    assert result.ood_metrics is None
```

- [ ] **Step 6: Run the whole file to confirm no other breakage yet expected**

Run: `uv run pytest tests/test_settings_ood.py tests/test_schema.py -v`
Expected: PASS (these two files are now fully migrated; other files will still fail until later tasks — that's expected at this point, don't fix them here)

- [ ] **Step 7: Commit**

```bash
git add src/schema.py tests/test_schema.py tests/test_settings_ood.py
git commit -m "feat: add OodMetrics + flatten_predict_result, nest OOD fields on PredictResult"
```

---

### Task 2: Update `BertTunningClassifier.predict_text` to build `OodMetrics`

**Files:**
- Modify: `src/inference/classify.py:298-348` (the `predict_text` method)
- Test: `tests/inference/test_pipeline.py:278-479`

**Interfaces:**
- Consumes: `OodMetrics` from Task 1 (`src/schema.py`).
- Produces: `predict_text(...) -> PredictResult` where `result.ood_metrics` is `None` when `self._ood_stats is None` (or has no k-NN training data), otherwise a fully-populated `OodMetrics`.

- [ ] **Step 1: Write the failing tests**

In `tests/inference/test_pipeline.py`, replace each OOD-field assertion with the equivalent `result.ood_metrics.x` form. Full replacements, by test:

Replace `test_predict_text_without_stats_leaves_ood_fields_none` (lines 278-284):

```python
def test_predict_text_without_stats_leaves_ood_fields_none() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None
```

Replace `test_predict_text_degrades_gracefully_when_no_knn_training_data` (lines 287-309, only the final assertion block changes):

```python
def test_predict_text_degrades_gracefully_when_no_knn_training_data() -> None:
    # ood_stats.npz with populated centroids but empty knn_train_embeddings/labels (e.g. a
    # hand-edited or partially-corrupted stats file) -- OOD scoring must be skipped entirely,
    # the same as when _ood_stats is None, not raise ValueError from downstream ranking code.
    clf = _make_mock_classifier()
    clf._ood_stats = ClassEmbeddingStats(  # noqa: SLF001
        class_names=["decreto", "ordenanza"],
        pca_mean=np.zeros(8),
        pca_components=np.eye(8),
        centroids=np.array([[0.0] * 8, [5.0] * 8]),
        covariance_inv=np.eye(8),
        cosine_calibration_mean=0.0,
        cosine_calibration_std=1.0,
        knn_train_embeddings=np.zeros((0, 8)),
        knn_train_labels=[],
    )
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None
```

Replace `test_predict_text_with_stats_populates_ood_fields` (lines 312-320):

```python
def test_predict_text_with_stats_populates_ood_fields() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert isinstance(result.ood_metrics.mahalanobis_p_value, float)
    assert isinstance(result.ood_metrics.mahalanobis_p_value_theoretical, float)
    assert isinstance(result.ood_metrics.cosine_z, float)
    assert isinstance(result.ood_metrics.in_distribution, bool)
```

Replace `test_predict_text_attaches_tfidf_cosine_z_when_available` (lines 323-343, only the final assertion changes):

```python
def test_predict_text_attaches_tfidf_cosine_z_when_available() -> None:
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
    # OodMetrics.tfidf_cosine_z stays Optional -- predict_text is where the internal NaN
    # sentinel translates back to None, same as save_stats/load_stats already do at the
    # .npz storage boundary for the scalar threshold fields.
    assert result.ood_metrics is not None
    assert result.ood_metrics.tfidf_cosine_z is not None
```

Replace `test_predict_text_leaves_tfidf_cosine_z_none_when_stats_predate_feature` (lines 346-351):

```python
def test_predict_text_leaves_tfidf_cosine_z_none_when_stats_predate_feature() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001 -- tfidf_vocabulary_terms defaults to []
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.tfidf_cosine_z is None
```

Replace `test_predict_text_mahalanobis_p_value_is_empirical` (lines 354-362):

```python
def test_predict_text_mahalanobis_p_value_is_empirical() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (mock hidden_states), exactly centroid A -- distance 0,
    # so the empirical p-value must be exactly 1.0 (all 1400 reference points have
    # distance >= 0).
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value == pytest.approx(1.0)
```

Replace `test_predict_text_in_distribution_when_matching_a_centroid_exactly` (lines 365-373):

```python
def test_predict_text_in_distribution_when_matching_a_centroid_exactly() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # first centroid in _make_stats() — i.e. a perfectly in-distribution point.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.in_distribution is True
    assert result.review_route == "accept"
```

Replace `test_predict_text_review_route_llm_judge_when_uncertain_and_in_distribution` (lines 376-387):

```python
def test_predict_text_review_route_llm_judge_when_uncertain_and_in_distribution() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # softmax([0.55, 0.45]) ≈ [0.525, 0.475], below the 0.70 threshold -- uncertain -- but
    # the [CLS] embedding (zeros, from the mock hidden_states) still matches the "decreto"
    # centroid exactly, so in_distribution stays True.
    clf.model.return_value.logits = torch.tensor([[0.55, 0.45]])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.certain is False
    assert result.ood_metrics is not None
    assert result.ood_metrics.in_distribution is True
    assert result.review_route == "llm_judge"
```

Replace `test_predict_text_review_route_accept_without_ood_stats` (lines 390-395):

```python
def test_predict_text_review_route_accept_without_ood_stats() -> None:
    clf = _make_mock_classifier()
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is None
    assert result.review_route == "accept"
```

Replace `test_predict_text_review_route_llm_judge_without_ood_stats_when_uncertain` (lines 398-405):

```python
def test_predict_text_review_route_llm_judge_without_ood_stats_when_uncertain() -> None:
    clf = _make_mock_classifier()
    clf.model.return_value.logits = torch.tensor([[0.55, 0.45]])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.certain is False
    assert result.ood_metrics is None
    assert result.review_route == "llm_judge"
```

Replace `test_predict_text_flags_out_of_distribution_via_mahalanobis_only` (lines 408-425):

```python
def test_predict_text_flags_out_of_distribution_via_mahalanobis_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats()  # noqa: SLF001
    # A point far from both centroids: with all 1400 reference (training) distances equal
    # to 0, the empirical p-value for any nonzero-distance query collapses to
    # 1 / (1400 + 1) ≈ 0.000714, safely below OOD_MAHALANOBIS_P_THRESHOLD (0.001) -- but
    # pointing in the exact same direction as centroid B, so cosine distance is ~0 and
    # only the Mahalanobis signal should fire.
    far_embedding = torch.full((1, 512, 8), 100.0)
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [far_embedding]
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value < Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"
```

Replace `test_predict_text_flags_out_of_distribution_via_cosine_only` (lines 428-445):

```python
def test_predict_text_flags_out_of_distribution_via_cosine_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_tight_cosine_stats()  # noqa: SLF001
    # A point close to centroid ["decreto"] ([5]*8) in Euclidean/Mahalanobis terms (squared
    # distance is 8, well under the chi-squared critical value for df=8) but rotated just
    # enough in direction to be several cosine-calibration standard deviations away — so
    # only the cosine signal should fire.
    embedding = torch.zeros(1, 512, 8)
    embedding[0, 0, :] = torch.tensor([6.0, 4.0, 6.0, 4.0, 6.0, 4.0, 6.0, 4.0])
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        clf.model.return_value.hidden_states = [embedding]
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value >= Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z > Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"
```

Replace `test_predict_text_flags_out_of_distribution_via_knn_only` (lines 448-463):

```python
def test_predict_text_flags_out_of_distribution_via_knn_only() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats_with_isolated_knn_cluster()  # noqa: SLF001
    # [CLS] embedding is all zeros (from the mock hidden_states), which is exactly the
    # "decreto" centroid — Mahalanobis and cosine both pass — but the "decreto" k-NN training
    # points are stored far away ([50]*8), so only the k-NN signal should fire.
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert result.ood_metrics.mahalanobis_p_value >= Settings.OOD_MAHALANOBIS_P_THRESHOLD
    assert result.ood_metrics.cosine_z <= Settings.OOD_COSINE_THRESHOLD
    assert result.ood_metrics.knn_distance > Settings.OOD_KNN_DISTANCE_THRESHOLD
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"
```

Replace `test_predict_text_flags_out_of_distribution_when_knn_distance_is_nan` (lines 466-478):

```python
def test_predict_text_flags_out_of_distribution_when_knn_distance_is_nan() -> None:
    clf = _make_mock_classifier()
    clf._ood_stats = _make_stats_with_no_knn_training_data_for_decreto()  # noqa: SLF001
    # [CLS] embedding is all zeros, exactly the "decreto" centroid — Mahalanobis and cosine
    # both pass — but "decreto" has zero k-NN training points, so knn_mean_distance returns
    # NaN. `nan > threshold` is False in Python, so without an explicit guard this would
    # silently pass as in-distribution; it must instead be treated as anomalous (fail safe).
    with patch("src.inference.classify.clean_text", return_value="cleaned text"):
        result = clf.predict_text("anything")
    assert result.ood_metrics is not None
    assert np.isnan(result.ood_metrics.knn_distance)
    assert result.ood_metrics.in_distribution is False
    assert result.review_route == "human_review"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_pipeline.py -k predict_text -v`
Expected: FAIL — `AttributeError: 'PredictResult' object has no attribute 'ood_metrics'` (the field doesn't populate yet; `predict_text` still sets the old flat fields, which no longer exist on the model after Task 1, so this will actually fail at `PredictResult` construction inside `classify.py` first — that's fine, confirms the test exercises the right code path)

- [ ] **Step 3: Update `predict_text` in `src/inference/classify.py`**

Add `OodMetrics` to the import from `src.schema` (currently `from src.schema import ClassEmbeddingStats, PredictResult`):

```python
from src.schema import ClassEmbeddingStats, OodMetrics, PredictResult
```

Replace the final `return result.model_copy(update={...})` block (the last statement of `predict_text`, currently):

```python
        return result.model_copy(
            update={
                "mahalanobis_p_value": round(scores.mahalanobis_p, 6),
                "mahalanobis_p_value_theoretical": round(maha_p_theoretical, 6),
                "cosine_z": round(scores.cosine_z, 4),
                "knn_distance": round(scores.knn_distance, 4),
                "tfidf_cosine_z": (
                    None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)
                ),
                "in_distribution": in_distribution,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(in_distribution=in_distribution),
                ),
            }
        )
```

with:

```python
        ood_metrics = OodMetrics(
            mahalanobis_p_value=round(scores.mahalanobis_p, 6),
            mahalanobis_p_value_theoretical=round(maha_p_theoretical, 6),
            cosine_z=round(scores.cosine_z, 4),
            knn_distance=round(scores.knn_distance, 4),
            tfidf_cosine_z=(
                None if np.isnan(scores.tfidf_cosine_z) else round(scores.tfidf_cosine_z, 4)
            ),
            in_distribution=in_distribution,
        )
        return result.model_copy(
            update={
                "ood_metrics": ood_metrics,
                "review_route": decide_review_route(
                    confidence_tier=confidence_tier,
                    ood_evidence=OodEvidence.from_in_distribution(in_distribution=in_distribution),
                ),
            }
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: PASS (all tests in this file, including the `is_out_of_distribution`/`OodEvidence` tests at the top which were never affected)

- [ ] **Step 5: Commit**

```bash
git add src/inference/classify.py tests/inference/test_pipeline.py
git commit -m "feat: build OodMetrics in predict_text instead of five flat updates"
```

---

### Task 3: Update the `/predict` API response

**Files:**
- Modify: `src/api/routes/predict/schemas.py`, `src/api/routes/predict/endpoints.py:47-64`
- Test: `tests/api/test_predict.py:31-91`

**Interfaces:**
- Consumes: `OodMetrics` from Task 1, `PredictResult.ood_metrics` from Task 2.
- Produces: `PredictResponse.ood_metrics: OodMetrics | None` — same type reused directly, no separate API-layer class.

- [ ] **Step 1: Write the failing tests**

Replace `tests/api/test_predict.py:31-40` (`test_predict_response_has_ood_fields`, `test_predict_response_has_knn_field`):

```python
def test_predict_response_has_ood_metrics_field() -> None:
    response = PredictResponse(filename="doc.pdf", label="decreto", confidence=0.9, certain=True)
    assert response.ood_metrics is None
```

Replace `tests/api/test_predict.py:69-91` (`test_predict_endpoint_returns_knn_distance`):

```python
def test_predict_endpoint_returns_ood_metrics() -> None:
    expected_knn_distance = 4.2
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=0.6,
            cosine_z=1.0,
            knn_distance=expected_knn_distance,
            in_distribution=True,
        ),
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
    assert response.json()["oodMetrics"]["knnDistance"] == expected_knn_distance
```

Update the import line at the top of `tests/api/test_predict.py` from:

```python
from src.schema import ExtractionMetadata, PredictResult
```

to:

```python
from src.schema import ExtractionMetadata, OodMetrics, PredictResult
```

Replace `tests/api/test_predict.py:135-158` (`test_predict_endpoint_returns_theoretical_mahalanobis_p_value`):

```python
def test_predict_endpoint_returns_theoretical_mahalanobis_p_value() -> None:
    expected_theoretical_p = 0.1708
    app = create_app(model_path="fake/path")
    mock_clf = MagicMock()
    mock_clf.predict_text.return_value = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.5,
            mahalanobis_p_value_theoretical=expected_theoretical_p,
            cosine_z=1.0,
            knn_distance=2.0,
            in_distribution=True,
        ),
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
    assert response.json()["oodMetrics"]["mahalanobisPValueTheoretical"] == expected_theoretical_p
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/api/test_predict.py -v`
Expected: FAIL — `PredictResponse`/`PredictResult` don't accept `ood_metrics` yet (or `OodMetrics` isn't importable from the fixture's perspective until Task 1/2 land, which they already have by this point — the failure here is specifically that `PredictResponse` has no `ood_metrics` field and still declares the five old flat ones)

- [ ] **Step 3: Update `src/api/routes/predict/schemas.py`**

Replace the whole file:

```python
from pydantic import Field

from src.api.schema import BaseSchema
from src.schema import OodMetrics


class PredictResponse(BaseSchema):
    filename: str
    label: str | None
    confidence: float
    certain: bool
    all_scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    ood_metrics: OodMetrics | None = None
    extracted_text: str = ""
    extractor_used: str = ""
    review_route: str = ""
```

- [ ] **Step 4: Update `src/api/routes/predict/endpoints.py`**

Replace `_to_predict_response` (lines 47-64):

```python
def _to_predict_response(result: PredictResult) -> PredictResponse:
    data = result.model_dump()
    return PredictResponse(
        filename=data["filename"],
        label=data["label"],
        confidence=data["confidence"],
        certain=data["certain"],
        all_scores=data["all_scores"],
        error=data["error"] or None,
        ood_metrics=result.ood_metrics,
        extracted_text=data["extracted_text"],
        extractor_used=data["extractor_used"],
        review_route=data["review_route"],
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/api/test_predict.py -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite to catch any remaining breakage**

Run: `uv run pytest -v 2>&1 | tail -60`
Expected: Only `tests/cli/test_commands.py` and `tests/test_wandb.py` still failing at this point (Tasks 4 and 5) — everything else passes

- [ ] **Step 7: Commit**

```bash
git add src/api/routes/predict/schemas.py src/api/routes/predict/endpoints.py tests/api/test_predict.py
git commit -m "feat: surface ood_metrics on the /predict API response"
```

---

### Task 4: Update the CLI's printed output

**Files:**
- Modify: `src/cli/predict.py:56-63`
- Test: `tests/cli/test_commands.py:66-84`

**Interfaces:**
- Consumes: `PredictResult.ood_metrics` from Task 2.

- [ ] **Step 1: Update `src/cli/predict.py`**

Replace the OOD-printing block inside `predict_cmd` (currently):

```python
    if result.mahalanobis_p_value is not None:
        click.echo(f"  Mahalanobis p: {result.mahalanobis_p_value:.6f}")
        theoretical = result.mahalanobis_p_value_theoretical
        theoretical_str = f"{theoretical:.6f}" if theoretical is not None else "n/a"
        click.echo(f"  Mahalanobis p (chi2, theoretical): {theoretical_str}")
        click.echo(f"  Cosine Z     : {result.cosine_z:.4f}")
        click.echo(f"  k-NN dist    : {result.knn_distance:.4f}")
        click.echo(f"  In-Dist.     : {result.in_distribution}")
```

with:

```python
    if result.ood_metrics is not None:
        m = result.ood_metrics
        click.echo(f"  Mahalanobis p: {m.mahalanobis_p_value:.6f}")
        click.echo(f"  Mahalanobis p (chi2, theoretical): {m.mahalanobis_p_value_theoretical:.6f}")
        click.echo(f"  Cosine Z     : {m.cosine_z:.4f}")
        click.echo(f"  k-NN dist    : {m.knn_distance:.4f}")
        click.echo(f"  In-Dist.     : {m.in_distribution}")
```

`mahalanobis_p_value_theoretical` no longer needs an `"n/a"` fallback: it's a required (non-`Optional`) field on `OodMetrics`, always populated whenever `ood_metrics` itself is not `None` — a state the old flat-field design could accidentally represent (fields set independently) but the new nested design cannot.

- [ ] **Step 2: Delete the now-impossible-to-construct test**

`tests/cli/test_commands.py:66-84` (`test_predict_cmd_echoes_n_a_when_theoretical_p_value_missing`) constructs a `PredictResult` with `mahalanobis_p_value_theoretical=None` while `mahalanobis_p_value`/`cosine_z`/`knn_distance`/`in_distribution` are set — exactly the physically-impossible-in-production state this refactor eliminates (see Task 1's docstring). Under `OodMetrics`, `mahalanobis_p_value_theoretical` is required, so this fixture no longer type-checks. Delete the whole test function:

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

- [ ] **Step 3: Add a replacement test that exercises the real code path**

Add to `tests/cli/test_commands.py` (needs `OodMetrics` added to the existing `from src.schema import ...` import):

```python
def test_predict_cmd_prints_ood_metrics_when_present(tmp_path: Path) -> None:
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake content")
    fake_result = PredictResult(
        label="decreto",
        confidence=0.9,
        certain=True,
        ood_metrics=OodMetrics(
            mahalanobis_p_value=0.005,
            mahalanobis_p_value_theoretical=0.017,
            cosine_z=0.1,
            knn_distance=1.0,
            in_distribution=False,
        ),
    )

    with patch("src.cli.predict.predict_pdf", return_value=fake_result):
        result = CliRunner().invoke(predict_cmd, [str(pdf_path)])

    assert result.exit_code == 0
    assert "Mahalanobis p (chi2, theoretical): 0.017000" in result.output
    assert "In-Dist.     : False" in result.output
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_commands.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/predict.py tests/cli/test_commands.py
git commit -m "feat: print ood_metrics in predict CLI, drop impossible n/a fallback"
```

---

### Task 5: Update the W&B predictions table

**Files:**
- Modify: `src/wandb.py:69-91`
- Test: `tests/test_wandb.py:39-129`

**Interfaces:**
- Consumes: `PredictResult.ood_metrics` from Task 2.
- Produces: `log_predict_folder_results` unchanged signature; `_PREDICTION_COLUMNS` unchanged names (per the approved design, no `ood_metrics_` prefix).

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_wandb.py:39-59` (`test_log_predict_folder_results_table_includes_knn_distance_column`):

```python
def test_log_predict_folder_results_table_includes_knn_distance_column() -> None:
    expected_knn_distance = 4.2
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            ood_metrics=OodMetrics(
                mahalanobis_p_value=0.5,
                mahalanobis_p_value_theoretical=0.6,
                cosine_z=1.0,
                knn_distance=expected_knn_distance,
                in_distribution=True,
            ),
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

    assert _logged_row(mock_table_cls, mock_table)["knn_distance"] == expected_knn_distance
```

Replace `tests/test_wandb.py:62-82` (`test_log_predict_folder_results_table_includes_tfidf_cosine_z_column`):

```python
def test_log_predict_folder_results_table_includes_tfidf_cosine_z_column() -> None:
    expected_tfidf_cosine_z = 3.28
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            ood_metrics=OodMetrics(
                mahalanobis_p_value=0.5,
                mahalanobis_p_value_theoretical=0.6,
                cosine_z=1.0,
                knn_distance=2.0,
                tfidf_cosine_z=expected_tfidf_cosine_z,
                in_distribution=True,
            ),
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

    assert _logged_row(mock_table_cls, mock_table)["tfidf_cosine_z"] == expected_tfidf_cosine_z
```

`tests/test_wandb.py:85-104` (`test_log_predict_folder_results_table_includes_review_route_column`) is unaffected — leave as-is, it only sets `review_route`, a field untouched by this refactor.

Replace `tests/test_wandb.py:107-128` (`test_log_predict_folder_results_table_includes_theoretical_mahalanobis_column`):

```python
def test_log_predict_folder_results_table_includes_theoretical_mahalanobis_column() -> None:
    expected_theoretical_p = 0.1708
    results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            ood_metrics=OodMetrics(
                mahalanobis_p_value=0.5,
                mahalanobis_p_value_theoretical=expected_theoretical_p,
                cosine_z=1.0,
                knn_distance=2.0,
                in_distribution=True,
            ),
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

Add `OodMetrics` to the existing `from src.schema import ...` import at the top of `tests/test_wandb.py`.

Also add one new test proving a `None` `ood_metrics` fills every OOD column with `None` rather than raising:

```python
def test_log_predict_folder_results_table_handles_missing_ood_metrics() -> None:
    results = [
        PredictResult(filename="a.pdf", label="decreto", confidence=0.9, certain=True),
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
    assert row["mahalanobis_p_value"] is None
    assert row["knn_distance"] is None
    assert row["in_distribution"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wandb.py -v`
Expected: FAIL — `PredictResult(..., ood_metrics=...)` works fine (Task 1/2 already landed), but `log_predict_folder_results`'s row-building still does a flat `model_dump()` lookup that no longer has top-level `knn_distance`/etc. keys, so `KeyError` is raised

- [ ] **Step 3: Update `src/wandb.py`**

Change the import line from:

```python
from src.schema import CalibrationReport, EvaluationResult, Hyperparams, PredictResult
```

to:

```python
from src.schema import (
    CalibrationReport,
    EvaluationResult,
    Hyperparams,
    PredictResult,
    flatten_predict_result,
)
```

Replace the loop body (currently lines 80-83):

```python
    table = wandb.Table(columns=_PREDICTION_COLUMNS)
    for r in results:
        row = r.model_dump()
        table.add_data(*(row[col] for col in _PREDICTION_COLUMNS))
```

with:

```python
    table = wandb.Table(columns=_PREDICTION_COLUMNS)
    for r in results:
        row = flatten_predict_result(r)
        table.add_data(*(row[col] for col in _PREDICTION_COLUMNS))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wandb.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/wandb.py tests/test_wandb.py
git commit -m "feat: flatten ood_metrics back into the W&B predictions table"
```

---

### Task 6: Fix the `predict-folder` CSV writer

**Files:**
- Modify: `src/cli/predict.py:84-85` (`_run_predict_folder`)
- Test: `tests/cli/test_commands.py`

**Interfaces:**
- Consumes: `flatten_predict_result` from Task 1.

**Why this task exists:** the approved design spec (`docs/superpowers/specs/2026-07-15-ood-metrics-nesting-design.md`) covered the API, CLI printing, and the W&B table, but missed that `_run_predict_folder` also flattens `PredictResult` via `pd.DataFrame([r.model_dump() for r in results])` to build the `predict-folder` CSV — found during this plan's self-review. Confirmed by direct test: a nested `ood_metrics` dict column serializes to CSV as one stringified-dict cell (`"{'mahalanobis_p_value': 0.5, ...}"`), not separate columns. `predict-folder`'s CSV is likely this project's most-used OOD output (no API deployment yet), so this is not optional.

- [ ] **Step 1: Write the failing test**

Add to `tests/cli/test_commands.py` (needs `OodMetrics` added to the existing `from src.schema import ...` import, and `import pandas as pd` at the top):

```python
def test_predict_folder_cmd_writes_flat_ood_columns_to_csv(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    output = tmp_path / "results.csv"
    fake_results = [
        PredictResult(
            filename="a.pdf",
            label="decreto",
            confidence=0.9,
            certain=True,
            ood_metrics=OodMetrics(
                mahalanobis_p_value=0.5,
                mahalanobis_p_value_theoretical=0.6,
                cosine_z=1.0,
                knn_distance=2.0,
                in_distribution=True,
            ),
        ),
    ]

    with patch("src.cli.predict.predict_folder", return_value=fake_results):
        result = CliRunner().invoke(
            predict_folder_cmd, ["--output", str(output), str(folder)]
        )

    assert result.exit_code == 0
    df = pd.read_csv(output)
    assert "ood_metrics" not in df.columns
    assert df["knnDistance"].iloc[0] == 2.0  # noqa: PLR2004
```

Note: `flatten_predict_result` returns snake_case keys (`knn_distance`), but `pd.DataFrame.to_csv` writes whatever keys the dict has — the existing CSV output has always used snake_case column headers (`model_dump()` without `by_alias=True`), not camelCase. Adjust the assertion to `df["knn_distance"].iloc[0]` to match the pre-existing convention — **do not** introduce a `by_alias=True` change here, that's a separate, unrelated concern outside this plan's scope.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cli/test_commands.py::test_predict_folder_cmd_writes_flat_ood_columns_to_csv -v`
Expected: FAIL — `assert "ood_metrics" not in df.columns` fails, since it's still a single nested-dict column

- [ ] **Step 3: Update `src/cli/predict.py`**

Add `flatten_predict_result` to the existing `from src.schema import ...`-style import — check the current import block at the top of `src/cli/predict.py` (it currently imports from `src.inference.pipeline`, `src.logger`, `src.settings`, `src.wandb`, but not directly from `src.schema`) and add:

```python
from src.schema import flatten_predict_result
```

Replace line 85 (currently `df = pd.DataFrame([r.model_dump() for r in results])`) with:

```python
    df = pd.DataFrame([flatten_predict_result(r) for r in results])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cli/test_commands.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cli/predict.py tests/cli/test_commands.py
git commit -m "fix: flatten ood_metrics back into individual predict-folder CSV columns"
```

---

### Task 7: Full verification, real-sample check, and PR

**Files:** none (verification only)

- [ ] **Step 1: Run the full check suite**

Run: `uv run poe check`
Expected: All lint/typecheck/test pass, 0 failures.

- [ ] **Step 2: Verify against real samples**

Run:

```bash
uv run python main.py predict-folder samples --model-path ./models/bert_tunning_model_beto_v2/final --output samples/predict_result_ood_metrics_check.csv
```

Then confirm the CSV's OOD columns are flat (Task 6 fixed the flattening), not a nested stringified dict:

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('samples/predict_result_ood_metrics_check.csv')
assert 'ood_metrics' not in df.columns, 'flatten_predict_result regression'
print(df[['filename', 'label', 'mahalanobis_p_value', 'cosine_z', 'knn_distance', 'in_distribution']].to_string(index=False))
"
```

- [ ] **Step 3: Clean up the verification CSV**

```bash
rm -f samples/predict_result_ood_metrics_check.csv
```

- [ ] **Step 4: Push and open the PR**

```bash
git push origin task/50-ood-metrics-nesting
gh pr create --base task/50-tfidf-ood-signal --head task/50-ood-metrics-nesting --title "feat: nest OOD score fields into OodMetrics to fix tfidf_cosine_z's overloaded None" --body "$(cat <<'EOF'
## Summary
- Found via a /stop-using-none audit: PredictResult.tfidf_cosine_z returned None for two indistinguishable reasons (no ood_stats.npz loaded at all, vs. loaded but this stats file predates TF-IDF).
- Replaces the five flat OOD score fields (mahalanobis_p_value, mahalanobis_p_value_theoretical, cosine_z, knn_distance, tfidf_cosine_z) plus in_distribution with one `ood_metrics: OodMetrics | None` field.
- Breaking change to the /predict API response shape (flat fields -> nested `oodMetrics` object) -- confirmed acceptable, no external consumers yet.
- Design spec: docs/superpowers/specs/2026-07-15-ood-metrics-nesting-design.md

## Test plan
- [x] `uv run poe check`
- [x] Verified against samples/

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Dispatch code review**

Per CLAUDE.md's review policy, run the `code-review` skill against the opened PR (Standards axis; Spec axis skipped, no tracked spec/issue for this conversational work) before considering the task done.
