# Mahalanobis Threshold Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ambiguous `mahalanobis_p_threshold is None` check with an explicit status field, so "never calibrated" and "calibration ran but the degenerate-threshold guard correctly refused to persist a value" are distinguishable in the type instead of only in prose comments.

**Architecture:** Add a `Literal["not_calibrated", "calibrated", "refused_degenerate"]` field to `ClassEmbeddingStats`, defaulting to `"not_calibrated"`. `evaluate-ood-calibration --write-thresholds` sets it explicitly. `save_stats`/`load_stats` persist/round-trip it (legacy files without the key default to `"not_calibrated"`, same backward-compat idiom as the existing threshold fields). `BertTunningClassifier._warn_on_uncalibrated_thresholds` branches on the status instead of `is None`, so the WARNING only fires for the genuinely actionable case and a separate non-actionable INFO line covers the expected-refusal case.

**Tech Stack:** Pydantic v2 (`Literal`), numpy `.npz` persistence, pytest.

## Global Constraints

- `ood_stats.npz` is written with `allow_pickle=False` — only primitive arrays/scalars may be persisted, no pickled objects. The new field must serialize as a plain string, matching the existing `model_type` field's pattern.
- Backward compatibility: any `ood_stats.npz` written before this change is missing the new key entirely (not even as an empty/NaN sentinel) — `load_stats` must not `KeyError` on it.
- `uv run poe check` (ruff + ruff format + mypy strict + pytest) must pass after every task.
- Do not touch `cosine_threshold`/`knn_distance_threshold` — only `mahalanobis_p_threshold` has the two-meanings problem (confirmed: the degenerate-threshold guard in `_write_calibrated_thresholds` only ever refuses to write the Mahalanobis value; cosine/knn are always written together on a successful `--write-thresholds` run).

---

### Task 1: Schema field + persistence round-trip

**Files:**
- Modify: `src/schema.py` (`ClassEmbeddingStats`)
- Modify: `src/ood.py` (`save_stats`, `load_stats`)
- Test: `tests/test_ood.py`

**Interfaces:**
- Produces: `ClassEmbeddingStats.mahalanobis_threshold_status: Literal["not_calibrated", "calibrated", "refused_degenerate"]`, default `"not_calibrated"`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ood.py` (near the existing `test_save_and_load_stats_roundtrip_includes_thresholds` / `test_load_stats_handles_legacy_file_without_threshold_fields`):

```python
def test_save_and_load_stats_roundtrip_includes_threshold_status() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8).model_copy(
        update={"mahalanobis_threshold_status": "refused_degenerate"}
    )
    path = Path("test_stats_threshold_status.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_threshold_status == "refused_degenerate"
    finally:
        path.unlink(missing_ok=True)


def test_save_and_load_stats_roundtrip_threshold_status_defaults_to_not_calibrated() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_threshold_status_default.npz")
    try:
        save_stats(stats, path)
        loaded = load_stats(path)
        assert loaded.mahalanobis_threshold_status == "not_calibrated"
    finally:
        path.unlink(missing_ok=True)


def test_load_stats_handles_legacy_file_without_threshold_status_field() -> None:
    # A pre-this-change ood_stats.npz has no mahalanobis_threshold_status key at all --
    # load_stats must not KeyError, and must default to "not_calibrated".
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_legacy_threshold_status.npz")
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
        assert loaded.mahalanobis_threshold_status == "not_calibrated"
    finally:
        path.unlink(missing_ok=True)


def test_load_stats_rejects_unknown_threshold_status() -> None:
    embeddings, labels, class_names = _synthetic_embeddings()
    stats = compute_class_stats(embeddings, labels, class_names, n_components=8)
    path = Path("test_stats_bad_threshold_status.npz")
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
            mahalanobis_threshold_status="not_a_real_status",
        )
        with pytest.raises(BertTunningError, match="mahalanobis_threshold_status"):
            load_stats(path)
    finally:
        path.unlink(missing_ok=True)
```

Add `from src.exceptions import BertTunningError` to `tests/test_ood.py`'s imports if not already present (check first — it is not).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ood.py -k threshold_status -v`
Expected: FAIL — `TypeError`/`ValidationError` (`mahalanobis_threshold_status` not a field of `ClassEmbeddingStats` yet) or `AttributeError`.

- [ ] **Step 3: Add the field to `ClassEmbeddingStats`**

In `src/schema.py`, add `Literal` to the `typing` import:

```python
from typing import Annotated, Literal
```

Add the field inside `ClassEmbeddingStats`, right after `knn_distance_threshold`:

```python
    knn_distance_threshold: float | None = None
    # Distinguishes *why* mahalanobis_p_threshold is None -- "not_calibrated" (nobody has run
    # evaluate-ood-calibration --write-thresholds yet, an operator should) from
    # "refused_degenerate" (it WAS run, but the degenerate-threshold guard in
    # cli/ood_calibration.py correctly refused to persist a floor-adjacent value -- expected,
    # no action needed, will keep recurring). Without this, both states collapse to the same
    # None and BertTunningClassifier's startup warning can't tell them apart -- see
    # _warn_on_uncalibrated_thresholds. cosine_threshold/knn_distance_threshold don't need an
    # equivalent field: the degenerate guard only ever applies to the Mahalanobis threshold.
    mahalanobis_threshold_status: Literal["not_calibrated", "calibrated", "refused_degenerate"] = (
        "not_calibrated"
    )
```

- [ ] **Step 4: Wire persistence in `src/ood.py`**

Add `from src.exceptions import BertTunningError` to `src/ood.py`'s imports (alongside `from src.schema import ClassEmbeddingStats`).

In `save_stats`, add the new field to the `np.savez` call (it's never `None`, so no sentinel needed — same treatment as a required string):

```python
                model_type=model_type,
                model_hidden_size=model_hidden_size,
                mahalanobis_threshold_status=stats.mahalanobis_threshold_status,
            )
```

Add a decoder function near `_optional_int` (same section):

```python
def _threshold_status(
    data: npt.NDArray[np.str_],
) -> Literal["not_calibrated", "calibrated", "refused_degenerate"]:
    value = str(data)
    match value:
        case "not_calibrated" | "calibrated" | "refused_degenerate":
            return value
        case _:
            msg = f"ood_stats.npz has an unrecognized mahalanobis_threshold_status: {value!r}"
            raise BertTunningError(msg)
```

Add `Literal` to `src/ood.py`'s `typing` import: `from typing import Literal, NamedTuple`.

In `load_stats`, add the field to the `ClassEmbeddingStats(...)` construction (after `model_hidden_size`):

```python
        model_hidden_size=_optional_int(data["model_hidden_size"])
        if "model_hidden_size" in data.files
        else None,
        mahalanobis_threshold_status=_threshold_status(data["mahalanobis_threshold_status"])
        if "mahalanobis_threshold_status" in data.files
        else "not_calibrated",
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_ood.py -k threshold_status -v`
Expected: PASS (4 tests)

Run: `uv run pytest tests/test_ood.py -v`
Expected: PASS (all — confirms the new field didn't break existing round-trip/legacy-load tests)

- [ ] **Step 6: Commit**

```bash
git add src/schema.py src/ood.py tests/test_ood.py
git commit -m "feat: add mahalanobis_threshold_status to disambiguate why a threshold is unset"
```

---

### Task 2: `evaluate-ood-calibration --write-thresholds` sets the status

**Files:**
- Modify: `src/cli/ood_calibration.py` (`_write_calibrated_thresholds`)
- Test: `tests/cli/test_ood_calibration.py`

**Interfaces:**
- Consumes: `ClassEmbeddingStats.mahalanobis_threshold_status` (Task 1).
- Produces: `_write_calibrated_thresholds` now also sets `mahalanobis_threshold_status` on the persisted stats — `"calibrated"` on a normal write, `"refused_degenerate"` only when the guard fires **and** there is still no usable value (an existing previously-calibrated value being kept counts as `"calibrated"`, not a refusal).

- [ ] **Step 1: Write the failing tests**

Add to `tests/cli/test_ood_calibration.py`, extending the two existing tests that exercise this path:

```python
def test_evaluate_ood_calibration_cmd_write_thresholds_persists_calibrated_status(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    result, _ = _run_successful_calibration(tmp_path, extra_args=["--write-thresholds"])
    assert result.exit_code == 0
    written = load_stats(stats_path)
    assert written.mahalanobis_threshold_status == "calibrated"


def test_evaluate_ood_calibration_cmd_write_thresholds_refused_status_when_degenerate(
    tmp_path: Path,
) -> None:
    tiny_stats = _make_stats().model_copy(
        update={
            "centroids": np.array([[3.0] * 8, [8.0] * 8]),
            "knn_train_embeddings": np.array([[3.0] * 8] * 2 + [[8.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_threshold_status == "refused_degenerate"


def test_evaluate_ood_calibration_cmd_write_thresholds_keeps_calibrated_status_when_refusal_keeps_existing_value(
    tmp_path: Path,
) -> None:
    # Guard refuses the new suggestion but a real value from a PRIOR successful calibration
    # already exists -- status must stay "calibrated", not flip to "refused_degenerate",
    # since nothing about the currently-persisted value actually changed or became invalid.
    tiny_stats = _make_stats().model_copy(
        update={
            "centroids": np.array([[3.0] * 8, [8.0] * 8]),
            "knn_train_embeddings": np.array([[3.0] * 8] * 2 + [[8.0] * 8] * 2),
            "knn_train_labels": [0, 0, 1, 1],
            "mahalanobis_p_threshold": 0.0005,
            "mahalanobis_threshold_status": "calibrated",
        }
    )
    result, _ = _run_calibration_with_stats_write_thresholds(tmp_path, tiny_stats)
    assert result.exit_code == 0
    stats_path = tmp_path / "fake-model" / "ood_stats.npz"
    written = load_stats(stats_path)
    assert written.mahalanobis_p_threshold == pytest.approx(0.0005)
    assert written.mahalanobis_threshold_status == "calibrated"
```

Check `tests/cli/test_ood_calibration.py`'s existing imports include `pytest` (it does, per current file).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/cli/test_ood_calibration.py -k "calibrated_status or refused_status or keeps_calibrated" -v`
Expected: FAIL — `mahalanobis_threshold_status` stays at its schema default (`"not_calibrated"`) regardless of path, since `_write_calibrated_thresholds` doesn't set it yet.

- [ ] **Step 3: Update `_write_calibrated_thresholds`**

In `src/cli/ood_calibration.py`, replace:

```python
    floor = 1 / (n_train + 1)
    suggested_maha_threshold = report.suggested_maha_threshold
    maha_threshold: float | None = suggested_maha_threshold
    if suggested_maha_threshold <= floor:
        log.warning(
            "Refusing to write suggested Mahalanobis threshold %.6f: at or below this "
            "model's empirical resolution floor %.6f (n_train=%d). The signal would never "
            "fire. Keeping the existing value (%s).",
            suggested_maha_threshold,
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
```

with:

```python
    floor = 1 / (n_train + 1)
    suggested_maha_threshold = report.suggested_maha_threshold
    maha_threshold: float | None = suggested_maha_threshold
    maha_status: Literal["calibrated", "refused_degenerate"] = "calibrated"
    if suggested_maha_threshold <= floor:
        log.warning(
            "Refusing to write suggested Mahalanobis threshold %.6f: at or below this "
            "model's empirical resolution floor %.6f (n_train=%d). The signal would never "
            "fire. Keeping the existing value (%s).",
            suggested_maha_threshold,
            floor,
            n_train,
            stats.mahalanobis_p_threshold,
        )
        maha_threshold = stats.mahalanobis_p_threshold
        # A kept prior value is still "calibrated" -- only truly unset (never calibrated
        # before, and now also refused) becomes "refused_degenerate".
        maha_status = "calibrated" if maha_threshold is not None else "refused_degenerate"

    updated = stats.model_copy(
        update={
            "mahalanobis_p_threshold": maha_threshold,
            "mahalanobis_threshold_status": maha_status,
            "cosine_threshold": report.suggested_cosine_threshold,
            "knn_distance_threshold": report.suggested_knn_threshold,
        }
    )
```

Add `Literal` to `src/cli/ood_calibration.py`'s imports: `from typing import Literal` (new import line, near the top with the other stdlib-ish imports).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/cli/test_ood_calibration.py -v`
Expected: PASS (all — including the 3 new tests and the pre-existing `test_evaluate_ood_calibration_cmd_write_thresholds_refuses_degenerate_maha_threshold`, which is unaffected since it doesn't assert on status)

- [ ] **Step 5: Commit**

```bash
git add src/cli/ood_calibration.py tests/cli/test_ood_calibration.py
git commit -m "feat: set mahalanobis_threshold_status when writing calibrated thresholds"
```

---

### Task 3: Classifier warning branches on status, not `is None`

**Files:**
- Modify: `src/inference/classify.py` (`_warn_on_uncalibrated_thresholds`)
- Modify: `tests/inference/test_pipeline.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: `ClassEmbeddingStats.mahalanobis_threshold_status` (Task 1).

- [ ] **Step 1: Write the failing test**

In `tests/inference/test_pipeline.py`, update `test_classifier_does_not_warn_when_thresholds_are_fully_calibrated` (around line 594) to also set the new status field — required now, since the classifier will branch on status instead of the raw value:

```python
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
            "mahalanobis_threshold_status": "calibrated",
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
```

Add a new test right after it:

```python
def test_classifier_logs_info_not_warning_when_mahalanobis_threshold_refused_degenerate(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    model = MagicMock()
    model.config.id2label = {0: "decreto", 1: "ordenanza"}
    model.config.max_position_embeddings = 512

    stats = _make_stats().model_copy(
        update={
            "mahalanobis_threshold_status": "refused_degenerate",
            "cosine_threshold": 13.7366,
            "knn_distance_threshold": 16.7908,
        }
    )
    save_stats(stats, tmp_path / "ood_stats.npz")

    with (
        patch("torch.cuda.is_available", return_value=False),
        caplog.at_level(logging.INFO),
    ):
        BertTunningClassifier(str(tmp_path), tokenizer=tokenizer, model=model)

    assert not any(
        record.levelno == logging.WARNING and "mahalanobis" in record.message.lower()
        for record in caplog.records
    )
    assert any(
        record.levelno == logging.INFO and "refused" in record.message.lower()
        for record in caplog.records
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/inference/test_pipeline.py -k "thresholds_are_fully_calibrated or refused_degenerate" -v`
Expected: FAIL — `test_classifier_does_not_warn_when_thresholds_are_fully_calibrated` now fails because the classifier still checks `mahalanobis_p_threshold is None` (which is `False` here, so this one might actually still pass — the real failure is the new `refused_degenerate` test, since nothing currently logs an INFO "refused" line, and today's code would still emit the WARNING for `mahalanobis_p_threshold` since it's `None` in that fixture).

- [ ] **Step 3: Update `_warn_on_uncalibrated_thresholds`**

In `src/inference/classify.py`, replace the method body:

```python
    def _warn_on_uncalibrated_thresholds(self) -> None:
        """resolve_ood_thresholds()'s silent Settings.OOD_* fallback is intentional backward
        compatibility, not something to hide from whoever operates this service -- a model
        that's never been through `evaluate-ood-calibration --write-thresholds` silently
        inherits whichever model Settings.OOD_* happens to be calibrated for. This does not
        fail startup -- an uncalibrated model is still usable, just with potentially
        miscalibrated OOD decisions -- but it must not be silent either. Runs once at
        construction, not per-request. mahalanobis_threshold_status distinguishes "never
        calibrated" (this WARNING) from "calibration ran, degenerate-threshold guard
        correctly refused to persist a value" (a separate, non-actionable INFO line below) --
        collapsing both into one message here is exactly the ambiguity that field exists to
        remove."""
        if self._ood_stats is None:
            return
        uncalibrated = [
            name
            for name, value in (
                ("cosine_threshold", self._ood_stats.cosine_threshold),
                ("knn_distance_threshold", self._ood_stats.knn_distance_threshold),
            )
            if value is None
        ]
        if self._ood_stats.mahalanobis_threshold_status == "not_calibrated":
            uncalibrated.append("mahalanobis_p_threshold")
        if uncalibrated:
            log.warning(
                "ood_stats.npz has no per-model value for %s -- falling back to "
                "Settings.OOD_* (calibrated for a specific model, not necessarily this "
                "one). Run evaluate-ood-calibration --write-thresholds for this model to "
                "silence this.",
                ", ".join(uncalibrated),
            )
        if self._ood_stats.mahalanobis_threshold_status == "refused_degenerate":
            log.info(
                "mahalanobis_p_threshold falls back to "
                "Settings.OOD_MAHALANOBIS_P_THRESHOLD because evaluate-ood-calibration's "
                "degenerate-threshold guard correctly refused to persist a floor-adjacent "
                "value for this model -- expected, no action needed."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/inference/test_pipeline.py -v`
Expected: PASS (all — including `test_classifier_warns_when_thresholds_fall_back_to_settings`, which is unaffected since `_make_stats()`'s default `mahalanobis_threshold_status` is `"not_calibrated"`)

- [ ] **Step 5: Update `CLAUDE.md`**

In the "Follow-up hardening" paragraph (search for `mahalanobis_p_threshold` is `None` because `--write-thresholds`'s degenerate-guard...), add one sentence noting the ambiguity is now resolved by the status field:

```
(3) `resolve_ood_thresholds()`'s Settings fallback was completely silent ... This is deliberately a warning, not a startup failure or a disabled signal: BETO v2's own `mahalanobis_p_threshold` is `None` because `--write-thresholds`'s degenerate-guard correctly refused to persist a floor-adjacent value, not because it was never calibrated -- so failing startup would break a correctly-configured model. **A follow-up (2026-07-13) replaced this prose-only distinction with an explicit `mahalanobis_threshold_status` field (`"not_calibrated"` / `"calibrated"` / `"refused_degenerate"`) on `ClassEmbeddingStats`, so `_warn_on_uncalibrated_thresholds` can log an actionable WARNING only for the genuinely-uncalibrated case and a separate non-actionable INFO line for the expected-refusal case, instead of one WARNING that had to explain both possibilities in its own text.**
```

- [ ] **Step 6: Run full check**

Run: `uv run poe check`
Expected: PASS — lint, format, mypy strict, full test suite (all 3 tasks combined)

- [ ] **Step 7: Commit**

```bash
git add src/inference/classify.py tests/inference/test_pipeline.py CLAUDE.md
git commit -m "fix: distinguish never-calibrated from refused-degenerate Mahalanobis threshold in startup logging"
```

---

## After all tasks: review before pushing

Per explicit instruction: **do not push or update the PR until the user has reviewed the diff.** After Task 3's commit, run `git log --oneline -3` and `git diff origin/task/49-ood-review-remediation..HEAD` (or `git show` per commit) and present the changes for review. Only after explicit go-ahead: `git push` and update PR #43's body with a new follow-up section (matching the style of the two existing follow-up sections already in that PR).
